"""Resumable fixed-dev precision bridges, isolated from running budget probes."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import sys
import time

# Reuse reference code without writing bytecode into frozen source trees. The
# current container's old ONNX protobuf stubs require this existing workaround.
sys.dont_write_bytecode = True
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')

from vlm_probe.evaluate import (
    complete_cache, content_sha, evaluate_batch, lock_file, now,
    package_versions, preflight_rows, private_path, processor_for, read_json,
    read_manifest, repo_path, score_prediction, sha_file, summarize,
    verify_dev_images, write_json,
)
from vlm_probe.evaluate import source_hashes as reference_source_hashes
from vlm_deep.model import load as reference_load

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / 'configs/vlm_precision_bridge_v1.json'
VISUAL_BUDGET = 256
PATH_SPECS = {
    'nf4_fp32': {'precision': 'fp32', 'vision_precision': 'fp32',
                 'quantization': 'nf4_language_only'},
    'original_fp32': {'precision': 'fp32', 'vision_precision': 'fp32',
                      'quantization': 'original'},
    'original_bf16': {'precision': 'bf16', 'vision_precision': 'bf16',
                      'quantization': 'original'},
}


def source_hashes():
    hashes = reference_source_hashes()
    for path in (Path(__file__), Path(__file__).parent / '__init__.py'):
        hashes[str(path.relative_to(REPO))] = sha_file(path)
    return hashes


def parameter_dtypes(model):
    """Record actual stored dtypes, rather than infer them from requested dtype."""
    counts, elements, lora_counts, lora_elements = Counter(), Counter(), Counter(), Counter()
    quantized = 0
    for name, parameter in model.named_parameters():
        dtype = str(parameter.dtype).removeprefix('torch.')
        counts[dtype] += 1
        elements[dtype] += parameter.numel()
        quantized += int(parameter.__class__.__name__ == 'Params4bit')
        if 'lora_' in name:
            lora_counts[dtype] += 1
            lora_elements[dtype] += parameter.numel()
    buffers = Counter(str(buffer.dtype).removeprefix('torch.') for buffer in model.buffers())
    return {'parameter_tensors_by_dtype': dict(sorted(counts.items())),
            'parameter_stored_elements_by_dtype': dict(sorted(elements.items())),
            'lora_tensors_by_dtype': dict(sorted(lora_counts.items())),
            'lora_elements_by_dtype': dict(sorted(lora_elements.items())),
            'buffer_tensors_by_dtype': dict(sorted(buffers.items())),
            'params4bit_tensors': quantized,
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'note': 'NF4 uint8 stored elements are packed and are not an original parameter count'}


def load_path(model_root, reference_config, device, path_name, adapter_dir):
    """FP32 paths preserve the frozen loader; BF16 is an independent original-weight path."""
    import torch
    spec = PATH_SPECS[path_name]
    if path_name != 'original_bf16':
        model, _ = reference_load(model_root, reference_config, device,
                                  adapter_dir=adapter_dir, quantization=spec['quantization'])
    else:
        if not torch.cuda.is_bf16_supported():
            raise ValueError('this hardware/runtime does not support the declared BF16 path')
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_root), local_files_only=True, torch_dtype=torch.bfloat16,
            quantization_config=None, attn_implementation='sdpa', low_cpu_mem_usage=True,
            device_map={'': str(device)})
        if adapter_dir is not None:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False,
                                              autocast_adapter_dtype=False)
            # Avoid PEFT-version-dependent adapter promotion. The same stored
            # checkpoint is deliberately rounded to BF16 for this engineering path.
            for name, parameter in model.named_parameters():
                if 'lora_' in name:
                    parameter.data = parameter.data.to(dtype=torch.bfloat16)
    model.requires_grad_(False)
    model.eval()
    model.config.use_cache = True
    model.generation_config.temperature = 1.0
    model.generation_config.top_p = 1.0
    model.generation_config.top_k = 50
    expected = torch.bfloat16 if spec['precision'] == 'bf16' else torch.float32
    for name, parameter in model.named_parameters():
        if parameter.is_floating_point() and parameter.dtype != expected:
            raise ValueError(f'{path_name} floating weight dtype differs: {name}: {parameter.dtype}')
    dtypes = parameter_dtypes(model)
    if dtypes['trainable_parameters']:
        raise ValueError('evaluation path unexpectedly has trainable parameters')
    if spec['quantization'] == 'original' and dtypes['params4bit_tensors']:
        raise ValueError('original-weight path contains 4-bit parameters')
    if spec['quantization'] == 'nf4_language_only' and not dtypes['params4bit_tensors']:
        raise ValueError('NF4 path has no 4-bit parameters')
    return model, dtypes


def run_preflight(model, args, reference, common_lock, rows, images, image_root,
                  device, image_token_id, path_name):
    directory = args.output_root / 'preflight' / args.role / path_name
    processor = processor_for(args.model_root, reference, VISUAL_BUDGET)
    selected = preflight_rows(rows, images, processor)
    lock = {**common_lock, 'scope': 'engineering_preflight_not_scientific_result',
            'selection': 'fixed eight input-only stress examples: longest questions, aspect extremes, largest image and teaching IDs; no answer/outcome selection',
            'question_ids': [row['question_id'] for row in selected]}
    lock_file(directory / 'INPUT_LOCK.json', lock)
    result_path = directory / 'PREFLIGHT.json'
    if result_path.exists():
        previous = read_json(result_path)
        if previous.get('input_lock_sha256') != content_sha(lock) or previous.get('status') != 'passed':
            raise ValueError('previous precision preflight binding/status is invalid')
        print('PREFLIGHT REUSED', args.role, path_name, flush=True)
        return previous
    import torch
    records, batches = [], []
    with torch.inference_mode():
        for begin in range(0, len(selected), args.batch_size):
            chunk = selected[begin:begin + args.batch_size]
            answers, resources = evaluate_batch(
                model, processor, chunk, images, image_root, device, VISUAL_BUDGET,
                image_token_id, reference['max_new_tokens'], check_logits=True)
            batches.append(resources)
            records.extend({'question_id': row['question_id'], 'image_id': row['image_id'], **answer}
                           for row, answer in zip(chunk, answers))
    result = {'schema_version': 'rarevlm.precision_preflight.v1', 'status': 'passed',
              'completed_at': now(), 'input_lock_sha256': content_sha(lock),
              'numerical_path': path_name, 'role': args.role,
              'scope': 'finite last-position logits/greedy generation/token contract/memory; no scores or scientific-cell predictions',
              'records': records, 'batches': batches,
              'peak_allocated_gib': max(batch['batch_peak_allocated_gib'] for batch in batches),
              'peak_reserved_gib': max(batch['batch_peak_reserved_gib'] for batch in batches)}
    write_json(result_path, result)
    print('PREFLIGHT PASSED', args.role, path_name,
          json.dumps({'n': len(records), 'peak_gib': result['peak_allocated_gib']}), flush=True)
    return result


def run_cell(model, args, reference, common_lock, rows, images, image_root,
             device, image_token_id, path_name, update_status):
    directory = args.output_root / 'runs' / args.role / path_name
    lock = {**common_lock, 'scope': 'full_internal_dev_fixed_input_numerical_bridge',
            'n': len(rows), 'question_order_sha256': content_sha([row['question_id'] for row in rows])}
    lock_file(directory / 'INPUT_LOCK.json', lock)
    lock_sha = content_sha(lock)
    cache_path = directory / 'predictions.jsonl'
    items, repaired = complete_cache(cache_path, rows, images, args.batch_size)
    for item in items:
        if item.get('input_lock_sha256') != lock_sha or item.get('numerical_path') != path_name:
            raise ValueError('cached numerical-path/input-lock binding changed')
    evaluation = directory / 'EVALUATION.json'
    if evaluation.exists():
        saved = read_json(evaluation)
        if (len(items) != len(rows) or saved.get('predictions_sha256') != sha_file(cache_path)
                or saved.get('input_lock_sha256') != lock_sha):
            raise ValueError('completed precision predictions no longer match evaluation')
        print('CELL REUSED', args.role, path_name, flush=True)
        update_status(path_name, len(items), 'completed')
        return saved
    import torch
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    processor = processor_for(args.model_root, reference, VISUAL_BUDGET)
    session_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    session = {'session_id': session_id, 'resumed_questions': len(items),
               'cache_repair': repaired, 'started_at': now()}
    begun = time.monotonic()
    start = len(items)
    with cache_path.open('a') as stream, torch.inference_mode():
        for begin in range(start, len(rows), args.batch_size):
            chunk = rows[begin:begin + args.batch_size]
            answers, resource = evaluate_batch(
                model, processor, chunk, images, image_root, device, VISUAL_BUDGET,
                image_token_id, reference['max_new_tokens'])
            resource_row = {key: value for key, value in resource.items() if key != 'finite_logits'}
            batch = [{'question_id': row['question_id'], 'image_id': row['image_id'],
                      'question_sha256': content_sha(row['question']),
                      'image_sha256': images[row['image_id']]['sha256'], **answer,
                      'score': score_prediction(answer['prediction'], row, 'textvqa'),
                      **resource_row, 'batch_start': begin, 'session_id': session_id,
                      'timing_warmup': begin == start, 'visual_token_budget': VISUAL_BUDGET,
                      'numerical_path': path_name, 'precision': common_lock['precision'],
                      'quantization': common_lock['quantization'], 'input_lock_sha256': lock_sha}
                     for row, answer in zip(chunk, answers)]
            stream.write(''.join(json.dumps(item, ensure_ascii=False, allow_nan=False) + '\n'
                                 for item in batch))
            stream.flush()
            os.fsync(stream.fileno())
            items.extend(batch)
            progress = {'schema_version': 'rarevlm.precision_progress.v1', 'updated_at': now(),
                        'status': 'running', 'role': args.role, 'numerical_path': path_name,
                        'completed_questions': len(items), 'total_questions': len(rows),
                        'new_questions': len(items) - start,
                        'session_elapsed_seconds': time.monotonic() - begun,
                        'last_batch': resource, 'session_id': session_id}
            write_json(directory / 'PROGRESS.json', progress)
            update_status(path_name, len(items), 'running')
            if begin == start or ((begin - start) // args.batch_size + 1) % 25 == 0:
                print('GENERATED', args.role, path_name, len(items), '/', len(rows), flush=True)
    session.update({'completed_at': now(), 'new_questions': len(items) - start,
                    'wall_seconds': time.monotonic() - begun})
    result = summarize(items, lock, cache_path, session)
    result.update(schema_version='rarevlm.precision_cell.v1', input_lock_sha256=lock_sha,
                  scope='fixed internal-dev numerical engineering bridge; one NF4-trained adapter; no original/BF16 retraining, official-val selection or pure-dtype causal claim')
    write_json(evaluation, result)
    write_json(directory / 'PROGRESS.json', {'schema_version': 'rarevlm.precision_progress.v1',
               'status': 'completed', 'updated_at': now(), 'completed_questions': len(items),
               'total_questions': len(rows), 'role': args.role, 'numerical_path': path_name})
    update_status(path_name, len(items), 'completed')
    print('CELL COMPLETE', args.role, path_name, result['score_percent'], flush=True)
    return result


def requested_paths(config, explicit=None):
    configured = config.get('paths', config.get('numerical_paths', list(PATH_SPECS)))
    names = list(configured)
    paths = explicit or names
    if (not paths or len(set(paths)) != len(paths) or any(path not in PATH_SPECS for path in paths)
            or any(path not in names for path in paths)):
        raise ValueError('expected distinct predeclared numerical paths')
    if isinstance(configured, dict):
        for name, spec in configured.items():
            for key in ('precision', 'vision_precision', 'quantization'):
                if key in spec and spec[key] != PATH_SPECS[name][key]:
                    raise ValueError(f'configured numerical path differs: {name}.{key}')
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', choices=('base', 'adapted'), required=True)
    parser.add_argument('--asset-root', type=Path, required=True)
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--adapter-dir', type=Path)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--paths', choices=tuple(PATH_SPECS), nargs='+')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    args.output_root = private_path(args.output_root)
    bridge = read_json(args.config)
    reference_path = repo_path(bridge['reference_config'])
    reference = read_json(reference_path)
    paths = requested_paths(bridge, args.paths)
    if (args.batch_size != 4 or bridge.get('batch_size', 4) != 4
            or bridge.get('visual_token_budget', 256) != VISUAL_BUDGET
            or bridge.get('split', 'dev') != 'dev'):
        raise ValueError('bridge requires batch4, fixed256 visual budget and frozen internal dev')
    if (reference['precision'] != 'fp32' or reference['training_quantization'] != 'nf4_language_only'
            or reference['min_pixels'] != 50176 or reference['max_pixels'] != 256 * 784
            or reference['max_new_tokens'] != 32 or reference['do_sample']):
        raise ValueError('reference numerical/input/generation contract differs')
    if (args.role == 'adapted') != (args.adapter_dir is not None):
        raise ValueError('only adapted role must receive --adapter-dir')
    adapter_files, train_binding = None, None
    if args.adapter_dir is not None:
        args.adapter_dir = args.adapter_dir.resolve()
        fixed = bridge.get('fixed_adapter_dir')
        if fixed and args.adapter_dir != repo_path(fixed).resolve():
            raise ValueError('adapter differs from predeclared checkpoint')
        train_path = args.adapter_dir.parent / 'TRAIN_RESULT.json'
        train = read_json(train_path)
        if (train['method'] != 'qv_r8' or train['seed'] != 42 or train['split'] != 'train_all'
                or train['completed_updates'] != train['planned_updates']):
            raise ValueError('expected completed stage2 qv_r8 seed42 checkpoint')
        adapter_files = {path.name: sha_file(path) for path in sorted(args.adapter_dir.iterdir()) if path.is_file()}
        if adapter_files != train['adapter_files_sha256']:
            raise ValueError('adapter differs from training completion binding')
        train_binding = {'path': str(train_path), 'sha256': sha_file(train_path), 'method': 'qv_r8', 'seed': 42}
    manifest_path = args.asset_root / 'prepared/manifest.json'
    manifest = read_manifest(manifest_path)
    rows, images = manifest['splits']['dev'], manifest['images']
    if len(rows) != 1617 or len({row['image_id'] for row in rows}) != 1024:
        raise ValueError('expected frozen internal dev with 1617 QA / 1024 images')
    import torch
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('one independent worker per visible GPU; no DDP')
    if not torch.cuda.is_available():
        raise RuntimeError('an A100 CUDA device is required')
    torch.cuda.set_device(0)
    device = torch.device('cuda', 0)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    props = torch.cuda.get_device_properties(device)
    if 'A100' not in props.name or props.total_memory < 70 * 2**30:
        raise ValueError('this bridge is declared for A100 80GB')
    hardware = {'name': props.name, 'total_memory_bytes': props.total_memory,
                'compute_capability': [props.major, props.minor],
                'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'torch_cuda_version': torch.version.cuda}
    shared_lock = {'schema_version': 'rarevlm.precision_input.v1', 'task': 'textvqa', 'split': 'dev',
                   'role': args.role, 'bridge_config_sha256': sha_file(args.config),
                   'reference_config_sha256': sha_file(reference_path),
                   'manifest_sha256': manifest['manifest_sha256'], 'manifest_file_sha256': sha_file(manifest_path),
                   'model_provenance_sha256': sha_file(args.model_root.parent / 'model_provenance.json'),
                   'model_config_sha256': sha_file(args.model_root / 'config.json'),
                   'model_revision': reference['model_revision'], 'adapter_files_sha256': adapter_files,
                   'training_binding': train_binding, 'source_sha256': source_hashes(),
                   'packages': package_versions(), 'hardware': hardware, 'batch_size': args.batch_size,
                   'attention_implementation': 'sdpa', 'tf32_matmul': False, 'tf32_cudnn': False,
                   'min_pixels': reference['min_pixels'], 'max_pixels': VISUAL_BUDGET * 784,
                   'visual_token_budget': VISUAL_BUDGET, 'max_new_tokens': 32, 'do_sample': False,
                   'head_last_position_only': True, 'autocast': False,
                   'input': 'image and question only; no reference answers, targets or OCR tokens',
                   'input_truncation': False, 'statistical_unit': 'image cluster; retain same-image questions',
                   'adapter_origin': 'same frozen NF4/FP32-trained qv_r8 seed42 checkpoint; not retrained per numerical path' if adapter_files else None,
                   'numerical_scope': 'separate full-weight FP32/BF16 versus quantized FP32 engineering paths; not claimed pure dtype/quantization effects',
                   'protobuf_implementation': os.environ.get('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION')}
    status_path = args.output_root / 'runs' / args.role / 'STATUS.json'
    status = {'schema_version': 'rarevlm.precision_worker_status.v1', 'role': args.role,
              'paths': paths, 'phase': 'starting', 'started_at': now(), 'hardware': hardware,
              'cells': {path: {'status': 'pending', 'completed_questions': 0,
                               'total_questions': len(rows)} for path in paths}}
    def update_status(path_name=None, count=None, state=None):
        if path_name is not None:
            status['current_path'] = path_name
            status['cells'][path_name].update(status=state, completed_questions=count)
        status['updated_at'] = now()
        write_json(status_path, status)
    update_status()
    try:
        status['image_validation'] = verify_dev_images(rows, images, manifest_path.parent)
        for path_name in paths:
            model, hook = None, None
            try:
                status.update(phase='loading_model', current_path=path_name)
                update_status()
                begun = time.monotonic()
                model, dtypes = load_path(args.model_root, reference, device, path_name, args.adapter_dir)
                status['current_model_load_seconds'] = time.monotonic() - begun
                status['actual_parameter_dtypes'] = dtypes
                lock = {**shared_lock, 'numerical_path': path_name, **PATH_SPECS[path_name],
                        'actual_parameter_dtypes': dtypes,
                        'adapter_autocast_dtype': False if path_name == 'original_bf16' else 'reference_loader_default',
                        'explicit_lora_bf16_conversion': path_name == 'original_bf16' and args.adapter_dir is not None}
                image_token_id = int(model.config.image_token_id)
                if image_token_id != 151655:
                    raise ValueError('unexpected model image-token id')
                hook = model.get_output_embeddings().register_forward_pre_hook(
                    lambda module, items: (items[0][:, -1:, :],))
                status['phase'] = 'preflight'
                update_status()
                run_preflight(model, args, reference, lock, rows, images,
                              manifest_path.parent, device, image_token_id, path_name)
                if not args.preflight_only:
                    status['phase'] = 'evaluating'
                    update_status()
                    run_cell(model, args, reference, lock, rows, images, manifest_path.parent,
                             device, image_token_id, path_name, update_status)
                else:
                    update_status(path_name, 0, 'preflight_completed')
            finally:
                if hook is not None:
                    hook.remove()
                del model
                gc.collect()
                torch.cuda.empty_cache()
        status['phase'] = 'preflight_completed' if args.preflight_only else 'completed'
        status['completed_at'] = now()
        update_status()
    except Exception as error:
        status.update(phase='failed', error_type=type(error).__name__, error=str(error))
        update_status()
        raise


if __name__ == '__main__':
    main()
