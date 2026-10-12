"""Resumable original-BF16 high-budget dev evaluation; no model training."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT
import subprocess
import sys
import time

sys.dont_write_bytecode = True
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')

from vlm_precision.evaluate import PATH_SPECS, load_path
from vlm_precision.evaluate import source_hashes as reference_source_hashes
from vlm_probe.evaluate import (
    complete_cache, content_sha, evaluate_batch, lock_file, now,
    package_versions, preflight_rows, private_path, processor_for, read_json,
    read_manifest, repo_path, score_prediction, sha_file, summarize,
    verify_dev_images, write_json,
)

REPO = PROJECT_ROOT
DEFAULT_CONFIG = REPO / 'configs/vlm_joint_bf16_v1.json'
NUMERICAL_PATH = 'original_bf16'
ALLOWED_BUDGETS = (512, 1024)


def source_hashes():
    hashes = reference_source_hashes()
    for path in (Path(__file__), Path(__file__).parent / '__init__.py'):
        hashes[str(path.relative_to(SOURCE_ROOT))] = sha_file(path)
    return hashes


def validate_config(joint, reference, explicit_budgets=None, batch_size=4):
    declared = joint.get('visual_token_budgets', joint.get('budgets', list(ALLOWED_BUDGETS)))
    budgets = list(explicit_budgets or declared)
    if (list(declared) != list(ALLOWED_BUDGETS) or not budgets
            or len(set(budgets)) != len(budgets)
            or any(value not in ALLOWED_BUDGETS for value in budgets)):
        raise ValueError('expected predeclared distinct 512/1024 visual budgets')
    expected = {'numerical_path': NUMERICAL_PATH, 'precision': 'bf16',
                'quantization': 'original', 'vision_precision': 'bf16',
                'split': 'dev', 'batch_size': 4, 'min_pixels': 50176,
                'max_new_tokens': 32, 'do_sample': False,
                'attention_implementation': 'sdpa', 'allow_tf32': False,
                'expected_questions': 1617, 'expected_images': 1024}
    for key, value in expected.items():
        if joint.get(key, value) != value:
            raise ValueError(f'joint configuration differs from fixed contract: {key}')
    roles = joint.get('roles', ['base', 'adapted'])
    if batch_size != 4 or len(roles) != 2 or set(roles) != {'base', 'adapted'}:
        raise ValueError('expected batch4 and base/adapted roles')
    if (reference['precision'] != 'fp32'
            or reference['training_quantization'] != 'nf4_language_only'
            or reference['min_pixels'] != 50176 or reference['max_pixels'] != 256 * 784
            or reference['max_new_tokens'] != 32 or reference['do_sample']):
        raise ValueError('frozen training/reference input contract differs')
    return budgets


def visible_gpu_binding():
    """Resolve the physical UUID without assuming which role uses card 0 or 1."""
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    tokens = [value.strip() for value in visible.split(',') if value.strip()]
    if len(tokens) != 1:
        raise ValueError('worker requires exactly one explicit CUDA_VISIBLE_DEVICES device')
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,uuid,pci.bus_id', '--format=csv,noheader,nounits'],
        check=True, capture_output=True, text=True, timeout=15)
    selected = []
    for line in result.stdout.splitlines():
        index, uuid, pci = [value.strip() for value in line.split(',')]
        if tokens[0] == index or uuid.startswith(tokens[0]):
            selected.append({'physical_index': int(index), 'uuid': uuid, 'pci_bus_id': pci})
    if len(selected) != 1:
        raise ValueError(f'could not uniquely bind visible CUDA device: {visible}')
    return {'visible_devices': visible, **selected[0]}


def model_metadata_binding(model_root, reference):
    """Bind earlier verified weight provenance and recheck inexpensive processor files."""
    path = model_root.parent / 'model_provenance.json'
    provenance = read_json(path)
    if provenance['official_hf_revision'] != reference['model_revision']:
        raise ValueError('model revision differs from frozen reference')
    small = {}
    for name, expected in provenance['files'].items():
        if name.endswith('.safetensors'):
            continue
        actual = sha_file(model_root / name)
        if actual != expected:
            raise ValueError(f'model/processor metadata differs from verified source: {name}')
        small[name] = actual
    return {'model_provenance_sha256': sha_file(path),
            'model_config_sha256': small['config.json'],
            'model_revision': reference['model_revision'],
            'model_weight_provenance_sha256': {
                name: value for name, value in provenance['files'].items()
                if name.endswith('.safetensors')},
            'processor_and_metadata_sha256': small,
            'model_binding_scope': 'prior completed weight verification retained; current small model/processor files rehashed'}


def run_preflight(model, args, reference, common_lock, rows, images, image_root,
                  device, image_token_id):
    budget = 1024
    directory = args.output_root / 'preflight' / args.role
    processor = processor_for(args.model_root, reference, budget)
    selected = preflight_rows(rows, images, processor)
    if len(selected) != 8:
        raise ValueError('expected eight input-only engineering stress examples')
    lock = {**common_lock, 'scope': 'engineering_preflight_not_scientific_result',
            'visual_token_budget': budget, 'max_pixels': budget * 784,
            'selection': 'two longest questions, aspect extremes, largest image and fixed teaching examples; no answer/outcome selection',
            'question_ids': [row['question_id'] for row in selected]}
    lock_file(directory / 'INPUT_LOCK.json', lock)
    result_path = directory / 'PREFLIGHT.json'
    if result_path.exists():
        previous = read_json(result_path)
        if previous.get('input_lock_sha256') != content_sha(lock) or previous.get('status') != 'passed':
            raise ValueError('previous joint preflight binding/status is invalid')
        print('PREFLIGHT REUSED', args.role, NUMERICAL_PATH, flush=True)
        return previous
    import torch
    records, batches = [], []
    with torch.inference_mode():
        for begin in range(0, len(selected), args.batch_size):
            chunk = selected[begin:begin + args.batch_size]
            answers, resource = evaluate_batch(
                model, processor, chunk, images, image_root, device, budget,
                image_token_id, reference['max_new_tokens'], check_logits=True)
            batches.append(resource)
            records.extend({'question_id': row['question_id'], 'image_id': row['image_id'], **answer}
                           for row, answer in zip(chunk, answers))
    result = {'schema_version': 'rarevlm.joint_bf16_preflight.v1', 'status': 'passed',
              'completed_at': now(), 'input_lock_sha256': content_sha(lock),
              'role': args.role, 'numerical_path': NUMERICAL_PATH,
              'visual_token_budget': budget,
              'scope': 'finite last-position logits, greedy generation, no truncation, visual-token contract and memory; no scoring or research-cache reuse',
              'records': records, 'batches': batches,
              'peak_allocated_gib': max(batch['batch_peak_allocated_gib'] for batch in batches),
              'peak_reserved_gib': max(batch['batch_peak_reserved_gib'] for batch in batches)}
    write_json(result_path, result)
    print('PREFLIGHT PASSED', args.role, NUMERICAL_PATH,
          json.dumps({'n': len(records), 'peak_gib': result['peak_allocated_gib']}), flush=True)
    return result


def run_cell(model, args, reference, common_lock, rows, images, image_root,
             device, image_token_id, budget, update_status):
    directory = args.output_root / 'runs' / args.role / f'budget_{budget}'
    lock = {**common_lock, 'scope': 'full_internal_dev_joint_numerical_input_engineering',
            'visual_token_budget': budget, 'max_pixels': budget * 784, 'n': len(rows),
            'question_order_sha256': content_sha([row['question_id'] for row in rows])}
    lock_file(directory / 'INPUT_LOCK.json', lock)
    lock_sha = content_sha(lock)
    cache_path = directory / 'predictions.jsonl'
    items, repaired = complete_cache(cache_path, rows, images, args.batch_size)
    for item in items:
        if (item.get('input_lock_sha256') != lock_sha
                or item.get('numerical_path') != NUMERICAL_PATH
                or item.get('visual_token_budget') != budget
                or item.get('precision') != 'bf16' or item.get('quantization') != 'original'):
            raise ValueError('cached input/budget/numerical-path binding changed')
    evaluation = directory / 'EVALUATION.json'
    if evaluation.exists():
        saved = read_json(evaluation)
        if (len(items) != len(rows) or saved.get('predictions_sha256') != sha_file(cache_path)
                or saved.get('input_lock_sha256') != lock_sha):
            raise ValueError('completed joint cell predictions no longer match evaluation')
        print('CELL REUSED', args.role, budget, flush=True)
        update_status(budget, len(items), 'completed')
        return saved
    import torch
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    processor = processor_for(args.model_root, reference, budget)
    session_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    session = {'session_id': session_id, 'resumed_questions': len(items),
               'cache_repair': repaired, 'started_at': now()}
    begun, start = time.monotonic(), len(items)
    with cache_path.open('a') as stream, torch.inference_mode():
        for begin in range(start, len(rows), args.batch_size):
            chunk = rows[begin:begin + args.batch_size]
            answers, resource = evaluate_batch(
                model, processor, chunk, images, image_root, device, budget,
                image_token_id, reference['max_new_tokens'])
            resource_row = {key: value for key, value in resource.items() if key != 'finite_logits'}
            batch = [{'question_id': row['question_id'], 'image_id': row['image_id'],
                      'question_sha256': content_sha(row['question']),
                      'image_sha256': images[row['image_id']]['sha256'], **answer,
                      'score': score_prediction(answer['prediction'], row, 'textvqa'),
                      **resource_row, 'batch_start': begin, 'session_id': session_id,
                      'timing_warmup': begin == start, 'visual_token_budget': budget,
                      'numerical_path': NUMERICAL_PATH, 'precision': 'bf16',
                      'quantization': 'original', 'input_lock_sha256': lock_sha}
                     for row, answer in zip(chunk, answers)]
            stream.write(''.join(json.dumps(item, ensure_ascii=False, allow_nan=False) + '\n'
                                 for item in batch))
            stream.flush()
            os.fsync(stream.fileno())
            items.extend(batch)
            write_json(directory / 'PROGRESS.json', {
                'schema_version': 'rarevlm.joint_bf16_progress.v1', 'updated_at': now(),
                'status': 'running', 'role': args.role, 'visual_token_budget': budget,
                'numerical_path': NUMERICAL_PATH, 'completed_questions': len(items),
                'total_questions': len(rows), 'new_questions': len(items) - start,
                'session_elapsed_seconds': time.monotonic() - begun,
                'last_batch': resource, 'session_id': session_id})
            update_status(budget, len(items), 'running')
            if begin == start or ((begin - start) // args.batch_size + 1) % 25 == 0:
                print('GENERATED', args.role, budget, len(items), '/', len(rows), flush=True)
    session.update(completed_at=now(), new_questions=len(items) - start,
                   wall_seconds=time.monotonic() - begun)
    result = summarize(items, lock, cache_path, session)
    result.update(schema_version='rarevlm.joint_bf16_cell.v1', input_lock_sha256=lock_sha,
                  scope='single-seed fixed-checkpoint internal-dev joint BF16/visual-budget engineering; no new training, official-val selection, equivalence or production-throughput claim')
    write_json(evaluation, result)
    write_json(directory / 'PROGRESS.json', {
        'schema_version': 'rarevlm.joint_bf16_progress.v1', 'status': 'completed',
        'updated_at': now(), 'completed_questions': len(items), 'total_questions': len(rows),
        'role': args.role, 'visual_token_budget': budget, 'numerical_path': NUMERICAL_PATH})
    update_status(budget, len(items), 'completed')
    print('CELL COMPLETE', args.role, budget, result['score_percent'], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', choices=('base', 'adapted'), required=True)
    parser.add_argument('--asset-root', type=Path, required=True)
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--adapter-dir', type=Path)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--budgets', type=int, nargs='+')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    for name in ('output_root', 'asset_root', 'model_root', 'config'):
        setattr(args, name, private_path(getattr(args, name)))
    joint = read_json(args.config)
    reference_path = private_path(repo_path(joint['reference_config']))
    reference = read_json(reference_path)
    budgets = validate_config(joint, reference, args.budgets, args.batch_size)
    if (args.role == 'adapted') != (args.adapter_dir is not None):
        raise ValueError('only adapted role must receive --adapter-dir')
    adapter_files, train_binding = None, None
    if args.adapter_dir is not None:
        args.adapter_dir = private_path(args.adapter_dir)
        fixed = joint.get('fixed_adapter_dir')
        if not fixed or args.adapter_dir != repo_path(fixed).resolve():
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
    hardware = {**visible_gpu_binding(), 'name': props.name,
                'total_memory_bytes': props.total_memory,
                'compute_capability': [props.major, props.minor],
                'torch_cuda_version': torch.version.cuda}
    common_lock = {
        'schema_version': 'rarevlm.joint_bf16_input.v1', 'task': 'textvqa', 'split': 'dev',
        'role': args.role, 'joint_config_sha256': sha_file(args.config),
        'reference_config_sha256': sha_file(reference_path),
        'manifest_sha256': manifest['manifest_sha256'], 'manifest_file_sha256': sha_file(manifest_path),
        **model_metadata_binding(args.model_root, reference),
        'adapter_files_sha256': adapter_files, 'training_binding': train_binding,
        'source_sha256': source_hashes(), 'packages': package_versions(), 'hardware': hardware,
        'batch_size': args.batch_size, 'numerical_path': NUMERICAL_PATH, **PATH_SPECS[NUMERICAL_PATH],
        'attention_implementation': 'sdpa', 'tf32_matmul': False, 'tf32_cudnn': False,
        'min_pixels': reference['min_pixels'], 'max_new_tokens': 32, 'do_sample': False,
        'head_last_position_only': True, 'autocast': False, 'adapter_autocast_dtype': False,
        'explicit_lora_bf16_conversion': args.adapter_dir is not None,
        'input': 'image and question only; no reference answers, targets or OCR tokens',
        'input_truncation': False, 'statistical_unit': 'image cluster; retain same-image questions',
        'adapter_origin': 'same frozen NF4/FP32-trained qv_r8 seed42 checkpoint; not retrained at BF16 or larger visual budgets' if adapter_files else None,
        'numerical_scope': 'joint original-weight BF16/input-budget engineering path; independent kernels and rounded adapter; not a pure-dtype causal comparison',
        'protobuf_implementation': os.environ.get('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION')}
    status_path = args.output_root / 'runs' / args.role / 'STATUS.json'
    status = {'schema_version': 'rarevlm.joint_bf16_worker_status.v1', 'role': args.role,
              'budgets': budgets, 'numerical_path': NUMERICAL_PATH, 'phase': 'starting',
              'started_at': now(), 'hardware': hardware,
              'cells': {str(t): {'status': 'pending', 'completed_questions': 0,
                                'total_questions': len(rows)} for t in budgets}}
    def update_status(budget=None, count=None, state=None):
        if budget is not None:
            status['current_budget'] = budget
            status['cells'][str(budget)].update(status=state, completed_questions=count)
        status['updated_at'] = now()
        write_json(status_path, status)
    update_status()
    model, hook = None, None
    try:
        status['image_validation'] = verify_dev_images(rows, images, manifest_path.parent)
        status['phase'] = 'loading_model'
        update_status()
        begun = time.monotonic()
        model, dtypes = load_path(args.model_root, reference, device, NUMERICAL_PATH, args.adapter_dir)
        status['model_load_seconds'] = time.monotonic() - begun
        status['actual_parameter_dtypes'] = dtypes
        common_lock['actual_parameter_dtypes'] = dtypes
        image_token_id = int(model.config.image_token_id)
        if image_token_id != 151655:
            raise ValueError('unexpected model image-token id')
        hook = model.get_output_embeddings().register_forward_pre_hook(
            lambda module, items: (items[0][:, -1:, :],))
        status['phase'] = 'preflight'
        update_status()
        run_preflight(model, args, reference, common_lock, rows, images,
                      manifest_path.parent, device, image_token_id)
        if not args.preflight_only:
            status['phase'] = 'evaluating'
            update_status()
            for budget in budgets:
                run_cell(model, args, reference, common_lock, rows, images,
                         manifest_path.parent, device, image_token_id, budget, update_status)
        status['phase'] = 'preflight_completed' if args.preflight_only else 'completed'
        status['completed_at'] = now()
        update_status()
    except Exception as error:
        status.update(phase='failed', error_type=type(error).__name__, error=str(error))
        update_status()
        raise
    finally:
        if hook is not None:
            hook.remove()
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
