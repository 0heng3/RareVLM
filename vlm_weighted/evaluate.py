"""Fixed original-BF16/1024 internal-dev evaluation of newly trained SFT conditions."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')

from vlm_deep.evaluate import aggregate, score_prediction
from vlm_formal.evaluate import complete_cache, question_prompt_sha, stress_rows, summarize_task
from vlm_formal.evaluate import source_hashes as reference_source_hashes
from vlm_joint.evaluate import model_metadata_binding, visible_gpu_binding
from vlm_precision.evaluate import PATH_SPECS, load_path
from vlm_probe.evaluate import (content_sha, evaluate_batch, lock_file, now, package_versions,
    private_path, processor_for, read_json, read_manifest, repo_path, sha_file, verify_dev_images, write_json)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / 'configs/vlm_weighted_sft_v1.json'
CONDITIONS = ('uniform', 'frequency')
BUDGET = 1024
METRIC = 'TextVQA leave-one-out consensus percent'


def model_key(condition, seed):
    return f'{condition}_seed{seed}'


def source_hashes():
    hashes = reference_source_hashes()
    for name in ('vlm_weighted/evaluate.py', 'vlm_weighted/__init__.py'):
        path = REPO / name
        if path.exists():
            hashes[name] = sha_file(path)
    return hashes


def validate_config(config, reference, batch_size=4):
    evaluation = config.get('evaluation', config.get('eval', config))
    expected = {'numerical_path': 'original_bf16', 'precision': 'bf16', 'quantization': 'original',
        'vision_precision': 'bf16', 'visual_token_budget': 1024, 'batch_size': 4,
        'min_pixels': 50176, 'max_pixels': 802816, 'max_new_tokens': 32, 'do_sample': False,
        'attention_implementation': 'sdpa', 'allow_tf32': False, 'input_truncation': False,
        'split': 'dev', 'expected_questions': 1617, 'expected_images': 1024}
    for key, value in expected.items():
        if evaluation.get(key, value) != value:
            raise ValueError(f'weighted dev evaluation contract differs: {key}')
    for key, value in (('questions', 1617), ('images', 1024)):
        if evaluation.get(key, value) != value:
            raise ValueError(f'weighted dev count differs: {key}')
    if batch_size != 4:
        raise ValueError('weighted dev requires batch4')
    if (reference['precision'] != 'fp32' or reference['training_quantization'] != 'nf4_language_only'
        or reference['min_pixels'] != 50176 or reference['max_pixels'] != 256 * 784
        or reference['max_new_tokens'] != 32 or reference['do_sample']):
        raise ValueError('frozen reference contract differs')
    return evaluation


def adapter_binding(args):
    """Reject historical adapter identities: the causal baseline is newly trained."""
    adapter = private_path(args.adapter_dir)
    train_path = adapter.parent / 'TRAIN_RESULT.json'
    train = read_json(train_path)
    expected = {'condition': args.condition, 'seed': args.seed, 'split': 'train_all',
        'planned_updates': 2062, 'completed_updates': 2062, 'actual_questions_seen': 32985,
        'frozen_parameters_unchanged': True}
    for key, value in expected.items():
        if train.get(key) != value:
            raise ValueError(f'new weighted-SFT final checkpoint differs: {key}')
    if train.get('config_sha256') != content_sha(read_json(args.config)):
        raise ValueError('new training scientific configuration differs')
    if train.get('method') != 'qv_r8':
        raise ValueError('expected the new final q/v rank8 adapter')
    if train.get('model_provenance_sha256') != sha_file(args.model_root.parent / 'model_provenance.json'):
        raise ValueError('adapter model provenance differs')
    files = {p.name: sha_file(p) for p in sorted(adapter.iterdir()) if p.is_file()}
    if not files or train.get('adapter_files_sha256') != files:
        raise ValueError('new adapter bytes differ from completed training')
    # Bind the entire new result, including loss/initialization/order/checkpoint lineage.
    binding = {'path': str(train_path), 'sha256': sha_file(train_path),
        **{key: train[key] for key in expected},
        'training_numerical_path': 'nf4_language_only_fp32', 'training_visual_token_budget': 256,
        'loss': train.get('loss', train.get('loss_condition', args.condition)),
        'config_sha256': train.get('config_sha256'), 'source_sha256': train.get('source_sha256'),
        'gpu_name': train.get('gpu_name'), 'hardware': train.get('hardware'),
        'runtime_layout': train.get('runtime_layout'),
        'initial_trainable_tensor_sha256': train.get('initial_trainable_tensor_sha256'),
        'initialization_contract_sha256': train.get('initialization_contract_sha256'),
        'global_question_order_sha256': train.get('global_question_order_sha256'),
        'checkpoint_selection': 'fixed completed final epoch; no official-val selection',
        'scope': 'new controlled SFT condition; not a reused historical stage2 checkpoint'}
    return files, binding


def dev_data(asset_root):
    path = asset_root / 'prepared/manifest.json'
    manifest = read_manifest(path)
    rows = manifest['splits']['dev']
    if len(rows) != 1617 or len({r['image_id'] for r in rows}) != 1024:
        raise ValueError('expected frozen full internal dev:1617QA/1024images')
    if any(len(r['answers']) != 10 for r in rows):
        raise ValueError('all ten TextVQA references required')
    return rows, manifest['images'], path.parent, {
        'task': 'textvqa', 'split': 'dev', 'source_split': 'frozen internal training-side dev',
        'manifest_sha256': manifest['manifest_sha256'], 'manifest_file_sha256': sha_file(path),
        'manifest_path': str(path), 'expected_questions': 1617, 'expected_images': 1024,
        'n': 1617, 'metric': METRIC,
        'question_order_sha256': content_sha([r['question_id'] for r in rows]),
        'question_prompt_order_sha256': content_sha([question_prompt_sha(r) for r in rows]),
        'pretraining_overlap': 'unknown'}


def run_preflight(model, processor, args, lock, rows, images, image_root, device, image_token_id):
    selected = stress_rows(rows, images, processor)
    directory = args.output_root / 'preflight' / args.condition / f'seed{args.seed}'
    prelock = {**lock, 'scope': 'engineering_preflight_not_scientific_result',
        'selection': 'eight input-only stress examples; longest prompts, aspect extremes, largest raw image',
        'question_ids': [r['question_id'] for r in selected]}
    lock_file(directory / 'INPUT_LOCK.json', prelock)
    result_path = directory / 'PREFLIGHT.json'
    if result_path.exists():
        previous = read_json(result_path)
        if previous.get('input_lock_sha256') != content_sha(prelock) or previous.get('status') != 'passed':
            raise ValueError('preflight lock/runtime differs')
        return previous
    import torch
    records, batches = [], []
    with torch.inference_mode():
        for start in range(0, len(selected), args.batch_size):
            chunk = selected[start:start+args.batch_size]
            answers, resources = evaluate_batch(model, processor, chunk, images, image_root,
                device, BUDGET, image_token_id, 32, check_logits=True)
            batches.append(resources)
            records.extend({'question_id': row['question_id'], 'image_id': row['image_id'], **answer}
                for row, answer in zip(chunk, answers))
    result = {'schema_version': 'rarevlm.weighted_sft.preflight.v1', 'status': 'passed',
        'completed_at': now(), 'input_lock_sha256': content_sha(prelock),
        'condition': args.condition, 'seed': args.seed, 'records': records, 'batches': batches,
        'peak_allocated_gib': max(r['batch_peak_allocated_gib'] for r in batches),
        'peak_reserved_gib': max(r['batch_peak_reserved_gib'] for r in batches),
        'scope': 'finite last-position logits and label-free greedy generation; no scored QA/cache reuse'}
    write_json(result_path, result)
    print('PREFLIGHT PASSED', args.model_key, result['peak_allocated_gib'], flush=True)
    return result


def run_cell(model, processor, args, lock, rows, images, image_root, device, image_token_id, update):
    directory = args.output_root / 'runs' / args.condition / f'seed{args.seed}' / 'dev'
    lock_file(directory / 'INPUT_LOCK.json', lock)
    path = directory / 'predictions.jsonl'
    items, repaired = complete_cache(path, rows, images, args.batch_size, 'textvqa', lock)
    result_path = directory / 'EVALUATION.json'
    if result_path.exists():
        result = read_json(result_path)
        value, _ = aggregate(items, 'textvqa')
        if (len(items) != len(rows) or result.get('predictions_sha256') != sha_file(path)
            or result.get('input_lock_sha256') != content_sha(lock)
            or not math.isclose(value, result['score_percent'], abs_tol=1e-12, rel_tol=0)):
            raise ValueError('completed dev result binding/score differs')
        update(len(items), 'completed')
        return result
    import torch
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    start, begun = len(items), time.monotonic()
    session_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    session = {'session_id': session_id, 'resumed_questions': start, 'cache_repair': repaired, 'started_at': now()}
    with path.open('a') as stream, torch.inference_mode():
        for begin in range(start, len(rows), args.batch_size):
            chunk = rows[begin:begin+args.batch_size]
            answers, resource = evaluate_batch(model, processor, chunk, images, image_root,
                device, BUDGET, image_token_id, 32)
            batch = []
            for row, answer in zip(chunk, answers):
                batch.append({'question_id': row['question_id'], 'image_id': row['image_id'],
                    'question_sha256': content_sha(row['question']), 'question_prompt_sha256': question_prompt_sha(row),
                    'image_sha256': images[row['image_id']]['sha256'], **answer,
                    'score': score_prediction(answer['prediction'], row, 'textvqa'),
                    'task': 'textvqa', 'subcategory': '', 'category': '',
                    'model_key': args.model_key, 'condition': args.condition, 'seed': args.seed, 'metric': METRIC,
                    'model_revision': lock['model_revision'], 'adapter_files_sha256': lock['adapter_files_sha256'],
                    'source_binding_sha256': content_sha(lock['source_sha256']),
                    'reference_score_source': 'frozen dev references used only after label-free generation',
                    **{key: value for key, value in resource.items() if key != 'finite_logits'},
                    'batch_start': begin, 'session_id': session_id, 'timing_warmup': begin == start,
                    'visual_token_budget': BUDGET, 'numerical_path': 'original_bf16',
                    'precision': 'bf16', 'quantization': 'original', 'input_lock_sha256': content_sha(lock)})
            stream.write(''.join(json.dumps(item, ensure_ascii=False, allow_nan=False)+'\n' for item in batch))
            stream.flush()
            os.fsync(stream.fileno())
            items.extend(batch)
            elapsed = time.monotonic() - begun
            write_json(directory / 'PROGRESS.json', {'schema_version': 'rarevlm.weighted_sft.eval_progress.v1',
                'status': 'running', 'updated_at': now(), 'condition': args.condition, 'seed': args.seed,
                'completed_questions': len(items), 'total_questions': len(rows),
                'new_questions': len(items)-start, 'session_elapsed_seconds': elapsed,
                'remaining_estimate_seconds': (len(rows)-len(items))*elapsed/(len(items)-start),
                'session_id': session_id, 'last_batch': resource})
            update(len(items), 'running')
            if begin == start or ((begin-start)//args.batch_size+1)%25 == 0:
                print('GENERATED', args.model_key, len(items), '/', len(rows), flush=True)
    session.update(completed_at=now(), new_questions=len(items)-start, wall_seconds=time.monotonic()-begun)
    result = summarize_task(items, lock, path, session)
    result.update(schema_version='rarevlm.weighted_sft.dev_cell.v1', condition=args.condition, seed=args.seed,
        scope='new full-data controlled SFT final checkpoint; fixed original-BF16/1024 internal dev; no official-val selection')
    write_json(result_path, result)
    write_json(directory / 'PROGRESS.json', {'schema_version': 'rarevlm.weighted_sft.eval_progress.v1',
        'status': 'completed', 'updated_at': now(), 'completed_questions': len(items),
        'total_questions': len(rows), 'condition': args.condition, 'seed': args.seed})
    update(len(items), 'completed')
    print('CELL COMPLETE', args.model_key, result['score_percent'], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition', choices=CONDITIONS, required=True)
    parser.add_argument('--seed', type=int, choices=(0, 42, 200), default=42)
    parser.add_argument('--asset-root', type=Path, required=True)
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--adapter-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    for name in ('asset_root', 'model_root', 'output_root', 'config', 'adapter_dir'):
        setattr(args, name, private_path(getattr(args, name)))
    args.model_key = model_key(args.condition, args.seed)
    config = read_json(args.config)
    reference_path = private_path(repo_path(config.get('reference_config', 'configs/vlm_deep_v2.json')))
    reference = read_json(reference_path)
    validate_config(config, reference, args.batch_size)
    files, training = adapter_binding(args)
    rows, images, image_root, metadata = dev_data(args.asset_root)
    import torch
    if int(os.environ.get('WORLD_SIZE', '1')) != 1 or not torch.cuda.is_available():
        raise ValueError('expected one independent admitted CUDA eval worker')
    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    device = torch.device('cuda', 0)
    props = torch.cuda.get_device_properties(device)
    if 'A100' not in props.name or props.total_memory < 70*2**30:
        raise ValueError('declared evaluation worker requires A10080GB')
    hardware = {**visible_gpu_binding(), 'name': props.name, 'total_memory_bytes': props.total_memory,
        'compute_capability': [props.major, props.minor], 'torch_cuda_version': torch.version.cuda}
    lock = {'schema_version': 'rarevlm.weighted_sft.dev_input.v1',
        'model_key': args.model_key, 'role': 'adapted', 'condition': args.condition, 'seed': args.seed,
        'weighted_config_sha256': sha_file(args.config), 'reference_config_sha256': sha_file(reference_path),
        **model_metadata_binding(args.model_root, reference), **metadata,
        'adapter_files_sha256': files, 'training_binding': training,
        'source_sha256': source_hashes(), 'packages': package_versions(), 'hardware': hardware,
        'batch_size': 4, 'numerical_path': 'original_bf16', **PATH_SPECS['original_bf16'],
        'visual_token_budget': BUDGET, 'max_pixels': BUDGET*784, 'min_pixels': 50176,
        'max_new_tokens': 32, 'do_sample': False, 'attention_implementation': 'sdpa',
        'tf32_matmul': False, 'tf32_cudnn': False, 'head_last_position_only': True,
        'autocast': False, 'adapter_autocast_dtype': False, 'explicit_lora_bf16_conversion': True,
        'input': 'image+question+fixed suffix only; no references, target or OCR tokens',
        'input_truncation': False, 'statistical_unit': 'image cluster',
        'adapter_origin': 'new controlled full-data NF4/FP32-training condition; cast to BF16 at eval',
        'scope': 'internal-dev exploratory single-seed contrast; predeclared loss candidate and gate',
        'protobuf_implementation': os.environ.get('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION')}
    status_path = args.output_root / 'runs' / args.condition / f'seed{args.seed}' / 'EVAL_STATUS.json'
    status = {'schema_version': 'rarevlm.weighted_sft.eval_status.v1', 'condition': args.condition,
        'seed': args.seed, 'started_at': now(), 'phase': 'starting', 'hardware': hardware,
        'training_binding': training, 'completed_questions': 0, 'total_questions': len(rows)}
    def update(count=None, state=None):
        if count is not None: status['completed_questions'] = count
        if state is not None: status['cell_status'] = state
        status['updated_at'] = now()
        write_json(status_path, status)
    model, hook = None, None
    update()
    try:
        status['phase'] = 'loading_model'; update()
        begun = time.monotonic()
        model, dtypes = load_path(args.model_root, reference, device, 'original_bf16', args.adapter_dir)
        status['model_load_seconds'] = time.monotonic()-begun
        lock['actual_parameter_dtypes'] = dtypes
        status['actual_parameter_dtypes'] = dtypes
        if int(model.config.image_token_id) != 151655:
            raise ValueError('image-token ID differs')
        hook = model.get_output_embeddings().register_forward_pre_hook(lambda module, items: (items[0][:, -1:, :],))
        processor = processor_for(args.model_root, reference, BUDGET)
        lock_file(args.output_root / 'runs' / args.condition / f'seed{args.seed}' / 'dev' / 'INPUT_LOCK.json', lock)
        status['phase'] = 'validating_images'; update()
        status['image_validation'] = verify_dev_images(rows, images, image_root)
        status['phase'] = 'preflight'; update()
        run_preflight(model, processor, args, lock, rows, images, image_root, device, 151655)
        if not args.preflight_only:
            status['phase'] = 'evaluating'; update()
            run_cell(model, processor, args, lock, rows, images, image_root, device, 151655, update)
        status['phase'] = 'preflight_completed' if args.preflight_only else 'completed'
        status['completed_at'] = now(); update()
    except Exception as error:
        status.update(phase='failed', error_type=type(error).__name__, error=str(error)); update()
        raise
    finally:
        if hook is not None: hook.remove()
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
