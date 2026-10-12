"""Resumable three-task BF16/1024-token formal evaluation, without training."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

sys.dont_write_bytecode = True
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')

from vlm_a100.evaluate import load_data, validate_prediction
from vlm_deep.common import PROMPT_SUFFIX
from vlm_deep.evaluate import aggregate, score_prediction
from vlm_joint.evaluate import model_metadata_binding, visible_gpu_binding
from vlm_joint.evaluate import source_hashes as reference_source_hashes
from vlm_precision.evaluate import PATH_SPECS, load_path
from vlm_probe.evaluate import (
    content_sha, evaluate_batch, lock_file, now, package_versions, private_path,
    processor_for, read_json, repo_path, sha_file, summarize, write_json,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / 'configs/vlm_formal_bf16_v1.json'
MODEL_KEYS = ('base', 'qv_seed0', 'qv_seed42', 'qv_seed200')
TASKS = ('textvqa', 'docvqa', 'mmstar')
COUNTS = {'textvqa': (5000, 3166), 'docvqa': (5349, 1285), 'mmstar': (1500, 1430)}
NUMERICAL_PATH = 'original_bf16'
VISUAL_BUDGET = 1024
METRICS = {
    'textvqa': 'TextVQA leave-one-out consensus percent',
    'docvqa': 'DocVQA ANLS percent',
    'mmstar': 'MMStar 18-subcategory macro accuracy percent',
}


def source_hashes():
    hashes = reference_source_hashes()
    for path in (Path(__file__), Path(__file__).parent / '__init__.py',
                 REPO / 'vlm_a100/evaluate.py', REPO / 'vlm_a100/__init__.py'):
        hashes[str(path.relative_to(REPO))] = sha_file(path)
    return hashes


def question_prompt_sha(row):
    return content_sha(row['question'] + row.get('prompt_suffix', PROMPT_SUFFIX))


def validate_config(formal, reference, tasks, batch_size):
    expected = {
        'numerical_path': NUMERICAL_PATH, 'precision': 'bf16', 'quantization': 'original',
        'vision_precision': 'bf16', 'visual_token_budget': VISUAL_BUDGET,
        'batch_size': 4, 'min_pixels': 50176, 'max_pixels': VISUAL_BUDGET * 784, 'max_new_tokens': 32,
        'do_sample': False, 'attention_implementation': 'sdpa', 'allow_tf32': False,
        'input_truncation': False,
    }
    for key, value in expected.items():
        if formal.get(key, value) != value:
            raise ValueError(f'formal configuration differs from frozen protocol: {key}')
    if (batch_size != 4 or not tasks or len(set(tasks)) != len(tasks)
            or any(task not in TASKS for task in tasks)
            or formal.get('tasks', list(TASKS)) != list(TASKS)
            or len(formal.get('model_keys', list(MODEL_KEYS))) != len(MODEL_KEYS)
            or set(formal.get('model_keys', list(MODEL_KEYS))) != set(MODEL_KEYS)):
        raise ValueError('expected batch4 and predeclared model/task grid')
    if (reference['precision'] != 'fp32'
            or reference['training_quantization'] != 'nf4_language_only'
            or reference['min_pixels'] != 50176 or reference['max_pixels'] != 256 * 784
            or reference['max_new_tokens'] != 32 or reference['do_sample']):
        raise ValueError('frozen training/reference contract differs')
    for task, (n, images) in COUNTS.items():
        for key, target in (('expected_questions', n), ('expected_images', images)):
            declared = formal.get(key, {})
            if declared and declared.get(task) != target:
                raise ValueError(f'formal count differs: {key}.{task}')


def adapter_binding(args, formal, reference):
    """Verify completed fixed checkpoints; preserve heterogeneous training hardware."""
    if args.model_key == 'base':
        if args.adapter_dir is not None:
            raise ValueError('base must not receive an adapter')
        return None, None, None
    seed = int(args.model_key.removeprefix('qv_seed'))
    if args.adapter_dir is None:
        raise ValueError('adapted model must receive --adapter-dir')
    args.adapter_dir = private_path(args.adapter_dir)
    fixed = formal.get('fixed_adapter_dirs', {}).get(args.model_key)
    if not fixed or args.adapter_dir != repo_path(fixed).resolve():
        raise ValueError('adapter differs from predeclared final checkpoint')
    train_path = args.adapter_dir.parent / 'TRAIN_RESULT.json'
    train = read_json(train_path)
    if (train['method'] != 'qv_r8' or train['seed'] != seed
            or train['split'] != 'train_all' or train['planned_updates'] != 2062
            or train['completed_updates'] != 2062
            or not train['frozen_parameters_unchanged']
            or train['actual_questions_seen'] != 32985):
        raise ValueError('expected completed full-data stage2 qv_r8 checkpoint')
    if train['config_sha256'] != content_sha(reference):
        raise ValueError('training configuration differs from reference')
    files = {path.name: sha_file(path) for path in sorted(args.adapter_dir.iterdir()) if path.is_file()}
    if files != train['adapter_files_sha256']:
        raise ValueError('adapter bytes differ from completed training binding')
    if train['model_provenance_sha256'] != sha_file(args.model_root.parent / 'model_provenance.json'):
        raise ValueError('adapter and base model provenance differ')
    binding = {
        'path': str(train_path), 'sha256': sha_file(train_path), 'method': 'qv_r8',
        'stage': 'stage2', 'seed': seed, 'split': 'train_all',
        'completed_updates': train['completed_updates'],
        'planned_updates': train['planned_updates'],
        'actual_questions_seen': train['actual_questions_seen'],
        'gpu_name': train.get('gpu_name'), 'hardware': train.get('hardware'),
        'runtime_layout': train.get('runtime_layout'),
        'scientific_config_origin': train.get('scientific_config_origin'),
        'checkpoint_selection': train.get('checkpoint_selection', 'fixed final epoch'),
        'training_numerical_path': 'nf4_language_only_fp32',
        'training_visual_token_budget': 256,
        'scope': 'existing complete checkpoint; no new training or official-benchmark checkpoint selection',
    }
    return files, binding, seed


def task_data(asset_root, task, formal):
    rows, images, image_root, manifest_sha = load_data(asset_root, task, 'final')
    n, nimages = COUNTS[task]
    if len(rows) != n or len({row['image_id'] for row in rows}) != nimages:
        raise ValueError(f'expected full frozen {task}: {n} QA / {nimages} images')
    manifest_path = image_root / 'manifest.json'
    manifest = read_json(manifest_path)
    expected_sha = formal.get('manifest_sha256', {}).get(task)
    if expected_sha and manifest_sha != expected_sha:
        raise ValueError(f'predeclared manifest differs: {task}')
    if task == 'textvqa' and any(len(row['answers']) != 10 for row in rows):
        raise ValueError('TextVQA must retain all ten human references')
    if task == 'docvqa' and any(not row['answers'] for row in rows):
        raise ValueError('DocVQA ANLS references cannot be empty')
    subcategories = sorted({row.get('subcategory', '') for row in rows}) if task == 'mmstar' else []
    if task == 'mmstar' and (len(subcategories) != 18 or '' in subcategories):
        raise ValueError('MMStar primary aggregation requires all eighteen subcategories')
    return rows, images, image_root, {
        'task': task, 'split': 'final',
        'source_split': 'official_benchmark' if task == 'mmstar' else 'official_validation',
        'manifest_sha256': manifest_sha, 'manifest_file_sha256': sha_file(manifest_path),
        'manifest_path': str(manifest_path),
        'dataset_repository': manifest.get('repository', manifest.get('dataset_id')),
        'dataset_revision': manifest.get('revision', manifest.get('dataset_revision')),
        'pretraining_overlap': manifest.get('pretraining_overlap', 'unknown'),
        'expected_questions': n, 'expected_images': nimages,
        'metric': METRICS[task], 'subcategories': subcategories,
        'question_order_sha256': content_sha([row['question_id'] for row in rows]),
        'question_prompt_order_sha256': content_sha([question_prompt_sha(row) for row in rows]),
    }


def verify_images(rows, images, image_root, task):
    from PIL import Image
    started = time.monotonic()
    ids = sorted({row['image_id'] for row in rows})
    for iid in ids:
        metadata = images[iid]
        path = private_path(image_root / metadata['path'])
        if sha_file(path) != metadata['sha256']:
            raise ValueError(f'{task} image bytes changed: {iid}')
        with Image.open(path) as image:
            if image.size != (int(metadata['width']), int(metadata['height'])):
                raise ValueError(f'{task} image dimensions changed: {iid}')
    return {'task': task, 'images': len(ids), 'seconds': time.monotonic() - started,
            'scope': 'current full task input bytes and dimensions verified once per model worker'}


def complete_cache(path, rows, images, batch_size, task, lock):
    """Rescore the task; only repair interrupted final writes and batch tails."""
    if not path.exists():
        return [], None
    lines = path.read_bytes().splitlines(keepends=True)
    repairs = []
    if lines and not lines[-1].endswith(b'\n'):
        lines.pop()
        repairs.append('discarded interrupted trailing JSONL line')
    items = [json.loads(line) for line in lines]
    if len(items) > len(rows):
        raise ValueError('cache contains more questions than the complete benchmark')
    if items:
        last_start = ((len(items) - 1) // batch_size) * batch_size
        expected = min(batch_size, len(rows) - last_start)
        if len(items) - last_start != expected:
            items, lines = items[:last_start], lines[:last_start]
            repairs.append('discarded incomplete final batch to preserve grouping')
    lock_sha = content_sha(lock)
    sources_sha = content_sha(lock['source_sha256'])
    for index, item in enumerate(items):
        row = rows[index]
        validate_prediction(item, row, images, task)
        start = (index // batch_size) * batch_size
        if item['batch_start'] != start or item['batch_size'] != min(batch_size, len(rows) - start):
            raise ValueError('cached batch grouping differs')
        expected = {
            'task': task, 'model_key': lock['model_key'], 'seed': lock['seed'],
            'input_lock_sha256': lock_sha, 'source_binding_sha256': sources_sha,
            'question_prompt_sha256': question_prompt_sha(row),
            'numerical_path': NUMERICAL_PATH, 'precision': 'bf16',
            'quantization': 'original', 'visual_token_budget': VISUAL_BUDGET,
            'metric': METRICS[task], 'model_revision': lock['model_revision'],
            'adapter_files_sha256': lock['adapter_files_sha256'],
        }
        for key, value in expected.items():
            if item.get(key) != value:
                raise ValueError(f'cached prediction binding differs: {key}')
        grid = item['image_grid_thw']
        visual = math.prod(grid) // 4
        if (len(grid) != 3 or grid[0] != 1 or math.prod(grid) % 4
                or item['visual_tokens'] != visual or item['counted_image_tokens'] != visual
                or visual > VISUAL_BUDGET or item['input_tokens'] < visual
                or item['input_truncated']):
            raise ValueError('cached actual visual/input token contract differs')
        for key in ('batch_generation_seconds', 'batch_preprocessing_seconds', 'batch_h2d_seconds',
                    'batch_peak_allocated_gib', 'batch_peak_reserved_gib'):
            if not math.isfinite(float(item[key])) or float(item[key]) < 0:
                raise ValueError('cached resource observation is invalid')
    if repairs:
        with path.open('wb') as stream:
            stream.write(b''.join(lines))
            stream.flush()
            os.fsync(stream.fileno())
    return items, repairs or None


def stress_rows(rows, images, processor):
    """Input-only deterministic pressure selection, without answer or score access."""
    longest = sorted(rows, key=lambda row: (
        -len(processor.tokenizer.encode(
            row['question'] + row.get('prompt_suffix', PROMPT_SUFFIX), add_special_tokens=False)),
        row['question_id']))
    ratio = lambda row: images[row['image_id']]['width'] / images[row['image_id']]['height']
    candidates = longest[:2] + [max(rows, key=ratio), min(rows, key=ratio),
                                max(rows, key=lambda row: images[row['image_id']]['width'] * images[row['image_id']]['height'])] + longest
    selected, seen = [], set()
    for row in candidates:
        if row['question_id'] not in seen:
            selected.append(row)
            seen.add(row['question_id'])
        if len(selected) == 8:
            return selected
    raise ValueError('engineering preflight requires eight unique input examples')


def run_preflight(model, processor, args, lock, rows, images, image_root, device, image_token_id):
    task = lock['task']
    directory = args.output_root / 'preflight' / args.model_key / task
    selected = stress_rows(rows, images, processor)
    prelock = {**lock, 'scope': 'engineering_preflight_not_scientific_result',
               'selection': 'longest complete question+suffix, aspect extremes, largest raw image; input-only, no answers or outcomes',
               'question_ids': [row['question_id'] for row in selected]}
    lock_file(directory / 'INPUT_LOCK.json', prelock)
    result_path = directory / 'PREFLIGHT.json'
    if result_path.exists():
        previous = read_json(result_path)
        if previous.get('input_lock_sha256') != content_sha(prelock) or previous.get('status') != 'passed':
            raise ValueError('preflight source/runtime binding changed')
        print('PREFLIGHT REUSED', args.model_key, task, flush=True)
        return previous
    import torch
    records, batches = [], []
    with torch.inference_mode():
        for begin in range(0, len(selected), args.batch_size):
            chunk = selected[begin:begin + args.batch_size]
            answers, resource = evaluate_batch(
                model, processor, chunk, images, image_root, device, VISUAL_BUDGET,
                image_token_id, 32, check_logits=True)
            batches.append(resource)
            records.extend({'question_id': row['question_id'], 'image_id': row['image_id'], **answer}
                           for row, answer in zip(chunk, answers))
    result = {'schema_version': 'rarevlm.formal_bf16_preflight.v1', 'status': 'passed',
              'completed_at': now(), 'task': task, 'model_key': args.model_key,
              'input_lock_sha256': content_sha(prelock), 'records': records, 'batches': batches,
              'peak_allocated_gib': max(batch['batch_peak_allocated_gib'] for batch in batches),
              'peak_reserved_gib': max(batch['batch_peak_reserved_gib'] for batch in batches),
              'scope': 'finite last-position logits, greedy generation, no truncation and actual tokens; no benchmark scoring/cache reuse'}
    write_json(result_path, result)
    print('PREFLIGHT PASSED', args.model_key, task, 'peak_gib', result['peak_allocated_gib'], flush=True)
    return result


def summarize_task(items, lock, cache_path, session):
    # Resource summaries depend on batches, not on the task scorer. Override all
    # TextVQA-specific metric and scientific-scope fields from the frozen helper.
    result = summarize(items, lock, cache_path, session)
    task = lock['task']
    value, by_group = aggregate(items, task)
    if task == 'mmstar' and len(by_group) != 18:
        raise ValueError('completed MMStar must contain eighteen subcategory scores')
    result.update(
        schema_version='rarevlm.formal_bf16_cell.v1', input_lock_sha256=content_sha(lock),
        score_percent=value, question_mean_score_percent=100 * statistics.mean(item['score'] for item in items),
        subcategory_score_percent=by_group, metric=METRICS[task],
        scope='complete official benchmark under independently dev-fixed BF16/1024 protocol; fixed existing checkpoint, no new training; historical protocol separate')
    resources = result['resources']
    # Count preprocessing/H2D on exactly the same timed non-warmup batches.
    unique = {(item['session_id'], item['batch_start']): item for item in items}
    timed = [item for item in unique.values() if not item['timing_warmup']]
    total = sum(item['batch_preprocessing_seconds'] + item['batch_h2d_seconds'] + item['batch_generation_seconds'] for item in timed)
    resources['preprocessing_h2d_generation_seconds'] = total
    resources['preprocessing_h2d_generation_questions_per_second'] = resources['timed_questions'] / total if total else None
    resources['scope'] = ('recorded admitted A100 worker/batch4; model load, validation, preflight, writer and scoring excluded; '
                          'per-session first batch excluded from matched timings; batch-amortized latency, not production service')
    return result


def run_cell(model, processor, args, lock, rows, images, image_root, device, image_token_id, update_status):
    task = lock['task']
    directory = args.output_root / 'runs' / args.model_key / task
    lock_file(directory / 'INPUT_LOCK.json', lock)
    lock_sha = content_sha(lock)
    cache_path = directory / 'predictions.jsonl'
    items, repaired = complete_cache(cache_path, rows, images, args.batch_size, task, lock)
    evaluation = directory / 'EVALUATION.json'
    if evaluation.exists():
        saved = read_json(evaluation)
        if (len(items) != len(rows) or saved.get('predictions_sha256') != sha_file(cache_path)
                or saved.get('input_lock_sha256') != lock_sha):
            raise ValueError('completed formal cell no longer matches predictions/input lock')
        value, by_group = aggregate(items, task)
        if (not math.isclose(saved['score_percent'], value, rel_tol=0, abs_tol=1e-12)
                or saved.get('subcategory_score_percent') != by_group):
            raise ValueError('completed formal cell metric differs from rescored predictions')
        update_status(task, len(items), 'completed')
        print('CELL REUSED', args.model_key, task, flush=True)
        return saved
    import torch
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    session_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    session = {'session_id': session_id, 'resumed_questions': len(items), 'cache_repair': repaired, 'started_at': now()}
    begun, start = time.monotonic(), len(items)
    update_status(task, start, 'running')
    with cache_path.open('a') as stream, torch.inference_mode():
        for begin in range(start, len(rows), args.batch_size):
            chunk = rows[begin:begin + args.batch_size]
            answers, resource = evaluate_batch(model, processor, chunk, images, image_root,
                device, VISUAL_BUDGET, image_token_id, 32)
            resource_row = {key: value for key, value in resource.items() if key != 'finite_logits'}
            batch = []
            for row, answer in zip(chunk, answers):
                batch.append({
                    'question_id': row['question_id'], 'image_id': row['image_id'],
                    'question_sha256': content_sha(row['question']), 'question_prompt_sha256': question_prompt_sha(row),
                    'image_sha256': images[row['image_id']]['sha256'], **answer,
                    'score': score_prediction(answer['prediction'], row, task),
                    'task': task, 'subcategory': row.get('subcategory', ''), 'category': row.get('category', ''),
                    'model_key': args.model_key, 'seed': lock['seed'], 'metric': METRICS[task],
                    'model_revision': lock['model_revision'], 'adapter_files_sha256': lock['adapter_files_sha256'],
                    'source_binding_sha256': content_sha(lock['source_sha256']),
                    'reference_score_source': 'frozen manifest references used only after label-free generation',
                    **resource_row, 'batch_start': begin, 'session_id': session_id,
                    'timing_warmup': begin == start, 'visual_token_budget': VISUAL_BUDGET,
                    'numerical_path': NUMERICAL_PATH, 'precision': 'bf16', 'quantization': 'original',
                    'input_lock_sha256': lock_sha,
                })
            stream.write(''.join(json.dumps(item, ensure_ascii=False, allow_nan=False) + '\n' for item in batch))
            stream.flush()
            os.fsync(stream.fileno())
            items.extend(batch)
            write_json(directory / 'PROGRESS.json', {
                'schema_version': 'rarevlm.formal_bf16_progress.v1', 'updated_at': now(),
                'status': 'running', 'model_key': args.model_key, 'task': task,
                'completed_questions': len(items), 'total_questions': len(rows),
                'new_questions': len(items) - start, 'session_elapsed_seconds': time.monotonic() - begun,
                'last_batch': resource, 'session_id': session_id,
            })
            update_status(task, len(items), 'running')
            if begin == start or ((begin - start) // args.batch_size + 1) % 25 == 0:
                print('GENERATED', args.model_key, task, len(items), '/', len(rows), flush=True)
    session.update(completed_at=now(), new_questions=len(items) - start, wall_seconds=time.monotonic() - begun)
    result = summarize_task(items, lock, cache_path, session)
    write_json(evaluation, result)
    write_json(directory / 'PROGRESS.json', {
        'schema_version': 'rarevlm.formal_bf16_progress.v1', 'status': 'completed',
        'updated_at': now(), 'model_key': args.model_key, 'task': task,
        'completed_questions': len(items), 'total_questions': len(rows),
    })
    update_status(task, len(items), 'completed')
    print('CELL COMPLETE', args.model_key, task, result['score_percent'], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-key', choices=MODEL_KEYS, required=True)
    parser.add_argument('--tasks', choices=TASKS, nargs='+', default=list(TASKS))
    parser.add_argument('--asset-root', type=Path, required=True)
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--adapter-dir', type=Path)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    for key in ('asset_root', 'model_root', 'output_root', 'config'):
        setattr(args, key, private_path(getattr(args, key)))
    formal = read_json(args.config)
    reference_path = private_path(repo_path(formal['reference_config']))
    reference = read_json(reference_path)
    validate_config(formal, reference, args.tasks, args.batch_size)
    adapter_files, training_binding, seed = adapter_binding(args, formal, reference)
    datasets = {task: task_data(args.asset_root, task, formal) for task in args.tasks}
    import torch
    if int(os.environ.get('WORLD_SIZE', '1')) != 1 or not torch.cuda.is_available():
        raise ValueError('expected one independent CUDA worker, without DDP')
    torch.cuda.set_device(0)
    device = torch.device('cuda', 0)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    props = torch.cuda.get_device_properties(device)
    if 'A100' not in props.name or props.total_memory < 70 * 2**30:
        raise ValueError('formal runtime declared for an admitted A100 80GB')
    hardware = {**visible_gpu_binding(), 'name': props.name,
                'total_memory_bytes': props.total_memory,
                'compute_capability': [props.major, props.minor], 'torch_cuda_version': torch.version.cuda}
    common_lock = {
        'schema_version': 'rarevlm.formal_bf16_input.v1', 'model_key': args.model_key,
        'role': 'base' if args.model_key == 'base' else 'adapted', 'seed': seed,
        'formal_config_sha256': sha_file(args.config), 'reference_config_sha256': sha_file(reference_path),
        **model_metadata_binding(args.model_root, reference),
        'adapter_files_sha256': adapter_files, 'training_binding': training_binding,
        'source_sha256': source_hashes(), 'packages': package_versions(), 'hardware': hardware,
        'batch_size': args.batch_size, 'numerical_path': NUMERICAL_PATH, **PATH_SPECS[NUMERICAL_PATH],
        'visual_token_budget': VISUAL_BUDGET, 'max_pixels': VISUAL_BUDGET * 784,
        'min_pixels': reference['min_pixels'], 'max_new_tokens': 32, 'do_sample': False,
        'attention_implementation': 'sdpa', 'tf32_matmul': False, 'tf32_cudnn': False,
        'head_last_position_only': True, 'autocast': False, 'adapter_autocast_dtype': False,
        'explicit_lora_bf16_conversion': args.adapter_dir is not None,
        'input': 'declared image+question+suffix only; multiple-choice options retained; no target, answers or OCR input',
        'input_truncation': False, 'statistical_unit': 'image cluster; retain all same-image questions',
        'adapter_origin': 'fixed final NF4/FP32-trained qv_r8 checkpoint, not retrained at BF16 or 1024 visual budget' if adapter_files else None,
        'protocol_selection': formal.get('selection_rule', 'configuration fixed from internal dev before this formal evaluation; no official-metric retuning'),
        'scope': 'complete new-protocol official benchmark confirmation; original historical protocol separate',
        'protobuf_implementation': os.environ.get('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'),
    }
    status_path = args.output_root / 'runs' / args.model_key / 'STATUS.json'
    status = {
        'schema_version': 'rarevlm.formal_bf16_worker_status.v1', 'model_key': args.model_key,
        'seed': seed, 'tasks': args.tasks, 'phase': 'starting', 'started_at': now(),
        'hardware': hardware, 'training_binding': training_binding,
        'cells': {task: {'status': 'pending', 'completed_questions': 0,
                         'total_questions': COUNTS[task][0]} for task in args.tasks},
    }
    def update_status(task=None, count=None, state=None):
        if task is not None:
            status['current_task'] = task
            status['cells'][task].update(status=state, completed_questions=count)
        status['updated_at'] = now()
        write_json(status_path, status)
    update_status()
    model, hook = None, None
    try:
        status['phase'] = 'loading_model'
        update_status()
        begun = time.monotonic()
        model, dtypes = load_path(args.model_root, reference, device, NUMERICAL_PATH, args.adapter_dir)
        status['model_load_seconds'] = time.monotonic() - begun
        status['actual_parameter_dtypes'] = dtypes
        common_lock['actual_parameter_dtypes'] = dtypes
        image_token_id = int(model.config.image_token_id)
        if image_token_id != 151655:
            raise ValueError('unexpected model image-token ID')
        hook = model.get_output_embeddings().register_forward_pre_hook(lambda module, items: (items[0][:, -1:, :],))
        processor = processor_for(args.model_root, reference, VISUAL_BUDGET)
        status['image_validation'] = {}
        for task in args.tasks:
            rows, images, image_root, metadata = datasets[task]
            lock = {**common_lock, **metadata, 'n': len(rows)}
            # Fail before generating if any existing lock would mix a new path.
            lock_file(args.output_root / 'runs' / args.model_key / task / 'INPUT_LOCK.json', lock)
            status.update(phase='validating_images', current_task=task)
            update_status()
            status['image_validation'][task] = verify_images(rows, images, image_root, task)
            status['phase'] = 'preflight'
            update_status()
            run_preflight(model, processor, args, lock, rows, images, image_root, device, image_token_id)
            if not args.preflight_only:
                status['phase'] = 'evaluating'
                update_status()
                run_cell(model, processor, args, lock, rows, images, image_root, device, image_token_id, update_status)
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
