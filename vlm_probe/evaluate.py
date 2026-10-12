"""Resumable FP32/NF4 visual-budget probes, isolated from completed A+B runs."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Importing the reference contracts must not create artifacts in frozen trees.
sys.dont_write_bytecode = True
from vlm_deep.common import collate, content_sha, encode, read_manifest, sha_file, to_device, write_json
from vlm_deep.evaluate import aggregate, score_prediction
from vlm_deep.model import load

REPO = Path(__file__).resolve().parents[1]
PRIVATE = Path('/opt/data/private')
DEFAULT_CONFIG = REPO / 'configs/vlm_budget_probe_v1.json'
CASE_IDS = ('27482', '10476', '12257')


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def private_path(path):
    value = Path(path).resolve()
    if value != PRIVATE and PRIVATE not in value.parents:
        raise ValueError(f'probe artifacts must stay under {PRIVATE}: {value}')
    return value


def repo_path(value):
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def inference_row(row):
    # References and training targets are deliberately absent from this object.
    clean = {key: row[key] for key in ('question_id', 'image_id', 'question')}
    if 'prompt_suffix' in row:
        clean['prompt_suffix'] = row['prompt_suffix']
    return clean


def source_hashes():
    paths = [Path(__file__), Path(__file__).parent / '__init__.py', REPO / 'vlm_sft/common.py']
    paths.extend(sorted((REPO / 'vlm_deep').rglob('*.py')))
    return {str(path.relative_to(REPO)): sha_file(path) for path in paths}


def package_versions():
    result = {}
    for name in ('torch', 'transformers', 'peft', 'bitsandbytes', 'Pillow', 'safetensors'):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def lock_file(path, lock):
    if path.exists():
        if read_json(path) != lock:
            raise ValueError(f'probe input/code/runtime lock changed: {path}')
    else:
        write_json(path, lock)


def processor_for(model_root, config, budget):
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(
        str(model_root), local_files_only=True, use_fast=False,
        min_pixels=config['min_pixels'], max_pixels=budget * 28 * 28)
    processor.tokenizer.padding_side = 'left'
    if processor.image_processor.patch_size != 14 or processor.image_processor.merge_size != 2:
        raise ValueError('this probe requires the pinned patch14/merge2 processor')
    return processor


def describe_input(sample, row, images, processor, image_token_id, budget):
    grid = sample['image_grid_thw'].tolist()
    if len(grid) != 1 or grid[0][0] != 1:
        raise ValueError('expected exactly one static image per question')
    t, h, w = map(int, grid[0])
    patches = t * h * w
    if patches % 4:
        raise ValueError('image patches do not match spatial merger')
    visual = patches // 4
    counted = int((sample['input_ids'] == image_token_id).sum())
    if visual != counted or visual > budget or sample['pixel_values'].shape[0] != patches:
        raise ValueError(f'visual-token contract failed: {row["question_id"]}')
    image = images[row['image_id']]
    length = int(sample['input_ids'].shape[1])
    return {'image_grid_thw': [t, h, w], 'visual_tokens': visual,
            'counted_image_tokens': counted, 'premerge_patch_tokens': patches,
            'raw_width': int(image['width']), 'raw_height': int(image['height']),
            'resized_width': w * int(processor.image_processor.patch_size),
            'resized_height': h * int(processor.image_processor.patch_size),
            'input_tokens': length, 'input_exceeds_training_768': length > 768,
            'input_truncated': False}


def verify_dev_images(rows, images, image_root):
    from PIL import Image
    begun = time.monotonic()
    ids = sorted({row['image_id'] for row in rows})
    for iid in ids:
        item = images[iid]
        path = image_root / item['path']
        if sha_file(path) != item['sha256']:
            raise ValueError(f'dev image bytes changed: {iid}')
        with Image.open(path) as image:
            if image.size != (int(item['width']), int(item['height'])):
                raise ValueError(f'dev image dimensions changed: {iid}')
    return {'images': len(ids), 'seconds': time.monotonic() - begun,
            'scope': 'all internal-dev input bytes and raw dimensions checked once per worker'}


def complete_cache(path, rows, images, batch_size):
    """Repair only interrupted writes, preserving the original global batch boundaries."""
    if not path.exists():
        return [], None
    original = path.read_bytes()
    lines = original.splitlines(keepends=True)
    repaired = []
    if lines and not lines[-1].endswith(b'\n'):
        lines.pop()
        repaired.append('discarded interrupted trailing JSONL line')
    items = [json.loads(line) for line in lines]
    if len(items) > len(rows):
        raise ValueError('cache contains more questions than internal dev')
    if items:
        start = ((len(items) - 1) // batch_size) * batch_size
        expected = min(batch_size, len(rows) - start)
        if len(items) - start != expected:
            items = items[:start]
            lines = lines[:start]
            repaired.append('discarded incomplete final batch to preserve fixed grouping')
    for index, item in enumerate(items):
        row = rows[index]
        if item['question_id'] != row['question_id'] or item['image_id'] != row['image_id']:
            raise ValueError('cached prediction order or identity changed')
        if item['question_sha256'] != content_sha(row['question']) or item['image_sha256'] != images[row['image_id']]['sha256']:
            raise ValueError('cached question/image hash changed')
        score = float(item['score'])
        if not math.isfinite(score) or not math.isclose(score, score_prediction(item['prediction'], row, 'textvqa'), abs_tol=1e-12, rel_tol=0):
            raise ValueError('cached consensus score changed')
        start = (index // batch_size) * batch_size
        if item['batch_start'] != start or item['batch_size'] != min(batch_size, len(rows) - start):
            raise ValueError('cached batch grouping changed')
    if repaired:
        # Only newly-created probe caches are ever modified here.
        with path.open('wb') as stream:
            stream.write(b''.join(lines))
            stream.flush()
            os.fsync(stream.fileno())
    return items, repaired or None


def preflight_rows(rows, images, processor):
    selected = []
    seen = set()
    def take(row):
        if row['question_id'] not in seen:
            selected.append(row)
            seen.add(row['question_id'])
    longest = sorted(rows, key=lambda row: (-len(processor.tokenizer.encode(row['question'], add_special_tokens=False)), row['question_id']))
    for row in longest[:2]:
        take(row)
    ratio = lambda row: images[row['image_id']]['width'] / images[row['image_id']]['height']
    take(max(rows, key=ratio))
    take(min(rows, key=ratio))
    take(max(rows, key=lambda row: images[row['image_id']]['width'] * images[row['image_id']]['height']))
    lookup = {row['question_id']: row for row in rows}
    for qid in CASE_IDS:
        if qid in lookup:
            take(lookup[qid])
    for row in longest:
        if len(selected) >= 8:
            break
        take(row)
    return selected[:8]


def evaluate_batch(model, processor, chunk, images, image_root, device, budget, image_token_id,
                   max_new_tokens, check_logits=False):
    import torch
    begun = time.monotonic()
    samples = [encode(processor, inference_row(row), image_root / images[row['image_id']]['path'], False) for row in chunk]
    details = [describe_input(sample, row, images, processor, image_token_id, budget) for sample, row in zip(samples, chunk)]
    cpu_batch = collate(samples, processor.tokenizer.pad_token_id)
    preprocessing = time.monotonic() - begun
    torch.cuda.synchronize(device)
    begun = time.monotonic()
    batch = to_device(cpu_batch, device)
    torch.cuda.synchronize(device)
    transfer = time.monotonic() - begun
    torch.cuda.reset_peak_memory_stats(device)
    finite = None
    if check_logits:
        result = model(**batch, use_cache=False)
        finite = bool(torch.isfinite(result.logits).all())
        del result
        if not finite:
            raise ValueError('non-finite preflight logits')
    torch.cuda.synchronize(device)
    begun = time.monotonic()
    generated = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
    torch.cuda.synchronize(device)
    elapsed = time.monotonic() - begun
    padded = int(batch['input_ids'].shape[1])
    answers = []
    for index in range(len(chunk)):
        ids = generated[index, padded:]
        answer = processor.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        # Keep the same nonpad-token definition as the frozen evaluator.
        count = int((ids != processor.tokenizer.pad_token_id).sum())
        answers.append({'prediction': answer, 'generated_nonpad_tokens': count,
                        'generated_tail_slots': int(ids.numel()), **details[index]})
    resource = {'batch_size': len(chunk), 'padded_input_tokens': padded,
                'batch_generation_seconds': elapsed, 'batch_preprocessing_seconds': preprocessing,
                'batch_h2d_seconds': transfer,
                'batch_peak_allocated_gib': torch.cuda.max_memory_allocated(device) / 2**30,
                'batch_peak_reserved_gib': torch.cuda.max_memory_reserved(device) / 2**30,
                'finite_logits': finite}
    return answers, resource


def run_preflight(model, args, config, common_lock, rows, images, image_root, device, image_token_id):
    directory = args.output_root / 'preflight' / args.role
    processor = processor_for(args.model_root, config, 1024)
    selected = preflight_rows(rows, images, processor)
    lock = {**common_lock, 'scope': 'engineering_preflight_not_scientific_result',
            'visual_token_budget': 1024, 'max_pixels': 1024 * 784,
            'selection': 'two longest tokenized questions, aspect extremes, largest image, fixed teaching examples; no answer/outcome selection',
            'question_ids': [row['question_id'] for row in selected]}
    lock_file(directory / 'INPUT_LOCK.json', lock)
    result_path = directory / 'PREFLIGHT.json'
    if result_path.exists():
        previous = read_json(result_path)
        if previous['input_lock_sha256'] != content_sha(lock) or previous.get('status') != 'passed':
            raise ValueError('previous preflight binding/status is invalid')
        print('PREFLIGHT REUSED', args.role, flush=True)
        return previous
    records, batches = [], []
    import torch
    with torch.inference_mode():
        for begin in range(0, len(selected), args.batch_size):
            chunk = selected[begin:begin + args.batch_size]
            answers, resource = evaluate_batch(model, processor, chunk, images, image_root, device, 1024,
                                               image_token_id, 32, check_logits=True)
            batches.append(resource)
            for row, answer in zip(chunk, answers):
                records.append({'question_id': row['question_id'], 'image_id': row['image_id'], **answer})
    result = {'schema_version': 'rarevlm.visual_budget_preflight.v1', 'status': 'passed',
              'completed_at': now(), 'input_lock_sha256': content_sha(lock),
              'scope': 'finite logits/greedy generation/token contract/memory only; no scores or research-cell cache',
              'records': records, 'batches': batches,
              'peak_allocated_gib': max(batch['batch_peak_allocated_gib'] for batch in batches),
              'peak_reserved_gib': max(batch['batch_peak_reserved_gib'] for batch in batches)}
    write_json(result_path, result)
    print('PREFLIGHT PASSED', args.role, json.dumps({'n': len(records), 'peak_gib': result['peak_allocated_gib']}), flush=True)
    return result


def distribution(values):
    import numpy as np
    return {'min': min(values), 'mean': statistics.mean(values), 'median': statistics.median(values),
            'p90': float(np.quantile(values, .9)), 'p95': float(np.quantile(values, .95)), 'max': max(values)}


def summarize(items, lock, cache_path, session):
    batches = {}
    for item in items:
        identity = (item['session_id'], item['batch_start'])
        if identity not in batches:
            batches[identity] = {key: item[key] for key in (
                'batch_size', 'batch_generation_seconds', 'batch_preprocessing_seconds', 'batch_h2d_seconds',
                'batch_peak_allocated_gib', 'batch_peak_reserved_gib', 'timing_warmup')}
    counted = [batch for batch in batches.values() if not batch['timing_warmup']]
    generated_seconds = sum(batch['batch_generation_seconds'] for batch in counted)
    n = sum(batch['batch_size'] for batch in counted)
    value, _ = aggregate(items, 'textvqa')
    return {'schema_version': 'rarevlm.visual_budget_cell.v1', 'status': 'completed',
            'completed_at': now(), **lock, 'score_percent': value, 'question_mean_score_percent': value,
            'unique_images': len({item['image_id'] for item in items}),
            'predictions_file': str(cache_path.resolve()), 'predictions_sha256': sha_file(cache_path),
            'input_statistics': {key: distribution([item[key] for item in items]) for key in (
                'visual_tokens', 'input_tokens', 'padded_input_tokens', 'generated_nonpad_tokens')},
            'questions_above_training_input_limit': sum(item['input_exceeds_training_768'] for item in items),
            'resources': {'timed_batches_excluding_each_session_first_batch': len(counted),
                          'timed_questions': n, 'model_generate_seconds': generated_seconds,
                          'pure_generation_questions_per_second': n / generated_seconds if generated_seconds else None,
                          'amortized_generation_seconds_per_question': distribution([
                              batch['batch_generation_seconds'] / batch['batch_size'] for batch in counted]) if counted else None,
                          'batch_generation_seconds': distribution([batch['batch_generation_seconds'] for batch in counted]) if counted else None,
                          'all_preprocessing_seconds': sum(batch['batch_preprocessing_seconds'] for batch in batches.values()),
                          'all_h2d_seconds': sum(batch['batch_h2d_seconds'] for batch in batches.values()),
                          'peak_allocated_gib': max(batch['batch_peak_allocated_gib'] for batch in batches.values()),
                          'peak_reserved_gib': max(batch['batch_peak_reserved_gib'] for batch in batches.values()),
                          'scope': 'same A100 worker/batch; model load, preflight, image validation and interrupted incomplete work excluded; per-question time is batch-amortized latency'},
            'current_session': session,
            'scope': 'internal dev exploratory input intervention; fixed single training seed; no formal-val selection or new-algorithm claim'}


def run_cell(model, args, config, common_lock, rows, images, image_root, device, image_token_id,
             budget, update_status):
    directory = args.output_root / 'runs' / args.role / f'budget_{budget}'
    lock = {**common_lock, 'scope': 'full_internal_dev', 'visual_token_budget': budget,
            'max_pixels': budget * 784, 'n': len(rows),
            'question_order_sha256': content_sha([row['question_id'] for row in rows])}
    lock_file(directory / 'EVAL_INPUT_LOCK.json', lock)
    cache_path = directory / 'predictions.jsonl'
    items, repaired = complete_cache(cache_path, rows, images, args.batch_size)
    evaluation = directory / 'EVALUATION.json'
    if evaluation.exists():
        saved = read_json(evaluation)
        if len(items) != len(rows) or saved['predictions_sha256'] != sha_file(cache_path):
            raise ValueError('completed cell predictions no longer match evaluation')
        print('CELL REUSED', args.role, budget, flush=True)
        update_status(budget, len(items), 'completed')
        return saved
    # Preflight runs at 1024 before the smaller research cells. Its freed blocks
    # must not inflate a later cell's reserved-memory measurement.
    import torch
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    processor = processor_for(args.model_root, config, budget)
    session_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    session = {'session_id': session_id, 'resumed_questions': len(items), 'cache_repair': repaired,
               'started_at': now()}
    begun = time.monotonic()
    start = len(items)
    with cache_path.open('a') as stream, torch.inference_mode():
        for begin in range(start, len(rows), args.batch_size):
            chunk = rows[begin:begin + args.batch_size]
            answers, resource = evaluate_batch(model, processor, chunk, images, image_root, device, budget,
                                               image_token_id, config['max_new_tokens'])
            batch = []
            for row, answer in zip(chunk, answers):
                resource_row = {key: value for key, value in resource.items() if key != 'finite_logits'}
                batch.append({'question_id': row['question_id'], 'image_id': row['image_id'],
                              'question_sha256': content_sha(row['question']),
                              'image_sha256': images[row['image_id']]['sha256'],
                              **answer, 'score': score_prediction(answer['prediction'], row, 'textvqa'),
                              **resource_row, 'batch_start': begin, 'session_id': session_id,
                              'timing_warmup': begin == start, 'visual_token_budget': budget})
            stream.write(''.join(json.dumps(item, ensure_ascii=False, allow_nan=False) + '\n' for item in batch))
            stream.flush()
            os.fsync(stream.fileno())
            items.extend(batch)
            progress = {'schema_version': 'rarevlm.visual_budget_progress.v1', 'updated_at': now(),
                        'status': 'running', 'role': args.role, 'visual_token_budget': budget,
                        'completed_questions': len(items), 'total_questions': len(rows),
                        'new_questions': len(items) - start, 'session_elapsed_seconds': time.monotonic() - begun,
                        'last_batch': resource, 'session_id': session_id}
            write_json(directory / 'PROGRESS.json', progress)
            update_status(budget, len(items), 'running')
            if begin == start or ((begin - start) // args.batch_size + 1) % 25 == 0:
                print('GENERATED', args.role, budget, len(items), '/', len(rows), flush=True)
    session.update({'completed_at': now(), 'new_questions': len(items) - start,
                    'wall_seconds': time.monotonic() - begun})
    result = summarize(items, lock, cache_path, session)
    write_json(evaluation, result)
    write_json(directory / 'PROGRESS.json', {'schema_version': 'rarevlm.visual_budget_progress.v1',
               'status': 'completed', 'updated_at': now(), 'completed_questions': len(items),
               'total_questions': len(rows), 'role': args.role, 'visual_token_budget': budget})
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
    args.output_root = private_path(args.output_root)
    probe_config = read_json(args.config)
    reference_path = repo_path(probe_config['reference_config'])
    config = read_json(reference_path)
    budgets = args.budgets or probe_config.get('visual_token_budgets', probe_config.get('budgets', [256, 512, 1024]))
    if args.batch_size != 4 or not budgets or len(set(budgets)) != len(budgets) or any(t not in (256, 512, 1024) for t in budgets):
        raise ValueError('locked probe requires batch4 and distinct budgets from 256/512/1024')
    if config['precision'] != 'fp32' or config['training_quantization'] != 'nf4_language_only' or config['min_pixels'] != 50176 or config['max_new_tokens'] != 32 or config['do_sample']:
        raise ValueError('reference numerical/input/generation contract does not match this probe')
    if (args.role == 'adapted') != (args.adapter_dir is not None):
        raise ValueError('only the adapted role must receive --adapter-dir')
    adapter_files = None
    train_binding = None
    if args.adapter_dir is not None:
        args.adapter_dir = args.adapter_dir.resolve()
        fixed = probe_config.get('fixed_adapter_dir')
        if fixed and args.adapter_dir != repo_path(fixed).resolve():
            raise ValueError('adapter differs from predeclared checkpoint')
        train_path = args.adapter_dir.parent / 'TRAIN_RESULT.json'
        train = read_json(train_path)
        if train['method'] != 'qv_r8' or train['seed'] != 42 or train['split'] != 'train_all' or train['completed_updates'] != train['planned_updates']:
            raise ValueError('expected completed stage2 qv_r8 seed42 checkpoint')
        adapter_files = {path.name: sha_file(path) for path in sorted(args.adapter_dir.iterdir()) if path.is_file()}
        if adapter_files != train['adapter_files_sha256']:
            raise ValueError('adapter bytes differ from training completion binding')
        train_binding = {'path': str(train_path), 'sha256': sha_file(train_path), 'method': 'qv_r8', 'seed': 42}
    manifest_path = args.asset_root / 'prepared/manifest.json'
    manifest = read_manifest(manifest_path)
    rows = manifest['splits']['dev']
    images = manifest['images']
    if len(rows) != 1617 or len({row['image_id'] for row in rows}) != 1024:
        raise ValueError('expected frozen internal dev with 1617 QA / 1024 images')
    import torch
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('launch one independent worker per visible GPU; no DDP')
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
        raise ValueError('this pilot is declared for A100 80GB')
    hardware = {'name': props.name, 'total_memory_bytes': props.total_memory,
                'compute_capability': [props.major, props.minor], 'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'torch_cuda_version': torch.version.cuda}
    common_lock = {'schema_version': 'rarevlm.visual_budget_input.v1', 'task': 'textvqa', 'split': 'dev',
                   'role': args.role, 'probe_config_sha256': sha_file(args.config),
                   'reference_config_sha256': sha_file(reference_path), 'manifest_sha256': manifest['manifest_sha256'],
                   'manifest_file_sha256': sha_file(manifest_path),
                   'model_provenance_sha256': sha_file(args.model_root.parent / 'model_provenance.json'),
                   'model_config_sha256': sha_file(args.model_root / 'config.json'),
                   'model_revision': config['model_revision'], 'adapter_files_sha256': adapter_files,
                   'training_binding': train_binding, 'source_sha256': source_hashes(),
                   'packages': package_versions(), 'hardware': hardware,
                   'batch_size': args.batch_size, 'precision': 'fp32', 'vision_precision': 'fp32',
                   'quantization': 'nf4_language_only', 'attention_implementation': 'sdpa',
                   'tf32_matmul': False, 'tf32_cudnn': False, 'min_pixels': config['min_pixels'],
                   'max_new_tokens': 32, 'do_sample': False, 'head_last_position_only': True,
                   'input': 'image and question only; no reference answers, targets or OCR tokens',
                   'input_truncation': False, 'statistical_unit': 'image cluster; retain all same-image questions'}
    role_root = args.output_root / 'runs' / args.role
    status_path = role_root / 'STATUS.json'
    status = {'schema_version': 'rarevlm.visual_budget_worker_status.v1', 'role': args.role,
              'budgets': budgets, 'phase': 'starting', 'started_at': now(),
              'cells': {str(t): {'status': 'pending', 'completed_questions': 0, 'total_questions': len(rows)} for t in budgets},
              'hardware': hardware}
    def update_status(budget=None, count=None, state=None):
        if budget is not None:
            status['current_budget'] = budget
            status['cells'][str(budget)].update(status=state, completed_questions=count)
        status['updated_at'] = now()
        write_json(status_path, status)
    update_status()
    try:
        validation = verify_dev_images(rows, images, manifest_path.parent)
        status['image_validation'] = validation
        status['phase'] = 'loading_model'
        update_status()
        begun = time.monotonic()
        model, _ = load(args.model_root, config, device, adapter_dir=args.adapter_dir, quantization='nf4_language_only')
        status['model_load_seconds'] = time.monotonic() - begun
        model.config.use_cache = True
        model.generation_config.temperature = 1.0
        model.generation_config.top_p = 1.0
        model.generation_config.top_k = 50
        image_token_id = int(model.config.image_token_id)
        if image_token_id != 151655:
            raise ValueError('unexpected model image-token id')
        hook = model.get_output_embeddings().register_forward_pre_hook(lambda module, items: (items[0][:, -1:, :],))
        try:
            status['phase'] = 'preflight'
            update_status()
            run_preflight(model, args, config, common_lock, rows, images, manifest_path.parent, device, image_token_id)
            if not args.preflight_only:
                status['phase'] = 'evaluating'
                update_status()
                for budget in budgets:
                    run_cell(model, args, config, common_lock, rows, images, manifest_path.parent, device, image_token_id,
                             budget, update_status)
            status['phase'] = 'preflight_completed' if args.preflight_only else 'completed'
            status['completed_at'] = now()
            update_status()
        finally:
            hook.remove()
    except Exception as error:
        status.update(phase='failed', error_type=type(error).__name__, error=str(error))
        update_status()
        raise


if __name__ == '__main__':
    main()
