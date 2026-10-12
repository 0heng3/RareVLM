"""Resume frozen FP32 VLM evaluation on A100, including legacy shard caches."""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from vlm_deep.common import collate, content_sha, encode, read_manifest, sha_file, to_device, write_json
from vlm_deep.evaluate import aggregate, score_prediction
from vlm_deep.model import load


FROZEN_ROOT = Path(__file__).resolve().parents[1] / 'vlm_deep'
SCIENCE_KEYS = ('task', 'split', 'manifest_sha256', 'config_sha256',
                'model_provenance_sha256', 'quantization', 'precision',
                'adapter_files_sha256', 'generation', 'input')


def frozen_sources():
    return {str(f.relative_to(FROZEN_ROOT)): sha_file(f)
            for f in FROZEN_ROOT.rglob('*.py')}


def load_data(asset_root, task, split):
    if task == 'textvqa':
        manifest = asset_root / 'prepared/manifest.json'
        data = read_manifest(manifest)
        rows = data['splits'][split]
    else:
        manifest = asset_root / 'transfer' / task / 'manifest.json'
        data = json.loads(manifest.read_text())
        if content_sha({k: v for k, v in data.items() if k != 'manifest_sha256'}) != data['manifest_sha256']:
            raise ValueError('transfer manifest changed')
        rows = data['records']
    if len({r['question_id'] for r in rows}) != len(rows):
        raise ValueError('duplicate manifest question IDs')
    return rows, data['images'], manifest.parent, data['manifest_sha256']


def complete_jsonl(path):
    """Read complete writer records, leaving an interrupted trailing line untouched."""
    lines = path.read_text().splitlines(keepends=True)
    ignored = bool(lines and not lines[-1].endswith('\n'))
    if ignored:
        lines = lines[:-1]
    return [json.loads(line) for line in lines], ignored


def validate_prediction(item, row, images, task):
    if item['question_id'] != row['question_id'] or item['image_id'] != row['image_id']:
        raise ValueError('cache question/image identity changed')
    if item['question_sha256'] != content_sha(row['question']):
        raise ValueError('cache question hash changed')
    if item['image_sha256'] != images[row['image_id']]['sha256']:
        raise ValueError('cache image hash changed')
    expected = score_prediction(item['prediction'], row, task)
    score = float(item['score'])
    if not math.isfinite(score) or not math.isclose(score, expected, rel_tol=0, abs_tol=1e-12):
        raise ValueError('cache task score changed')
    if item.get('subcategory', '') != row.get('subcategory', ''):
        raise ValueError('cache subcategory changed')


def read_cache(directory, science, full_rows, images, task, legacy=False):
    """Validate all complete cached records against the original benchmark order."""
    lock_path = directory / 'EVAL_INPUT_LOCK.json'
    lock = json.loads(lock_path.read_text())
    for key in SCIENCE_KEYS:
        if lock.get(key) != science.get(key):
            raise ValueError(f'cache science lock changed: {key}')
    if lock.get('source_sha256') != frozen_sources():
        raise ValueError('frozen evaluation source changed from cache lock')
    if legacy and (lock.get('schema_version') != 'rarevlm.vlm_eval.v2' or lock.get('n') != len(full_rows)):
        raise ValueError('expected complete-benchmark legacy v2 input lock')
    world = lock['world_size']
    selected = full_rows if legacy else full_rows[:lock['n']]
    lookup = {r['question_id']: r for r in selected}
    cache, files, ignored_tails = {}, {}, []
    lock_sha = sha_file(lock_path)
    for rank in range(world):
        path = directory / f'predictions_rank{rank}.jsonl'
        if not path.exists():
            continue
        files[path.name] = sha_file(path)
        items, ignored = complete_jsonl(path)
        if ignored:
            ignored_tails.append(path.name)
        assigned = selected[rank::world]
        positions = {r['question_id']: i for i, r in enumerate(assigned)}
        indices = []
        for line_number, item in enumerate(items, 1):
            qid = item['question_id']
            if qid not in positions or qid not in lookup or qid in cache:
                raise ValueError('cache duplicate or wrong shard question ID')
            indices.append(positions[qid])
            validate_prediction(item, lookup[qid], images, task)
            value = dict(item)
            if legacy:
                value['prediction_source'] = {
                    'kind': 'reused_legacy', 'directory': str(directory.resolve()),
                    'lock_sha256': lock_sha, 'file_sha256': files[path.name],
                    'rank': rank, 'line_number': line_number,
                    'hardware': '4 x Tesla V100-PCIE-16GB (recorded prior run)',
                }
            elif 'prediction_source' not in value:
                raise ValueError('A100 cache lacks prediction source')
            cache[qid] = value
        if indices != sorted(indices):
            raise ValueError('cache shard order changed')
        if legacy and indices != list(range(len(items))):
            raise ValueError('legacy cache is not the original contiguous shard prefix')
    return cache, {'directory': str(directory.resolve()), 'lock_sha256': lock_sha,
                   'prediction_files_sha256': files, 'complete_questions': len(cache),
                   'ignored_incomplete_trailing_lines': ignored_tails}


def save_partial(path, assigned, cache):
    """Atomic, ordered sparse cache; unlike a prefix, it can contain legacy holes."""
    temporary = path.with_suffix('.jsonl.tmp')
    with temporary.open('w') as stream:
        for row in assigned:
            if row['question_id'] in cache:
                stream.write(json.dumps(cache[row['question_id']], ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-root', type=Path, required=True)
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--adapter-dir', type=Path)
    parser.add_argument('--reuse-dir', type=Path)
    parser.add_argument('--quantization', choices=['original', 'nf4_language_only'], default='nf4_language_only')
    parser.add_argument('--task', choices=['textvqa', 'docvqa', 'mmstar'], default='textvqa')
    parser.add_argument('--split', choices=['dev', 'final'], default='final')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'configs/vlm_deep_v2.json')
    parser.add_argument('--limit-questions', type=int, help='Separate preflight subset; never emits EVALUATION.json.')
    parser.add_argument('--stop-after-new-questions', type=int,
                        help='Global new-question budget for a resumable timing probe, outside the science lock.')
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit_questions is not None and args.limit_questions < 1) or (
            args.stop_after_new_questions is not None and args.stop_after_new_questions < 1):
        raise ValueError('batch size and optional question limits must be positive')
    if args.reuse_dir and args.reuse_dir.resolve() == args.output_dir.resolve():
        raise ValueError('legacy reuse directory must remain separate and read-only')
    config = json.loads(args.config.read_text())
    full_rows, images, image_root, data_sha = load_data(args.asset_root, args.task, args.split)
    rows = full_rows[:args.limit_questions] if args.limit_questions is not None else full_rows
    adapter_files = {f.name: sha_file(f) for f in args.adapter_dir.iterdir() if f.is_file()} if args.adapter_dir else None
    science = {'task': args.task, 'split': args.split, 'manifest_sha256': data_sha,
               'config_sha256': content_sha(config),
               'model_provenance_sha256': sha_file(args.model_root.parent / 'model_provenance.json'),
               'quantization': args.quantization, 'precision': 'fp32', 'adapter_files_sha256': adapter_files,
               'generation': {'max_new_tokens': config['max_new_tokens'], 'do_sample': False,
                              'min_pixels': config['min_pixels'], 'max_pixels': config['max_pixels'],
                              'head_last_position_only': True},
               'input': 'image/question only; references scored after generation'}
    legacy_cache, reuse = ({}, None)
    if args.reuse_dir:
        legacy_cache, reuse = read_cache(args.reuse_dir, science, full_rows, images, args.task, legacy=True)

    import torch
    import torch.distributed as dist
    rank = int(os.environ.get('RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    local = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    torch.set_num_threads(2)
    # A100 enables TF32 in some software versions; the scientific path remains FP32.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    if world > 1:
        dist.init_process_group('nccl')
    props = torch.cuda.get_device_properties(local)
    gpu = {'rank': rank, 'name': props.name, 'total_memory_bytes': props.total_memory,
           'compute_capability': [props.major, props.minor]}
    hardware = [None] * world
    if world > 1:
        dist.all_gather_object(hardware, gpu)
    else:
        hardware[0] = gpu
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    runtime = {'world_size': world, 'batch_size_per_rank': args.batch_size,
               'hardware': hardware, 'tf32_matmul': False, 'tf32_cudnn': False,
               'dtype': 'fp32', 'cache_partition': 'question_id lookup, sparse ordered rank shards'}
    lock = {'schema_version': 'rarevlm.vlm_eval.a100.v1', **science,
            'world_size': world, 'batch_size': args.batch_size, 'n': len(rows),
            'benchmark_n': len(full_rows), 'runtime_layout': runtime,
            'evaluation_scope': 'preflight_subset' if args.limit_questions is not None else 'full_benchmark',
            'source_sha256': frozen_sources(),
            'migration_source_sha256': {'vlm_a100/evaluate.py': sha_file(Path(__file__))},
            'reuse_source': reuse}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = args.output_dir / 'EVAL_INPUT_LOCK.json'
    if lock_path.exists() and json.loads(lock_path.read_text()) != lock:
        raise ValueError('evaluation science/runtime/code/reuse lock changed')
    if rank == 0:
        write_json(lock_path, lock)
    if world > 1:
        dist.barrier(device_ids=[local])
    current_cache, _ = read_cache(args.output_dir, science, full_rows, images, args.task)
    cache = dict(legacy_cache)
    for qid, item in current_cache.items():
        if qid in cache and any(cache[qid][key] != item[key] for key in (
                'prediction', 'score', 'question_sha256', 'image_sha256')):
            raise ValueError('current cache conflicts with legacy prediction')
        cache[qid] = item
    assigned = rows[rank::world]
    assigned_ids = {r['question_id'] for r in assigned}
    cache = {qid: item for qid, item in cache.items() if qid in assigned_ids}
    reused_legacy = sum(item['prediction_source']['kind'] == 'reused_legacy' for item in cache.values())
    resumed_a100 = len(cache) - reused_legacy
    image_checked = set()
    # The cached image SHA is checked above, and its current file bytes are checked once.
    for row in assigned:
        if row['question_id'] in cache and row['image_id'] not in image_checked:
            image = images[row['image_id']]
            if sha_file(image_root / image['path']) != image['sha256']:
                raise ValueError('cached image bytes changed')
            image_checked.add(row['image_id'])
    partial = args.output_dir / f'predictions_rank{rank}.jsonl'
    save_partial(partial, assigned, cache)
    missing = [r for r in assigned if r['question_id'] not in cache]
    local_budget = len(missing)
    if args.stop_after_new_questions is not None:
        local_budget = min(local_budget, max(0, (args.stop_after_new_questions + world - 1 - rank) // world))
    to_generate = missing[:local_budget]
    model = processor = None
    for active_rank in range(world):
        if rank == active_rank and to_generate:
            model, processor = load(args.model_root, config, device, adapter_dir=args.adapter_dir,
                                    quantization=args.quantization)
            model.config.use_cache = True
            model.generation_config.temperature = 1.0
            model.generation_config.top_p = 1.0
            model.generation_config.top_k = 50
            print('model loaded', args.task, args.quantization, 'rank', rank, flush=True)
        if world > 1:
            dist.barrier(device_ids=[local])
    head_hook = None
    if model is not None:
        head_hook = model.get_output_embeddings().register_forward_pre_hook(
            lambda module, items: (items[0][:, -1:, :],))
    torch.cuda.reset_peak_memory_stats()
    begun = time.monotonic()
    generation_seconds = 0.0
    newly_generated = 0
    with torch.inference_mode():
        for begin in range(0, len(to_generate), args.batch_size):
            chunk = to_generate[begin:begin + args.batch_size]
            for row in chunk:
                if row['image_id'] not in image_checked:
                    image = images[row['image_id']]
                    if sha_file(image_root / image['path']) != image['sha256']:
                        raise ValueError('input image bytes changed')
                    image_checked.add(row['image_id'])
            samples = [encode(processor, r, image_root / images[r['image_id']]['path'], False) for r in chunk]
            batch = to_device(collate(samples, processor.tokenizer.pad_token_id), device)
            torch.cuda.synchronize()
            started = time.monotonic()
            generated = model.generate(**batch, max_new_tokens=config['max_new_tokens'], do_sample=False, use_cache=True)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - started
            generation_seconds += elapsed
            for index, row in enumerate(chunk):
                ids = generated[index, batch['input_ids'].shape[1]:]
                answer = processor.tokenizer.decode(ids, skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=False).strip()
                cache[row['question_id']] = {
                    'question_id': row['question_id'], 'image_id': row['image_id'], 'prediction': answer,
                    'score': score_prediction(answer, row, args.task),
                    'question_sha256': content_sha(row['question']),
                    'image_sha256': images[row['image_id']]['sha256'],
                    'batch_generation_seconds': elapsed, 'batch_size': len(chunk),
                    'generated_nonpad_tokens': int((ids != processor.tokenizer.pad_token_id).sum()),
                    'timing_warmup': begin == 0, 'subcategory': row.get('subcategory', ''),
                    'prediction_source': {'kind': 'generated_a100', 'run_id': run_id,
                                          'rank': rank, 'hardware': gpu,
                                          'runtime_layout_sha256': content_sha(runtime)},
                }
            newly_generated += len(chunk)
            save_partial(partial, assigned, cache)
            if rank == 0 and (begin == 0 or newly_generated % 100 < args.batch_size):
                print('generated', args.task, args.split, len(cache), '/', len(assigned),
                      'rank-local questions; new', newly_generated, flush=True)
    if head_hook is not None:
        head_hook.remove()
    resource = {'rank': rank, 'run_id': run_id, 'hardware': gpu,
                'new_generation_seconds': time.monotonic() - begun if newly_generated else 0.0,
                'new_model_generate_seconds': generation_seconds, 'new_generated_questions': newly_generated,
                'reused_legacy_questions': reused_legacy, 'resumed_a100_questions': resumed_a100,
                'complete_questions': len(cache), 'assigned_questions': len(assigned),
                'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
                'timing_scope': 'A100 new work only; excludes reused legacy timing and model load'}
    write_json(args.output_dir / f'RANK{rank}_RESOURCE.json', resource)
    write_json(args.output_dir / f'RANK{rank}_RESOURCE_{run_id}.json', resource)
    if world > 1:
        dist.barrier(device_ids=[local])
    if rank == 0:
        observed = {}
        for worker in range(world):
            items, ignored = complete_jsonl(args.output_dir / f'predictions_rank{worker}.jsonl')
            if ignored:
                raise ValueError('atomic output unexpectedly has incomplete line')
            worker_assigned = rows[worker::world]
            position = {row['question_id']: i for i, row in enumerate(worker_assigned)}
            indices = []
            for item in items:
                qid = item['question_id']
                if qid in observed or qid not in position:
                    raise ValueError('output duplicate/wrong shard question')
                validate_prediction(item, worker_assigned[position[qid]], images, args.task)
                indices.append(position[qid])
                observed[qid] = item
            if indices != sorted(indices):
                raise ValueError('output cache order changed')
        resources = [json.loads((args.output_dir / f'RANK{r}_RESOURCE.json').read_text()) for r in range(world)]
        ordered = [observed[r['question_id']] for r in rows if r['question_id'] in observed]
        counts = dict(Counter(item['prediction_source']['kind'] for item in ordered))
        progress = {**lock, 'status': 'complete' if len(ordered) == len(rows) else 'incomplete',
                    'completed_questions': len(ordered), 'remaining_questions': len(rows) - len(ordered),
                    'prediction_source_counts': counts, 'mixed_hardware_predictions': len(counts) > 1,
                    'resources': resources, 'stop_after_new_questions': args.stop_after_new_questions,
                    'resource_history_files': sorted(p.name for p in args.output_dir.glob('RANK*_RESOURCE_*Z.json'))}
        write_json(args.output_dir / 'BENCHMARK_PROGRESS.json', progress)
        if len(ordered) == len(rows) and args.limit_questions is None:
            if len({r['question_id'] for r in ordered}) != len(rows):
                raise ValueError('incomplete/duplicate benchmark predictions')
            value, group_scores = aggregate(ordered, args.task)
            write_json(args.output_dir / 'predictions.json', ordered)
            result = {**progress, 'score_percent': value,
                      'question_mean_score_percent': 100 * statistics.mean(r['score'] for r in ordered),
                      'category_scores_percent': group_scores,
                      'unique_images': len({r['image_id'] for r in rows}),
                      'predictions_sha256': sha_file(args.output_dir / 'predictions.json'),
                      'scope': 'public validation benchmark; pretraining overlap unknown; no final-based checkpoint selection'}
            write_json(args.output_dir / 'EVALUATION.json', result)
            print('EVALUATION COMPLETE', json.dumps({k: result[k] for k in (
                'task', 'split', 'n', 'unique_images', 'score_percent')}), flush=True)
        else:
            print('BENCHMARK PROGRESS', json.dumps({k: progress[k] for k in (
                'status', 'completed_questions', 'remaining_questions', 'prediction_source_counts')}), flush=True)
    if world > 1:
        dist.barrier(device_ids=[local])
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
