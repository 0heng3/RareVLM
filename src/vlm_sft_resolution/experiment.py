"""Complete-data resolution SFT; reuse existing loaders, loss, resume and scorer."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import json
import math
import os
from pathlib import Path
import random
import sys
import time

from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT


def read(path):
    return json.loads(Path(path).read_text())


def blocks(order, size):
    return [order[start:start + size] for start in range(0, len(order), size)]


def paired_interval(before, after, replicates, seed):
    import numpy as np
    by_image = defaultdict(lambda: [0., 0])
    if len(before) != len(after) or not before:
        raise ValueError('paired predictions required')
    for a, b in zip(before, after):
        if (a['question_id'], a['image_id']) != (b['question_id'], b['image_id']):
            raise ValueError('prediction identities differ')
        by_image[a['image_id']][0] += b['score'] - a['score']
        by_image[a['image_id']][1] += 1
    data = np.asarray(list(by_image.values()))
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(replicates):
        value = data[rng.integers(len(data), size=len(data))].sum(axis=0)
        samples.append(100 * value[0] / value[1])
    return {'delta_pp': float(100 * data[:, 0].sum() / data[:, 1].sum()),
            'image_cluster_95_ci_pp': np.quantile(samples, [.025, .975]).tolist(),
            'qa': len(before), 'images': len(data)}


def analyze(config):
    from vlm_deep.common import NORMALIZER, write_json
    root = Path(config['output_root'])
    initial = [json.loads(line) for line in Path(config['initial_predictions']).read_text().splitlines()]
    predictions = {'initial': initial}
    for name in config['train_budgets']:
        predictions[name] = [json.loads(line) for line in (root / name / 'dev/predictions.jsonl').read_text().splitlines()]
    manifest = read(config['manifest'])
    dev = manifest['splits']['dev']
    for name, rows in predictions.items():
        if [(str(r['question_id']), r['image_id']) for r in rows] != [(str(r['question_id']), r['image_id']) for r in dev]:
            raise ValueError(f'incomplete/unpaired dev: {name}')
    reference = predictions['initial']
    candidate = predictions['high1024']
    control = predictions['low256']
    comparison = paired_interval(control, candidate, config['bootstrap_replicates'], config['bootstrap_seed'])
    versus_initial = paired_interval(reference, candidate, config['bootstrap_replicates'], config['bootstrap_seed'])
    frequency = Counter(NORMALIZER(row['target']) for row in manifest['splits']['train_all'])
    slices = {}
    for label, select in [('rare_1_5', lambda n: 1 <= n <= 5), ('frequent_gt20', lambda n: n > 20)]:
        mask = [select(frequency[NORMALIZER(r['target'])]) for r in dev]
        left = [r for r, keep in zip(reference, mask) if keep]
        right = [r for r, keep in zip(candidate, mask) if keep]
        slices[label] = paired_interval(left, right, config['bootstrap_replicates'], config['bootstrap_seed'])
    tests = {'beats_fresh_control_by_minimum': comparison['delta_pp'] >= config['minimum_gain_pp'],
             'paired_interval_positive': comparison['image_cluster_95_ci_pp'][0] > 0,
             'beats_initial_by_minimum': versus_initial['delta_pp'] >= config['minimum_gain_pp'],
             **{label + '_point_protection': item['delta_pp'] >= -config['slice_point_margin_pp'] for label, item in slices.items()}}
    result = {'status': 'completed', 'pilot_passed': all(tests.values()), 'goal_complete': False,
              'scores_percent': {name: 100 * sum(r['score'] for r in rows) / len(rows) for name, rows in predictions.items()},
              'high_vs_low': comparison, 'high_vs_initial': versus_initial, 'slices': slices, 'criteria': tests,
              'scope': 'single continuation seed42, explored internal dev; not formal or general capability evidence',
              'next': 'matched seed replication then frozen formal evaluation' if all(tests.values()) else 'stop this candidate; no hyperparameter grid'}
    write_json(root / 'ANALYSIS.json', result)
    return result


def worker(config, condition, input_lock):
    import numpy as np
    import torch
    from vlm_deep.common import answer_loss, collate, consensus, content_sha, encode, read_manifest, sha_file, to_device, write_json
    from vlm_object_train.train import learning_rate, load_actor, lock_json, package_lock, resume_checkpoint, save_checkpoint, trainable_hash
    from vlm_precision.evaluate import load_path
    from vlm_probe.evaluate import evaluate_batch, processor_for

    directory = Path(config['output_root']) / condition
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'COMPLETE.json').exists():
        return
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed']); torch.cuda.manual_seed_all(config['seed'])
    device = torch.device('cuda:0')
    reference = read(config['reference_config'])
    manifest = read_manifest(config['manifest'])
    rows, images = manifest['splits']['train_all'], manifest['images']
    image_root = Path(config['manifest']).parent
    if len(rows) != config['expected_questions']:
        raise ValueError('full training set required')
    budget = config['train_budgets'][condition]
    order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(config['seed'])).tolist()
    updates = blocks(order, config['effective_batch'])
    schedule_sha = content_sha(updates)
    lineage = {'study_lock_sha256': sha_file(input_lock), 'condition': condition, 'budget': budget,
               'schedule_sha256': schedule_sha, 'gpu_uuid': os.environ['CUDA_VISIBLE_DEVICES'],
               'packages': package_lock(), 'planned_updates': len(updates)}
    lock_json(directory / 'INPUT_LOCK.json', lineage)
    lineage_sha = content_sha(lineage)
    print('LOADING_MODEL', condition, flush=True)
    model, parameters, dtypes = load_actor(config, reference, device)
    initial_hash = trainable_hash(model)
    processor = processor_for(Path(config['model_root']), reference, budget)
    optimizer = torch.optim.AdamW(parameters, lr=config['lr'], weight_decay=config['weight_decay'])
    completed = 0
    stats = {'questions': 0, 'loss_sum': 0., 'active_seconds': 0., 'visual_tokens': 0, 'assistant_tokens': 0}
    if (directory / 'LATEST.json').exists():
        completed, stats = resume_checkpoint(directory, model, optimizer, lineage_sha, schedule_sha, len(updates))
    torch.cuda.reset_peak_memory_stats(device)
    loop_start = time.monotonic()
    start_active = stats['active_seconds']
    for step in range(completed, len(updates)):
        indices = updates[step]
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group['lr'] = learning_rate(step, len(updates), config)
        update_loss, update_visual, update_answers = 0., 0, 0
        first_inputs = []
        for begin in range(0, len(indices), config['microbatch']):
            chunk = [rows[i] for i in indices[begin:begin + config['microbatch']]]
            encoded = []
            for row in chunk:
                item = encode(processor, row, image_root / images[row['image_id']]['path'], True)
                visual = int(item['image_grid_thw'].prod()) // 4
                if item['input_ids'].shape[1] > config['max_sequence_length'] or visual > budget:
                    raise ValueError('training input exceeds declared limit; no truncation')
                update_visual += visual
                if step == 0:
                    first_inputs.append({'question_id': row['question_id'], 'image_id': row['image_id'],
                        'image_grid_thw': item['image_grid_thw'].tolist(), 'visual_tokens': visual,
                        'sequence_length': int(item['input_ids'].shape[1]),
                        'assistant_tokens': int((item['labels'] != -100).sum())})
                encoded.append(item)
            batch = to_device(collate(encoded, processor.tokenizer.pad_token_id), device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                losses, _, tokens = answer_loss(model, batch, tail=True)
                if not bool(torch.isfinite(losses).all()):
                    raise FloatingPointError('nonfinite training loss')
                loss = losses.sum() / len(indices)
            loss.backward()
            update_loss += float(losses.detach().sum())
            update_answers += tokens
            del batch, encoded, losses, loss
        norm = torch.nn.utils.clip_grad_norm_(parameters, config['gradient_clip_norm'], error_if_nonfinite=True)
        if step == 0 and not float(norm) > 0:
            raise ValueError('first formal update has no gradient')
        if torch.cuda.max_memory_allocated() / 2**30 > 65:
            raise RuntimeError('new job exceeds its65GiB memory allowance')
        optimizer.step()
        if step == 0:
            write_json(directory / 'FIRST_UPDATE.json', {'status': 'completed', 'formal_updates': 1,
                'mean_question_loss': update_loss / len(indices), 'gradient_norm_before_clip': float(norm),
                'initial_lora_sha256': initial_hash, 'after_update_lora_sha256': trainable_hash(model),
                'inputs': first_inputs, 'scope': 'first real training update; no quality conclusion'})
        stats['questions'] += len(indices)
        stats['loss_sum'] += update_loss
        stats['visual_tokens'] += update_visual
        stats['assistant_tokens'] += update_answers
        stats['active_seconds'] = start_active + time.monotonic() - loop_start
        if step == 0 or (step + 1) % 10 == 0 or step + 1 == len(updates):
            seconds = stats['active_seconds'] / (step + 1)
            progress = {'status': 'training', 'condition': condition, 'completed_updates': step + 1,
                        'planned_updates': len(updates), **stats, 'mean_question_loss': stats['loss_sum'] / stats['questions'],
                        'gradient_norm_before_clip': float(norm), 'remaining_train_seconds_estimate': seconds * (len(updates) - step - 1),
                        'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30}
            write_json(directory / 'PROGRESS.json', progress)
            print(json.dumps(progress), flush=True)
        if step == 0 or (step + 1) % config['checkpoint_every_updates'] == 0 or step + 1 == len(updates):
            save_checkpoint(directory, model, optimizer, step + 1, stats, lineage_sha, schedule_sha)
    model.save_pretrained(directory / 'adapter', safe_serialization=True)
    write_json(directory / 'TRAIN_RESULT.json', {'status': 'completed', 'condition': condition, 'train_visual_budget': budget,
        'completed_updates': len(updates), **stats, 'initial_lora_sha256': initial_hash, 'final_lora_sha256': trainable_hash(model),
        'initial_dtypes': dtypes, 'schedule_sha256': schedule_sha, 'scope': 'CE continuation; elapsed includes input processing and checkpoint IO'})
    del model, optimizer, parameters
    gc.collect(); torch.cuda.empty_cache()
    # Reload using the same BF16 inference policy as the cached initial baseline.
    model, _ = load_path(Path(config['model_root']), reference, device, 'original_bf16', directory / 'adapter')
    processor = processor_for(Path(config['model_root']), reference, config['evaluation_budget'])
    eval_root = directory / 'dev'
    eval_root.mkdir(exist_ok=True)
    prediction_file = eval_root / 'predictions.jsonl'
    dev = manifest['splits']['dev']
    predictions = []
    if prediction_file.exists():
        from vlm_probe.evaluate import complete_cache
        predictions, _ = complete_cache(prediction_file, dev, images, config['evaluation_batch_size'])
    image_token_id = processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
    with prediction_file.open('a') as stream, torch.inference_mode():
        for start in range(len(predictions), len(dev), config['evaluation_batch_size']):
            chunk = dev[start:start + config['evaluation_batch_size']]
            answers, resources = evaluate_batch(model, processor, chunk, images, image_root, device,
                config['evaluation_budget'], image_token_id, 32)
            for row, answer in zip(chunk, answers):
                result = {'question_id': row['question_id'], 'image_id': row['image_id'],
                          'question_sha256': content_sha(row['question']), 'image_sha256': images[row['image_id']]['sha256'],
                          **answer, 'score': consensus(answer['prediction'], row['answers']), **resources,
                          'batch_start': start, 'condition': condition}
                stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
                predictions.append(result)
            stream.flush(); os.fsync(stream.fileno())
            write_json(directory / 'PROGRESS.json', {'status': 'evaluating', 'condition': condition,
                'completed_questions': len(predictions), 'expected_questions': len(dev), 'training_updates': len(updates)})
    result = {'status': 'completed', 'condition': condition, 'questions': len(predictions),
              'score_percent': 100 * sum(row['score'] for row in predictions) / len(predictions),
              'prediction_sha256': sha_file(prediction_file), 'formal_evaluation': False, 'goal_complete': False}
    write_json(directory / 'COMPLETE.json', result)
    print(json.dumps(result), flush=True)


def run(config_file):
    from research_runtime.queue import run_queue
    from vlm_deep.common import sha_file, write_json
    from vlm_object_train.train import lock_json
    config = read(config_file)
    root = Path(config['output_root']); root.mkdir(parents=True, exist_ok=True)
    protocol = PROJECT_ROOT / 'docs/VLM_SFT_RESOLUTION_V1_PROTOCOL.md'
    files = [config_file, Path(__file__), protocol, SOURCE_ROOT / 'vlm_sft_resolution/__init__.py',
             SOURCE_ROOT / 'vlm_deep/common.py', SOURCE_ROOT / 'vlm_object_train/train.py',
             SOURCE_ROOT / 'vlm_precision/evaluate.py', SOURCE_ROOT / 'vlm_probe/evaluate.py',
             Path(config['manifest']), Path(config['reference_config']), Path(config['initial_predictions'])]
    files.extend(sorted(Path(config['initial_adapter']).glob('*')))
    binding = {str(p.resolve()): sha_file(p) for p in files if p.is_file()}
    input_lock = root / 'STUDY_INPUT_LOCK.json'
    lock_json(input_lock, {'files_sha256': binding, 'config': config})
    jobs = [{'key': condition, 'command': [sys.executable, '-u', '-m', 'vlm_sft_resolution.experiment', '--config',
             str(config_file), '--worker', condition], 'completion_marker': str(root / condition / 'COMPLETE.json')}
            for condition in config['train_budgets']]
    run_queue(root / 'queue', jobs, {'study_input_lock_sha256': sha_file(input_lock)})
    result = analyze(config)
    write_json(root / 'COMPLETION.json', result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--worker', choices=['low256', 'high1024'])
    parser.add_argument('--analyze', action='store_true')
    args = parser.parse_args()
    config = read(args.config)
    if args.worker:
        worker(config, args.worker, Path(config['output_root']) / 'STUDY_INPUT_LOCK.json')
    elif args.analyze:
        print(json.dumps(analyze(config)), flush=True)
    else:
        run(args.config.resolve())


if __name__ == '__main__':
    main()
