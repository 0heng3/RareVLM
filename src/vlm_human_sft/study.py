"""Full-data matched continuation; reuse FP32 LoRA/CE, checkpoints and evaluator."""
import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
import random
import sys
import time

from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT
from .evaluate import evaluate, lora_hash, primary_score, read, source_files as eval_sources


def slots(text, human, condition, seed):
    from vlm_deep.common import content_sha
    if condition not in ('control', 'human') or len(human) > len(text):
        raise ValueError('Invalid matched training population')
    extra = human if condition == 'human' else sorted(text, key=lambda row:
        content_sha(['human_sft_text_repeat_v1', seed, row['question_id']]))[:len(human)]
    return [(row, 'text') for row in text] + [(row, 'human' if condition == 'human' else 'repeat_text') for row in extra]


def update_plan(n, batch, seed):
    import torch
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    return [order[start:start+batch] for start in range(0, n, batch)]


def data(config):
    from vlm_deep.common import read_manifest
    reference = read(config['baseline_config'])
    manifests = {task: read_manifest(reference[task+'_manifest']) for task in ('human', 'text')}
    if not manifests['human'].get('training_allowed'):
        raise ValueError('Human source is not admitted training data')
    for task, expected in [('text', 32985), ('human', 3893)]:
        if len(manifests[task]['splits']['train_all']) != expected:
            raise ValueError('The complete preregistered training corpus is required')
    # New cross-source combination: check both hash axes once, not just ID namespaces.
    for key in ('sha256', 'pixel_sha256'):
        train, heldout = set(), set()
        for manifest in manifests.values():
            train.update(manifest['images'][row['image_id']][key] for row in manifest['splits']['train_all'])
            heldout.update(manifest['images'][row['image_id']][key]
                           for split in ('dev', 'final') for row in manifest['splits'][split])
        if train & heldout:
            raise ValueError('Combined training/heldout content overlaps')
    return reference, manifests


def scientific_files(config_file, config):
    reference = read(config['baseline_config'])
    paths = [Path(config_file), Path(config['baseline_config']), Path(config['protocol']),
             PROJECT_ROOT/'docs/VLM_HUMAN_SFT_BASELINE_V1.md', Path(__file__),
             SOURCE_ROOT/'vlm_human_sft/__init__.py', SOURCE_ROOT/'vlm_fp32_adapter/loss.py',
             SOURCE_ROOT/'vlm_object_train/train.py', SOURCE_ROOT/'research_runtime/queue.py',
             SOURCE_ROOT/'vlm_sft_resolution/experiment.py',
             SOURCE_ROOT/'vlm_deep/third_party/vqa_eval_metric.py',
             PROJECT_ROOT/'tools/run_human_sft_repair_v1.py',
             Path(reference['human_manifest']), Path(reference['text_manifest']), Path(reference['reference_config'])]
    paths += eval_sources()
    paths += [Path(reference['initial_adapter'])/name for name in ('adapter_config.json', 'adapter_model.safetensors')]
    paths += sorted(Path(reference['model_root']).glob('*.json'))
    return sorted(set(paths))


def register(config_file, config):
    from vlm_deep.common import sha_file, write_json
    from vlm_object_train.train import lock_json, package_lock
    root = Path(config['output_root'])
    initial = {task: read(root/'initial'/task/'COMPLETE.json') for task in ('human', 'text')}
    if any(result['status'] != 'completed' for result in initial.values()):
        raise ValueError('Both full initial baselines must finish before registration')
    if len({result['lora_sha256'] for result in initial.values()}) != 1:
        raise ValueError('Initial baselines used different LoRA values')
    if 100-initial['human']['score_percent']+1e-12 < config['minimum_human_gain_pp']:
        write_json(root/'COMPLETION.json', {'status': 'completed', 'training_started': False,
            'pilot_passed': False, 'goal_complete': False, 'reason': 'insufficient headroom under the preregistered gate'})
        return False
    data(config)
    files = scientific_files(config_file, config)
    for task in ('human', 'text'):
        files += [root/'initial'/task/name for name in ('COMPLETE.json', 'predictions.jsonl', 'INPUT_LOCK.json')]
    binding = {'files': {str(path): sha_file(path) for path in files}, 'packages': package_lock(),
               'initial_lora_sha256': initial['human']['lora_sha256'],
               'source_counts': {'control': {'text': 32985, 'repeat_text': 3893},
                                 'human': {'text': 32985, 'human': 3893}},
               'updates': 2305, 'qa_exposures_per_arm': 36878,
               'base_weight_files': {path.name: {'bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}
                    for path in sorted(Path(read(config['baseline_config'])['model_root']).glob('*.safetensors'))}}
    lock_json(root/'STUDY_INPUT_LOCK.json', binding)
    return True


def train(config, condition, reference, manifests, directory):
    import numpy as np
    import torch
    from vlm_deep.common import collate, content_sha, encode, sha_file, to_device, write_json
    from vlm_fp32_adapter.model import load
    from vlm_fp32_adapter.loss import answer_loss
    from vlm_object_train.train import learning_rate, lock_json, resume_checkpoint, save_checkpoint
    from vlm_probe.evaluate import processor_for
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/'TRAIN_COMPLETE.json').exists():
        return read(directory/'TRAIN_COMPLETE.json')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed']); torch.cuda.manual_seed_all(config['seed'])
    records = slots(manifests['text']['splits']['train_all'], manifests['human']['splits']['train_all'], condition, config['seed'])
    plan = update_plan(len(records), config['effective_batch'], config['seed'])
    schedule_sha = content_sha(plan)
    lineage = {'study_input_lock_sha256': sha_file(Path(config['output_root'])/'STUDY_INPUT_LOCK.json'),
               'condition': condition, 'schedule_sha256': schedule_sha,
               'gpu_uuid': os.environ['CUDA_VISIBLE_DEVICES'], 'planned_updates': len(plan)}
    lock_json(directory/'INPUT_LOCK.json', lineage); lineage_sha = content_sha(lineage)
    device = torch.device('cuda:0')
    model, binding = load(reference['model_root'], reference['initial_adapter'], device,
                           training=True, dropout=0., gradient_checkpointing=True)
    initial_hash = lora_hash(model)
    if initial_hash != read(Path(config['output_root'])/'STUDY_INPUT_LOCK.json')['initial_lora_sha256']:
        raise ValueError('Training initial LoRA differs from fresh baselines')
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if sum(parameter.numel() for parameter in parameters) != config['expected_trainable']:
        raise ValueError('Trainable parameter budget changed')
    processor = processor_for(Path(reference['model_root']), read(reference['reference_config']), config['train_budget'])
    optimizer = torch.optim.AdamW(parameters, lr=config['lr'], weight_decay=config['weight_decay'])
    completed, stats = 0, {'questions': 0, 'loss_sum': 0., 'active_seconds': 0., 'visual_tokens': 0,
                           'assistant_tokens': 0, 'trace_bytes': 0, 'source_counts': {}, 'source_loss_sums': {},
                           'peak_allocated_gib': 0., 'peak_reserved_gib': 0.}
    if (directory/'LATEST.json').exists():
        completed, stats = resume_checkpoint(directory, model, optimizer, lineage_sha, schedule_sha, len(plan))
    trace_path = directory/'INPUT_TRACES.jsonl'
    if not trace_path.exists():
        trace_path.touch()
    if trace_path.stat().st_size < stats['trace_bytes']:
        raise ValueError('Training input trace is shorter than its checkpoint')
    started, prior_seconds, durations = time.monotonic(), stats['active_seconds'], []
    torch.cuda.reset_peak_memory_stats(device)
    with trace_path.open('r+b') as trace:
        trace.truncate(stats['trace_bytes']); trace.seek(0, os.SEEK_END)
        for step in range(completed, len(plan)):
            step_started = time.monotonic()
            model.train(); model.get_base_model().visual.eval(); model.config.use_cache = False
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group['lr'] = learning_rate(step, len(plan), config)
            indices = plan[step]
            for start in range(0, len(indices), config['microbatch']):
                selected = [records[index] for index in indices[start:start+config['microbatch']]]
                encoded, evidence = [], []
                for row, source in selected:
                    task = 'human' if source == 'human' else 'text'
                    manifest = manifests[task]; image = manifest['images'][row['image_id']]
                    path = Path(reference[task+'_manifest']).parent/image['path']
                    sample = encode(processor, row, path, True)
                    visual = int(sample['image_grid_thw'].prod())//4
                    length, supervised = int(sample['input_ids'].shape[1]), int(sample['labels'].ne(-100).sum())
                    if length > config['max_sequence_length'] or visual > config['train_budget']:
                        raise ValueError('Actual input exceeds budget; no drop/truncation permitted')
                    encoded.append(sample)
                    evidence.append({'update': step+1, 'source': source, 'question_id': row['question_id'],
                        'image_id': row['image_id'], 'question_sha256': content_sha(row['question']),
                        'image_sha256': image['sha256'], 'sequence_length': length,
                        'visual_tokens': visual, 'assistant_tokens': supervised,
                        'image_grid_thw': sample['image_grid_thw'].tolist()})
                batch = to_device(collate(encoded, processor.tokenizer.pad_token_id), device)
                # No global AMP: preserve FP32 adapter computation; CE itself explicitly FP32.
                losses, _, _ = answer_loss(model, batch)
                if losses.dtype != torch.float32 or not bool(torch.isfinite(losses).all()):
                    raise FloatingPointError('Training CE is not finite FP32')
                loss = losses.sum()/len(indices); loss.backward()
                values = losses.detach().cpu().tolist()
                for entry, value in zip(evidence, values):
                    source = entry['source']; entry['padded_sequence_length'] = int(batch['input_ids'].shape[1])
                    trace.write((json.dumps(entry, allow_nan=False)+'\n').encode())
                    stats['source_counts'][source] = stats['source_counts'].get(source, 0)+1
                    stats['source_loss_sums'][source] = stats['source_loss_sums'].get(source, 0.)+value
                    stats['visual_tokens'] += entry['visual_tokens']; stats['assistant_tokens'] += entry['assistant_tokens']
                stats['loss_sum'] += sum(values)
                del batch, encoded, losses, loss
            norm = torch.nn.utils.clip_grad_norm_(parameters, config['gradient_clip_norm'], error_if_nonfinite=True)
            if float(norm) <= 0:
                raise ValueError('Training gradient is zero')
            optimizer.step(); stats['questions'] += len(indices)
            stats['active_seconds'] = prior_seconds+time.monotonic()-started
            stats['trace_bytes'] = trace.tell()
            stats['peak_allocated_gib'] = max(stats['peak_allocated_gib'], torch.cuda.max_memory_allocated(device)/2**30)
            stats['peak_reserved_gib'] = max(stats['peak_reserved_gib'], torch.cuda.max_memory_reserved(device)/2**30)
            if stats['peak_allocated_gib'] > config['memory_limit_gib']:
                raise RuntimeError('Training exceeds declared allocated-memory allowance')
            if step == 0 or (step+1)%config['checkpoint_every_updates'] == 0 or step+1 == len(plan):
                trace.flush(); os.fsync(trace.fileno())
                save_checkpoint(directory, model, optimizer, step+1, stats, lineage_sha, schedule_sha)
            durations.append(time.monotonic()-step_started)
            durations = durations[-50:]
            if step == 0 or (step+1)%10 == 0 or step+1 == len(plan):
                recent = sum(durations)/len(durations)
                progress = {'status': 'training', 'condition': condition, 'completed_updates': step+1,
                    'planned_updates': len(plan), 'recent_seconds_per_update': recent,
                    'remaining_training_seconds_estimate': recent*(len(plan)-step-1),
                    'gradient_norm': float(norm), 'ce_dtype': 'float32', 'lr': optimizer.param_groups[0]['lr'],
                    **stats}
                write_json(directory/'PROGRESS.json', progress); print(json.dumps(progress), flush=True)
    model.save_pretrained(directory/'adapter', safe_serialization=True)
    stats['active_seconds'] = prior_seconds+time.monotonic()-started
    expected = {'text': 32985, 'human': 3893} if condition == 'human' else {'text': 32985, 'repeat_text': 3893}
    if stats['source_counts'] != expected or stats['questions'] != 36878:
        raise ValueError('Full source exposure counts differ')
    result = {'status': 'completed', 'condition': condition, 'updates': len(plan), **stats,
              'binding': binding, 'initial_lora_sha256': initial_hash, 'final_lora_sha256': lora_hash(model),
              'schedule_sha256': schedule_sha, 'input_trace_sha256': sha_file(trace_path),
              'quality_gain_evaluated': False}
    write_json(directory/'TRAIN_COMPLETE.json', result); write_json(directory/'PROGRESS.json', result)
    del model, parameters, optimizer
    gc.collect(); torch.cuda.empty_cache()
    return result


def condition_worker(config, condition):
    import torch
    from vlm_deep.common import write_json
    from vlm_fp32_adapter.model import load
    from vlm_probe.evaluate import processor_for
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(config['memory_limit_gib']*2**30/torch.cuda.get_device_properties(0).total_memory)
    reference, manifests = data(config)
    directory = Path(config['output_root'])/'runs'/condition
    result = train(config, condition, reference, manifests, directory/'train')
    model, _ = load(reference['model_root'], directory/'train/adapter', torch.device('cuda:0'), training=False)
    if lora_hash(model) != result['final_lora_sha256']:
        raise ValueError('Final evaluation reloaded different LoRA values')
    processor = processor_for(Path(reference['model_root']), read(reference['reference_config']), reference['evaluation_budget'])
    scores = {task: evaluate(model, processor, reference, task, directory/task) for task in ('human', 'text')}
    write_json(directory/'COMPLETE.json', {'status': 'completed', 'condition': condition,
               'train': result, 'evaluations': scores, 'goal_complete': False})


def analyze(config):
    from vlm_deep.common import NORMALIZER, read_manifest, write_json
    from vlm_sft_resolution.experiment import paired_interval
    reference = read(config['baseline_config']); root = Path(config['output_root'])
    manifests = {task: read_manifest(reference[task+'_manifest']) for task in ('human', 'text')}
    predictions = {}
    for role in ('initial', 'control', 'human'):
        predictions[role] = {}
        for task in ('human', 'text'):
            directory = root/'initial'/task if role == 'initial' else root/'runs'/role/task
            rows = [json.loads(line) for line in (directory/'predictions.jsonl').read_text().splitlines()]
            original = manifests[task]['splits']['dev']
            if len(rows) != len(original) or [(row['question_id'], row['image_id']) for row in rows] != [
                    (row['question_id'], row['image_id']) for row in original]:
                raise ValueError('Final evaluation identities differ')
            for row, gold in zip(rows, original):
                value = primary_score(row['prediction'], gold, task)
                if row['primary_score'] != value:
                    raise ValueError('Saved primary score differs from predictions')
                row['score'] = value
            predictions[role][task] = rows
    def delta(left, right, task):
        return paired_interval(predictions[left][task], predictions[right][task], config['bootstrap_replicates'], config['bootstrap_seed'])
    comparisons = {'human_vs_initial': delta('initial', 'human', 'human'),
                   'human_vs_control': delta('control', 'human', 'human'),
                   'text_vs_initial': delta('initial', 'human', 'text'),
                   'text_vs_control': delta('control', 'human', 'text')}
    frequency = Counter(NORMALIZER(row['target']) for task in ('text', 'human')
                        for row in manifests[task]['splits']['train_all'])
    slices = {}
    for label, select in [('unseen_0', lambda n:n == 0), ('rare_1_5', lambda n:1 <= n <= 5), ('frequent_gt20', lambda n:n > 20)]:
        keep = [select(frequency[NORMALIZER(row['target'])]) for row in manifests['text']['splits']['dev']]
        before = [row for row, valid in zip(predictions['initial']['text'], keep) if valid]
        after = [row for row, valid in zip(predictions['human']['text'], keep) if valid]
        slices[label] = paired_interval(before, after, config['bootstrap_replicates'], config['bootstrap_seed'])
    criteria = {}
    for key in ('human_vs_initial', 'human_vs_control'):
        criteria[key+'_minimum_gain'] = comparisons[key]['delta_pp']+1e-12 >= config['minimum_human_gain_pp']
        criteria[key+'_positive_interval'] = comparisons[key]['image_cluster_95_ci_pp'][0] > 0
    criteria['text_initial_point_noninferior'] = comparisons['text_vs_initial']['delta_pp'] >= -config['text_margin_pp']
    criteria['text_initial_interval_noninferior'] = comparisons['text_vs_initial']['image_cluster_95_ci_pp'][0] > -config['text_margin_pp']
    for key in ('rare_1_5', 'frequent_gt20'):
        criteria[key+'_point_protection'] = slices[key]['delta_pp'] >= -config['text_margin_pp']
    scores = {role: {task:100*sum(row['score'] for row in rows)/len(rows) for task, rows in tasks.items()}
              for role, tasks in predictions.items()}
    human_slices = {}
    human_gold = manifests['human']['splits']['dev']
    for label, select in [('yes', lambda row:row['target'] == 'yes'), ('no', lambda row:row['target'] == 'no'),
            ('existence_prefix', lambda row:row['question'].lower().startswith(('is there ', 'are there '))),
            ('other_questions', lambda row:not row['question'].lower().startswith(('is there ', 'are there '))),
            ('votes9', lambda row:row['answer_vote_count'] == 9), ('votes10', lambda row:row['answer_vote_count'] == 10)]:
        keep = [select(row) for row in human_gold]
        human_slices[label] = paired_interval(
            [row for row, valid in zip(predictions['initial']['human'], keep) if valid],
            [row for row, valid in zip(predictions['human']['human'], keep) if valid],
            config['bootstrap_replicates'], config['bootstrap_seed'])
    changed = {}
    for task in ('human', 'text'):
        changed[task] = [{'question_id': before['question_id'], 'image_id': before['image_id'],
                          'before': before['prediction'], 'after': after['prediction'],
                          'before_score': before['score'], 'after_score': after['score']}
                         for before, after in zip(predictions['initial'][task], predictions['human'][task])
                         if before['prediction'] != after['prediction']]
    write_json(root/'CHANGED_CASES.json', changed)
    result = {'status': 'completed', 'pilot_passed': all(criteria.values()), 'goal_complete': False,
              'scores_percent': scores, 'comparisons': comparisons, 'text_slices': slices,
              'human_slices': human_slices, 'criteria': criteria,
              'frequency_basis': 'actual candidate training targets: 32985 TextVQA + 3893 human yes/no',
              'scope': 'single continuation seed42, previously exposed internal images; not formal or three-seed evidence',
              'next': 'replicate declared seeds and register formal transfer evaluation' if all(criteria.values()) else
                      'retain negative result; no LR/epoch grid or automatic seed/formal expansion'}
    write_json(root/'ANALYSIS.json', result)
    write_json(PROJECT_ROOT/'results/vlm_human_sft_repair_v1_metrics.json', result)
    lines = ['# 人工监督完整 SFT v1 结果', '', '单 continuation seed42，完整两分支；不称三种子或正式测试提升。', '',
             '| 模型 | 人工yes/no正确率 | TextVQA |', '| --- | ---: | ---: |']
    for role, values in scores.items():
        lines.append(f"| {role} | {values['human']:.6f} | {values['text']:.6f} |")
    lines += ['', '联合门槛通过：'+str(result['pilot_passed']), '',
              '完整区间、切片、全部变化预测、实际训练输入与资源记录见 outputs/vlm_human_sft_repair_v1。', '', result['next']]
    (PROJECT_ROOT/'docs/VLM_HUMAN_SFT_REPAIR_V1_RESULTS.md').write_text('\n'.join(lines)+'\n')
    write_json(root/'COMPLETION.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--condition', choices=['control', 'human'])
    args = parser.parse_args(); config = read(args.config)
    if args.condition:
        condition_worker(config, args.condition); return
    if not register(args.config.resolve(), config):
        return
    from research_runtime.queue import run_queue
    from vlm_deep.common import sha_file
    root = Path(config['output_root'])
    jobs = [{'key': condition, 'command': [sys.executable, '-u', '-m', 'vlm_human_sft.study',
              '--config', str(args.config.resolve()), '--condition', condition],
              'completion_marker': str(root/'runs'/condition/'COMPLETE.json')} for condition in ('control', 'human')]
    run_queue(root/'training_queue', jobs, {'study_input_lock_sha256': sha_file(root/'STUDY_INPUT_LOCK.json')})
    print(json.dumps(analyze(config)), flush=True)


if __name__ == '__main__':
    main()
