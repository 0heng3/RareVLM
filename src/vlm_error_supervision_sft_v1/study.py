"""Reviewed actual-error data -> fixed SFT reference -> matched CE / CE+DPO -> generation."""
import argparse
from collections import Counter
from pathlib import Path
import random
import sys
import time

from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT
from vlm_deep.common import NORMALIZER, content_sha, encode, read_manifest, sha_file, write_json
from .common import CONFIG, MINING, ROOT, evaluate, make_batch, paired, read, summarize

ROLES = ('sft', 'dpo_sft')


def prepare():
    import torch
    from vlm_probe.evaluate import processor_for
    torch.set_num_threads(2)
    if (ROOT / 'PRELAUNCH_INPUT_LOCK.json').exists():
        raise FileExistsError('Already frozen; inspect current queue')
    if read(MINING / 'COMPLETION.json')['status'] != 'completed':
        raise ValueError('Actual-error generation is not complete')
    review = read(MINING / 'VISIBLE_REVIEW.json')
    ids = review['admitted_training_question_ids']
    if review['status'] != 'completed' or len(ids) != 32 or len(set(ids)) != 32:
        raise ValueError('Exactly32 actually reviewed unique training pairs required')
    algorithmic = read(MINING / 'TRAINING_CANDIDATES.json')
    reviewed = {q['question_id']: q for q in review['rows']}
    candidates = {q['question_id']: q for q in algorithmic['pairs']}
    if any(q not in algorithmic['review_first'] or not reviewed[q]['supported'] or q not in candidates for q in ids):
        raise ValueError('Selection is outside the registered review cohort or lacks visible support')
    data = read(MINING / 'DATA.json')
    lookup = {q['question_id']: q for q in data['rows']}
    predictions = sum((read(MINING / f'shard_{i}/PREDICTIONS.json') for i in (0, 1)), [])
    prediction_by_id = {q['question_id']: q for q in predictions}
    hard = [{**lookup[q], 'rejected_target': candidates[q]['prediction'], 'study_source': 'preference'} for q in ids]
    # SHA rank, independent from candidate model outcomes; only initial full-score training rows qualify.
    import hashlib
    def key(q):
        return hashlib.sha256(('error_supervision_sft_v1:replay:' + q['image_id'] + ':' + q['question_id']).encode()).hexdigest()
    replay = []
    for stratum in ('singleword', 'multiword'):
        pool = sorted((q for q in predictions if q['fold'] == 'training_pool' and q['vqa_score'] >= .999999
                       and q['stratum'] == stratum and q['question_id'] not in ids), key=key)
        if len(pool) < 128:
            raise ValueError('Insufficient full-score replay; no fallback')
        replay += [{**lookup[q['question_id']], 'study_source': 'common'} for q in pool[:128]]
    c = read(CONFIG)
    base = read(c['baseline_config'])
    source = read_manifest(c['source_manifest'])
    frequency = Counter(NORMALIZER(q['target']) for q in source['splits']['train_all'])
    old_eval = read(ROOT.parent / 'vlm_scope_balanced_sft_v2/data/EVAL.json')
    retention = [{**q, 'stratum': 'singleword' if len(q['target'].split()) == 1 else 'multiword',
                  'answer_frequency': frequency[NORMALIZER(q['target'])]} for q in old_eval['retention_eval']]
    # Reuse the actual initial predictions only with exactly the same per-shard microbatch grouping.
    validation = sum(([q for q in data['rows'][i::2] if q['fold'] == 'validation'] for i in (0, 1)), [])
    if len(validation) != 512 or len(retention) != 64:
        raise ValueError('Fixed independent512/retention64 cohorts required')
    pixel = lambda iid: source['images'][iid].get('pixel_sha256', source['images'][iid]['sha256'])
    groups = [[*hard, *replay], validation, retention]
    pixel_sets = [{pixel(q['image_id']) for q in group} for group in groups]
    if any(pixel_sets[i] & pixel_sets[j] for i in range(3) for j in range(i)):
        raise ValueError('Training, validation or retention pixels overlap')
    all_rows = [*hard, *replay, *validation, *retention]
    images = {q['image_id']: {**source['images'][q['image_id']],
              'path': str((Path(c['source_manifest']).parent / source['images'][q['image_id']]['path']).resolve())}
              for q in all_rows}
    manifest = {'rows': hard + replay, 'images': images, 'hard_eval': hard,
                'validation_eval': validation, 'retention_eval': retention}
    write_json(ROOT / 'data/MANIFEST.json', manifest)
    processor = processor_for(Path(base['model_root']), read(base['reference_config']), 1024)
    cpu = {}
    for q in hard + replay:
        for kind in (('chosen', 'rejected') if q['study_source'] == 'preference' else ('chosen',)):
            sample = encode(processor, {**q, 'target': q['target'] if kind == 'chosen' else q['rejected_target']},
                            Path(images[q['image_id']]['path']), True)
            if sample['input_ids'].shape[1] > 2048:
                raise ValueError('Registered supervised sequence budget exceeded')
            cpu[q['question_id'] + ':' + kind] = {field + '_sha256': content_sha(sample[field].tolist())
                                                 for field in ('input_ids', 'labels')}
    write_json(ROOT / 'data/CPU_INPUTS.json', cpu)
    hard_order = []
    rng = random.Random(42)
    for _ in range(8):
        cycle = list(ids)
        rng.shuffle(cycle)
        hard_order += cycle
    replay_order = [q['question_id'] for q in replay]
    random.Random(43).shuffle(replay_order)
    schedule = []
    for step in range(32):
        offset = step * 8
        schedule.append({'micros': [replay_order[offset:offset + 4], hard_order[offset:offset + 4],
                                    replay_order[offset + 4:offset + 8], hard_order[offset + 4:offset + 8]]})
    write_json(ROOT / 'data/SCHEDULE.json', {'updates': schedule})
    initial_validation = [prediction_by_id[q['question_id']] for q in validation]
    write_json(ROOT / 'baseline/validation/PREDICTIONS.json', initial_validation)
    write_json(ROOT / 'DATA_COMPLETE.json', {'status': 'prepared', 'reviewed_error_pairs': 32, 'replay_qa': 256,
        'validation_qa': 512, 'retention_qa': 64, 'actual_supervised_cpu_encodings': len(cpu),
        'chosen_exposures_per_role': 512, 'negative_forward_sequences_per_role': 256,
        'validation_generation_reused_qa': 512, 'validation_labels_used_for_training_selection': 0,
        'reviewed_error_text_relations': dict(Counter(candidates[q]['text_relation'] for q in ids)),
        'reviewed_error_strata': dict(Counter(lookup[q]['stratum'] for q in ids))})
    paths = [CONFIG, Path(c['baseline_config']), Path(base['reference_config']),
             PROJECT_ROOT / 'docs/VLM_ERROR_SUPERVISION_SFT_V1_PROTOCOL.md',
             ROOT / 'DATA_COMPLETE.json', ROOT / 'baseline/validation/PREDICTIONS.json']
    paths += list((ROOT / 'data').glob('*.json')) + list((SOURCE_ROOT / 'vlm_error_supervision_sft_v1').glob('*.py'))
    paths += [MINING / f for f in ('DATA.json', 'PLAN.json', 'TRAINING_CANDIDATES.json', 'VISIBLE_REVIEW.json',
              'COMPLETION.json', 'shard_0/PREDICTIONS.json', 'shard_1/PREDICTIONS.json')]
    paths += [Path(c['initial_adapter']) / f for f in ('adapter_config.json', 'adapter_model.safetensors')]
    paths += [ROOT.parent / 'vlm_scope_balanced_sft_v2/data/EVAL.json']
    paths += [SOURCE_ROOT / f for f in ('vlm_region_preference_probe_v1/common.py', 'vlm_region_ocr_sft_v1/study.py',
              'vlm_alignment_signal/loss.py', 'vlm_fp32_readout_v1/model.py', 'vlm_fp32_adapter/model.py',
              'vlm_runtime_metadata_v1/model.py', 'vlm_probe/evaluate.py', 'vlm_attention_scope_repair_v1/generation.py',
              'vlm_sft/common.py', 'vlm_object_train/train.py', 'vlm_human_sft/evaluate.py', 'vlm_deep/common.py',
              'vlm_deep/evaluate.py', 'research_runtime/queue.py')]
    write_json(ROOT / 'PRELAUNCH_INPUT_LOCK.json', {'files': {str(path): sha_file(path) for path in paths},
        'images': {im['path']: im['sha256'] for im in images.values()},
        'initial_lora_sha256': read(MINING / 'PLAN.json')['initial_lora_sha256'],
        'scope': 'Reviewed actual-error single-seed adaptation study; no initial-unseen or benchmark gain.'})
    print(read(ROOT / 'DATA_COMPLETE.json'), flush=True)


def verify(training=False):
    lock = read(ROOT / 'PRELAUNCH_INPUT_LOCK.json')
    for path, digest in {**lock['files'], **lock['images']}.items():
        if sha_file(path) != digest:
            raise ValueError('Frozen actual-error study input changed: ' + path)
    if training:
        for path, digest in read(ROOT / 'STUDY_INPUT_LOCK.json')['reference_and_baseline_files'].items():
            if sha_file(path) != digest:
                raise ValueError('Frozen reference/baseline changed: ' + path)
    return lock


def reference():
    import torch
    from vlm_region_ocr_sft_v1.study import actor
    from vlm_human_sft.evaluate import lora_hash
    from vlm_alignment_signal.loss import sequence_logps
    from vlm_sft.common import to_device
    lock = verify()
    c = read(CONFIG)
    model, processor, _ = actor(c['initial_adapter'])
    if lora_hash(model) != lock['initial_lora_sha256']:
        raise ValueError('Original SFT identity differs')
    metrics = evaluate(model, processor, ROOT / 'baseline', names=('hard', 'retention'))
    metrics['validation'] = summarize(read(ROOT / 'baseline/validation/PREDICTIONS.json'))
    write_json(ROOT / 'baseline/COMPLETE.json', {'status': 'completed', 'training_updates': 0,
               'metrics': metrics, 'lora_sha256': lock['initial_lora_sha256'], 'new_generation_qa': 96,
               'reused_validation_generation_qa': 512})
    model.eval()
    model.config.use_cache = False
    data = read(ROOT / 'data/MANIFEST.json')
    lookup = {q['question_id']: q for q in data['rows']}
    cpu = read(ROOT / 'data/CPU_INPUTS.json')
    schedule = read(ROOT / 'data/SCHEDULE.json')['updates']
    cache = {'reference_lora_sha256': lock['initial_lora_sha256'], 'batches': {}}
    started = time.monotonic()
    with torch.inference_mode():
        for step, update in enumerate(schedule):
            for micro, ids in enumerate(update['micros']):
                rows = [lookup[q] for q in ids]
                if rows[0]['study_source'] != 'preference':
                    continue
                batch, binding, _, _ = make_batch(processor, data, rows, cpu)
                gpu = to_device(batch, next(model.parameters()).device)
                values, counts = sequence_logps(model, gpu)
                if values.dtype != torch.float32 or not bool(torch.isfinite(values).all()):
                    raise FloatingPointError('Reference log-probabilities are not finite FP32')
                cache['batches'][f'{step + 1}:{micro}'] = {'binding': binding,
                    'chosen_logps': values[:4].cpu().tolist(), 'rejected_logps': values[4:].cpu().tolist(),
                    'assistant_counts': counts.cpu().tolist()}
                write_json(ROOT / 'reference/CACHE.json', cache)
                write_json(ROOT / 'reference/PROGRESS.json', {'status': 'caching_reference',
                    'completed_batches': len(cache['batches']), 'expected_batches': 64,
                    'active_seconds': time.monotonic() - started})
                del batch, gpu, values, counts
                if time.monotonic() - started > c['maximum_reference_seconds']:
                    raise TimeoutError('Reference generation budget exceeded')
    if len(cache['batches']) != 64 or lora_hash(model) != lock['initial_lora_sha256']:
        raise ValueError('Reference coverage or original SFT identity changed')
    write_json(ROOT / 'reference/COMPLETE.json', {'status': 'completed', 'training_updates': 0,
        'reference_batches': 64, 'reference_sequences': 512, 'active_seconds': time.monotonic() - started,
        'lora_sha256': lock['initial_lora_sha256']})


def condition(role):
    from .train import train
    from vlm_human_sft.evaluate import lora_hash
    verify(True)
    model, processor, result = train(role)
    final = lora_hash(model)
    metrics = evaluate(model, processor, ROOT / 'runs' / role)
    if lora_hash(model) != final or final != result['final_lora_sha256']:
        raise ValueError('Final evaluation changed the adapter')
    write_json(ROOT / 'runs' / role / 'COMPLETE.json', {'status': 'completed', 'training_updates': 32,
        'metrics': metrics, 'final_lora_sha256': final, 'training_active_seconds': result['active_seconds']})


def analyze():
    records = {}
    metrics = {'initial': read(ROOT / 'baseline/COMPLETE.json')['metrics']}
    for role in ('initial',) + ROLES:
        directory = ROOT / 'baseline' if role == 'initial' else ROOT / 'runs' / role
        records[role] = {name: read(directory / name / 'PREDICTIONS.json') for name in ('hard', 'validation', 'retention')}
        if role != 'initial':
            metrics[role] = read(directory / 'COMPLETE.json')['metrics']
    comparisons, passed = {}, {}
    for role in ROLES:
        change = {name: paired(records['initial'][name], records[role][name]) for name in ('hard', 'validation', 'retention')}
        strata = {}
        for stratum in ('singleword', 'multiword'):
            strata[stratum] = paired([q for q in records['initial']['validation'] if q['stratum'] == stratum],
                                    [q for q in records[role]['validation'] if q['stratum'] == stratum])
        hard_net = metrics[role]['hard']['full_score_qa'] - metrics['initial']['hard']['full_score_qa']
        checks = {'training_learning': hard_net >= 8, 'validation_gain_at_least_half_pp': change['validation']['delta_pp'] >= .5,
                  'validation_lower_positive': change['validation']['ci95'][0] > 0,
                  'singleword_protection': strata['singleword']['delta_pp'] >= -.5,
                  'multiword_protection': strata['multiword']['delta_pp'] >= -.5,
                  'retention_protection': change['retention']['delta_pp'] >= -.5}
        comparisons[role] = {'changes_vs_initial': change, 'validation_strata': strata,
                             'hard_full_score_net': hard_net, 'checks': checks}
        passed[role] = all(checks.values())
    extra = paired(records['sft']['validation'], records['dpo_sft']['validation'])
    result = {'status': 'completed', 'metrics': metrics, 'comparisons': comparisons,
              'candidate_gate_passed': passed, 'dpo_vs_same_data_sft_validation_exploratory': extra,
              'dpo_extra_advantage_exploratory': passed['dpo_sft'] and extra['delta_pp'] >= .5 and extra['ci95'][0] > 0,
              'scope': 'Single-seed initial-training-whitelist adaptation study; not original main VQA/rare/cross-task gain.',
              'automatic_main_evaluation': False, 'automatic_more_training': False,
              'next': 'Only passing candidates qualify for separately registered original main-task screening; otherwise stop this branch.'}
    write_json(ROOT / 'COMPLETION.json', result)
    lines = ['# 真实错误监督SFT与DPO：完整结果', '', result['scope'], '',
             '|模型|32训练错误满分|512独立验证VQA|64保持VQA|', '|---|---:|---:|---:|']
    for role, value in metrics.items():
        lines.append(f'|{role}|{value["hard"]["full_score_qa"]}|{value["validation"]["vqa_percent"]:.6f}%|{value["retention"]["vqa_percent"]:.6f}%|')
    lines += ['', f'固定门槛：{passed}；DPO额外优势（探索）：{result["dpo_extra_advantage_exploratory"]}。',
              '完整配对区间、长度分层与保持结果见输出COMPLETION.json；不据训练题提分改简历。']
    (PROJECT_ROOT / 'docs/VLM_ERROR_SUPERVISION_SFT_V1_RESULTS.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--reference', action='store_true')
    parser.add_argument('--role', choices=ROLES)
    args = parser.parse_args()
    if args.prepare:
        prepare()
        return
    if args.reference:
        reference()
        return
    if args.role:
        condition(args.role)
        return
    from research_runtime.queue import run_queue
    verify()
    run_queue(ROOT / 'initial_queue', [{'key': 'reference',
        'command': [sys.executable, '-u', '-m', __package__ + '.study', '--reference'],
        'completion_marker': str(ROOT / 'reference/COMPLETE.json')}],
        {'prelaunch_sha256': sha_file(ROOT / 'PRELAUNCH_INPUT_LOCK.json')})
    paths = [ROOT / 'reference/CACHE.json', ROOT / 'reference/COMPLETE.json', ROOT / 'baseline/COMPLETE.json']
    paths += [ROOT / 'baseline' / name / 'PREDICTIONS.json' for name in ('hard', 'validation', 'retention')]
    lock = {'initial_lora_sha256': read(ROOT / 'PRELAUNCH_INPUT_LOCK.json')['initial_lora_sha256'],
            'reference_and_baseline_files': {str(p): sha_file(p) for p in paths}}
    if (ROOT / 'STUDY_INPUT_LOCK.json').exists() and read(ROOT / 'STUDY_INPUT_LOCK.json') != lock:
        raise ValueError('Existing actual-error training reference differs')
    write_json(ROOT / 'STUDY_INPUT_LOCK.json', lock)
    run_queue(ROOT / 'training_queue', [{'key': role,
        'command': [sys.executable, '-u', '-m', __package__ + '.study', '--role', role],
        'completion_marker': str(ROOT / 'runs' / role / 'COMPLETE.json')} for role in ROLES],
        {'study_sha256': sha_file(ROOT / 'STUDY_INPUT_LOCK.json')})
    analyze()


if __name__ == '__main__':
    main()
