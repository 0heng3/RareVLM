"""Bounded native replay-distribution repair; reuse the verified CE trainer."""
import argparse
from collections import Counter, defaultdict
import copy
import importlib
from pathlib import Path
import random
import re
import sys
import time

from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import content_sha, sha_file, write_json
from vlm_grpo_iteration_v1.common import read
from vlm_iiit_scope_v1.data import canonical

BASE = PROJECT_ROOT.parents[1] / 'outputs'
OLD = BASE / 'vlm_iiit_scope_v1'
TRANSFER = BASE / 'vlm_iiit_native_transfer_v1'
ROOT = BASE / 'vlm_iiit_replay_repair_v1'
PROTOCOL = PROJECT_ROOT / 'docs/VLM_IIIT_REPLAY_REPAIR_V1_PROTOCOL.md'
FUNCTION_WORDS = set('the a an of to in on and for at with by is be are you your it no not from as or we our'.split())
NAMES = ('hard', 'literal_validation', 'seen_validation', 'retention', 'transfer_dev')
UNITS = ('repair42', 'control0', 'repair0', 'control200', 'repair200')


def bucket(q):
    return min(3, len(canonical(q['target']).split()))


def prepare():
    import torch
    from vlm_probe.evaluate import processor_for
    from vlm_grpo_iteration_v1.common import chosen_sample
    if (ROOT / 'PLAN.json').exists():
        raise FileExistsError('Study already frozen')
    torch.set_num_threads(2)
    old = read(OLD / 'DATA.json')
    pool_data = read(BASE / 'vlm_replay_diversity_v1/DATA.json')
    hard = {q['question_id'] for q in pool_data['hard_rows']}
    pool = [q for q in pool_data['training_rows']['unique1024'] if q['question_id'] not in hard]
    if len(pool) != 1024 or len({q['question_id'] for q in pool}) != 1024:
        raise ValueError('Expected original correct native1024')
    validation_words = {canonical(q['target']) for q in old['evaluations']['literal_validation']}
    blocked = validation_words - FUNCTION_WORDS
    eligible = [q for q in pool if not blocked.intersection(re.findall('[a-z0-9]+', canonical(q['target'])))]
    groups = defaultdict(list)
    for q in eligible:
        groups[bucket(q)].append(q)
    for k in groups:
        groups[k].sort(key=lambda q: content_sha(['iiit_replay_repair_v1:native', q['question_id']]))
    quotas = {1: 64, 2: 152, 3: 40}
    if any(len(groups[k]) < n for k, n in quotas.items()):
        raise ValueError('Insufficient fixed length quota')
    selected = {k: groups[k][:n] for k, n in quotas.items()}
    cursors = Counter()
    repaired = []
    # Every native micro has one single-word and at least two two-word QAs.
    for i in range(64):
        for k in ([1, 2, 2, 3] if i < 40 else [1, 2, 2, 2]):
            repaired.append({**selected[k][cursors[k]], 'training_domain': 'native'})
            cursors[k] += 1
    original = [q for q in old['training_rows'] if q['training_domain'] == 'native']
    schedule = read(OLD / 'SCHEDULE.json')['sparse_language']
    first_ids = list(dict.fromkeys(qid for u in schedule for m in u['micros'] for qid in m if not qid.startswith('iiit5k:')))
    if len(first_ids) != 256 or set(first_ids) != {q['question_id'] for q in original}:
        raise ValueError('Original native order differs')
    mapping = dict(zip(first_ids, [q['question_id'] for q in repaired]))
    transfer = read(TRANSFER / 'DATA.json')
    images = {**pool_data['images'], **old['images'], **transfer['images']}
    evaluations = {**old['evaluations'], **transfer['evaluations']}
    held_ids = {q['image_id'] for rows in evaluations.values() for q in rows if not q['question_id'].startswith('iiit5k:')}
    held_pixels = {images[i]['pixel_sha256'] for i in held_ids}
    if any(q['image_id'] in held_ids or images[q['image_id']]['pixel_sha256'] in held_pixels for q in repaired + original):
        raise ValueError('Native replay/evaluation image overlap')
    ocr_ids = set(old['sparse_question_ids'])
    ocr = [q for q in old['training_rows'] if q['question_id'] in ocr_ids]
    old_cpu = read(OLD / 'CPU_INPUTS.json')
    reference = read(BASE / 'vlm_reference_replay_v1/PLAN.json')
    processor = processor_for(Path(reference['model_root']), read(reference['reference_config']), 1024)
    cpu = {q['question_id']: old_cpu[q['question_id']] for q in ocr + original}
    for i, q in enumerate(repaired):
        sample = chosen_sample(processor, q, images)
        record = {k: content_sha(sample[k].tolist()) for k in ('input_ids', 'labels')}
        record['assistant_ids'] = sample['labels'][sample['labels'] != -100].tolist()
        if q['question_id'] in old_cpu and record != old_cpu[q['question_id']]:
            raise ValueError('Shared native QA encoding changed')
        cpu[q['question_id']] = record
        if (i + 1) % 64 == 0:
            write_json(ROOT / 'CPU_PROGRESS.json', {'completed_new_replay': i + 1, 'expected': 256})
    content_ids = [q['question_id'] for q in evaluations['literal_validation'] if canonical(q['target']) not in FUNCTION_WORDS]
    write_json(ROOT / 'CONTENT_VALIDATION_IDS.json', content_ids)
    def describe(rows):
        return {'qa': len(rows), 'length_buckets': dict(Counter(bucket(q) for q in rows)),
                'answers_containing_the': sum('the' in canonical(q['target']).split() for q in rows)}
    audit = {'status': 'passed', 'original_pool': describe(pool), 'eligible_relaxed': describe(eligible),
             'old_native256': describe(original), 'repaired_native256': describe(repaired),
             'restored_common_words_overlapping_ocr_dev': sorted(validation_words & FUNCTION_WORDS),
             'content_only_ocr_validation_qa': len(content_ids), 'native_image_and_pixel_overlap': 0,
             'selection_uses_new_model_predictions': False,
             'original_pool_selected_using_historical_initial_correctness': True,
             'policy_is_bundle_of_length_and_content_changes': True}
    write_json(ROOT / 'DATA_AUDIT.json', audit)
    files = dict(read(OLD / 'PLAN.json')['files'])
    for name in UNITS:
        policy = 'repair' if name.startswith('repair') else 'control'
        seed = int(name.removeprefix(policy))
        native = repaired if policy == 'repair' else original
        updates = copy.deepcopy(schedule)
        if policy == 'repair':
            for u in updates:
                u['micros'] = [[mapping.get(qid, qid) for qid in m] for m in u['micros']]
        permutation = list(range(64))
        if seed != 42:
            random.Random(seed).shuffle(permutation)
        updates = [updates[i] for i in permutation]
        counts = Counter(qid for u in updates for m in u['micros'] for qid in m)
        if sum(counts.values()) != 1536 or any(counts[q['question_id']] != 2 for q in native) or any(counts[qid] != 8 for qid in ocr_ids):
            raise ValueError('Matched exposure budget changed')
        unit = ROOT / 'units' / name
        data = {'training_rows': ocr + native, 'images': images, 'evaluations': evaluations,
                'sparse_question_ids': old['sparse_question_ids'], 'supported_validation_qids': old['supported_validation_qids']}
        write_json(unit / 'DATA.json', data)
        write_json(unit / 'CPU_INPUTS.json', {q['question_id']: cpu[q['question_id']] for q in ocr + native})
        write_json(unit / 'SCHEDULE.json', {'sparse_language': updates})
        write_json(unit / 'INITIALIZATION.json', read(OLD / 'INITIALIZATION.json'))
        write_json(unit / 'PLAN.json', {'initial_lora_sha256': read(OLD / 'PLAN.json')['initial_lora_sha256'],
                   'policy': policy, 'order_seed': seed, 'macro_update_permutation': permutation,
                   'seed_axis': 'Fixed initializer; complete24-QA update-bundle order, not upstream SFT seed.'})
        for leaf in ('DATA', 'CPU_INPUTS', 'SCHEDULE', 'INITIALIZATION', 'PLAN'):
            path = unit / (leaf + '.json'); files[str(path)] = sha_file(path)
    # Bind proven control42 instead of rerunning its optimizer or generations.
    for path in [Path(__file__), PROTOCOL, ROOT / 'DATA_AUDIT.json', ROOT / 'CONTENT_VALIDATION_IDS.json',
                 TRANSFER / 'PLAN.json', OLD / 'runs/sparse_language/train/TRAIN_COMPLETE.json']:
        files[str(path)] = sha_file(path)
    for name in NAMES[:-1]:
        for directory in [OLD / 'baseline', OLD / 'runs/sparse_language']:
            path = directory / name / 'PREDICTIONS.json'; files[str(path)] = sha_file(path)
    # Pin the saved native-transfer implementation and its not-yet-complete outputs' lineage.
    files.update(read(TRANSFER / 'PLAN.json')['files'])
    used = {q['image_id'] for q in ocr + original + repaired + [q for rows in evaluations.values() for q in rows]}
    old_plan = read(OLD / 'PLAN.json')
    plan = {'status': 'prepared', 'files': files, 'packages': old_plan['packages'],
            'base_weight_files': old_plan['base_weight_files'], 'units': list(UNITS),
            'new_image_files': {images[i]['path']: images[i]['sha256'] for i in used},
            'phase1_new_updates': 64, 'phase1_new_generations': 2577, 'conditional_extra_updates': 256,
            'conditional_extra_internal_generations': 10308, 'conditional_ocr_test_generations': 15000,
            'primary_native_cohort': 'historically_used_internal_transfer1617', 'all_checkpoints_final64': True,
            'independent_upstream_seed_replication': False, 'official_ocr_test_already_exposed': True,
            'selection_gate_uses_official_test': False}
    write_json(ROOT / 'PLAN.json', plan)
    print('REPLAY_REPAIR_PREPARED', audit, flush=True)


def verify():
    from vlm_object_train.train import package_lock
    plan = read(ROOT / 'PLAN.json')
    for path, digest in {**plan['files'], **plan['new_image_files']}.items():
        if sha_file(path) != digest:
            raise ValueError('Frozen input changed: ' + path)
    for path, meta in plan['base_weight_files'].items():
        stat = Path(path).stat()
        if stat.st_size != meta['bytes'] or stat.st_mtime_ns != meta['mtime_ns']:
            raise ValueError('Base model identity changed')
    if package_lock() != plan['packages']:
        raise ValueError('Runtime versions changed')
    return plan


def worker(name, test=False):
    from vlm_human_sft.evaluate import lora_hash
    from vlm_grpo_iteration_v1.common import actor
    scope = importlib.import_module('run_iiit_scope_v1')
    trainer = importlib.import_module('vlm_iiit_scope_v1.train')
    verify(); unit = ROOT / 'units' / name
    scope.ROOT = trainer.ROOT = unit
    directory = unit / 'runs/sparse_language'
    if test:
        result = read(directory / 'train/TRAIN_COMPLETE.json')
        model, processor, _ = actor(adapter=result['adapter'])
        if lora_hash(model) != result['final_lora_sha256']:
            raise ValueError('Wrong frozen Final64 for repeated test')
        binding = read(OLD / 'TEST_PLAN.json')
        if sha_file(OLD / 'TEST_DATA.json') != binding['data_sha256']:
            raise ValueError('Old official test changed')
        for path, digest in binding['test_images_sha256'].items():
            if sha_file(path) != digest:
                raise ValueError('Official test image changed')
        scope.evaluate(model, processor, read(OLD / 'TEST_DATA.json'), directory / 'test', ('official_test',), binding['data_sha256'])
        write_json(directory / 'TEST_COMPLETE.json', {'status': 'completed', 'qa': 3000, 'test_is_repeated_exposed_benchmark': True})
        return
    model, processor, result = trainer.train('sparse_language', read(unit / 'PLAN.json'))
    if lora_hash(model) != result['final_lora_sha256']:
        raise ValueError('Wrong trained Final64')
    data = read(unit / 'DATA.json'); digest = sha_file(unit / 'DATA.json')
    scope.evaluate(model, processor, data, directory, NAMES[:-1], digest)
    scope.evaluate(model, processor, data, directory / 'transfer', ('transfer_dev',), digest)
    for cohort in NAMES:
        target = directory / 'transfer/transfer_dev' if cohort == 'transfer_dev' else directory / cohort
        preds = read(target / 'PREDICTIONS.json')
        rows = data['evaluations'][cohort]
        if len(preds) != len(rows) or any(a['question_id'] != q['question_id'] or a['input_sha256'] != q['inference_input_sha256'] for a, q in zip(preds, rows)):
            raise ValueError('Actual evaluation prefixes changed')
    write_json(directory / 'COMPLETE.json', {'status': 'completed', 'unit': name, 'updates': 64, 'evaluated_qa': 2577,
               'final_lora_sha256': result['final_lora_sha256']})


def records(name):
    directory = OLD / ('baseline' if name == 'initial' else 'runs/sparse_language') if name in ('initial', 'control42') else ROOT / 'units' / name / 'runs/sparse_language'
    result = {n: read(directory / n / 'PREDICTIONS.json') for n in NAMES[:-1]}
    path = TRANSFER / 'runs' / ('initial' if name == 'initial' else 'sparse_language') if name in ('initial', 'control42') else directory / 'transfer/transfer_dev'
    if name in ('initial', 'control42') and not (path / 'COMPLETE.json').exists():
        raise ValueError('Original native1617 evaluation incomplete')
    result['transfer_dev'] = read(path / 'PREDICTIONS.json')
    return result


def summarize(name, control):
    from vlm_grpo_iteration_v1.study import paired
    native_meta = read(OLD / 'DATA.json')['evaluations']['seen_validation']
    content_ids = set(read(ROOT / 'CONTENT_VALIDATION_IDS.json'))
    all_records = {r: records(r) for r in dict.fromkeys(['initial', control, name])}
    for rec in all_records.values():
        rec['content_validation'] = [q for q in rec['literal_validation'] if q['question_id'] in content_ids]
        for group in ('singleword', 'multiword', 'rare_1_5'):
            ids = {q['question_id'] for q in native_meta if (q['stratum'] == group if group != 'rare_1_5' else 1 <= q['answer_frequency'] <= 5)}
            rec[group] = [q for q in rec['seen_validation'] if q['question_id'] in ids]
    def norm(rows, cohort):
        if cohort in ('hard', 'literal_validation', 'content_validation'):
            return [{**q, 'image_id': q['filename_family'], 'vqa_score': float(q['iiit_alnum_exact'])} for q in rows]
        return rows
    def comparison(a, b):
        return {cohort: paired(norm(rows, cohort), norm(all_records[b][cohort], cohort)) for cohort, rows in all_records[a].items()}
    against_initial = comparison('initial', name); against_control = comparison(control, name)
    gate = {'transfer_vs_initial_at_least_half_pp': against_initial['transfer_dev']['delta_pp'] >= .5,
            'transfer_not_below_matched_control': against_control['transfer_dev']['delta_pp'] >= 0,
            'native_point_protections': all(against_initial[n]['delta_pp'] >= -.5 for n in ('seen_validation', 'retention', 'singleword', 'multiword', 'rare_1_5')),
            'content_ocr_dev_gain_at_least_one_pp': against_initial['content_validation']['delta_pp'] >= 1.,
            'content_ocr_dev_within_half_pp_of_control': against_control['content_validation']['delta_pp'] >= -.5}
    metrics = {r: {n: {'qa': len(rows), 'score_percent': 100 * sum(q['vqa_score'] for q in norm(rows, n)) / len(rows)} for n, rows in rec.items()} for r, rec in all_records.items()}
    result = {'status': 'completed', 'unit': name, 'control': control, 'metrics': metrics, 'vs_initial': against_initial,
              'vs_control': against_control, 'continuation_checks': gate, 'continue': all(gate.values()),
              'goal_complete': False, 'scope': 'Exploratory point-value continuation gate; historically used dev; no independent upstream seeds.'}
    write_json(ROOT / (name + '_ANALYSIS.json'), result)
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true'); parser.add_argument('--worker'); parser.add_argument('--test', action='store_true'); args = parser.parse_args()
    if args.prepare:
        prepare(); return
    if args.worker:
        worker(args.worker, args.test); return
    from research_runtime.queue import run_queue
    from vlm_object_train.train import lock_json
    verify(); identity = {'plan_sha256': sha_file(ROOT / 'PLAN.json')}; entry = str(Path(__file__).resolve())
    def job(name, test=False):
        directory = ROOT / 'units' / name / 'runs/sparse_language'
        return {'key': name, 'command': [sys.executable, '-u', entry, '--worker', name] + (['--test'] if test else []),
                'completion_marker': str(directory / ('TEST_COMPLETE.json' if test else 'COMPLETE.json'))}
    run_queue(ROOT / 'pilot_queue', [job('repair42')], identity)
    while not all((TRANSFER / 'runs' / r / 'COMPLETE.json').exists() for r in ('initial', 'sparse_language')):
        status = read(TRANSFER / 'queue/STATUS.json')
        if status['status'] == 'failed':
            raise RuntimeError('Native reference evaluation failed; no automatic restart')
        write_json(ROOT / 'STATUS.json', {'status': 'waiting_for_fixed_native_reference', 'plan_sha256': identity['plan_sha256']})
        time.sleep(40)
    paths = [TRANSFER / 'runs' / r / leaf for r in ('initial', 'sparse_language') for leaf in ('COMPLETE.json', 'INPUT_LOCK.json', 'PREDICTIONS.json')]
    lock_json(ROOT / 'CONTROL_REUSE.json', {'native_transfer_plan_sha256': sha_file(TRANSFER / 'PLAN.json'), 'files': {str(p): sha_file(p) for p in paths}, 'new_control42_updates': 0})
    pilot = summarize('repair42', 'control42')
    if not pilot['continue']:
        write_json(ROOT / 'COMPLETION.json', {'status': 'completed', 'phase': 'pilot_negative_no_expansion', 'new_updates': 64, 'new_generations': 2577,
                   'goal_complete': False, 'analysis': str(ROOT / 'repair42_ANALYSIS.json')}); return
    run_queue(ROOT / 'replication_queue', [job(n) for n in UNITS[1:]], identity)
    additional = {str(seed): summarize('repair' + str(seed), 'control' + str(seed)) for seed in (0, 200)}
    if not all(r['continue'] for r in additional.values()):
        write_json(ROOT / 'COMPLETION.json', {'status': 'completed', 'phase': 'order_replication_not_stable_no_test_expansion',
                   'new_updates': 320, 'new_generations': 12885, 'goal_complete': False}); return
    # Repeated OCR3000 is descriptive replication only, never a configuration gate.
    run_queue(ROOT / 'repeated_test_queue', [job(n, True) for n in UNITS], identity)
    from vlm_grpo_iteration_v1.study import paired
    before = read(OLD / 'official_test/initial/official_test/PREDICTIONS.json')
    def norm(rows):
        return [{**q, 'image_id': q['filename_family'], 'vqa_score': float(q['iiit_alnum_exact'])} for q in rows]
    test_results = {}
    for n in UNITS:
        rows = read(ROOT / 'units' / n / 'runs/sparse_language/test/official_test/PREDICTIONS.json')
        test_results[n] = {'score_percent': 100 * sum(q['iiit_alnum_exact'] for q in rows) / 3000, 'vs_initial': paired(norm(before), norm(rows))}
    write_json(ROOT / 'COMPLETION.json', {'status': 'completed', 'phase': 'fixed_start_order_replication_complete', 'new_updates': 320,
               'new_generations': 27885, 'official_test': test_results, 'goal_complete': False,
               'scope': 'Same SFT42 initializer, order seeds0/42/200; repeated exposed OCR test; broader original project gates remain unfulfilled.'})


if __name__ == '__main__':
    main()
