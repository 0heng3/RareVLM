"""Versioned post-evaluation metadata correction; never restart the pilot."""
import argparse
import importlib
from pathlib import Path
import sys
import time

from vlm_deep.common import sha_file, write_json
from vlm_grpo_iteration_v1.common import read
import run_iiit_replay_repair_v1 as original

ROOT = original.ROOT / 'recovery_v1'
PROTOCOL = original.PROJECT_ROOT / 'docs/VLM_IIIT_REPLAY_REPAIR_V1_METADATA_RECOVERY.md'


def prepare():
    original.verify()
    data = read(original.ROOT / 'units/repair42/DATA.json')
    missing = {n: sum('inference_input_sha256' not in q for q in data['evaluations'][n]) for n in original.NAMES}
    if missing != {'hard': 0, 'literal_validation': 0, 'seen_validation': 0, 'retention': 64, 'transfer_dev': 0}:
        raise ValueError('Unexpected missing metadata; correction only accepts original64 retention')
    paths = [Path(__file__), PROTOCOL, original.ROOT / 'PLAN.json', original.OLD / 'baseline/retention/PREDICTIONS.json']
    from vlm_object_train.train import lock_json
    lock_json(ROOT / 'PLAN.json', {'status': 'prepared', 'files': {str(p): sha_file(p) for p in paths},
              'missing_field_counts': missing, 'pilot_restart_permitted': False,
              'correction': 'Only retention64 absent metadata uses frozen initial actual input_sha256; all other prefixes strict.'})


def verify():
    original.verify()
    for path, digest in read(ROOT / 'PLAN.json')['files'].items():
        if sha_file(path) != digest:
            raise ValueError('Recovery source/input changed: ' + path)


def validate(name):
    unit = original.ROOT / 'units' / name; directory = unit / 'runs/sparse_language'
    data = read(unit / 'DATA.json'); result = read(directory / 'train/TRAIN_COMPLETE.json')
    if result['updates'] != 64 or result['sources'] != {'native': 512, 'ocr': 1024} or not result['frozen_adapter_hash_unchanged']:
        raise ValueError('Training budget/scope differs')
    identity = {'plan_sha256': sha_file(unit / 'PLAN.json'), 'data_sha256': sha_file(unit / 'DATA.json'),
                'model_lora_sha256': result['final_lora_sha256'], 'shard': None}
    if read(directory / 'INPUT_LOCK.json') != identity or read(directory / 'transfer/INPUT_LOCK.json') != identity:
        raise ValueError('Actual evaluation model/data identity differs')
    fallback = {q['question_id']: q for q in read(original.OLD / 'baseline/retention/PREDICTIONS.json')}
    files = {}
    for cohort in original.NAMES:
        path = (directory / 'transfer/transfer_dev' if cohort == 'transfer_dev' else directory / cohort) / 'PREDICTIONS.json'
        rows = read(path); expected = data['evaluations'][cohort]
        if len(rows) != len(expected):
            raise ValueError('Incomplete predictions')
        for answer, q in zip(rows, expected):
            if cohort == 'retention':
                if 'inference_input_sha256' in q:
                    raise ValueError('Expected unchanged original64 metadata representation')
                baseline = fallback[q['question_id']]
                if baseline['image_id'] != q['image_id'] or baseline['target'] != q['target']:
                    raise ValueError('Retention identity changed')
                prefix = baseline['input_sha256']
            else:
                prefix = q['inference_input_sha256']
            if (answer['question_id'], answer['image_id'], answer['target'], answer['input_sha256']) != (q['question_id'], q['image_id'], q['target'], prefix):
                raise ValueError('Actual prediction/input identity differs')
        files[str(path)] = sha_file(path)
    write_json(ROOT / 'units' / name / 'COMPLETE.json', {'status': 'completed', 'unit': name, 'qa': 2577,
               'prefixes_matched': 2577, 'retention_baseline_prefix_fallback': 64, 'files': files,
               'final_lora_sha256': result['final_lora_sha256'], 'recovery_plan_sha256': sha_file(ROOT / 'PLAN.json')})


def worker(name, test=False):
    verify()
    if test:
        original.worker(name, test=True); return
    # Same optimizer, data, schedule and generation functions. Only validate() changes.
    trainer = importlib.import_module('vlm_iiit_scope_v1.train')
    scope = importlib.import_module('run_iiit_scope_v1')
    unit = original.ROOT / 'units' / name; directory = unit / 'runs/sparse_language'
    trainer.ROOT = scope.ROOT = unit
    model, processor, _ = trainer.train('sparse_language', read(unit / 'PLAN.json'))
    data = read(unit / 'DATA.json'); digest = sha_file(unit / 'DATA.json')
    scope.evaluate(model, processor, data, directory, original.NAMES[:-1], digest)
    scope.evaluate(model, processor, data, directory / 'transfer', ('transfer_dev',), digest)
    validate(name)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true'); parser.add_argument('--worker'); parser.add_argument('--test', action='store_true'); args = parser.parse_args()
    if args.prepare:
        prepare(); return
    if args.worker:
        worker(args.worker, args.test); return
    verify(); started = time.monotonic()
    # Existing worker finishes all generations before its known final metadata error.
    while True:
        status = read(original.ROOT / 'pilot_queue/STATUS.json')
        if status['status'] == 'failed':
            log = original.ROOT / 'pilot_queue/logs/repair42.log'
            if "KeyError: 'inference_input_sha256'" not in log.read_text():
                raise RuntimeError('Different pilot error; do not auto-recover')
            break
        if status['status'] == 'completed':
            raise RuntimeError('Unexpected original completion; inspect before recovery')
        if time.monotonic() - started > 3600:
            raise TimeoutError('No original pilot completion; no restart')
        write_json(ROOT / 'STATUS.json', {'status': 'waiting_for_existing_generation', 'new_pilot_updates': 0, 'new_pilot_generations': 0})
        time.sleep(40)
    validate('repair42')
    from vlm_object_train.train import lock_json
    paths = [original.TRANSFER / 'runs' / r / leaf for r in ('initial', 'sparse_language') for leaf in ('COMPLETE.json', 'INPUT_LOCK.json', 'PREDICTIONS.json')]
    paths += [original.ROOT / 'pilot_queue/STATUS.json', original.ROOT / 'pilot_queue/logs/repair42.log']
    lock_json(ROOT / 'PILOT_REUSE.json', {'files': {str(p): sha_file(p) for p in paths}, 'new_pilot_updates': 0, 'new_pilot_generations': 0})
    pilot = original.summarize('repair42', 'control42')
    write_json(ROOT / 'PILOT_ANALYSIS.json', pilot)
    if not pilot['continue']:
        write_json(ROOT / 'COMPLETION.json', {'status': 'completed', 'phase': 'pilot_negative_no_expansion',
                   'pilot_updates': 64, 'pilot_generations': 2577, 'recovery_extra_updates': 0, 'recovery_extra_generations': 0,
                   'goal_complete': False, 'analysis': str(ROOT / 'PILOT_ANALYSIS.json'), 'original_failed_state_preserved': True}); return
    from research_runtime.queue import run_queue
    identity = {'parent_plan_sha256': sha_file(original.ROOT / 'PLAN.json'), 'recovery_plan_sha256': sha_file(ROOT / 'PLAN.json')}
    entry = str(Path(__file__).resolve())
    def job(name, test=False):
        marker = original.ROOT / 'units' / name / 'runs/sparse_language/TEST_COMPLETE.json' if test else ROOT / 'units' / name / 'COMPLETE.json'
        return {'key': name, 'command': [sys.executable, '-u', entry, '--worker', name] + (['--test'] if test else []), 'completion_marker': str(marker)}
    run_queue(ROOT / 'replication_queue', [job(n) for n in original.UNITS[1:]], identity)
    more = {str(seed): original.summarize('repair' + str(seed), 'control' + str(seed)) for seed in (0, 200)}
    if not all(r['continue'] for r in more.values()):
        write_json(ROOT / 'COMPLETION.json', {'status': 'completed', 'phase': 'order_replication_not_stable_no_test_expansion', 'new_updates': 320, 'new_generations': 12885, 'goal_complete': False}); return
    run_queue(ROOT / 'repeated_test_queue', [job(n, True) for n in original.UNITS], identity)
    from vlm_grpo_iteration_v1.study import paired
    def norm(rows):
        return [{**q, 'image_id': q['filename_family'], 'vqa_score': float(q['iiit_alnum_exact'])} for q in rows]
    before = read(original.OLD / 'official_test/initial/official_test/PREDICTIONS.json'); metrics = {}
    for n in original.UNITS:
        rows = read(original.ROOT / 'units' / n / 'runs/sparse_language/test/official_test/PREDICTIONS.json')
        metrics[n] = {'score_percent': 100 * sum(q['iiit_alnum_exact'] for q in rows) / 3000, 'vs_initial': paired(norm(before), norm(rows))}
    write_json(ROOT / 'COMPLETION.json', {'status': 'completed', 'phase': 'fixed_start_order_replication_complete', 'new_updates': 320,
               'new_generations': 27885, 'official_test': metrics, 'goal_complete': False,
               'scope': 'Unchanged algorithm/protocol; metadata validation corrected; fixed SFT42 start, not independent upstream seeds.'})


if __name__ == '__main__':
    main()
