"""Prepare fixed same-image replay views, a fresh scope probe, and two32-step CE arms."""
import argparse
from collections import Counter
from pathlib import Path
import hashlib
import json
import sys
from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT
from vlm_deep.common import sha_file, write_json, read_manifest, encode, content_sha
from vlm_probe.evaluate import processor_for
from vlm_human_sft.evaluate import lora_hash
from vlm_object_train.train import package_lock
from vlm_region_ocr_sft_v1.study import actor
from .common import ROOT, PARENT, CONFIG, read, evaluate

ROLES = ('empirical_mix', 'scope_balanced')
PREP = ROOT.parent / 'vlm_scope_balanced_replay_v1'
MARKER = 'answering does not require reading text in the image'


def prepare():
    import random
    from shutil import copyfile
    previous = ROOT.parent / 'vlm_scope_balanced_sft_v1'
    if (ROOT / 'STUDY_INPUT_LOCK.json').exists():
        raise FileExistsError('Already frozen; inspect existing v2 study')
    c = read(CONFIG)
    prior_lock = read(previous / 'STUDY_INPUT_LOCK.json')
    if package_lock() != prior_lock['packages']:
        raise ValueError('Runtime differs from completed initial readout')
    stop = read(previous / 'STOPPED_DESIGN_CORRECTION.json')
    if stop['status'] != 'stopped_for_design_correction':
        raise ValueError('Original study not archived as stopped')
    for item in stop['signals_only_to_verified_own_processes']:
        path = Path('/proc') / str(item['pid']) / 'cmdline'
        if path.exists() and b'vlm_scope_balanced_sft_v1.study' in path.read_bytes():
            raise ValueError('Original study process still active')
    for rel in ('data/EVAL.json', 'data/RETENTION_CPU_INPUTS.json',
                'baseline/COMPLETE.json', 'baseline/EVAL_COMPLETE.json',
                'baseline/hard/PREDICTIONS.json', 'baseline/hard/PAIRS.json',
                'baseline/retention/PREDICTIONS.json'):
        target = ROOT / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        copyfile(previous / rel, target)
    if sha_file(ROOT / 'data/EVAL.json') != sha_file(previous / 'data/EVAL.json'):
        raise ValueError('Reused initial cohort differs')
    if read(ROOT / 'baseline/COMPLETE.json')['lora_sha256'] != prior_lock['initial_lora_sha256']:
        raise ValueError('Reused initial model differs')
    views = read(PREP / 'MATCHED_REPLAY_VIEWS.json')
    template = read(PARENT / 'data/SCHEDULE.json')
    old = read(PARENT / 'data/MANIFEST.json')
    replay_ids = {q['question_id'] for q in old['rows'] if q['study_source'] == 'common'}
    occurrence = [qid for update in template['updates'] for micro in update['micros'] for qid in micro if qid in replay_ids]
    order = list(range(256))
    random.Random(42).shuffle(order)
    stats = {}
    actual_images = {}
    for role in ROLES:
        folder = ROOT / 'data' / role
        folder.mkdir(parents=True, exist_ok=True)
        for name in ('MANIFEST.json', 'CPU_INPUTS.json'):
            copyfile(previous / 'data' / role / name, folder / name)
        replay = [views['views'][role][index] for index in order]
        mapping = {qid: replay[index]['question_id'] for index, qid in enumerate(occurrence)}
        schedule = {**template, 'updates': [{'micros': [[mapping.get(qid, qid) for qid in micro] for micro in update['micros']]} for update in template['updates']]}
        write_json(folder / 'SCHEDULE.json', schedule)
        long = [len(q['target'].split()) >= 2 for q in replay]
        stats[role] = {'multiword_replay_count': sum(long),
                       'multiword_per8_replay_slots': [sum(long[i:i + 8]) for i in range(0, 256, 8)],
                       'multiword_per_quarter': [sum(long[i:i + 64]) for i in range(0, 256, 64)]}
        if any(n == 0 for n in stats[role]['multiword_per_quarter']):
            raise ValueError('Registered shuffle unexpectedly leaves a quarter without multiword examples; stop, no seed search')
        actual_images[role] = [q['image_id'] for q in replay]
    if actual_images['empirical_mix'] != actual_images['scope_balanced']:
        raise ValueError('Replay image order unpaired')
    write_json(ROOT / 'ORDER_REVIEW.json', {'status': 'passed', 'permutation_seed': 42,
               'selection_order_independently_shuffled': True, 'shared_replay_image_sequence': True,
               'slot_permutation': order, 'roles': stats, 'baseline92_reused_without_model_execution': True})
    data = read(ROOT / 'data/EVAL.json')
    paths = [CONFIG, PROJECT_ROOT / 'docs/VLM_SCOPE_BALANCED_SFT_V2_PROTOCOL.md',
             PREP / 'MATCHED_REPLAY_VIEWS.json', PREP / 'REPLAY_CPU_INPUTS.json', PREP / 'DATA_COMPLETE.json',
             previous / 'STUDY_INPUT_LOCK.json', previous / 'STOPPED_DESIGN_CORRECTION.json',
             previous / 'data/EVAL.json', previous / 'baseline/COMPLETE.json', ROOT / 'ORDER_REVIEW.json',
             PARENT / 'reference/CACHE.json', PARENT / 'data/SCHEDULE.json',
             Path(c['initial_adapter']) / 'adapter_config.json', Path(c['initial_adapter']) / 'adapter_model.safetensors']
    paths += list((ROOT / 'data').rglob('*.json')) + list((ROOT / 'baseline').rglob('*.json'))
    paths += list((SOURCE_ROOT / 'vlm_scope_balanced_sft_v2').glob('*.py'))
    paths += [SOURCE_ROOT / name for name in ('vlm_region_preference_probe_v1/common.py', 'vlm_alignment_signal/loss.py',
              'vlm_fp32_readout_v1/model.py', 'vlm_fp32_adapter/model.py', 'vlm_runtime_metadata_v1/model.py',
              'vlm_object_train/train.py', 'vlm_deep/common.py', 'vlm_deep/evaluate.py', 'vlm_probe/evaluate.py',
              'vlm_attention_scope_repair_v1/generation.py', 'research_runtime/queue.py')]
    write_json(ROOT / 'STUDY_INPUT_LOCK.json', {'files': {str(path): sha_file(path) for path in set(paths)},
               'images': {im['path']: im['sha256'] for im in data['images'].values()},
               'base_weight_files': prior_lock['base_weight_files'], 'initial_lora_sha256': prior_lock['initial_lora_sha256'],
               'packages': prior_lock['packages']})
    print({'status': 'prepared', 'baseline_reused': True, 'roles': stats})


def verify():
    lock = read(ROOT / 'STUDY_INPUT_LOCK.json')
    for path, digest in {**lock['files'], **lock['images']}.items():
        if sha_file(path) != digest:
            raise ValueError('Frozen scope-study input changed: ' + path)
    for path, value in lock['base_weight_files'].items():
        stat = Path(path).stat()
        if {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns} != value:
            raise ValueError('Base storage changed')


def baseline():
    verify()
    model, processor, _ = actor(read(CONFIG)['initial_adapter'])
    before = lora_hash(model)
    metrics = evaluate(model, processor, ROOT / 'baseline')
    if lora_hash(model) != before:
        raise ValueError('Initial readout changed model')
    write_json(ROOT / 'baseline/COMPLETE.json', {'status': 'completed', 'training_updates': 0,
               'lora_sha256': before, 'metrics': metrics})


def condition(role):
    from .train import train
    verify()
    model, processor, trained = train(role)
    before = lora_hash(model)
    metrics = evaluate(model, processor, ROOT / 'runs' / role)
    if before != trained['final_lora_sha256'] or lora_hash(model) != before:
        raise ValueError('Final identity changed')
    write_json(ROOT / 'runs' / role / 'COMPLETE.json', {'status': 'completed', 'training_updates': 32,
               'metrics': metrics, 'final_lora_sha256': before, 'training_active_seconds': trained['active_seconds']})


def analyze():
    from vlm_sft_resolution.experiment import paired_interval
    metrics = {'initial': read(ROOT / 'baseline/COMPLETE.json')['metrics'],
               **{role: read(ROOT / 'runs' / role / 'COMPLETE.json')['metrics'] for role in ROLES}}
    initial, control, candidate = (metrics[k] for k in ('initial', 'empirical_mix', 'scope_balanced'))
    records = {role: read((ROOT / 'baseline' if role == 'initial' else ROOT / 'runs' / role) / 'retention/PREDICTIONS.json')
               for role in ('initial',) + ROLES}
    def interval(before, stratum):
        def transform(rows):
            return [{'question_id': q['question_id'], 'image_id': q['image_id'], 'score': q['vqa_score']}
                    for q in rows if stratum == 'all' or q['scope_stratum'] == stratum]
        return paired_interval(transform(records[before]), transform(records['scope_balanced']), 6000, 20261020)
    comparisons = {before: {name: interval(before, name) for name in ('all', 'short', 'multiword')} for before in ('initial', 'empirical_mix')}
    gates = {'multiword_over_control_plus1pp': candidate['retention']['multiword']['score_percent'] >= control['retention']['multiword']['score_percent'] + 1.,
             'multiword_prefix_truncations_no_more_than_control': candidate['retention']['multiword']['prefix_truncated'] <= control['retention']['multiword']['prefix_truncated'],
             'retention_all_minus_halfpp': candidate['retention']['score_percent'] >= initial['retention']['score_percent'] - .5,
             'retention_short_minus_halfpp': candidate['retention']['short']['score_percent'] >= initial['retention']['short']['score_percent'] - .5,
             'region_net4': candidate['hard']['casefold_correct'] >= initial['hard']['casefold_correct'] + 4,
             'region_pair_plus2': candidate['hard']['both_correct'] >= initial['hard']['both_correct'] + 2,
             'region_not_below_new_control': candidate['hard']['casefold_correct'] >= control['hard']['casefold_correct'],
             'region_pairs_not_below_new_control': candidate['hard']['both_correct'] >= control['hard']['both_correct']}
    result = {'status': 'completed', 'scope': 'Single-seed32-step scope-mixture probe, retained questions initially SFT-seen, no main-task or resume gain',
              'metrics': metrics, 'comparisons': comparisons, 'criteria': gates, 'short_probe_passed': all(gates.values()),
              'next': 'No automatic longer training/main-screen/seed/CV update; main-task thresholds remain unchanged.'}
    write_json(ROOT / 'COMPLETION.json', result)
    lines = ['# 长答案回放匹配对照：32步结果', '', result['scope'], '',
             '|模型|区域正确|双框全对|保持整体|保持短答案|保持多词|多词前缀截短|', '|---|---:|---:|---:|---:|---:|---:|']
    lines += [f'|{role}|{v["hard"]["casefold_correct"]}/28|{v["hard"]["both_correct"]}/14|{v["retention"]["score_percent"]:.6f}|{v["retention"]["short"]["score_percent"]:.6f}|{v["retention"]["multiword"]["score_percent"]:.6f}|{v["retention"]["multiword"]["prefix_truncated"]}|' for role, v in metrics.items()]
    lines += ['', f'短探针联合通过={result["short_probe_passed"]}；原主VQA门槛保持，未新做正式评测。']
    (PROJECT_ROOT / 'docs/VLM_SCOPE_BALANCED_SFT_V2_RESULTS.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--role', choices=ROLES)
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.baseline:
        baseline()
    elif args.role:
        try:
            condition(args.role)
        except Exception as error:
            write_json(ROOT / 'runs' / args.role / 'ERROR.json', {'status': 'failed', 'error': repr(error)})
            raise
    else:
        verify()
        from research_runtime.queue import run_queue
        jobs = [{'key': role, 'command': [sys.executable, '-u', '-m', __package__ + '.study', '--role', role],
                 'completion_marker': str(ROOT / 'runs' / role / 'COMPLETE.json')} for role in ROLES]
        run_queue(ROOT / 'queue', jobs, {'study_input_sha256': sha_file(ROOT / 'STUDY_INPUT_LOCK.json')})
        analyze()


if __name__ == '__main__':
    main()
