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
    if (ROOT / 'STUDY_INPUT_LOCK.json').exists():
        raise FileExistsError('Already frozen; inspect existing study')
    c = read(CONFIG)
    views = read(PREP / 'MATCHED_REPLAY_VIEWS.json')
    replay_cpu = read(PREP / 'REPLAY_CPU_INPUTS.json')
    old = read(PARENT / 'data/MANIFEST.json')
    old_cpu = read(PARENT / 'data/CPU_INPUTS.json')
    schedule = read(PARENT / 'data/SCHEDULE.json')
    source = read_manifest(c['source_manifest'])
    hard = views['shared_region_rows']
    training_images = {q['image_id'] for q in hard} | set(views['images'])
    images = {iid: old['images'][iid] for iid in {q['image_id'] for q in hard}}
    images.update(views['images'])
    image_key = lambda im: im.get('pixel_sha256', im['sha256'])
    excluded = training_images | {q['image_id'] for q in old['retention_eval']}
    excluded.update(read(ROOT.parent / 'vlm_readout_transfer_v1/DATA.json')['images'])
    excluded_pixels = {image_key(im) for im in images.values()}
    excluded_pixels.update(image_key(source['images'][iid]) for iid in excluded if iid in source['images'])
    candidates = [q for q in source['splits']['train_all'] if q['image_id'] not in excluded
        and image_key(source['images'][q['image_id']]) not in excluded_pixels
        and Counter(q['answers'])[q['target']] >= 8 and q['target'] != MARKER
        and 1 <= len(q['target'].split()) <= 6 and any(ch.isalnum() for ch in q['target'])]
    candidates.sort(key=lambda q: hashlib.sha256(('scope_probe_v1:' + q['image_id'] + ':' + q['question_id']).encode()).hexdigest())
    retained = []
    used_images = set()
    used_pixels = set()
    counts = Counter()
    for q in candidates:
        stratum = 'short' if len(q['target'].split()) == 1 else 'multiword'
        pixel = image_key(source['images'][q['image_id']])
        if counts[stratum] >= 32 or q['image_id'] in used_images or pixel in used_pixels:
            continue
        retained.append({**q, 'scope_stratum': stratum})
        counts[stratum] += 1
        used_images.add(q['image_id'])
        used_pixels.add(pixel)
        if len(retained) == 64:
            break
    if counts != {'short': 32, 'multiword': 32}:
        raise ValueError('Fresh image-disjoint32+32 probe unavailable')
    for iid in used_images:
        im = source['images'][iid]
        images[iid] = {**im, 'path': str((Path(c['source_manifest']).parent / im['path']).resolve())}
    base = read(c['baseline_config'])
    processor = processor_for(Path(base['model_root']), read(base['reference_config']), 1024)
    retained_cpu = {}
    for q in retained:
        item = encode(processor, q, Path(images[q['image_id']]['path']), False)
        q['expected_input_grid'] = item['image_grid_thw'][0].tolist()
        q['expected_input_token_count'] = int(item['input_ids'].shape[1])
        retained_cpu[q['question_id']] = {'input_ids_sha256': content_sha(item['input_ids'].tolist()),
            'grid': q['expected_input_grid'], 'input_tokens': q['expected_input_token_count']}
    write_json(ROOT / 'data/RETENTION_CPU_INPUTS.json', retained_cpu)
    write_json(ROOT / 'data/EVAL.json', {'images': images, 'hard_eval': old['hard_eval'], 'retention_eval': retained})
    # Replay-image sequence stays equal across roles; only the78 genuine QA substitutions differ.
    old_ids = {q['question_id'] for q in old['rows'] if q['study_source'] == 'common'}
    occurrence = [qid for update in schedule['updates'] for micro in update['micros'] for qid in micro if qid in old_ids]
    if len(occurrence) != 256 or len(set(occurrence)) != 256:
        raise ValueError('Expected original256 unique replay slots')
    region_cpu = {k: v for k, v in old_cpu.items() if k.startswith('region-pref:')}
    for role in ROLES:
        replay = views['views'][role]
        mapping = {qid: replay[i]['question_id'] for i, qid in enumerate(occurrence)}
        role_schedule = {**schedule, 'updates': [{'micros': [[mapping.get(qid, qid) for qid in micro] for micro in update['micros']]} for update in schedule['updates']]}
        role_cpu = {**region_cpu, **{q['question_id'] + ':chosen': replay_cpu[q['question_id']] for q in replay}}
        write_json(ROOT / 'data' / role / 'MANIFEST.json', {'images': images, 'rows': hard + replay})
        write_json(ROOT / 'data' / role / 'CPU_INPUTS.json', role_cpu)
        write_json(ROOT / 'data' / role / 'SCHEDULE.json', role_schedule)
    assert not used_images & training_images
    parent_lock = read(PARENT / 'PRELAUNCH_INPUT_LOCK.json')
    if package_lock() != parent_lock['packages']:
        raise ValueError('Pinned runtime packages differ')
    paths = [CONFIG, PROJECT_ROOT / 'docs/VLM_SCOPE_BALANCED_SFT_V1_PROTOCOL.md',
             PREP / 'MATCHED_REPLAY_VIEWS.json', PREP / 'REPLAY_CPU_INPUTS.json', PREP / 'DATA_COMPLETE.json',
             PARENT / 'reference/CACHE.json', PARENT / 'data/SCHEDULE.json', PARENT / 'data/MANIFEST.json',
             ROOT.parent / 'vlm_plain_sft_main_v1/COMPLETION.json', Path(c['source_manifest']),
             Path(c['initial_adapter']) / 'adapter_config.json', Path(c['initial_adapter']) / 'adapter_model.safetensors']
    paths += list((ROOT / 'data').rglob('*.json')) + list((SOURCE_ROOT / 'vlm_scope_balanced_sft_v1').glob('*.py'))
    paths += [SOURCE_ROOT / name for name in ('vlm_region_preference_probe_v1/common.py', 'vlm_alignment_signal/loss.py',
              'vlm_fp32_readout_v1/model.py', 'vlm_fp32_adapter/model.py', 'vlm_runtime_metadata_v1/model.py',
              'vlm_object_train/train.py', 'vlm_deep/common.py', 'vlm_deep/evaluate.py', 'vlm_probe/evaluate.py',
              'vlm_attention_scope_repair_v1/generation.py', 'research_runtime/queue.py')]
    write_json(ROOT / 'STUDY_INPUT_LOCK.json', {'files': {str(p): sha_file(p) for p in set(paths)},
               'images': {im['path']: im['sha256'] for im in images.values()},
               'base_weight_files': parent_lock['base_weight_files'], 'initial_lora_sha256': parent_lock['initial_lora_sha256'],
               'packages': parent_lock['packages']})
    write_json(ROOT / 'DATA_COMPLETE.json', {'status': 'prepared', 'replay_images': 256,
               'new_retention_images': 64, 'short_retention_qa': 32, 'multiword_retention_qa': 32,
               'actual_new_retention_cpu_prefix_encodings': len(retained_cpu),
               'retention_training_image_overlap': 0, 'reference_reused_only_for_fixed_region_diagnostics': True})
    print(read(ROOT / 'DATA_COMPLETE.json'))


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
    (PROJECT_ROOT / 'docs/VLM_SCOPE_BALANCED_SFT_V1_RESULTS.md').write_text('\n'.join(lines) + '\n')


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
        jobs = [{'key': 'baseline', 'command': [sys.executable, '-u', '-m', __package__ + '.study', '--baseline'],
                 'completion_marker': str(ROOT / 'baseline/COMPLETE.json')}]
        jobs += [{'key': role, 'command': [sys.executable, '-u', '-m', __package__ + '.study', '--role', role],
                  'completion_marker': str(ROOT / 'runs' / role / 'COMPLETE.json')} for role in ROLES]
        run_queue(ROOT / 'queue', jobs, {'study_input_sha256': sha_file(ROOT / 'STUDY_INPUT_LOCK.json')})
        analyze()


if __name__ == '__main__':
    main()
