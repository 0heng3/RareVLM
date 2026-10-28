"""Change the sampling distribution, reusing the validated numerical SFT backend."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT
from vlm_deep.common import NORMALIZER, content_sha, read_manifest, sha_file, write_json
from vlm_human_sft.evaluate import binary_answer, primary_score, read

OUTPUTS = PROJECT_ROOT.parents[1] / 'outputs'
ROOT = OUTPUTS / 'vlm_hard_visual_sft_v1'
PROTOCOL = PROJECT_ROOT / 'docs/VLM_HARD_VISUAL_SFT_V1_PROTOCOL.md'
ROLES = ('uniform', 'hard')


def freeze(path, document):
    path = Path(path)
    if path.exists() and read(path) != document:
        raise ValueError(f'Frozen content differs: {path}')
    write_json(path, document)


def hashed(rows, salt):
    return sorted(rows, key=lambda row: content_sha([salt, row['question_id']]))


def predictions(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def prepare():
    """One deterministic data preparation, without GPU access or heldout selection."""
    reference = read(PROJECT_ROOT / 'configs/vlm_human_sft_baseline_v1.json')
    human = read_manifest(reference['human_manifest'])
    text_path = Path(reference['text_manifest'])
    text = read_manifest(text_path)
    plan = read(ROOT / 'DATA_PLAN.json')
    reviews = read(ROOT / 'SEMANTIC_REVIEW.json')['rows']
    train_ids = set(plan['question_ids']['train'])
    pool = [row for row in human['splits']['train_all'] if row['question_id'] in train_ids]
    baseline = {row['question_id']: row for shard in ('mine0', 'mine1') for row in
                predictions(ROOT / f'mining/{shard}/initial/human/predictions.jsonl')}
    if set(baseline) != train_ids:
        raise ValueError('Training mining incomplete')
    admitted = {row['question_id']: row for row in reviews if row['admitted']}
    easy = {label: [] for label in ('yes', 'no')}
    hard = {label: [] for label in ('yes', 'no')}
    for row in pool:
        answer = binary_answer(baseline[row['question_id']]['prediction'])
        if answer == row['target']:
            easy[row['target']].append(row)
        elif row['question_id'] in admitted:
            if answer not in ('yes', 'no') or admitted[row['question_id']]['visual_answer'] != row['target']:
                raise ValueError('Invalid hard-example admission')
            hard[row['target']].append(row)
    for label in ('yes', 'no'):
        if len(hard[label]) < 12 or len({row['image_id'] for row in hard[label]}) < 10:
            raise ValueError('Hard-pool minimum not met; no training permitted')
        if len(easy[label]) < 800:
            raise ValueError('Insufficient nonrepeated uniform/easy examples')
    frequency = Counter(NORMALIZER(row['target']) for row in text['splits']['train_all'])
    selected_text, used_images, text_counts = [], set(), {}
    for name, count, select in [('rare', 400, lambda n: 1 <= n <= 5),
                              ('mid', 400, lambda n: 6 <= n <= 20),
                              ('frequent', 800, lambda n: n > 20)]:
        choices = [row for row in text['splits']['train_all']
                   if select(frequency[NORMALIZER(row['target'])]) and
                   sum(NORMALIZER(a) == NORMALIZER(row['target']) for a in row['answers']) >= 8]
        chosen = []
        for row in hashed(choices, 'hard_visual_text_replay_v1'):
            if row['image_id'] not in used_images:
                chosen.append(row); used_images.add(row['image_id'])
            if len(chosen) == count:
                break
        if len(chosen) != count:
            raise ValueError(f'Insufficient high-consensus unique-image Text replay: {name} {len(chosen)}')
        selected_text.extend(chosen); text_counts[name] = len(chosen)
    images = {key: {**item, 'path': str((text_path.parent / item['path']).resolve())}
              for key, item in text['images'].items()}
    if set(images) & set(human['images']):
        raise ValueError('Image namespaces overlap')
    images.update(human['images'])
    human_rows = {split: [row for row in human['splits']['train_all']
                         if row['question_id'] in set(plan['question_ids'][split])]
                  for split in ('screen', 'confirm')}
    # Combine sources by actual file and pixel identities, not just ID namespaces.
    heldout = human_rows['screen'] + human_rows['confirm'] + human['splits']['dev'] + human['splits']['final']
    heldout += text['splits']['dev'] + text['splits']['final']
    for key in ('sha256', 'pixel_sha256'):
        if {images[row['image_id']][key] for row in pool + selected_text} & {
                images[row['image_id']][key] for row in heldout}:
            raise ValueError(f'Combined training/heldout overlap: {key}')
    ordered_easy = {label: hashed(easy[label], ['hard_visual_easy_v1', label]) for label in easy}
    ordered_hard = {label: hashed(hard[label], ['hard_visual_hard_v1', label]) for label in hard}
    ordered_uniform = {label: hashed(easy[label] + hard[label], ['hard_visual_uniform_v1', label])[:800]
                       for label in easy}
    configs, summaries = {}, {}
    for role in ROLES:
        rows = []
        def exposure(original, source, stratum):
            rows.append({**original, 'question_id': f'hard_visual_v1:exposure{len(rows):04d}',
                         'ancestor_question_id': original['question_id'], 'study_source': source,
                         'sampling_stratum': stratum})
        for row in selected_text:
            exposure(row, 'text', 'common_replay')
        for number in range(800):
            for label in ('yes', 'no'):
                if role == 'uniform':
                    row = ordered_uniform[label][number]
                elif number < 200:
                    row = ordered_hard[label][number % len(ordered_hard[label])]
                else:
                    row = ordered_easy[label][number - 200]
                stratum = 'verified_hard' if row['question_id'] in admitted else 'baseline_correct'
                exposure(row, 'human', stratum)
        assert len(rows) == 3200
        active = rows + human_rows['screen'] + human_rows['confirm']
        manifest = {'schema_version': 'rarevlm.vlm_data.v2', 'training_allowed': True,
                    'purpose': '200-update exposure schedule; repeated IDs retain original ancestry',
                    'images': {row['image_id']: images[row['image_id']] for row in active},
                    'splits': {'train_all': rows, 'train_10k': rows,
                               'dev': human_rows['screen'], 'final': human_rows['confirm']}}
        manifest['manifest_sha256'] = content_sha(manifest)
        manifest_path = ROOT / f'data/{role.upper()}_TRAIN.json'
        freeze(manifest_path, manifest)
        read_manifest(manifest_path)
        counts = Counter(row['ancestor_question_id'] for row in rows)
        summaries[role] = {'exposures': len(rows), 'unique_qa': len(counts),
            'unique_images': len({row['image_id'] for row in rows}), 'maximum_repetitions': max(counts.values()),
            'source_counts': dict(Counter(row['study_source'] for row in rows)),
            'human_labels': dict(Counter(row['target'] for row in rows if row['study_source'] == 'human')),
            'hard_exposures': sum(row['sampling_stratum'] == 'verified_hard' for row in rows),
            'ancestor_exposure_counts': dict(counts)}
        config = {**reference, 'manifest': str(manifest_path), 'output_root': str(ROOT / f'runs/{role}/train'),
            'validation_only': False, 'expected_questions': 3200, 'expected_trainable': 1843200,
            'seed': 42, 'epochs': 1, 'effective_batch': 16, 'microbatch': 4, 'train_budget': 1024,
            'max_sequence_length': 2048, 'lr': 2e-5, 'warmup_ratio': .1, 'weight_decay': 0.,
            'gradient_clip_norm': 1., 'global_bf16_autocast': False, 'checkpoint_every_updates': 25,
            'roundtrip_questions': 1, 'human_manifest': str(ROOT / 'data/SCREEN_EVAL.json'),
            'text_manifest': str(ROOT / 'data/TEXT_SCREEN_EVAL.json'),
            'expected_eval_questions': {'human': 368, 'text': 512}}
        path = ROOT / f'data/{role.upper()}_CONFIG.json'; freeze(path, config); configs[role] = str(path)
    summary = {'roles': summaries, 'hard_pool_qa': {label: len(hard[label]) for label in hard},
        'hard_pool_images': {label: len({row['image_id'] for row in hard[label]}) for label in hard},
        'text_replay_parent_frequency_counts': text_counts,
        'parent_frequency_basis': 'normalized actual targets in original 32985 TextVQA training QA',
        'training_updates_per_arm': 200, 'training_cap_seconds_per_arm': 2700,
        'configs': configs, 'confirm_generated': False, 'quality_improvement': 'not yet measured'}
    freeze(ROOT / 'TRAINING_DATA.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'roles'}), flush=True)
    # Actual CPU processor encoding, once per unique QA; not a GPU runtime trace.
    encode_inputs(configs)
    files = [Path(__file__), Path(__file__).with_name('__init__.py'), PROTOCOL,
        PROJECT_ROOT / 'tools/run_hard_visual_sft_v1.py', ROOT / 'DATA_PLAN.json',
        ROOT / 'SEMANTIC_REVIEW.json', ROOT / 'TRAINING_DATA.json',
        ROOT / 'data/CPU_INPUTS.jsonl', ROOT / 'data/CPU_INPUTS_COMPLETE.json',
        Path(reference['human_manifest']), Path(reference['text_manifest']), Path(reference['reference_config']),
        Path(reference['initial_adapter']) / 'adapter_model.safetensors',
        Path(reference['initial_adapter']) / 'adapter_config.json']
    files += [SOURCE_ROOT / name for name in ('vlm_fp32_adapter/train.py', 'vlm_fp32_adapter/model.py',
        'vlm_fp32_adapter/loss.py', 'vlm_object_train/train.py', 'vlm_deep/common.py',
        'vlm_sft/common.py', 'vlm_probe/evaluate.py', 'vlm_human_sft/evaluate.py',
        'vlm_sft_resolution/experiment.py', 'research_runtime/queue.py')]
    files += [ROOT / f'data/{role.upper()}_{kind}.json' for role in ROLES for kind in ('TRAIN', 'CONFIG')]
    files += [ROOT / f'data/{name}.json' for name in ('SCREEN_EVAL', 'CONFIRM_EVAL', 'TEXT_SCREEN_EVAL', 'INITIAL_TEXT_SCREEN')]
    files += [ROOT / 'mining/screen/initial/human' / name for name in ('COMPLETE.json', 'INPUT_LOCK.json', 'predictions.jsonl')]
    freeze(ROOT / 'STUDY_INPUT_LOCK.json', {'files': {str(path): sha_file(path) for path in sorted(set(files))},
        'scope': 'single-seed 200-update screening; no automatic training extension',
        'base_weight_files': {path.name: {'bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}
                              for path in sorted(Path(reference['model_root']).glob('*.safetensors'))}})
    return summary


def encode_inputs(configs):
    from concurrent.futures import ThreadPoolExecutor
    from vlm_deep.common import encode
    from vlm_probe.evaluate import processor_for
    import torch
    complete = ROOT / 'data/CPU_INPUTS_COMPLETE.json'
    if complete.exists():
        if sha_file(ROOT / 'data/CPU_INPUTS.jsonl') != read(complete)['sha256']:
            raise ValueError('Encoded input records changed')
        return
    torch.set_num_threads(2)
    config = read(configs['hard'])
    processor = processor_for(Path(config['model_root']), read(config['reference_config']), 1024)
    unique = {}
    images = {}
    for path in configs.values():
        manifest = read_manifest(read(path)['manifest']); images.update(manifest['images'])
        for row in manifest['splits']['train_all']:
            unique[row['ancestor_question_id']] = row
    def one(item):
        question_id, row = item
        batch = encode(processor, row, images[row['image_id']]['path'], True)
        length = batch['input_ids'].shape[1]
        grid = batch['image_grid_thw'].flatten().tolist()
        visual = int(batch['image_grid_thw'].prod()) // 4
        assistant = int((batch['labels'] != -100).sum())
        if length > 2048 or visual > 1024 or assistant == 0:
            raise ValueError(f'Input exceeds training contract: {question_id}')
        return {'ancestor_question_id': question_id, 'image_id': row['image_id'],
                'input_tokens': length, 'visual_tokens': visual, 'image_grid_thw': grid,
                'assistant_tokens': assistant, 'input_ids_sha256': content_sha(batch['input_ids'].tolist()),
                'labels_sha256': content_sha(batch['labels'].tolist()),
                'scope': 'fresh CPU encoding with exact frozen training processor; not GPU-observed trace'}
    started = time.monotonic()
    path = ROOT / 'data/CPU_INPUTS.jsonl'
    with path.open('w') as stream, ThreadPoolExecutor(max_workers=8) as executor:
        for number, result in enumerate(executor.map(one, sorted(unique.items())), 1):
            stream.write(json.dumps(result) + '\n')
            if number % 250 == 0:
                stream.flush(); print('CPU_ENCODED', number, len(unique), flush=True)
    freeze(complete, {'status': 'completed', 'unique_qa': len(unique), 'sha256': sha_file(path),
                     'elapsed_seconds': time.monotonic() - started,
                     'scope': 'real CPU inputs, no model/GPU/optimizer; exposures refer to ancestor IDs'})


def verify_lock():
    lock = read(ROOT / 'STUDY_INPUT_LOCK.json')
    for path, expected in lock['files'].items():
        if sha_file(path) != expected:
            raise ValueError(f'Study inputs changed: {path}')
    reference = read(PROJECT_ROOT / 'configs/vlm_human_sft_baseline_v1.json')
    for name, expected in lock['base_weight_files'].items():
        stat = (Path(reference['model_root']) / name).stat()
        if {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns} != expected:
            raise ValueError('Base model files changed')


def worker(role):
    verify_lock()
    config_path = ROOT / f'data/{role.upper()}_CONFIG.json'
    config = read(config_path); directory = ROOT / f'runs/{role}'
    directory.mkdir(parents=True, exist_ok=True)
    if not (directory / 'train/COMPLETE.json').exists():
        command = [sys.executable, '-u', '-m', 'vlm_fp32_adapter.train', '--config', str(config_path), '--worker']
        with (directory / 'train.log').open('a') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write_json(directory / 'TRAIN_PROCESS.json', {'pid': process.pid, 'cap_seconds': 2700,
                       'started_unix': time.time(), 'scope': 'this study child process only'})
            try:
                code = process.wait(timeout=2700)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); process.wait()
                write_json(directory / 'BUDGET_STOP.json', {'status': 'stopped_at_budget',
                           'reason': '45-minute per-arm process cap; no automatic extension'})
                raise RuntimeError('Training reached declared budget cap')
        if code != 0:
            raise RuntimeError(f'Numerical backend failed with exit {code}; see {directory / "train.log"}')
    import torch
    from vlm_fp32_adapter.model import load
    from vlm_probe.evaluate import processor_for
    from vlm_human_sft.evaluate import evaluate
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    model, _ = load(config['model_root'], directory / 'train/adapter', torch.device('cuda:0'), training=False)
    processor = processor_for(Path(config['model_root']), read(config['reference_config']), 1024)
    for task in ('human', 'text'):
        evaluate(model, processor, config, task, directory / task)
    write_json(directory / 'COMPLETE.json', {'status': 'completed', 'updates': 200,
               'quality_improvement': 'await paired screening analysis'})


def analyze():
    from vlm_sft_resolution.experiment import paired_interval
    reference = read(PROJECT_ROOT / 'configs/vlm_human_sft_baseline_v1.json')
    human = read_manifest(ROOT / 'data/SCREEN_EVAL.json')['splits']['dev']
    text = read_manifest(ROOT / 'data/TEXT_SCREEN_EVAL.json')['splits']['dev']
    gold = {'human': human, 'text': text}
    records = {'initial': {'human': predictions(ROOT / 'mining/screen/initial/human/predictions.jsonl'),
                          'text': read(ROOT / 'data/INITIAL_TEXT_SCREEN.json')['records']}}
    for role in ROLES:
        records[role] = {task: predictions(ROOT / f'runs/{role}/{task}/predictions.jsonl') for task in gold}
    for role, tasks in records.items():
        for task, values in tasks.items():
            if [(r['question_id'], r['image_id']) for r in values] != [(r['question_id'], r['image_id']) for r in gold[task]]:
                raise ValueError('Screen evaluation identities differ')
            for row, original in zip(values, gold[task]):
                value = primary_score(row['prediction'], original, task)
                if value != row['primary_score']:
                    raise ValueError('Saved score differs from prediction')
                row['score'] = value
    def compare(before, after, task, keep=None):
        left, right = records[before][task], records[after][task]
        if keep is not None:
            left = [row for row, valid in zip(left, keep) if valid]
            right = [row for row, valid in zip(right, keep) if valid]
        if not left:
            raise ValueError('Empty prespecified screen slice')
        result = paired_interval(left, right, 4000, 20261018)
        if task == 'human':
            result.update(corrected_qa=sum(b['score'] > a['score'] for a, b in zip(left, right)),
                          broken_qa=sum(b['score'] < a['score'] for a, b in zip(left, right)))
            result['net_corrected_qa'] = result['corrected_qa'] - result['broken_qa']
        return result
    comparisons = {f'hard_vs_{role}': compare(role, 'hard', 'human') for role in ('initial', 'uniform')}
    comparisons['uniform_vs_initial'] = compare('initial', 'uniform', 'human')
    frequency = Counter(NORMALIZER(row['target']) for row in read_manifest(reference['text_manifest'])['splits']['train_all'])
    text_slices = {}
    for name, select in [('overall', lambda n: True), ('rare_1_5', lambda n: 1 <= n <= 5),
                         ('frequent_gt20', lambda n: n > 20), ('unseen_0', lambda n: n == 0)]:
        keep = [select(frequency[NORMALIZER(row['target'])]) for row in text]
        if any(keep):
            text_slices[name] = compare('initial', 'hard', 'text', keep)
    label_slices = {label: compare('initial', 'hard', 'human', [row['target'] == label for row in human])
                    for label in ('yes', 'no')}
    criteria = {}
    for name in ('hard_vs_initial', 'hard_vs_uniform'):
        result = comparisons[name]
        criteria[name + '_gain_1pp'] = result['delta_pp'] >= 1. - 1e-12
        criteria[name + '_positive_95ci'] = result['image_cluster_95_ci_pp'][0] > 0
        criteria[name + '_net_6qa'] = result['net_corrected_qa'] >= 6
    for label, result in label_slices.items():
        criteria[label + '_protection'] = result['delta_pp'] >= -.5
    for name in ('overall', 'rare_1_5', 'frequent_gt20'):
        criteria['text_' + name + '_protection'] = text_slices[name]['delta_pp'] >= -.5
    changed = {}
    for comparison in ('initial', 'uniform'):
        changed[comparison] = {task: [{'question_id': a['question_id'], 'image_id': a['image_id'],
            'before': a['prediction'], 'after': b['prediction'], 'before_score': a['score'], 'after_score': b['score']}
            for a, b in zip(records[comparison][task], records['hard'][task]) if a['prediction'] != b['prediction']]
            for task in gold}
    scores = {role: {task: 100 * sum(row['score'] for row in values) / len(values)
                    for task, values in tasks.items()} for role, tasks in records.items()}
    result = {'status': 'completed', 'screen_passed': all(criteria.values()), 'goal_complete': False,
              'scores_percent': scores, 'comparisons': comparisons, 'text_parent_frequency_slices': text_slices,
              'human_label_slices': label_slices, 'criteria': criteria, 'confirm_generated': False,
              'scope': 'single continuation seed42 development screening; prior project exposure disclosed; no CV claim',
              'next': 'freeze adapters and run independent confirmation evaluation' if all(criteria.values()) else
                      'stop: no confirmation generation or longer training; retain complete negative result'}
    write_json(ROOT / 'SCREEN_CHANGED_CASES.json', changed)
    write_json(ROOT / 'SCREEN_ANALYSIS.json', result)
    write_json(ROOT / 'COMPLETION.json', result)
    lines = ['# 困难样本SFT 200步筛选结果', '', '| 模型 | 人工screen368 | Text screen512 |', '| --- | ---: | ---: |']
    lines += [f'| {role} | {values["human"]:.6f} | {values["text"]:.6f} |' for role, values in scores.items()]
    lines += ['', f'筛选通过：{result["screen_passed"]}。', '', result['next'], '',
              '这是新划分上的单种子探索结果；独立题数、曝光、原图复核和全部升降预测均保存在输出目录。']
    (PROJECT_ROOT / 'docs/VLM_HARD_VISUAL_SFT_V1_RESULTS.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--worker', choices=ROLES)
    parser.add_argument('--analyze', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare(); return
    if args.worker:
        worker(args.worker); return
    verify_lock()
    if not args.analyze:
        from research_runtime.queue import run_queue
        jobs = [{'key': role, 'command': [sys.executable, '-u', '-m', 'vlm_hard_visual_sft.study', '--worker', role],
                 'completion_marker': str(ROOT / f'runs/{role}/COMPLETE.json')} for role in ROLES]
        run_queue(ROOT / 'training_queue', jobs, {'study_lock_sha256': sha_file(ROOT / 'STUDY_INPUT_LOCK.json'),
                   'training_updates_per_arm': 200, 'no_automatic_training_extension': True})
    analyze()


if __name__ == '__main__':
    main()
