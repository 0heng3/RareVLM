"""Prepare, run and score the registered attention-scope CE comparison."""

import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import sys

from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT
from vlm_deep.common import NORMALIZER, content_sha, read_manifest, sha_file, write_json
from vlm_human_sft.evaluate import evaluate, lora_hash, primary_score, read

from .model import ROLES, expand_qv


OUTPUTS = PROJECT_ROOT.parents[1] / 'outputs'
ROOT = OUTPUTS / 'vlm_attention_scope_v1'
OLD = OUTPUTS / 'vlm_complementary_rank_v1'
CONFIG = PROJECT_ROOT / 'configs/vlm_attention_scope_v1.json'


def freeze(path, value):
    path = Path(path)
    if path.exists() and read(path) != value:
        raise ValueError('Refusing to change frozen input: ' + str(path))
    write_json(path, value)


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def prepare():
    import torch
    from safetensors.torch import load_file, save_file
    from vlm_object_train.train import package_lock
    from .train import _validate_schedule

    config = read(CONFIG)
    reference = read(config['baseline_config'])
    admitted = read_manifest(config['admitted_manifest'])
    if (not admitted['training_allowed'] or admitted['splits']['dev'] or admitted['splits']['final']
            or len(admitted['splits']['train_all']) != 1186):
        raise ValueError('Expected the repaired train-only population')
    parent = Path(reference['initial_adapter'])
    original = load_file(str(parent / 'adapter_model.safetensors'))
    expanded, adapter_config, init_receipt = expand_qv(original, read(parent / 'adapter_config.json'), config['seed'])
    if init_receipt['total_adapter_parameters'] != 3686400:
        raise ValueError('Unexpected expanded parameter count')
    adapter = Path(config['initial_adapter'])
    adapter.mkdir(parents=True, exist_ok=True)
    weights = adapter / 'adapter_model.safetensors'
    if weights.exists():
        existing = load_file(str(weights))
        if set(existing) != set(expanded) or any(not torch.equal(existing[k], expanded[k]) for k in expanded):
            raise ValueError('Shared initialization already differs')
    else:
        save_file(expanded, str(weights))
    freeze(adapter / 'adapter_config.json', adapter_config)
    freeze(ROOT / 'initialization/MANIFEST.json', {**init_receipt,
        'original_weights_sha256': sha_file(parent / 'adapter_model.safetensors'),
        'expanded_weights_sha256': sha_file(weights), 'capability_improvement': False})
    replay_path = OUTPUTS / 'vlm_hard_visual_sft_v1/data/UNIFORM_TRAIN.json'
    replay = read_manifest(replay_path)
    text = [{**row, 'question_id': row['ancestor_question_id'], 'study_source': 'text'}
            for row in replay['splits']['train_all'] if row['study_source'] == 'text']
    if len(text) != 1600 or len({row['question_id'] for row in text}) != 1600:
        raise ValueError('Expected the original 1600 unique Text anchors')
    human = [{**row, 'study_source': 'human'} for row in admitted['splits']['train_all']]
    rows = text + human
    images = dict(admitted['images'])
    for row in text:
        item = replay['images'][row['image_id']]
        images[row['image_id']] = {**item, 'path': str((replay_path.parent / item['path']).resolve(strict=True))}
    replay_only = {'schema_version': admitted['schema_version'],
        'purpose': 'Only original Text training anchors; legacy nontraining folds are not exported or used',
        'images': {row['image_id']: images[row['image_id']] for row in text},
        'splits': {'train_all': text, 'train_10k': [], 'dev': [], 'final': []}}
    replay_only['manifest_sha256'] = content_sha(replay_only)
    freeze(ROOT / 'data/TEXT_REPLAY_TRAIN_ONLY.json', replay_only)
    if any(not Path(image['path']).is_absolute() or not Path(image['path']).is_file() for image in images.values()):
        raise ValueError('Training image paths must be absolute originals')
    manifest = {**admitted, 'purpose': 'Common unique training catalog; actual order in SCHEDULE.json',
                'images': images, 'splits': {'train_all': rows, 'train_10k': [], 'dev': [], 'final': []}}
    manifest.pop('manifest_sha256', None)
    manifest['manifest_sha256'] = content_sha(manifest)
    freeze(config['manifest'], manifest)
    pairs = admitted['complementary_pairs']['train']
    pair_order = []
    for cycle in (0, 1):
        pair_order += torch.randperm(len(pairs), generator=torch.Generator().manual_seed(config['seed'] + cycle)).tolist()
    old_schedule = read(OLD / 'data/SCHEDULE.json')['updates']
    updates = []
    for step in range(200):
        micros = [micro for micro in old_schedule[step]['micros'] if micro['source'] == 'text']
        for begin in (0, 2):
            chosen = [pairs[pair_order[step * 4 + begin + i]] for i in range(2)]
            micros.append({'source': 'human', 'question_ids': [pair[key] for pair in chosen
                for key in ('positive_question_id', 'negative_question_id')],
                'pair_indices': [[0, 1], [2, 3]], 'pair_ids': [pair['pair_id'] for pair in chosen]})
        updates.append({'micros': micros})
    freeze(config['schedule_path'], {'updates': updates, 'seed': config['seed'], 'qa_exposures': 3200,
        'pair_exposures': 800, 'text_exposures': 1600, 'train_pairs': len(pairs)})
    _validate_schedule(updates, {row['question_id']: row for row in rows}, config)
    for task, old_name in [('human', 'SCREEN_EVAL.json'), ('text', 'TEXT_SCREEN_EVAL.json')]:
        freeze(ROOT / f'data/SCREEN_{task.upper()}.json', read_manifest(OLD / 'data' / old_name))
    frozen_readout = read(OUTPUTS / 'vlm_fixed_error_readout_v1/data/COHORT.json')
    freeze(ROOT / 'data/SCREEN_PAIRS.json', frozen_readout['pairs']['screen'])
    cached_text = read(OLD / 'data/INITIAL_TEXT_BASELINES.json')['screen']['records']
    freeze(ROOT / 'data/ORIGINAL_TEXT_SCREEN.json', cached_text)
    paths = [CONFIG, Path(config['protocol']), Path(config['admitted_manifest']), Path(config['baseline_config']), replay_path,
             Path(reference['reference_config']), Path(config['manifest']), Path(config['schedule_path']),
             adapter / 'adapter_config.json', weights, ROOT / 'initialization/MANIFEST.json',
             OLD / 'initial/screen/human/predictions.jsonl',
             OUTPUTS / 'vlm_semantic_admission_v2/PREPARATION_RESULT.json']
    paths += list((ROOT / 'data').glob('*.json')) + list((SOURCE_ROOT / 'vlm_attention_scope_v1').glob('*.py'))
    paths += [PROJECT_ROOT / 'tests/test_vlm_attention_scope_v1.py']
    paths += [SOURCE_ROOT / name for name in ['vlm_runtime_metadata_v1/model.py', 'vlm_fp32_adapter/model.py',
        'vlm_complementary_rank/loss.py', 'vlm_object_train/train.py', 'vlm_human_sft/evaluate.py',
        'vlm_deep/common.py', 'vlm_sft/common.py', 'vlm_probe/evaluate.py',
        'vlm_sft_resolution/experiment.py', 'research_runtime/queue.py']]
    freeze(ROOT / 'PRELAUNCH_INPUT_LOCK.json', {'files': {str(p): sha_file(p) for p in sorted(set(paths))},
        'base_weight_files': {p.name: {'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
                              for p in sorted(Path(reference['model_root']).glob('*.safetensors'))},
        'packages': package_lock(), 'new_training_updates_per_role': 200,
        'old_source_or_predictions_modified': False})
    print(json.dumps({'status': 'prepared', 'train_pairs': len(pairs), 'unique_training_qa': len(rows),
                      'updates_per_role': 200, 'initializer': init_receipt}), flush=True)


def verify():
    from vlm_object_train.train import package_lock
    lock = read(ROOT / 'PRELAUNCH_INPUT_LOCK.json')
    for path, expected in lock['files'].items():
        if sha_file(path) != expected:
            raise ValueError('Frozen input changed: ' + path)
    config = read(CONFIG)
    reference = read(config['baseline_config'])
    for name, expected in lock['base_weight_files'].items():
        stat = (Path(reference['model_root']) / name).stat()
        if expected != {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}:
            raise ValueError('Base-model storage changed')
    if package_lock() != lock['packages']:
        raise ValueError('Installed package versions changed')


def actor(adapter):
    import torch
    from vlm_runtime_metadata_v1.model import load
    from vlm_probe.evaluate import processor_for
    config = read(CONFIG)
    reference = read(config['baseline_config'])
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, binding = load(reference['model_root'], adapter, torch.device('cuda:0'), training=False)
    processor = processor_for(Path(reference['model_root']), read(reference['reference_config']), 1024)
    return model, processor, binding


def eval_config(stage):
    config = read(CONFIG)
    reference = read(config['baseline_config'])
    if stage == 'screen':
        human, text = ROOT / 'data/SCREEN_HUMAN.json', ROOT / 'data/SCREEN_TEXT.json'
    else:
        # These heldout manifests are only opened after a successful screen.
        if not read(ROOT / 'SCREEN_ANALYSIS.json')['passed']:
            raise ValueError('Confirmation not permitted')
        human, text = OLD / 'data/CONFIRM_EVAL.json', OLD / 'data/TEXT_FULL_EVAL.json'
    return {**reference, 'initial_adapter': config['initial_adapter'],
            'human_manifest': str(human), 'text_manifest': str(text),
            'expected_eval_questions': {'human': 512, 'text': 512 if stage == 'screen' else 1617}}


def baseline(task):
    verify()
    config = read(CONFIG)
    model, processor, binding = actor(config['initial_adapter'])
    freeze(ROOT / f'initial/{task}/MODEL_IDENTITY.json', {'lora_sha256': lora_hash(model), 'binding': binding,
        'initializer_weights_sha256': sha_file(Path(config['initial_adapter']) / 'adapter_model.safetensors')})
    evaluate(model, processor, eval_config('screen'), task, ROOT / 'initial' / task)


def register():
    identities = {read(ROOT / f'initial/{task}/MODEL_IDENTITY.json')['lora_sha256'] for task in ('human', 'text')}
    if len(identities) != 1:
        raise ValueError('Baseline tasks loaded different initial states')
    comparisons = {}
    for task in ('human', 'text'):
        old = jsonl(OLD / 'initial/screen/human/predictions.jsonl') if task == 'human' else read(ROOT / 'data/ORIGINAL_TEXT_SCREEN.json')
        new = jsonl(ROOT / f'initial/{task}/predictions.jsonl')
        if [(r['question_id'], r['image_id']) for r in old] != [(r['question_id'], r['image_id']) for r in new]:
            raise ValueError('Fresh baseline input identities differ')
        fields = ('prediction', 'question_sha256', 'image_sha256', 'input_tokens', 'visual_tokens', 'image_grid_thw')
        differences = [{key: [a.get(key), b.get(key)] for key in fields if a.get(key) != b.get(key)} for a, b in zip(old, new)]
        differences = [d for d in differences if d]
        comparisons[task] = {'qa': len(new), 'differences': differences, 'passed': not differences}
    result = {'status': 'completed', 'passed': all(v['passed'] for v in comparisons.values()),
              'comparisons': comparisons, 'scope': 'new zero-output graph against complete original baseline, not capability gain'}
    freeze(ROOT / 'BASELINE_PARITY.json', result)
    if not result['passed']:
        write_json(ROOT / 'COMPLETION.json', {**result, 'training_started': False})
        return False
    freeze(ROOT / 'STUDY_INPUT_LOCK.json', {'initial_lora_sha256': identities.pop(),
        'prelaunch_sha256': sha_file(ROOT / 'PRELAUNCH_INPUT_LOCK.json'),
        'baseline_parity_sha256': sha_file(ROOT / 'BASELINE_PARITY.json'),
        'updates_per_role': 200, 'trainable_by_role': read(CONFIG)['expected_trainable_by_role']})
    return True


def condition(role):
    verify()
    from .train import train
    config = read(CONFIG)
    result = train(config, role, ROOT / 'runs' / role / 'train')
    model, processor, _ = actor(result['adapter'])
    if lora_hash(model) != result['final_lora_sha256']:
        raise ValueError('Evaluation loaded a different trained adapter')
    values = {task: evaluate(model, processor, eval_config('screen'), task, ROOT / 'runs' / role / task)
              for task in ('human', 'text')}
    write_json(ROOT / 'runs' / role / 'COMPLETE.json', {'status': 'completed', 'train': result, 'evaluations': values})


def confirm(role):
    verify()
    config = read(CONFIG)
    adapter = config['initial_adapter'] if role == 'initial' else str(ROOT / 'runs' / role / 'train/adapter')
    model, processor, _ = actor(adapter)
    for task in ('human', 'text'):
        evaluate(model, processor, eval_config('confirm'), task, ROOT / 'confirmation' / role / task)
    write_json(ROOT / 'confirmation' / role / 'COMPLETE.json', {'status': 'completed', 'training_updates': 0})


def pair_values(predictions, pairs):
    mapping = {row['question_id']: row for row in predictions}
    return [{'question_id': pair['pair_id'], 'image_id': pair['component_id'],
             'score': min(mapping[pair['positive_question_id']]['primary_score'],
                          mapping[pair['negative_question_id']]['primary_score'])} for pair in pairs]


def analyze(stage):
    from vlm_sft_resolution.experiment import paired_interval
    config = read(CONFIG)
    ec = eval_config(stage)
    gold = read_manifest(ec['human_manifest'])['splits']['dev']
    text_gold = read_manifest(ec['text_manifest'])['splits']['dev']
    if stage == 'screen':
        pairs = read(ROOT / 'data/SCREEN_PAIRS.json')
    else:
        # Original component identities, consumed only after screen gate passed.
        source_path = read(OLD / 'PRELAUNCH_INPUT_LOCK.json')['files']
        original_path = read(PROJECT_ROOT / 'configs/vlm_complementary_rank_v1.json')['dataset_manifest']
        if sha_file(original_path) != source_path[original_path]:
            raise ValueError('Original confirmation components changed')
        pairs = read(original_path)['complementary_pairs']['confirm']
    records = {}
    for role in ('initial',) + ROLES:
        directory = ROOT / 'confirmation' / role if stage == 'confirm' else ROOT / 'initial' if role == 'initial' else ROOT / 'runs' / role
        records[role] = {task: jsonl(directory / task / 'predictions.jsonl') for task in ('human', 'text')}
    for tasks in records.values():
        for task, rows in [('human', gold), ('text', text_gold)]:
            values = tasks[task]
            if [(r['question_id'], r['image_id']) for r in values] != [(r['question_id'], r['image_id']) for r in rows]:
                raise ValueError('Scored question/image order differs')
            if any(row['primary_score'] != primary_score(row['prediction'], original, task) for row, original in zip(values, rows)):
                raise ValueError('Saved score differs')
    component = {pair[key]: pair['component_id'] for pair in pairs for key in ('positive_question_id', 'negative_question_id')}
    metrics = {role: {'qa': [{**r, 'score': r['primary_score'], 'image_id': component[r['question_id']]} for r in tasks['human']],
                     'pair': pair_values(tasks['human'], pairs),
                     'text': [{**r, 'score': r['primary_score']} for r in tasks['text']]} for role, tasks in records.items()}
    def compare(before, metric, keep=None):
        left, right = metrics[before][metric], metrics['qkvo'][metric]
        if keep is not None:
            left = [r for r, flag in zip(left, keep) if flag]
            right = [r for r, flag in zip(right, keep) if flag]
        result = paired_interval(left, right, 4000, 20261019)
        if metric != 'text':
            result.update(corrected=sum(b['score'] > a['score'] for a, b in zip(left, right)),
                          broken=sum(b['score'] < a['score'] for a, b in zip(left, right)))
            result['net_corrected'] = result['corrected'] - result['broken']
        return result
    comparisons = {f'{metric}_vs_{role}': compare(role, metric) for metric in ('qa', 'pair') for role in ('initial', 'qv')}
    reference = read(config['baseline_config'])
    frequency = Counter(NORMALIZER(row['target']) for row in read_manifest(reference['text_manifest'])['splits']['train_all'])
    text_slices, human_slices, criteria = {}, {}, {}
    for name, value in comparisons.items():
        criteria[name + '_gain'] = value['delta_pp'] >= (1. if name.startswith('qa') else 2.) - 1e-12
        criteria[name + '_net6'] = value['net_corrected'] >= 6
        criteria[name + '_positive_ci'] = value['image_cluster_95_ci_pp'][0] > 0
    for before in ('initial', 'qv'):
        human_slices[before] = {label: compare(before, 'qa', [q['target'] == label for q in gold]) for label in ('yes', 'no')}
        text_slices[before] = {label: compare(before, 'text', [select(frequency[NORMALIZER(q['target'])]) for q in text_gold])
            for label, select in [('overall', lambda n: True), ('rare_1_5', lambda n: 1 <= n <= 5), ('frequent_gt20', lambda n: n > 20)]}
        for label, value in human_slices[before].items():
            criteria[f'{before}_{label}_protection'] = value['delta_pp'] >= -.5
        for label, value in text_slices[before].items():
            criteria[f'{before}_text_{label}_protection'] = value['delta_pp'] >= -.5
        if stage == 'confirm':
            criteria[f'{before}_text_overall_noninferior'] = text_slices[before]['overall']['image_cluster_95_ci_pp'][0] > -.5
    scores = {role: {metric: 100 * sum(r['score'] for r in values) / len(values)
                     for metric, values in tasks.items()} for role, tasks in metrics.items()}
    changed = {before: {task: [{'question_id': x['question_id'], 'image_id': x['image_id'], 'before': x['prediction'],
        'after': y['prediction'], 'before_score': x['primary_score'], 'after_score': y['primary_score']}
        for x, y in zip(records[before][task], records['qkvo'][task]) if x['prediction'] != y['prediction']]
        for task in ('human', 'text')} for before in ('initial', 'qv')}
    result = {'status': 'completed', 'stage': stage, 'passed': all(criteria.values()), 'scores_percent': scores,
        'comparisons': comparisons, 'human_slices': human_slices, 'text_slices': text_slices, 'criteria': criteria,
        'scope': 'single-seed matched attention-adapter capacity screen; no claim of three-seed or general capability gain',
        'confirm_generated': stage == 'confirm', 'training_updates_per_role': 200}
    write_json(ROOT / f'{stage.upper()}_CHANGED_CASES.json', changed)
    write_json(ROOT / f'{stage.upper()}_ANALYSIS.json', result)
    if not result['passed'] or stage == 'confirm':
        write_json(ROOT / 'COMPLETION.json', {**result, 'next': 'no automatic additional training or CV claim'})
    lines = ['# Q/V与Q/K/V/O可训练范围结果', '', f'阶段：{stage}；联合通过：{result["passed"]}。', '',
             '|模型|QA准确率|两图全对率|TextVQA|', '|---|---:|---:|---:|']
    lines += [f'|{role}|{v["qa"]:.6f}|{v["pair"]:.6f}|{v["text"]:.6f}|' for role, v in scores.items()]
    lines += ['', result['scope'], '', '新增参数、实际训练成本、逐题升降与完整区间见outputs/vlm_attention_scope_v1。']
    (PROJECT_ROOT / 'docs/VLM_ATTENTION_SCOPE_V1_RESULTS.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--baseline', choices=('human', 'text'))
    parser.add_argument('--condition', choices=ROLES)
    parser.add_argument('--confirm', choices=('initial',) + ROLES)
    args = parser.parse_args()
    if args.prepare:
        prepare(); return
    if args.baseline:
        baseline(args.baseline); return
    if args.condition:
        condition(args.condition); return
    if args.confirm:
        confirm(args.confirm); return
    from research_runtime.queue import run_queue
    verify()
    run_queue(ROOT / 'baseline_queue', [{'key': task, 'command': [sys.executable, '-u', '-m',
        'vlm_attention_scope_v1.study', '--baseline', task], 'completion_marker': str(ROOT / 'initial' / task / 'COMPLETE.json')}
        for task in ('human', 'text')], {'prelaunch_sha256': sha_file(ROOT / 'PRELAUNCH_INPUT_LOCK.json')})
    if not register():
        return
    run_queue(ROOT / 'training_queue', [{'key': role, 'command': [sys.executable, '-u', '-m',
        'vlm_attention_scope_v1.study', '--condition', role], 'completion_marker': str(ROOT / 'runs' / role / 'COMPLETE.json')}
        for role in ROLES], {'study_sha256': sha_file(ROOT / 'STUDY_INPUT_LOCK.json')})
    if not analyze('screen')['passed']:
        return
    run_queue(ROOT / 'confirmation_queue', [{'key': role, 'command': [sys.executable, '-u', '-m',
        'vlm_attention_scope_v1.study', '--confirm', role], 'completion_marker': str(ROOT / 'confirmation' / role / 'COMPLETE.json')}
        for role in ('initial',) + ROLES], {'screen_sha256': sha_file(ROOT / 'SCREEN_ANALYSIS.json')})
    analyze('confirm')


if __name__ == '__main__':
    main()
