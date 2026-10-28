"""Official complementary-image supervision with matched CE and CE+rank arms."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT
from vlm_deep.common import NORMALIZER, content_sha, read_manifest, sha_file, write_json
from vlm_human_sft.evaluate import evaluate, lora_hash, primary_score, read

OUTPUTS = PROJECT_ROOT.parents[1] / 'outputs'
ROOT = OUTPUTS / 'vlm_complementary_rank_v1'
CONFIG = PROJECT_ROOT / 'configs/vlm_complementary_rank_v1.json'
ROLES = ('ce', 'ce_rank')


def freeze(path, value):
    path = Path(path)
    if path.exists() and read(path) != value:
        raise ValueError(f'Frozen content changed: {path}')
    write_json(path, value)


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def make_view(rows, manifest, source_path, purpose):
    images = {}
    for row in rows:
        item = manifest['images'][row['image_id']]
        path = (Path(source_path).parent / item['path']).resolve(strict=True)
        images[row['image_id']] = {**item, 'path': str(path)}
    result = {'schema_version': 'rarevlm.vlm_data.v2', 'purpose': purpose, 'images': images,
              'splits': {'train_all': [], 'train_10k': [], 'dev': rows, 'final': []}}
    result['manifest_sha256'] = content_sha(result)
    return result


def prepare():
    import torch
    from .train import _validate_schedule
    from vlm_deep.common import encode
    from vlm_object_train.train import package_lock
    from vlm_probe.evaluate import processor_for
    config = read(CONFIG)
    reference = read(config['baseline_config'])
    data_path = Path(config['dataset_manifest'])
    data = read_manifest(data_path)
    quality_path = data_path.parent / 'DATA_REVIEW.json'
    quality = read(quality_path)
    if (quality['status'] != 'passed' or quality['manifest_sha256'] != sha_file(data_path)
            or quality['train_passed_pairs'] < 14 or quality['screen_passed_pairs'] < 7):
        raise ValueError('Fixed image-review quality gate not passed')
    for split, expected in [('train', 600), ('screen', 256), ('confirm', 256)]:
        if len(data['complementary_pairs'][split]) != expected:
            raise ValueError('Full registered complementary pair count required')
    if [len(data['splits'][name]) for name in ('train_all', 'dev', 'final')] != [1200, 512, 512]:
        raise ValueError('Pair/QA counts differ')
    # Reuse the exact 1600 Text anchors already prepared in the prior study.
    replay_path = Path(config['replay_manifest'])
    replay = read_manifest(replay_path)
    text = [{**row, 'question_id': row['ancestor_question_id'], 'study_source': 'text'}
            for row in replay['splits']['train_all'] if row['study_source'] == 'text']
    if len(text) != 1600 or len({row['question_id'] for row in text}) != 1600:
        raise ValueError('Common Text replay must contain 1600 unique QA')
    rows = text + [{**row, 'study_source': 'human'} for row in data['splits']['train_all']]
    images = dict(data['images'])
    for row in text:
        item = replay['images'][row['image_id']]
        images[row['image_id']] = {**item, 'path': str((replay_path.parent/item['path']).resolve(strict=True))}
    for item in images.values():
        if not Path(item['path']).is_absolute() or not Path(item['path']).is_file():
            raise ValueError('Every training/evaluation image must have an existing absolute path')
    train_images = {row['image_id'] for row in rows}
    heldout_images = {row['image_id'] for name in ('dev', 'final') for row in data['splits'][name]}
    for key in ('sha256', 'pixel_sha256'):
        if {images[key_id][key] for key_id in train_images} & {images[key_id][key] for key_id in heldout_images}:
            raise ValueError('Combined replay and complementary heldout pixels overlap')
    manifest = {**data, 'training_allowed': True,
        'semantic_admission': {'status': 'passed_fixed_sample_quality_gate',
            'source_manifest_sha256': sha_file(data_path), 'review_sha256': sha_file(quality_path),
            'scope': 'official >=9/10 human labels plus small assistant image-review gate, not expert verification of every QA'},
        'purpose': 'Fixed unique QA catalog; actual exposure order is SCHEDULE.json',
        'images': images, 'splits': {**data['splits'], 'train_all': rows, 'train_10k': rows}}
    manifest.pop('manifest_sha256', None); manifest['manifest_sha256'] = content_sha(manifest)
    freeze(config['manifest'], manifest); read_manifest(config['manifest'])
    text_order = torch.randperm(1600, generator=torch.Generator().manual_seed(42)).tolist()
    pairs = data['complementary_pairs']['train']
    pair_order = []
    for cycle in (0, 1):
        pair_order.extend(torch.randperm(len(pairs), generator=torch.Generator().manual_seed(42+cycle)).tolist())
    updates = []
    for step in range(200):
        micros = []
        for begin in (0, 4):
            selected = [text[text_order[step*8+begin+i]]['question_id'] for i in range(4)]
            micros.append({'source': 'text', 'question_ids': selected, 'pair_indices': [], 'pair_ids': []})
        for begin in (0, 2):
            chosen = [pairs[pair_order[step*4+begin+i]] for i in range(2)]
            micros.append({'source': 'human', 'question_ids': [qid for pair in chosen for qid in
                (pair['positive_question_id'], pair['negative_question_id'])],
                'pair_indices': [[0, 1], [2, 3]], 'pair_ids': [pair['pair_id'] for pair in chosen]})
        updates.append({'micros': micros})
    freeze(config['schedule_path'], {'updates': updates, 'seed': 42,
           'qa_exposures': 3200, 'pair_exposures': 800, 'text_exposures': 1600})
    _validate_schedule(updates, {row['question_id']: row for row in rows}, config)
    processor = processor_for(Path(reference['model_root']), read(reference['reference_config']), 1024)
    check_ids = updates[0]['micros'][0]['question_ids'] + updates[0]['micros'][2]['question_ids']
    lookup = {row['question_id']: row for row in rows}
    input_evidence = []
    for qid in check_ids:
        row = lookup[qid]; sample = encode(processor, row, images[row['image_id']]['path'], True)
        target_ids = sample['labels'][sample['labels'] != -100].tolist()
        if row['study_source'] == 'human' and target_ids[0] != processor.tokenizer.encode(row['target'], add_special_tokens=False)[0]:
            raise ValueError('New human dataset assistant target differs from local canonical token')
        visual = int(sample['image_grid_thw'].prod())//4
        length = int(sample['input_ids'].shape[1])
        if not target_ids or visual > 1024 or length > 2048:
            raise ValueError('New actual processor input violates the registered budget')
        input_evidence.append({'question_id': qid, 'source': row['study_source'], 'visual_tokens': visual,
            'input_tokens': length, 'assistant_target_ids': target_ids,
            'scope': 'fresh CPU data integration check; no model, gradient or gain claim'})
    freeze(ROOT/'data/INPUT_PREPARATION.json', {'status': 'passed', 'schedule_updates': len(updates),
           'real_cpu_encoded_samples': input_evidence, 'new_loss_cpu_tests': '15 passed; loss/math only'})
    for phase, split in [('train', 'train_all'), ('screen', 'dev'), ('confirm', 'final')]:
        freeze(ROOT/f'data/{phase.upper()}_EVAL.json', make_view(data['splits'][split], data, data_path,
            'official complementary internal '+phase+'; no original heldout377 consumption'))
    screen_path = OUTPUTS/'vlm_hard_visual_sft_v1/data/TEXT_SCREEN_EVAL.json'
    screen = read_manifest(screen_path)
    freeze(ROOT/'data/TEXT_SCREEN_EVAL.json', make_view(screen['splits']['dev'], screen, screen_path,
           'same complete original four-QA groups, absolute image paths'))
    full_path = Path(reference['text_manifest']); full = read_manifest(full_path)
    freeze(ROOT/'data/TEXT_FULL_EVAL.json', make_view(full['splits']['dev'], full, full_path,
           'same full Text1617 dev population; conditional confirmation only'))
    prior = OUTPUTS/'vlm_human_sft_repair_v1/initial/text'
    from vlm_human_sft.evaluate import source_files as evaluation_sources
    from vlm_probe.evaluate import package_versions
    cached_binding = read(prior/'INPUT_LOCK.json')
    expected_binding = {'budget': 1024, 'batch_size': 4, 'max_new_tokens': 32,
        'numerics': 'BF16 base/FP32 LoRA/no global AMP', 'packages': package_versions(),
        'source_files': {str(path): sha_file(path) for path in evaluation_sources()}}
    if any(cached_binding[key] != value for key, value in expected_binding.items()):
        raise ValueError('Cached Text baseline profile differs; cannot silently reuse')
    freeze(ROOT/'data/INITIAL_TEXT_BASELINES.json', {
        'screen': read(OUTPUTS/'vlm_hard_visual_sft_v1/data/INITIAL_TEXT_SCREEN.json'),
        'full_source_sha256': sha_file(prior/'predictions.jsonl'), 'full': jsonl(prior/'predictions.jsonl'),
        'original_input_lock_sha256': sha_file(prior/'INPUT_LOCK.json'),
        'initial_lora_sha256': read(prior/'COMPLETE.json')['lora_sha256'],
        'scope': 'derived exact raw-FP32 baseline cache; no independent baseline repetition'})
    paths = [CONFIG, Path(config['protocol']), Path(config['dataset_manifest']),
        Path(config['manifest']), Path(config['schedule_path']), Path(config['baseline_config']),
        Path(config['replay_manifest']), quality_path,
        data_path.parent/'FINAL_SELECTION_PLAN.json', ROOT/'data_preparation/SOURCE_LOCK.json',
        PROJECT_ROOT/'tools/prepare_complementary_vqa_v1.py', PROJECT_ROOT/'tools/run_complementary_rank_v1.py',
        PROJECT_ROOT/'tests/test_vlm_complementary_rank_loss.py', Path(reference['reference_config'])]
    paths += list((SOURCE_ROOT/'vlm_complementary_rank').glob('*.py'))
    paths += [SOURCE_ROOT/name for name in ('vlm_fp32_adapter/model.py', 'vlm_fp32_adapter/loss.py',
        'vlm_object_train/train.py', 'vlm_human_sft/evaluate.py', 'vlm_deep/common.py',
        'vlm_probe/evaluate.py', 'vlm_sft/common.py', 'vlm_sft_resolution/experiment.py', 'research_runtime/queue.py')]
    paths += list((ROOT/'data').glob('*EVAL.json')) + [ROOT/'data/INITIAL_TEXT_BASELINES.json', ROOT/'data/INPUT_PREPARATION.json']
    paths += [Path(reference['initial_adapter'])/name for name in ('adapter_model.safetensors', 'adapter_config.json')]
    paths += list(Path(reference['model_root']).glob('*.json'))
    freeze(ROOT/'PRELAUNCH_INPUT_LOCK.json', {'files': {str(path): sha_file(path) for path in sorted(set(paths))},
        'base_weight_files': {path.name: {'bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}
            for path in sorted(Path(reference['model_root']).glob('*.safetensors'))},
        'packages': package_lock(), 'training_qa_exposures': 3200, 'updates_per_arm': 200,
        'no_automatic_training_extension': True, 'no_model_based_data_selection': True})
    print('PREPARED', json.dumps({'train_pairs': 600, 'screen_pairs': 256, 'confirm_pairs': 256,
        'updates': 200, 'unique_training_qa': len(rows), 'quality': quality['status']}), flush=True)


def verify():
    lock = read(ROOT/'PRELAUNCH_INPUT_LOCK.json')
    for path, expected in lock['files'].items():
        if sha_file(path) != expected:
            raise ValueError(f'Frozen study inputs changed: {path}')
    reference = read(read(CONFIG)['baseline_config'])
    for name, expected in lock['base_weight_files'].items():
        stat = (Path(reference['model_root'])/name).stat()
        if {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns} != expected:
            raise ValueError('Base model storage changed')


def eval_config(config, phase):
    reference = read(config['baseline_config'])
    return {**reference, 'human_manifest': str(ROOT/f'data/{phase.upper()}_EVAL.json'),
        'text_manifest': str(ROOT/'data/TEXT_FULL_EVAL.json' if phase == 'confirm' else ROOT/'data/TEXT_SCREEN_EVAL.json'),
        'expected_eval_questions': {'human': 1200 if phase == 'train' else 512,
                                   'text': 1617 if phase == 'confirm' else 512}}


def actor(config, adapter):
    import torch
    from vlm_fp32_adapter.model import load
    from vlm_probe.evaluate import processor_for
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    reference = read(config['baseline_config'])
    model, _ = load(reference['model_root'], adapter, torch.device('cuda:0'), training=False)
    processor = processor_for(Path(reference['model_root']), read(reference['reference_config']), 1024)
    return model, processor


def baseline(phase):
    verify(); config = read(CONFIG); reference = read(config['baseline_config'])
    model, processor = actor(config, reference['initial_adapter'])
    evaluate(model, processor, eval_config(config, phase), 'human', ROOT/f'initial/{phase}/human')


def pair_values(predictions, pairs):
    mapping = {row['question_id']: row for row in predictions}
    return [{'question_id': pair['pair_id'], 'image_id': pair['component_id'],
             'score': min(mapping[pair['positive_question_id']]['primary_score'],
                          mapping[pair['negative_question_id']]['primary_score'])} for pair in pairs]


def register():
    config = read(CONFIG); data = read_manifest(config['dataset_manifest'])
    train = jsonl(ROOT/'initial/train/human/predictions.jsonl')
    screen = jsonl(ROOT/'initial/screen/human/predictions.jsonl')
    remaining = {split: sum(row['score'] == 0 for row in pair_values(values, data['complementary_pairs'][split]))
                 for split, values in [('train', train), ('screen', screen)]}
    components = len({pair['component_id'] for pair in data['complementary_pairs']['screen']})
    gates = {'train_24_wrong_pairs': remaining['train'] >= 24,
             'screen_12_wrong_pairs': remaining['screen'] >= 12, 'screen_32_components': components >= 32}
    result = {'status': 'completed', 'training_permitted': all(gates.values()),
              'wrong_pairs': remaining, 'screen_components': components, 'criteria': gates,
              'scope': 'preregistered learning/measurement headroom, not evidence of gain'}
    freeze(ROOT/'BASELINE_HEADROOM.json', result)
    if not all(gates.values()):
        write_json(ROOT/'COMPLETION.json', {**result, 'training_started': False, 'goal_complete': False,
            'next': 'stop; no resampling of screen/confirm or automatic training'})
        print(json.dumps(result), flush=True); return False
    receipts = [read(ROOT/f'initial/{phase}/human/COMPLETE.json') for phase in ('train', 'screen')]
    hashes = {x['lora_sha256'] for x in receipts}
    hashes.add(read(ROOT/'data/INITIAL_TEXT_BASELINES.json')['initial_lora_sha256'])
    if len(hashes) != 1:
        raise ValueError('Fresh human baseline and cached Text baseline used different initial LoRA')
    freeze(ROOT/'STUDY_INPUT_LOCK.json', {'prelaunch_sha256': sha_file(ROOT/'PRELAUNCH_INPUT_LOCK.json'),
        'initial_lora_sha256': hashes.pop(), 'headroom_sha256': sha_file(ROOT/'BASELINE_HEADROOM.json'),
        'initial_prediction_files': {phase: sha_file(ROOT/f'initial/{phase}/human/predictions.jsonl')
            for phase in ('train', 'screen')}, 'updates': 200, 'lambda_by_role': config['lambda_by_role']})
    return True


def condition_worker(role):
    verify(); config = read(CONFIG)
    from .train import train
    result = train(config, role, ROOT/f'runs/{role}/train')
    model, processor = actor(config, result['adapter'])
    if lora_hash(model) != result['final_lora_sha256']:
        raise ValueError('Evaluation loaded different trained LoRA values')
    evaluations = {task: evaluate(model, processor, eval_config(config, 'screen'), task,
                                  ROOT/f'runs/{role}/{task}') for task in ('human', 'text')}
    write_json(ROOT/f'runs/{role}/COMPLETE.json', {'status': 'completed', 'updates': 200,
               'role': role, 'train': result, 'evaluations': evaluations, 'goal_complete': False})


def confirm_worker(role):
    verify(); config = read(CONFIG)
    if not read(ROOT/'SCREEN_ANALYSIS.json')['passed']:
        raise ValueError('Confirmation not permitted after failed screening')
    adapter = read(config['baseline_config'])['initial_adapter'] if role == 'initial' else str(ROOT/f'runs/{role}/train/adapter')
    model, processor = actor(config, adapter)
    tasks = ('human',) if role == 'initial' else ('human', 'text')
    for task in tasks:
        evaluate(model, processor, eval_config(config, 'confirm'), task, ROOT/f'confirmation/{role}/{task}')
    write_json(ROOT/f'confirmation/{role}/COMPLETE.json', {'status': 'completed', 'role': role, 'new_training': False})


def analyze(stage):
    from vlm_sft_resolution.experiment import paired_interval
    config = read(CONFIG); data = read_manifest(config['dataset_manifest'])
    split = 'screen' if stage == 'screen' else 'confirm'
    gold = read_manifest(ROOT/f'data/{split.upper()}_EVAL.json')['splits']['dev']
    text_manifest = read_manifest(eval_config(config, split)['text_manifest'])
    text_gold = text_manifest['splits']['dev']
    cache = read(ROOT/'data/INITIAL_TEXT_BASELINES.json')
    initial_text = cache['screen']['records'] if stage == 'screen' else cache['full']
    initial_human = ROOT/'initial/screen/human' if stage == 'screen' else ROOT/'confirmation/initial/human'
    records = {'initial': {'human': jsonl(initial_human/'predictions.jsonl'), 'text': initial_text}}
    for role in ROLES:
        directory = ROOT/f'runs/{role}' if stage == 'screen' else ROOT/f'confirmation/{role}'
        records[role] = {task: jsonl(directory/task/'predictions.jsonl') for task in ('human', 'text')}
    for tasks in records.values():
        for task, original in [('human', gold), ('text', text_gold)]:
            values = tasks[task]
            if [(r['question_id'], r['image_id']) for r in values] != [(r['question_id'], r['image_id']) for r in original]:
                raise ValueError('Evaluation question/image identities differ')
            for row, target in zip(values, original):
                if row['primary_score'] != primary_score(row['prediction'], target, task):
                    raise ValueError('Saved primary score differs')
    pairs = data['complementary_pairs'][split]
    component = {pair[key]: pair['component_id'] for pair in pairs
                 for key in ('positive_question_id', 'negative_question_id')}
    metrics = {role: {'qa': [{**row, 'score': row['primary_score'], 'image_id': component[row['question_id']]}
                            for row in tasks['human']], 'pair': pair_values(tasks['human'], pairs),
                      'text': [{**row, 'score': row['primary_score']} for row in tasks['text']]}
               for role, tasks in records.items()}
    def compare(before, after, metric, keep=None):
        left, right = metrics[before][metric], metrics[after][metric]
        if keep is not None:
            left = [row for row, valid in zip(left, keep) if valid]
            right = [row for row, valid in zip(right, keep) if valid]
        result = paired_interval(left, right, 4000, 20261018)
        if metric != 'text':
            result.update(corrected=sum(b['score'] > a['score'] for a, b in zip(left, right)),
                          broken=sum(b['score'] < a['score'] for a, b in zip(left, right)))
            result['net_corrected'] = result['corrected']-result['broken']
        return result
    comparisons = {f'{metric}_vs_{role}': compare(role, 'ce_rank', metric)
                   for metric in ('qa', 'pair') for role in ('initial', 'ce')}
    reference = read(config['baseline_config'])
    frequency = Counter(NORMALIZER(row['target']) for row in read_manifest(reference['text_manifest'])['splits']['train_all'])
    text_slices = {label: compare('initial', 'ce_rank', 'text', [select(frequency[NORMALIZER(row['target'])]) for row in text_gold])
                   for label, select in [('overall', lambda n: True), ('rare_1_5', lambda n: 1 <= n <= 5),
                                         ('frequent_gt20', lambda n: n > 20)]}
    human_slices = {label: compare('initial', 'ce_rank', 'qa', [row['target'] == label for row in gold])
                    for label in ('yes', 'no')}
    criteria = {}
    for name, result in comparisons.items():
        criteria[name+'_gain'] = result['delta_pp'] >= (1. if name.startswith('qa') else 2.)-1e-12
        criteria[name+'_positive_ci'] = result['image_cluster_95_ci_pp'][0] > 0
        criteria[name+'_net_6'] = result['net_corrected'] >= 6
    for name, result in human_slices.items():
        criteria[name+'_protection'] = result['delta_pp'] >= -.5
    for name, result in text_slices.items():
        criteria['text_'+name+'_protection'] = result['delta_pp'] >= -.5
    if stage == 'confirm':
        criteria['text_full_95ci_noninferior'] = text_slices['overall']['image_cluster_95_ci_pp'][0] > -.5
    scores = {role: {metric: 100*sum(row['score'] for row in values)/len(values)
                    for metric, values in tasks.items()} for role, tasks in metrics.items()}
    changed = {before: {task: [{'question_id': x['question_id'], 'image_id': x['image_id'],
        'before': x['prediction'], 'after': y['prediction'], 'before_score': x['primary_score'], 'after_score': y['primary_score']}
        for x, y in zip(records[before][task], records['ce_rank'][task]) if x['prediction'] != y['prediction']]
        for task in ('human', 'text')} for before in ('initial', 'ce')}
    write_json(ROOT/f'{stage.upper()}_CHANGED_CASES.json', changed)
    result = {'status': 'completed', 'stage': stage, 'passed': all(criteria.values()), 'goal_complete': False,
        'scores_percent': scores, 'comparisons': comparisons, 'human_slices': human_slices,
        'text_parent_frequency_slices': text_slices, 'criteria': criteria,
        'confirm_generated': stage == 'confirm', 'training_updates_per_arm': 200,
        'scope': 'single continuation seed42, official-train-derived internal image-disjoint data; not official benchmark or three-seed claim'}
    write_json(ROOT/f'{stage.upper()}_ANALYSIS.json', result)
    if not result['passed'] or stage == 'confirm':
        write_json(ROOT/'COMPLETION.json', {**result, 'next': 'retain result; no automatic training extension or CV claim'})
    lines = ['# 同问异图CE＋跨图排序 v1 结果', '', f'阶段：{stage}；联合通过：{result["passed"]}。', '',
        '| 模型 | QA准确率 | 两图全对率 | TextVQA |', '| --- | ---: | ---: | ---: |']
    lines += [f'| {role} | {values["qa"]:.6f} | {values["pair"]:.6f} | {values["text"]:.6f} |' for role, values in scores.items()]
    lines += ['', result['scope'], '', '完整区间、升降预测、实际训练输入与成本见outputs/vlm_complementary_rank_v1。']
    (PROJECT_ROOT/'docs/VLM_COMPLEMENTARY_RANK_V1_RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--baseline', choices=('train', 'screen'))
    parser.add_argument('--condition', choices=ROLES)
    parser.add_argument('--confirm', choices=('initial',)+ROLES)
    args = parser.parse_args()
    if args.prepare:
        prepare(); return
    if args.baseline:
        baseline(args.baseline); return
    if args.condition:
        condition_worker(args.condition); return
    if args.confirm:
        confirm_worker(args.confirm); return
    from research_runtime.queue import run_queue
    verify()
    jobs = [{'key': phase, 'command': [sys.executable, '-u', '-m', 'vlm_complementary_rank.study', '--baseline', phase],
             'completion_marker': str(ROOT/f'initial/{phase}/human/COMPLETE.json')} for phase in ('train', 'screen')]
    run_queue(ROOT/'baseline_queue', jobs, {'prelaunch_sha256': sha_file(ROOT/'PRELAUNCH_INPUT_LOCK.json')})
    if not register():
        return
    jobs = [{'key': role, 'command': [sys.executable, '-u', '-m', 'vlm_complementary_rank.study', '--condition', role],
             'completion_marker': str(ROOT/f'runs/{role}/COMPLETE.json')} for role in ROLES]
    run_queue(ROOT/'training_queue', jobs, {'study_sha256': sha_file(ROOT/'STUDY_INPUT_LOCK.json'), 'updates': 200})
    if not analyze('screen')['passed']:
        return
    jobs = [{'key': role, 'command': [sys.executable, '-u', '-m', 'vlm_complementary_rank.study', '--confirm', role],
             'completion_marker': str(ROOT/f'confirmation/{role}/COMPLETE.json')} for role in ('initial',)+ROLES]
    run_queue(ROOT/'confirmation_queue', jobs, {'screen_analysis_sha256': sha_file(ROOT/'SCREEN_ANALYSIS.json'), 'new_training': False})
    analyze('confirm')


if __name__ == '__main__':
    main()
