"""Matched termination-loss ablation, using frozen completed controls unchanged."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT
from vlm_deep.common import NORMALIZER, read_manifest, sha_file, write_json

ROOT = PROJECT_ROOT.parents[1] / 'outputs/vlm_region_ocr_termination_v1'
PARENT = PROJECT_ROOT.parents[1] / 'outputs/vlm_region_ocr_sft_v1'
DIAG = PROJECT_ROOT.parents[1] / 'outputs/vlm_region_ocr_diagnosis_v1'
CONFIG = PROJECT_ROOT / 'configs/vlm_region_ocr_termination_v1.json'


def read(path):
    return json.loads(Path(path).read_text())


def prepare():
    from vlm_region_ocr_sft_v1.study import verify as verify_parent
    verify_parent()
    diagnosis = read(DIAG / 'RESULTS.json')
    if diagnosis['status'] != 'completed':
        raise ValueError('The registered diagnosis must finish before this experiment')
    models = diagnosis['models']
    if models['region']['normalized_correct'] <= models['initial']['normalized_correct']:
        raise ValueError('No observed on-task learning; termination-transfer hypothesis not admitted')
    for cohort in ('common', 'control_aux'):
        key = 'first_word_eos_minus_gt_logit'
        if models['region']['teacher'][cohort][key] <= models['control']['teacher'][cohort][key]:
            raise ValueError('Missing the registered multiword premature-termination observation')
    c = read(CONFIG)
    files = [CONFIG, PROJECT_ROOT / 'docs/VLM_REGION_OCR_TERMINATION_V1_PROTOCOL.md',
             PROJECT_ROOT / 'tests/test_vlm_region_ocr_termination_v1.py', DIAG / 'RESULTS.json',
             PARENT / 'COMPLETION.json', PARENT / 'STUDY_INPUT_LOCK.json', PARENT / 'PRELAUNCH_INPUT_LOCK.json']
    files += list((SOURCE_ROOT / 'vlm_region_ocr_termination_v1').glob('*.py'))
    for role in ('initial', 'control', 'region'):
        directory = PARENT / 'initial' if role == 'initial' else PARENT / 'runs' / role
        files += [directory / task / 'predictions.jsonl' for task in ('text', 'human')]
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / 'STUDY_INPUT_LOCK.json').exists():
        raise FileExistsError('Study already frozen; use existing queue and checkpoint')
    write_json(ROOT / 'STUDY_INPUT_LOCK.json', {
        'initial_lora_sha256': read(PARENT / 'STUDY_INPUT_LOCK.json')['initial_lora_sha256'],
        'files': {str(p): sha_file(p) for p in sorted(set(files))},
        'parent_locked_files': read(PARENT / 'PRELAUNCH_INPUT_LOCK.json')['files'],
        'initial_adapter': c['initial_adapter'], 'single_change': c['loss'],
        'controls': 'Exact completed parent initial, original-QA control and full-region-CE models',
        'old_scientific_files_modified': False})


def verify():
    lock = read(ROOT / 'STUDY_INPUT_LOCK.json')
    for path, expected in {**lock['files'], **lock['parent_locked_files']}.items():
        if sha_file(path) != expected:
            raise ValueError('Frozen candidate or parent source/input changed: ' + path)


def region_probe(model, processor):
    import torch
    from vlm_probe.evaluate import evaluate_batch
    from vlm_attention_scope_repair_v1.generation import raw_greedy_config
    data = read(PARENT / 'data/REGION_TRAIN_PROBE.json')
    directory = ROOT / 'runs/region/region_train_probe'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'PREDICTIONS.json'
    predictions = read(path) if path.exists() else []
    if len(predictions) % 4 or [q['question_id'] for q in predictions] != [q['question_id'] for q in data['qa'][:len(predictions)]]:
        raise ValueError('Fixed training probe grouping changed')
    model.eval(); model.config.use_cache = True
    model.generation_config = raw_greedy_config(model.generation_config, 32)
    with torch.inference_mode():
        for start in range(len(predictions), 64, 4):
            rows = data['qa'][start:start+4]
            answers, _ = evaluate_batch(model, processor, rows, data['images'], PARENT / 'data',
                next(model.parameters()).device, 1024, processor.tokenizer.convert_tokens_to_ids('<|image_pad|>'), 32)
            predictions += [{'question_id': q['question_id'], 'image_id': q['image_id'], 'target': q['target'], **a,
                'raw_exact': a['prediction'] == q['target'],
                'casefold_exact': a['prediction'].casefold() == q['target'].casefold(),
                'normalized_correct': bool(a['prediction'].strip()) and NORMALIZER(a['prediction']) == NORMALIZER(q['target'])}
                for q, a in zip(rows, answers)]
            write_json(path, predictions)
    result = {'status': 'completed', 'qa': 64, 'scope': 'Fixed in-training task learning; not a heldout gain',
              **{key: sum(q[key] for q in predictions) for key in ('raw_exact', 'casefold_exact', 'normalized_correct')}}
    write_json(directory / 'COMPLETE.json', result)


def worker():
    from .train import train
    from vlm_region_ocr_sft_v1.study import eval_config
    from vlm_attention_scope_repair_v1.generation import evaluate
    from vlm_human_sft.evaluate import lora_hash
    verify(); c = read(CONFIG)
    model, processor, result = train(c, 'region', ROOT / 'STUDY_INPUT_LOCK.json')
    if lora_hash(model) != result['final_lora_sha256']:
        raise ValueError('Final adapter identity differs')
    for task in ('text', 'human'):
        evaluate(model, processor, eval_config('screen'), task, ROOT / 'runs/region' / task)
    region_probe(model, processor)
    if lora_hash(model) != result['final_lora_sha256']:
        raise ValueError('Read-only evaluation changed adapter')
    write_json(ROOT / 'runs/region/COMPLETE.json', {'status': 'completed', 'updates': 200,
        'final_lora_sha256': result['final_lora_sha256'], 'training_active_seconds': result['active_seconds']})


def analyze():
    from vlm_sft_resolution.experiment import paired_interval
    c = read(CONFIG)
    def jsonl(path):
        return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    records = {}
    for role in ('initial', 'control', 'region', 'candidate'):
        directory = (ROOT / 'runs/region' if role == 'candidate' else
                     PARENT / 'initial' if role == 'initial' else PARENT / 'runs' / role)
        records[role] = {task: jsonl(directory / task / 'predictions.jsonl') for task in ('text', 'human')}
    gold = {task: read_manifest(PARENT / f'data/SCREEN_{task.upper()}.json')['splits']['dev'] for task in ('text', 'human')}
    for tasks in records.values():
        for task in gold:
            if [(q['question_id'], q['image_id']) for q in tasks[task]] != [(q['question_id'], q['image_id']) for q in gold[task]]:
                raise ValueError('Unpaired screening questions')
    frequency = Counter(NORMALIZER(q['target']) for q in read_manifest(c['source_manifest'])['splits']['train_all'])
    components = {p[k]: p['component_id'] for p in read(PARENT / 'data/SCREEN_PAIRS.json')
                  for k in ('positive_question_id', 'negative_question_id')}
    comparisons = {}; gates = {}; changes = {}
    for before in ('initial', 'control', 'region'):
        comparisons[before] = {}; changes[before] = {}
        for task in ('text', 'human'):
            changes[before][task] = [{'question_id': a['question_id'], 'image_id': a['image_id'],
                'before': a['prediction'], 'after': b['prediction'],
                'before_score': a['primary_score'], 'after_score': b['primary_score']}
                for a,b in zip(records[before][task], records['candidate'][task]) if a['prediction'] != b['prediction']]
            masks = {'overall': [True] * len(gold[task])}
            if task == 'text':
                masks.update(rare_1_5=[1 <= frequency[NORMALIZER(q['target'])] <= 5 for q in gold[task]],
                             frequent_gt20=[frequency[NORMALIZER(q['target'])] > 20 for q in gold[task]])
            else:
                masks.update({answer: [q['target'] == answer for q in gold[task]] for answer in ('yes', 'no')})
            for name, mask in masks.items():
                def transformed(role):
                    return [{'question_id': q['question_id'], 'image_id': components[q['question_id']] if task == 'human' else q['image_id'],
                             'score': q['primary_score']} for q, keep in zip(records[role][task], mask) if keep]
                left, right = transformed(before), transformed('candidate')
                value = paired_interval(left, right, c['bootstrap_replicates'], c['bootstrap_seed'])
                value.update(corrected_qa=sum(b['score'] > a['score'] for a,b in zip(left,right)),
                             broken_qa=sum(b['score'] < a['score'] for a,b in zip(left,right)),
                             net_score_points=sum(b['score']-a['score'] for a,b in zip(left,right)))
                key = task + '_' + name; comparisons[before][key] = value
                gain = task == 'text' and name in ('overall', 'rare_1_5')
                gates[before + '_' + key] = value['delta_pp'] >= (c['minimum_gain_pp'] if gain else -c['protection_margin_pp'])
                if task == 'text' and name == 'overall':
                    gates[before + '_positive_text_ci'] = value['image_cluster_95_ci_pp'][0] > 0
    result = {'status': 'completed', 'stage': 'screen', 'passed': all(gates.values()), 'criteria': gates,
        'scores_percent': {role: {task: 100*sum(q['primary_score'] for q in rows)/len(rows) for task,rows in tasks.items()} for role,tasks in records.items()},
        'comparisons': comparisons, 'scope': 'Single seed42, historically exposed exploratory screen; no blind or replicated capability claim',
        'next': 'Independent confirmation and replication required if passed; otherwise stop this candidate without more training'}
    write_json(ROOT / 'CHANGED_CASES.json', changes)
    write_json(ROOT / 'SCREEN_ANALYSIS.json', result)
    write_json(ROOT / 'COMPLETION.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true'); parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare(); return
    if args.worker:
        try:
            worker()
        except Exception as exc:
            write_json(ROOT / 'ERROR.json', {'status': 'failed', 'error': repr(exc), 'other_tasks_untouched': True})
            raise
        return
    verify()
    from research_runtime.queue import run_queue
    run_queue(ROOT / 'queue', [{'key': 'candidate', 'command': [sys.executable, '-u', '-m', __package__ + '.study', '--worker'],
        'completion_marker': str(ROOT / 'runs/region/COMPLETE.json')}], {'study_sha256': sha_file(ROOT / 'STUDY_INPUT_LOCK.json')})
    analyze()


if __name__ == '__main__':
    main()
