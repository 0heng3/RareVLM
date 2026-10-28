"""Zero-update main-task screen after ordinary-SFT auxiliary-region transfer; no module sweep."""
import argparse
from collections import Counter
from pathlib import Path
import sys
from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT
from vlm_deep.common import read_manifest, sha_file, write_json, NORMALIZER
from vlm_region_ocr_sft_v1.study import read, eval_config, actor, CONFIG as ORIGINAL_CONFIG

ROOT = PROJECT_ROOT.parents[1] / 'outputs/vlm_plain_sft_main_v1'
BASELINE = ROOT.parent / 'vlm_region_ocr_sft_v1'
SOURCE = ROOT.parent / 'vlm_region_preference_probe_v1'
TASKS = ('text', 'human')


def jsonl(path):
    import json
    return [json.loads(line) for line in path.read_text().splitlines()]


def prepare():
    if (ROOT / 'PLAN.json').exists():
        raise FileExistsError('Already prepared; inspect existing queue')
    c = eval_config('screen')
    c.update(initial_adapter=str(SOURCE / 'runs/sft/train/adapter'), output_root=str(ROOT))
    train = read(SOURCE / 'runs/sft/train/TRAIN_COMPLETE.json')
    current = read(SOURCE / 'data/MANIFEST.json')
    training_images = {q['image_id'] for q in current['rows']}
    for task in TASKS:
        original = read(BASELINE / 'initial' / task / 'INPUT_LOCK.json')
        if original['manifest_sha256'] != sha_file(Path(c[task + '_manifest'])) or original['budget'] != 1024 or original['batch_size'] != 4:
            raise ValueError('Reused baseline inputs differ')
        policy = read(BASELINE / 'initial' / task / 'GENERATION_POLICY.json')
        if policy['policy']['repetition_penalty'] != 1. or policy['policy']['max_new_tokens'] != 32:
            raise ValueError('Reused baseline generation differs')
        rows = read_manifest(c[task + '_manifest'])['splits']['dev']
        if training_images & {q['image_id'] for q in rows}:
            raise ValueError('Current adaptation images overlap main screen')
    ROOT.mkdir(parents=True, exist_ok=True)
    write_json(ROOT / 'CONFIG.json', c)
    files = [Path(__file__).resolve(), ROOT / 'CONFIG.json', ORIGINAL_CONFIG,
             PROJECT_ROOT / 'docs/VLM_PLAIN_SFT_MAIN_V1_PROTOCOL.md',
             SOURCE / 'runs/sft/train/TRAIN_COMPLETE.json', SOURCE / 'data/MANIFEST.json',
             Path(c['initial_adapter']) / 'adapter_config.json', Path(c['initial_adapter']) / 'adapter_model.safetensors',
             ROOT.parent / 'vlm_readout_transfer_v1/COMPLETION.json', ROOT.parent / 'vlm_readout_transfer_v1/PLAIN_SFT_POSTHOC.json',
             Path(read(ORIGINAL_CONFIG)['source_manifest']), BASELINE / 'data/SCREEN_PAIRS.json']
    files += [Path(c[task + '_manifest']) for task in TASKS]
    files += [BASELINE / 'initial' / task / name for task in TASKS for name in ('predictions.jsonl', 'COMPLETE.json', 'INPUT_LOCK.json', 'GENERATION_POLICY.json')]
    files += [SOURCE_ROOT / name for name in ('vlm_region_ocr_sft_v1/study.py', 'vlm_fp32_readout_v1/model.py',
              'vlm_fp32_adapter/model.py', 'vlm_runtime_metadata_v1/model.py', 'vlm_human_sft/evaluate.py',
              'vlm_attention_scope_repair_v1/generation.py', 'vlm_probe/evaluate.py', 'vlm_deep/common.py', 'research_runtime/queue.py')]
    write_json(ROOT / 'PLAN.json', {'training_updates': 0, 'new_predictions': 1024, 'baseline_reused': True,
               'candidate_lora_sha256': train['final_lora_sha256'],
               'scope': 'Historically exposed internal512+512 screen, single seed42, no new training; not formal benchmark or attribution to region-only supervision.',
               'files': {str(path): sha_file(path) for path in files}})
    print({'status': 'prepared', 'new_predictions': 1024, 'training_updates': 0, 'baseline': 'reused exact raw-greedy initial'})


def worker(task):
    from vlm_attention_scope_repair_v1.generation import evaluate, raw_greedy_config
    from vlm_human_sft.evaluate import lora_hash
    plan = read(ROOT / 'PLAN.json')
    for path, digest in plan['files'].items():
        if sha_file(path) != digest:
            raise ValueError('Main-screen input changed: ' + path)
    c = read(ROOT / 'CONFIG.json')
    model, processor, _ = actor(c['initial_adapter'])
    if lora_hash(model) != plan['candidate_lora_sha256']:
        raise ValueError('Wrong ordinary SFT checkpoint')
    policy = raw_greedy_config(model.generation_config, 32).to_dict()
    if policy != read(BASELINE / 'initial' / task / 'GENERATION_POLICY.json')['policy']:
        raise ValueError('Candidate and reused initial policies differ')
    evaluate(model, processor, c, task, ROOT / task)
    if lora_hash(model) != plan['candidate_lora_sha256']:
        raise ValueError('Read-only screen changed model')


def analyze():
    from vlm_sft_resolution.experiment import paired_interval
    c = read(ROOT / 'CONFIG.json')
    source = read_manifest(read(ORIGINAL_CONFIG)['source_manifest'])
    frequency = Counter(NORMALIZER(q['target']) for q in source['splits']['train_all'])
    component = {}
    for pair in read(BASELINE / 'data/SCREEN_PAIRS.json'):
        for key in ('positive_question_id', 'negative_question_id'):
            component[pair[key]] = pair['component_id']
    initial = {task: jsonl(BASELINE / 'initial' / task / 'predictions.jsonl') for task in TASKS}
    candidate = {task: jsonl(ROOT / task / 'predictions.jsonl') for task in TASKS}
    gold = {task: read_manifest(c[task + '_manifest'])['splits']['dev'] for task in TASKS}
    for task in TASKS:
        expected = [(q['question_id'], q['image_id']) for q in gold[task]]
        if [(q['question_id'], q['image_id']) for q in initial[task]] != expected or [(q['question_id'], q['image_id']) for q in candidate[task]] != expected:
            raise ValueError('Main-screen results unpaired')
    def compare(task, mask):
        def transform(rows):
            return [{'question_id': q['question_id'], 'image_id': component[q['question_id']] if task == 'human' else q['image_id'],
                     'score': q['primary_score']} for q, keep in zip(rows, mask) if keep]
        a, b = transform(initial[task]), transform(candidate[task])
        return {**paired_interval(a, b, 6000, 20261020),
                'increased_score_qa': sum(y['score'] > x['score'] for x, y in zip(a, b)),
                'decreased_score_qa': sum(y['score'] < x['score'] for x, y in zip(a, b))}
    values, gates = {}, {}
    for name, select in (('overall', lambda n: True), ('rare_1_5', lambda n: 1 <= n <= 5), ('frequent_gt20', lambda n: n > 20)):
        v = compare('text', [select(frequency[NORMALIZER(q['target'])]) for q in gold['text']])
        values['text_' + name] = v
        gates['text_' + name] = v['delta_pp'] >= (-.5 if name == 'frequent_gt20' else .5)
    gates['text_overall_positive_ci'] = values['text_overall']['image_cluster_95_ci_pp'][0] > 0
    for name in ('overall', 'yes', 'no'):
        v = compare('human', [name == 'overall' or q['target'] == name for q in gold['human']])
        values['human_' + name] = v
        gates['human_' + name] = v['delta_pp'] >= -.5
    scores = {role: {task: 100 * sum(q['primary_score'] for q in rows[task]) / len(rows[task]) for task in TASKS}
              for role, rows in (('initial_reused', initial), ('ordinary_sft32', candidate))}
    result = {'status': 'completed', 'scope': read(ROOT / 'PLAN.json')['scope'], 'new_training_updates': 0,
              'scores_percent': scores, 'comparisons': values, 'criteria': gates, 'screen_passed': all(gates.values()),
              'next': 'No automatic seeds, training, confirmation or CV update; full strong-control attribution still needed.'}
    write_json(ROOT / 'COMPLETION.json', result)
    write_json(ROOT / 'CHANGED_CASES.json', {task: [{'qid': a['question_id'], 'before': a['prediction'], 'after': b['prediction'],
               'before_score': a['primary_score'], 'after_score': b['primary_score']} for a, b in zip(initial[task], candidate[task]) if a['prediction'] != b['prediction']] for task in TASKS})
    lines = ['# 普通SFT32：原始VQA输入迁移筛选', '', result['scope'], '', '|模型|TextVQA|Human|', '|---|---:|---:|']
    lines += [f'|{role}|{v["text"]:.6f}|{v["human"]:.6f}|' for role, v in scores.items()]
    lines += ['', f'联合筛选通过={result["screen_passed"]}；无自动长训/确认/正式简历结论。']
    (PROJECT_ROOT / 'docs/VLM_PLAIN_SFT_MAIN_V1_RESULTS.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--task', choices=TASKS)
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.task:
        worker(args.task)
    else:
        from research_runtime.queue import run_queue
        jobs = [{'key': task, 'command': [sys.executable, '-u', str(Path(__file__).resolve()), '--task', task],
                 'completion_marker': str(ROOT / task / 'COMPLETE.json')} for task in TASKS]
        run_queue(ROOT / 'queue', jobs, {'plan_sha256': sha_file(ROOT / 'PLAN.json')})
        analyze()


if __name__ == '__main__':
    main()
