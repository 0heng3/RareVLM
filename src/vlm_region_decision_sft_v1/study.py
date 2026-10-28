"""Freeze the chosen-token objective comparison, reuse completed baseline, run two short arms."""
import argparse
from pathlib import Path
import sys
from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT
from vlm_deep.common import encode, sha_file, write_json
from vlm_probe.evaluate import processor_for
from vlm_human_sft.evaluate import lora_hash
from .common import ROOT, DATA_ROOT, CONFIG, read, evaluate

ROLES = ('qa_weighted', 'decision')


def prepare():
    if (ROOT / 'STUDY_INPUT_LOCK.json').exists():
        raise FileExistsError('Already frozen; do not duplicate study')
    c = read(CONFIG)
    base = read(c['baseline_config'])
    data = read(DATA_ROOT / 'data/MANIFEST.json')
    initial = read(DATA_ROOT / 'baseline/hard/PREDICTIONS.json')
    rows = {q['question_id']: q for q in data['hard_eval']}
    cpu = read(DATA_ROOT / 'data/CPU_INPUTS.json')
    processor = processor_for(Path(base['model_root']), read(base['reference_config']), 1024)
    decisions = {}
    for prediction in initial:
        if prediction['casefold_exact']:
            continue
        q = rows[prediction['question_id']]
        pair = [encode(processor, {**q, 'target': text}, Path(data['images'][q['image_id']]['path']), True)
                for text in (q['target'], prediction['prediction'])]
        from vlm_deep.common import content_sha
        for key in ('input_ids', 'labels'):
            if content_sha(pair[0][key].tolist()) != cpu[q['question_id'] + ':chosen'][key + '_sha256']:
                raise ValueError('Chosen inputs differ from previous matched study')
        ids = [s['labels'][0][s['labels'][0] != -100].tolist() for s in pair]
        divergence = next((i for i, (a, b) in enumerate(zip(*ids)) if a != b), None)
        if divergence is None:
            raise ValueError('No differing token in initially wrong answer')
        starts = [int((s['labels'][0] != -100).nonzero()[0]) for s in pair]
        if not pair[0]['input_ids'][0, :starts[0] + divergence].equal(pair[1]['input_ids'][0, :starts[1] + divergence]):
            raise ValueError('Chosen/wrong prefixes disagree before decision')
        decisions[q['question_id']] = {'divergence_index': divergence, 'correct_token': ids[0][divergence],
                                      'wrong_token': ids[1][divergence], 'initial_prediction': prediction['prediction']}
    if len(decisions) != 20:
        raise ValueError('Expected fixed20 initial errors, without resampling')
    ROOT.mkdir(parents=True, exist_ok=True)
    write_json(ROOT / 'DECISIONS.json', decisions)
    paths = [Path(__file__), CONFIG, ROOT / 'DECISIONS.json',
             PROJECT_ROOT / 'docs/VLM_REGION_DECISION_SFT_V1_PROTOCOL.md',
             DATA_ROOT / 'data/MANIFEST.json', DATA_ROOT / 'data/CPU_INPUTS.json', DATA_ROOT / 'data/SCHEDULE.json',
             DATA_ROOT / 'reference/CACHE.json', DATA_ROOT / 'baseline/COMPLETE.json',
             DATA_ROOT / 'baseline/hard/PREDICTIONS.json', DATA_ROOT / 'baseline/retention/PREDICTIONS.json',
             DATA_ROOT / 'runs/sft/COMPLETE.json', DATA_ROOT / 'COMPLETION.json',
             Path(c['initial_adapter']) / 'adapter_config.json', Path(c['initial_adapter']) / 'adapter_model.safetensors',
             Path(c['baseline_config']), Path(base['reference_config'])]
    paths += list((SOURCE_ROOT / 'vlm_region_decision_sft_v1').glob('*.py'))
    paths += [SOURCE_ROOT / name for name in ('vlm_region_preference_probe_v1/common.py', 'vlm_alignment_signal/loss.py',
              'vlm_fp32_readout_v1/model.py', 'vlm_fp32_adapter/model.py', 'vlm_runtime_metadata_v1/model.py',
              'vlm_object_train/train.py', 'vlm_deep/common.py', 'vlm_probe/evaluate.py', 'research_runtime/queue.py')]
    parent = read(DATA_ROOT / 'PRELAUNCH_INPUT_LOCK.json')
    write_json(ROOT / 'STUDY_INPUT_LOCK.json', {'files': {str(p): sha_file(p) for p in set(paths)},
               'images': {im['path']: im['sha256'] for im in data['images'].values()},
               'initial_lora_sha256': parent['initial_lora_sha256'], 'base_weight_files': parent['base_weight_files'],
               'scope': 'Training-only32-step objective comparison; original SFT baseline reused, two new arms.'})
    print({'status': 'prepared', 'frozen_initial_wrong_qa': len(decisions), 'actual_cpu_encodings': 40,
           'new_arms': ROLES, 'updates_per_arm': 32})


def verify():
    lock = read(ROOT / 'STUDY_INPUT_LOCK.json')
    for path, digest in {**lock['files'], **lock['images']}.items():
        if sha_file(path) != digest:
            raise ValueError('New objective study input changed: ' + path)
    for path, expected in lock['base_weight_files'].items():
        stat = Path(path).stat()
        if {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns} != expected:
            raise ValueError('Base weights changed')


def condition(role):
    from .train import train
    verify()
    model, processor, trained = train(role)
    before = lora_hash(model)
    metrics = evaluate(model, processor, ROOT / 'runs' / role)
    if before != trained['final_lora_sha256'] or lora_hash(model) != before:
        raise ValueError('Evaluation changed trained adapter')
    write_json(ROOT / 'runs' / role / 'COMPLETE.json', {'status': 'completed', 'training_updates': 32,
               'metrics': metrics, 'final_lora_sha256': before, 'training_active_seconds': trained['active_seconds']})


def analyze():
    initial = read(DATA_ROOT / 'baseline/COMPLETE.json')['metrics']
    metrics = {'initial': initial, 'reused_plain_sft': read(DATA_ROOT / 'runs/sft/COMPLETE.json')['metrics'],
               **{role: read(ROOT / 'runs' / role / 'COMPLETE.json')['metrics'] for role in ROLES}}
    retention_initial = read(DATA_ROOT / 'baseline/retention/PREDICTIONS.json')
    changes = {}
    for role in ROLES:
        rows = read(ROOT / 'runs' / role / 'retention/PREDICTIONS.json')
        if [q['question_id'] for q in rows] != [q['question_id'] for q in retention_initial]:
            raise ValueError('Retention QA unpaired')
        changes[role] = sum(q['vqa_score'] < a['vqa_score'] for q, a in zip(rows, retention_initial))
    candidate, control = metrics['decision'], metrics['qa_weighted']
    learn = (candidate['hard']['casefold_correct'] >= initial['hard']['casefold_correct'] + 4
             and candidate['hard']['both_correct'] >= initial['hard']['both_correct'] + 2
             and candidate['retention']['score_points'] >= initial['retention']['score_points'] - 1.
             and changes['decision'] <= 2)
    advantage = (learn and candidate['hard']['casefold_correct'] >= control['hard']['casefold_correct'] + 2
                 and candidate['hard']['both_correct'] >= control['hard']['both_correct'] + 2
                 and candidate['retention']['score_points'] >= control['retention']['score_points'] - 1.)
    result = {'status': 'completed', 'scope': 'Training-only selected cohort, not independent capability or CV gain',
              'metrics': metrics, 'retention_new_lower_score_qa': changes,
              'learning_control_passed': learn, 'decision_advantage_for_further_research': advantage,
              'next': 'No automatic longer training; report all results before choosing next hypothesis.'}
    write_json(ROOT / 'COMPLETION.json', result)
    lines = ['# 关键决策监督32步：最终结果', '', result['scope'], '',
             '|模型|难例正确|双框全对|保持VQA总分|', '|---|---:|---:|---:|']
    lines += [f'|{role}|{v["hard"]["casefold_correct"]}/28|{v["hard"]["both_correct"]}/14|{v["retention"]["score_points"]}/64|' for role, v in metrics.items()]
    lines += ['', f'短学习门槛={learn}；超过等QA权重CE的准入={advantage}。没有独立泛化或自动扩大训练结论。']
    (PROJECT_ROOT / 'docs/VLM_REGION_DECISION_SFT_V1_RESULTS.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--role', choices=ROLES)
    args = parser.parse_args()
    if args.prepare:
        prepare()
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
