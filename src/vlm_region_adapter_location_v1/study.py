"""Two parameter-matched32-step adapter locations, ordinary full CE throughout."""
import argparse
from pathlib import Path
import sys
import torch
from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT
from vlm_deep.common import sha_file, write_json
from vlm_human_sft.evaluate import lora_hash
from .common import ROOT, DATA_ROOT, CONFIG, read, evaluate
from .model import Residual, residual_hash

ROLES = ('readout', 'visual')


def prepare():
    if (ROOT / 'STUDY_INPUT_LOCK.json').exists():
        raise FileExistsError('Already frozen; inspect existing queue')
    c = read(CONFIG)
    # New module-specific identity and gradient checks, without a model/optimizer step.
    residual = Residual()
    value = torch.randn(5, 2048, generator=torch.Generator().manual_seed(7)).bfloat16()
    assert torch.equal(residual(value), value)
    residual(value).float().square().mean().backward()
    assert residual.up.grad.norm() > 0 and residual.down.grad.norm() == 0
    assert sum(p.numel() for p in residual.parameters()) == 131072
    base = read(c['baseline_config'])
    parent = read(DATA_ROOT / 'PRELAUNCH_INPUT_LOCK.json')
    ROOT.mkdir(parents=True, exist_ok=True)
    write_json(ROOT / 'NEW_MODULE_CPU_CHECK.json', {'status': 'passed', 'zero_output_bitwise_identity_bf16': True,
               'up_gradient_positive_down_gradient_zero_at_zero_up': True, 'parameters_each_arm': 131072,
               'optimizer_updates': 0, 'not_a_model_capability_result': True})
    paths = [CONFIG, PROJECT_ROOT / 'docs/VLM_REGION_ADAPTER_LOCATION_V1_PROTOCOL.md',
             DATA_ROOT / 'data/MANIFEST.json', DATA_ROOT / 'data/CPU_INPUTS.json', DATA_ROOT / 'data/SCHEDULE.json',
             DATA_ROOT / 'reference/CACHE.json', DATA_ROOT / 'baseline/COMPLETE.json',
             DATA_ROOT / 'baseline/hard/PREDICTIONS.json', DATA_ROOT / 'baseline/retention/PREDICTIONS.json',
             DATA_ROOT / 'runs/sft/COMPLETE.json', DATA_ROOT / 'COMPLETION.json',
             ROOT.parent / 'vlm_region_decision_sft_v1/COMPLETION.json',
             Path(c['initial_adapter']) / 'adapter_config.json', Path(c['initial_adapter']) / 'adapter_model.safetensors',
             Path(c['baseline_config']), Path(base['reference_config'])]
    paths += list((SOURCE_ROOT / 'vlm_region_adapter_location_v1').glob('*.py'))
    paths += [SOURCE_ROOT / name for name in ('vlm_region_preference_probe_v1/common.py', 'vlm_alignment_signal/loss.py',
              'vlm_fp32_readout_v1/model.py', 'vlm_fp32_adapter/model.py', 'vlm_runtime_metadata_v1/model.py',
              'vlm_object_train/train.py', 'vlm_deep/common.py', 'vlm_probe/evaluate.py', 'research_runtime/queue.py')]
    data = read(DATA_ROOT / 'data/MANIFEST.json')
    write_json(ROOT / 'STUDY_INPUT_LOCK.json', {'files': {str(p): sha_file(p) for p in paths},
               'images': {im['path']: im['sha256'] for im in data['images'].values()},
               'initial_lora_sha256': parent['initial_lora_sha256'], 'base_weight_files': parent['base_weight_files'],
               'scope': 'Only training-positive-control evidence; no main task or CV gain'})
    print({'status': 'prepared', 'new_parameters_each_arm': 131072, 'total_trainable_each_arm': 1974272,
           'new_module_cpu_checks': 'passed', 'updates_each_arm': 32})


def verify():
    lock = read(ROOT / 'STUDY_INPUT_LOCK.json')
    for path, digest in {**lock['files'], **lock['images']}.items():
        if sha_file(path) != digest:
            raise ValueError('New location study input changed: ' + path)
    for path, expected in lock['base_weight_files'].items():
        stat = Path(path).stat()
        if {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns} != expected:
            raise ValueError('Base weights changed')


def condition(role):
    from .train import train
    verify()
    model, processor, trained = train(role)
    before, extra = lora_hash(model), residual_hash(model)
    metrics = evaluate(model, processor, ROOT / 'runs' / role)
    if before != trained['final_lora_sha256'] or lora_hash(model) != before or residual_hash(model) != extra:
        raise ValueError('Evaluation changed model')
    write_json(ROOT / 'runs' / role / 'COMPLETE.json', {'status': 'completed', 'training_updates': 32,
               'metrics': metrics, 'final_lora_sha256': before, 'residual_sha256': extra,
               'training_active_seconds': trained['active_seconds']})


def analyze():
    metrics = {'initial': read(DATA_ROOT / 'baseline/COMPLETE.json')['metrics'],
               'reused_plain_sft': read(DATA_ROOT / 'runs/sft/COMPLETE.json')['metrics'],
               **{role: read(ROOT / 'runs' / role / 'COMPLETE.json')['metrics'] for role in ROLES}}
    initial_rows = read(DATA_ROOT / 'baseline/hard/PREDICTIONS.json')
    anchors = read(DATA_ROOT / 'baseline/retention/PREDICTIONS.json')
    comparisons = {}
    for role in ROLES:
        rows = read(ROOT / 'runs' / role / 'hard/PREDICTIONS.json')
        retention = read(ROOT / 'runs' / role / 'retention/PREDICTIONS.json')
        if [q['question_id'] for q in rows] != [q['question_id'] for q in initial_rows] or [q['question_id'] for q in retention] != [q['question_id'] for q in anchors]:
            raise ValueError('Generated populations differ')
        comparisons[role] = {'corrected': sum(not a['casefold_exact'] and q['casefold_exact'] for a, q in zip(initial_rows, rows)),
             'broken': sum(a['casefold_exact'] and not q['casefold_exact'] for a, q in zip(initial_rows, rows)),
             'retention_new_lower_score_qa': sum(q['vqa_score'] < a['vqa_score'] for q, a in zip(retention, anchors))}
    initial, candidate, control = metrics['initial'], metrics['visual'], metrics['readout']
    learning = candidate['hard']['casefold_correct'] >= initial['hard']['casefold_correct'] + 4 and candidate['hard']['both_correct'] >= initial['hard']['both_correct'] + 2 and candidate['retention']['score_points'] >= initial['retention']['score_points'] - 1. and comparisons['visual']['retention_new_lower_score_qa'] <= 2
    advantage = learning and candidate['hard']['casefold_correct'] >= control['hard']['casefold_correct'] + 2 and candidate['hard']['both_correct'] >= control['hard']['both_correct'] + 2 and candidate['retention']['score_points'] >= control['retention']['score_points'] - 1.
    result = {'status': 'completed', 'scope': 'Training-only adapter-location control, not generalization or CV gain',
              'metrics': metrics, 'comparisons': comparisons, 'visual_learning_passed': learning,
              'visual_advantage_for_further_research': advantage, 'new_parameters_each_arm': 131072,
              'next': 'No automatic extension, sweep, main evaluation or resume change.'}
    write_json(ROOT / 'COMPLETION.json', result)
    lines = ['# 等参数残差适配位置：32步结果', '', result['scope'], '', '|模型|难例正确|双框全对|保持VQA总分|', '|---|---:|---:|---:|']
    lines += [f'|{role}|{v["hard"]["casefold_correct"]}/28|{v["hard"]["both_correct"]}/14|{v["retention"]["score_points"]}/64|' for role, v in metrics.items()]
    lines += ['', f'视觉侧学习准入={learning}；超过等参数读出侧控制准入={advantage}。未做独立泛化或自动扩训。']
    (PROJECT_ROOT / 'docs/VLM_REGION_ADAPTER_LOCATION_V1_RESULTS.md').write_text('\n'.join(lines) + '\n')


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
