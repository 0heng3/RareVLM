"""Zero-update factorial restoration of the completed KD language/vision factors."""
import argparse
from datetime import datetime
import importlib
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import sha_file, write_json
from vlm_grpo_iteration_v1.common import read

BASE = PROJECT_ROOT.parents[1] / 'outputs'
PARENT = BASE / 'vlm_full_teacher_kd_v1'
ROOT = BASE / 'vlm_kd_component_restore_v1'
PROTOCOL = PROJECT_ROOT / 'docs/VLM_KD_COMPONENT_RESTORE_V1_PROTOCOL.md'
ROLES = ('language_only', 'visual_only')
COHORTS = ('transfer_dev', 'native_hard', 'retention')


def mixed_factors(initial, final, role):
    """A complete factor pair is restored; never disable the original adaptation."""
    import torch
    if role not in ROLES or set(initial) != set(final) or len(initial) != 204:
        raise ValueError('Wrong intervention or factor inventory')
    state = {}
    for name, value in final.items():
        original = initial[name]
        if value.dtype != torch.float32 or original.dtype != torch.float32 or value.shape != original.shape:
            raise ValueError('Factor shape/dtype differs: ' + name)
        retain_final = ('.visual.' in name) == (role == 'visual_only')
        state[name] = value if retain_final else original
    return state


def prepare():
    import torch
    from safetensors.torch import load_file
    if (ROOT / 'STUDY_INPUT_LOCK.json').exists():
        raise FileExistsError('Component diagnostic already frozen')
    parent = read(PARENT / 'COMPLETION.json')
    if parent['phase'] != 'completed_negative' or parent['student_signal']:
        raise ValueError('Require the completed negative parent')
    cfg = read(PARENT / 'CONFIG.json')
    initial = Path(cfg['initial_adapter'])
    final = Path(read(PARENT / 'runs/kd/train/TRAIN_COMPLETE.json')['adapter'])
    initial_state = load_file(str(initial / 'adapter_model.safetensors'))
    final_state = load_file(str(final / 'adapter_model.safetensors'))
    checks = {}
    for role in ROLES:
        state = mixed_factors(initial_state, final_state, role)
        retained = [k for k in state if ('.visual.' in k) == (role == 'visual_only')]
        restored = [k for k in state if k not in retained]
        if len(retained) != (60 if role == 'visual_only' else 144):
            raise ValueError('Incorrect final-side inventory')
        if sum(v.numel() for v in state.values()) != 3686400:
            raise ValueError('Stored parameter count differs')
        if any(not torch.equal(state[k], final_state[k]) for k in retained) or any(not torch.equal(state[k], initial_state[k]) for k in restored):
            raise ValueError('Actual CPU intervention differs')
        for side in (retained, restored):
            stems = {k.replace('.lora_A.', '.lora_PAIR.').replace('.lora_B.', '.lora_PAIR.') for k in side}
            if len(stems) * 2 != len(side):
                raise ValueError('Incomplete A/B pair')
        checks[role] = {'retained_final_factors': len(retained), 'restored_start_factors': len(restored),
            'actual_source_tensors_exact': True, 'complete_AB_pairs': True, 'stored_parameters': 3686400}
    write_json(ROOT / 'CPU_INTERVENTION_CHECK.json', {'status': 'passed', 'roles': checks, 'new_model_calls': 0})
    manifest = read(PARENT / 'data/MANIFEST.json')
    data = {'rows': [], 'images': {}, 'evaluations': {n: manifest['evaluations'][n] for n in COHORTS}}
    used = {q['image_id'] for rows in data['evaluations'].values() for q in rows}
    data['images'] = {k: manifest['images'][k] for k in used}
    if [len(data['evaluations'][n]) for n in COHORTS] != [1617, 128, 64]:
        raise ValueError('Require all fixed parent cohorts')
    write_json(ROOT / 'data/MANIFEST.json', data)
    dependencies = [Path(__file__), PROTOCOL, ROOT / 'data/MANIFEST.json', ROOT / 'CPU_INTERVENTION_CHECK.json',
        PARENT / 'COMPLETION.json', PARENT / 'DECISION.json', PARENT / 'CONFIG.json',
        PARENT / 'runs/kd/train/TRAIN_COMPLETE.json']
    dependencies += [PROJECT_ROOT / p for p in ('tools/run_native_correction_sft_dpo_v2.py',
        'src/vlm_fit_positive_control_v1/study.py', 'src/vlm_native_scope_v1/model.py',
        'src/vlm_batch_stable_v1/model.py', 'src/research_runtime/queue.py')]
    dependencies += [directory / name for directory in (initial, final) for name in ('adapter_model.safetensors', 'adapter_config.json')]
    dependencies += [Path(v['path']) for v in data['images'].values()]
    for role in ('start', 'kd'):
        for name in COHORTS:
            source = PARENT / ('baseline' if role == 'start' else 'runs/kd/' + ('transfer' if name == 'transfer_dev' else 'internal')) / name / 'PREDICTIONS.json'
            original = read(source); rows = data['evaluations'][name]
            if len(original) != len(rows) or [(q['question_id'], q['input_sha256']) for q in original] != [(q['question_id'], q['inference_input_sha256']) for q in rows]:
                raise ValueError('Reused endpoints differ')
            destination = ROOT / 'reused' / role / name / 'PREDICTIONS.json'
            destination.parent.mkdir(parents=True, exist_ok=True); destination.write_bytes(source.read_bytes())
            dependencies += [source, destination]
    plan = {'status': 'frozen', 'recorded_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'initial_adapter': str(initial), 'final_adapter': str(final), 'roles': list(ROLES),
        'cohorts': list(COHORTS), 'qa_per_role': 1809, 'new_generations': 3618,
        'reused_endpoint_generations': 3618, 'optimizer_updates': 0, 'new_adapters': 0,
        'automatic_training': False, 'original_KD_stop_unchanged': True,
        'evaluation_cap_seconds': 1500, 'peak_gib_cap': 65,
        'files': {str(p): sha_file(p) for p in dependencies}}
    write_json(ROOT / 'STUDY_INPUT_LOCK.json', plan)
    print('COMPONENT_RESTORE_PREPARED', {k: plan[k] for k in ('qa_per_role', 'new_generations', 'optimizer_updates')}, flush=True)


def verify():
    plan = read(ROOT / 'STUDY_INPUT_LOCK.json')
    for path, digest in plan['files'].items():
        if sha_file(path) != digest:
            raise ValueError('Frozen restore diagnostic input changed: ' + path)
    return plan


def worker(role):
    import torch
    from safetensors.torch import load_file
    from vlm_native_scope_v1.model import load
    from vlm_human_sft.evaluate import lora_hash
    if role not in ROLES:
        raise ValueError('Unregistered restore role')
    plan = verify(); started = time.monotonic()
    model, processor, _ = load(plan['final_adapter'], 'visual', False)
    initial = load_file(str(Path(plan['initial_adapter']) / 'adapter_model.safetensors'))
    final = load_file(str(Path(plan['final_adapter']) / 'adapter_model.safetensors'))
    state = mixed_factors(initial, final, role); touched = []; found = set()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if 'lora_' not in name:
                continue
            key = name.replace('.default.', '.')
            if key not in state or parameter.shape != state[key].shape or parameter.dtype != torch.float32:
                raise ValueError('Intervention model inventory differs')
            if ('.visual.' in key) != (role == 'visual_only'):
                parameter.copy_(state[key]); touched.append(key)
            if not torch.equal(parameter.detach().cpu(), state[key]):
                raise ValueError('Actual in-memory factor differs')
            found.add(key)
    if found != set(state) or len(touched) != (144 if role == 'visual_only' else 60):
        raise ValueError('Wrong component restored')
    model.requires_grad_(False); model.eval(); model.config.use_cache = True
    before = lora_hash(model); torch.cuda.reset_peak_memory_stats()
    directory = ROOT / 'runs' / role
    write_json(directory / 'INTERVENTION.json', {'status': 'passed', 'role': role, 'restored_factors': len(touched),
        'restored_names': touched, 'model_lora_sha256': before, 'actual_factors_exact': True,
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'optimizer_updates': 0, 'new_adapters': 0})
    engine = importlib.import_module('run_native_correction_sft_dpo_v2'); engine.ROOT = ROOT
    engine.evaluate(model, processor, directory, COHORTS, read(ROOT / 'data/MANIFEST.json'))
    if lora_hash(model) != before:
        raise ValueError('Zero-update intervention changed during evaluation')
    write_json(directory / 'COMPLETE.json', {'status': 'completed', 'role': role,
        'new_generations': 1809, 'optimizer_updates': 0, 'active_seconds': time.monotonic() - started,
        'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
        'model_lora_sha256': before, 'read_only_model_hash_unchanged': True})


def contrast(records, weights, cohort):
    import numpy as np
    reference = records['start'][cohort]; clusters = {}
    for i, q in enumerate(reference):
        value = sum(w * records[role][cohort][i]['vqa_score'] for role, w in weights.items())
        item = clusters.setdefault(q['image_id'], [0., 0]); item[0] += value; item[1] += 1
    values = np.asarray(list(clusters.values()), dtype=np.float64)
    indices = np.random.default_rng(20261021).integers(0, len(values), size=(6000, len(values)))
    bootstrap = 100 * values[indices, 0].sum(1) / values[indices, 1].sum(1)
    return {'delta_pp': float(100 * values[:, 0].sum() / values[:, 1].sum()),
        'ci95': np.quantile(bootstrap, [.025, .975]).tolist(),
        'ci99_five_main_comparisons': np.quantile(bootstrap, [.005, .995]).tolist(),
        'image_clusters': len(values)}


def analyze():
    data = read(ROOT / 'data/MANIFEST.json'); records = {}
    for role in ('start', 'language_only', 'visual_only', 'kd'):
        directory = ROOT / ('reused' if role in ('start', 'kd') else 'runs') / role
        records[role] = {n: read(directory / n / 'PREDICTIONS.json') for n in COHORTS}
    identity = lambda rows: [(q['question_id'], q['image_id'], q['input_sha256']) for q in rows]
    for name in COHORTS:
        if any(identity(rec[name]) != identity(records['start'][name]) for rec in records.values()):
            raise ValueError('Factorial endpoint frames differ')
    rare = {q['question_id'] for q in data['evaluations']['transfer_dev'] if 1 <= q['answer_frequency'] <= 5}
    for rec in records.values():
        rec['full_rare342'] = [q for q in rec['transfer_dev'] if q['question_id'] in rare]
    comparisons = {'restore_language_01_minus_11': {'visual_only': 1, 'kd': -1},
        'restore_visual_10_minus_11': {'language_only': 1, 'kd': -1},
        'language_update_10_minus_00': {'language_only': 1, 'start': -1},
        'visual_update_01_minus_00': {'visual_only': 1, 'start': -1},
        'interaction_11_minus_10_minus_01_plus_00': {'kd': 1, 'language_only': -1, 'visual_only': -1, 'start': 1}}
    metrics = {role: {n: 100 * sum(q['vqa_score'] for q in rows) / len(rows) for n, rows in rec.items()} for role, rec in records.items()}
    results = {key: {n: contrast(records, weights, n) for n in records['start']} for key, weights in comparisons.items()}
    result = {'status': 'completed', 'phase': 'zero_update_component_diagnosis_complete', 'metrics': metrics,
        'contrasts': results, 'new_generations': 3618, 'optimizer_updates': 0, 'new_adapters': 0,
        'automatic_training': False, 'original_KD_stop_unchanged': True, 'goal_complete': False,
        'scope': 'Post-training fixed-model component intervention; reused internal development cohorts, no independently retrained scope controls or external/seed confirmation.'}
    write_json(ROOT / 'ANALYSIS.json', result)
    write_json(ROOT / 'DECISION.json', {'status': 'completed', 'decision': 'diagnosis_complete_no_automatic_training_or_model_promotion', 'automatic_training': False})
    write_json(ROOT / 'COMPLETION.json', result)
    print('COMPONENT_RESTORE_COMPLETE', result, flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true'); parser.add_argument('--worker', choices=ROLES); args = parser.parse_args()
    if args.prepare:
        prepare(); return
    if args.worker:
        worker(args.worker); return
    verify()
    from research_runtime.queue import run_queue
    entry = str(Path(__file__).resolve())
    run_queue(ROOT / 'queue', [{'key': role, 'command': [sys.executable, '-u', entry, '--worker', role],
        'completion_marker': str(ROOT / 'runs' / role / 'COMPLETE.json')} for role in ROLES],
        {'study_sha256': sha_file(ROOT / 'STUDY_INPUT_LOCK.json')})
    analyze()


if __name__ == '__main__':
    main()
