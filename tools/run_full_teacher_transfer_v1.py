"""Evaluate the already verified full teacher on the entire fixed native task."""
import argparse
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import sys
import time
from zoneinfo import ZoneInfo

from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import sha_file, write_json
from vlm_grpo_iteration_v1.common import read

BASE = PROJECT_ROOT.parents[1] / 'outputs'
ROOT = BASE / 'vlm_full_teacher_transfer_v1'
PARENT = BASE / 'vlm_full_teacher_kd_v1'
ASSETS = BASE / 'vlm_full_teacher_assets_v1'
SIGNAL = BASE / 'vlm_full_teacher_signal_v1'
PROTOCOL = PROJECT_ROOT / 'docs/VLM_FULL_TEACHER_TRANSFER_V1_PROTOCOL.md'
COHORTS = ('transfer_dev', 'retention')


def prepare():
    if (ROOT / 'PLAN.json').exists():
        raise FileExistsError('Full teacher transfer already registered')
    assets = read(ASSETS / 'TRANSPORT_COMPLETE.json'); original = read(SIGNAL / 'PLAN.json')
    if not assets['all_pinned_official_hashes_matched'] or assets['revision'] != original['teacher_revision']:
        raise ValueError('Previously verified teacher asset receipt missing')
    if read(BASE / 'vlm_kd_component_restore_v1/COMPLETION.json')['status'] != 'completed':
        raise ValueError('Require completed prior component evidence')
    parent = read(PARENT / 'data/MANIFEST.json')
    data = {'rows': [], 'evaluations': {n: parent['evaluations'][n] for n in COHORTS}, 'images': {}}
    if [len(data['evaluations'][n]) for n in COHORTS] != [1617, 64]:
        raise ValueError('Full fixed cohorts required')
    used = {q['image_id'] for rows in data['evaluations'].values() for q in rows}
    data['images'] = {image_id: parent['images'][image_id] for image_id in used}
    write_json(ROOT / 'DATA.json', data)
    paths = [Path(__file__), PROTOCOL, ROOT / 'DATA.json', ASSETS / 'TRANSPORT_COMPLETE.json',
        ASSETS / 'MODEL_SOURCE.json', SIGNAL / 'PLAN.json', PARENT / 'COMPLETION.json',
        BASE / 'vlm_kd_component_restore_v1/COMPLETION.json', Path(original['reference_config'])]
    paths += [PROJECT_ROOT / p for p in ('src/vlm_fp32_readout_v1/model.py', 'src/vlm_batch_stable_v1/model.py',
        'src/vlm_fit_positive_control_v1/study.py', 'src/research_runtime/queue.py')]
    paths += [p for p in (ASSETS / 'model').iterdir() if p.is_file() and p.suffix != '.safetensors' and not p.name.endswith(('.partial', '.prefetch'))]
    paths += [Path(meta['path']) for meta in data['images'].values()]
    for name in COHORTS:
        source = PARENT / 'baseline' / name / 'PREDICTIONS.json'; rows = read(source)
        if [(q['question_id'], q['input_sha256']) for q in rows] != [(q['question_id'], q['inference_input_sha256']) for q in data['evaluations'][name]]:
            raise ValueError('Reused strong student frames differ')
        destination = ROOT / 'baseline' / name / 'PREDICTIONS.json'
        destination.parent.mkdir(parents=True, exist_ok=True); destination.write_bytes(source.read_bytes()); paths += [source, destination]
    shards = {}
    for shard in range(4):
        subset = {'rows': [], 'images': data['images'], 'evaluations': {}}
        for name in COHORTS:
            rows = data['evaluations'][name]
            subset['evaluations'][name] = [q for start in range(0, len(rows), 4) if (start // 4) % 4 == shard for q in rows[start:start + 4]]
        path = ROOT / 'shards' / str(shard) / 'DATA.json'; write_json(path, subset); paths.append(path)
        shards[str(shard)] = {n: len(rows) for n, rows in subset['evaluations'].items()}
    for name in COHORTS:
        all_ids = [q['question_id'] for shard in range(4) for q in read(ROOT / 'shards' / str(shard) / 'DATA.json')['evaluations'][name]]
        if len(all_ids) != len(set(all_ids)) or set(all_ids) != {q['question_id'] for q in data['evaluations'][name]}:
            raise ValueError('Sharded full coverage differs')
        original_batches = [[q['question_id'] for q in data['evaluations'][name][i:i + 4]] for i in range(0, len(data['evaluations'][name]), 4)]
        for shard in range(4):
            subset = read(ROOT / 'shards' / str(shard) / 'DATA.json')['evaluations'][name]
            batches = [[q['question_id'] for q in subset[i:i + 4]] for i in range(0, len(subset), 4)]
            if batches != original_batches[shard::4]:
                raise ValueError('Original batch composition changed')
    plan = {'status': 'frozen', 'recorded_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'teacher_repository': original['teacher_repository'], 'teacher_revision': original['teacher_revision'],
        'teacher_weights': original['teacher_weights'], 'model_root': str(ASSETS / 'model'),
        'reference_config': original['reference_config'], 'shards': shards, 'cohorts': list(COHORTS),
        'new_generations': 1681, 'optimizer_updates': 0, 'new_adapters': 0,
        'teacher_forced_sequences': 0, 'automatic_training': False,
        'generation_cap_seconds_per_job': 1200, 'peak_gib_cap': 65,
        'files': {str(p): sha_file(p) for p in paths}}
    write_json(ROOT / 'PLAN.json', plan)
    print('FULL_TEACHER_TRANSFER_PREPARED', {'shards': shards, 'new_generations': 1681, 'optimizer_updates': 0}, flush=True)


def verify():
    plan = read(ROOT / 'PLAN.json')
    for path, digest in plan['files'].items():
        if sha_file(path) != digest:
            raise ValueError('Frozen teacher-transfer input differs: ' + path)
    for item in plan['teacher_weights']:
        if (Path(plan['model_root']) / item['name']).stat().st_size != item['size']:
            raise ValueError('Previously verified teacher weight size differs')
    return plan


def worker(shard):
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration
    from vlm_fp32_readout_v1.model import install_fp32_readout
    from vlm_batch_stable_v1.model import install
    from vlm_probe.evaluate import processor_for
    from vlm_fit_positive_control_v1.study import generated
    plan = verify(); directory = ROOT / 'shards' / str(shard); data = read(directory / 'DATA.json')
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    begun = time.monotonic()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(plan['model_root'], local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation='sdpa', low_cpu_mem_usage=True, device_map={'': 'cuda:0'})
    model.eval().requires_grad_(False)
    binding = {'readout': install_fp32_readout(model), 'numerics': install(SimpleNamespace(get_base_model=lambda: model), 'imagewise_fp32_language')}
    model.eval().requires_grad_(False); model.config.use_cache = True
    processor = processor_for(Path(plan['model_root']), read(plan['reference_config']), 1024)
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError('Teacher must remain completely frozen')
    write_json(directory / 'BINDING.json', {'status': 'passed', 'binding': binding, 'trainable_parameters': 0,
        'optimizer_updates': 0, 'teacher_revision': plan['teacher_revision']})
    torch.cuda.reset_peak_memory_stats(); started = time.monotonic(); total = 0
    for name in COHORTS:
        rows = data['evaluations'][name]; path = directory / name / 'PREDICTIONS.json'; predictions = read(path) if path.exists() else []
        if len(predictions) > len(rows) or (len(predictions) % 4 and len(predictions) != len(rows)) or [q['question_id'] for q in predictions] != [q['question_id'] for q in rows[:len(predictions)]]:
            raise ValueError('Whole original-batch resume differs')
        total += len(predictions)
        for start in range(len(predictions), len(rows), 4):
            chunk = rows[start:start + 4]; answers = generated(model, processor, chunk, data)
            if [(q['question_id'], q['input_sha256']) for q in answers] != [(q['question_id'], q['inference_input_sha256']) for q in chunk]:
                raise ValueError('Actual teacher generation input differs')
            predictions.extend(answers); total += len(answers)
            temporary = path.with_suffix('.tmp'); write_json(temporary, predictions); temporary.replace(path)
            progress = {'status': 'evaluating_full_teacher', 'shard': shard, 'cohort': name,
                'completed': total, 'expected': sum(plan['shards'][str(shard)].values()),
                'generation_seconds': time.monotonic() - started, 'active_seconds': time.monotonic() - begun,
                'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30, 'optimizer_updates': 0}
            write_json(directory / 'PROGRESS.json', progress)
            if progress['generation_seconds'] > 1200 or progress['peak_allocated_gib'] > 65:
                raise RuntimeError('Fixed full-teacher generation budget exceeded')
    write_json(directory / 'COMPLETE.json', {'status': 'completed', 'shard': shard, 'completed': total,
        'expected': sum(plan['shards'][str(shard)].values()), 'new_generations': total,
        'active_seconds': time.monotonic() - begun, 'generation_seconds': time.monotonic() - started,
        'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30, 'optimizer_updates': 0,
        'teacher_forced_sequences': 0, 'new_adapters': 0, 'trainable_parameters': 0})


def analyze():
    from vlm_grpo_iteration_v1.study import paired
    data = read(ROOT / 'DATA.json'); records = {'start': {}, 'teacher': {}}
    for name in COHORTS:
        records['start'][name] = read(ROOT / 'baseline' / name / 'PREDICTIONS.json')
        lookup = {q['question_id']: q for shard in range(4) for q in read(ROOT / 'shards' / str(shard) / name / 'PREDICTIONS.json')}
        order = data['evaluations'][name]
        if set(lookup) != {q['question_id'] for q in order}:
            raise ValueError('Incomplete full teacher endpoint')
        records['teacher'][name] = [lookup[q['question_id']] for q in order]
        if [(q['question_id'], q['image_id'], q['input_sha256']) for q in records['teacher'][name]] != [(q['question_id'], q['image_id'], q['input_sha256']) for q in records['start'][name]]:
            raise ValueError('Full teacher and strong student frames differ')
        write_json(ROOT / 'teacher' / name / 'PREDICTIONS.json', records['teacher'][name])
    rare = {q['question_id'] for q in data['evaluations']['transfer_dev'] if 1 <= q['answer_frequency'] <= 5}
    for record in records.values():
        record['full_rare342'] = [q for q in record['transfer_dev'] if q['question_id'] in rare]
    metrics = {role: {name: 100 * sum(q['vqa_score'] for q in rows) / len(rows) for name, rows in record.items()} for role, record in records.items()}
    comparisons = {name: paired(records['start'][name], records['teacher'][name]) for name in records['start']}
    main = comparisons['transfer_dev']
    gates = {'full1617_plus_point5_positive_ci': main['delta_pp'] >= .5 and main['ci95'][0] > 0,
        'rare342_point_protection': comparisons['full_rare342']['delta_pp'] >= -.5,
        'retention64_point_protection': comparisons['retention']['delta_pp'] >= -.5}
    changes = {}
    for name in records['start']:
        pairs = list(zip(records['start'][name], records['teacher'][name]))
        changes[name] = {'score_improved': sum(b['vqa_score'] > a['vqa_score'] + 1e-9 for a, b in pairs),
            'score_regressed': sum(b['vqa_score'] < a['vqa_score'] - 1e-9 for a, b in pairs),
            'teacher_unterminated': sum(not b['terminated'] for a, b in pairs)}
    result = {'status': 'completed', 'metrics': metrics, 'teacher_vs_strong_student': comparisons, 'changes': changes,
        'gates': gates, 'full_task_teacher_opportunity': all(gates.values()), 'new_generations': 1681,
        'optimizer_updates': 0, 'new_adapters': 0, 'teacher_forced_sequences': 0,
        'automatic_training': False, 'student_improved': False, 'goal_complete': False,
        'scope': 'Different frozen 7B teacher versus specialized 3B student on reused internal cohorts; not pure size causality, a new student improvement or external/seed confirmation.'}
    write_json(ROOT / 'ANALYSIS.json', result)
    write_json(ROOT / 'DECISION.json', {'status': 'completed', 'decision': 'teacher_opportunity_requires_separate_student_protocol' if all(gates.values()) else 'full_task_teacher_opportunity_not_established', 'automatic_training': False})
    write_json(ROOT / 'COMPLETION.json', result)
    print('FULL_TEACHER_TRANSFER_COMPLETE', result, flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true'); parser.add_argument('--worker', type=int, choices=range(4)); args = parser.parse_args()
    if args.prepare:
        prepare(); return
    if args.worker is not None:
        worker(args.worker); return
    verify()
    from research_runtime.queue import run_queue
    entry = str(Path(__file__).resolve())
    run_queue(ROOT / 'queue', [{'key': 'teacher_' + str(s), 'command': [sys.executable, '-u', entry, '--worker', str(s)],
        'completion_marker': str(ROOT / 'shards' / str(s) / 'COMPLETE.json')} for s in range(4)],
        {'plan_sha256': sha_file(ROOT / 'PLAN.json')})
    analyze()


if __name__ == '__main__': main()
