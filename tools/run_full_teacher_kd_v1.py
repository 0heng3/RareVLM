"""Matched joint-LoRA student training with hard, uniform and real-teacher targets."""
import argparse
from collections import Counter
import importlib
import json
import os
from pathlib import Path
import random
import time
import sys

from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import content_sha, sha_file, write_json
from vlm_grpo_iteration_v1.common import read

BASE = PROJECT_ROOT.parents[1] / 'outputs'
ROOT = BASE / 'vlm_full_teacher_kd_v1'
PARENT = BASE / 'vlm_stvqa_replay_v1'
TEACHER = BASE / 'vlm_full_teacher_signal_v1'
PROTOCOL = PROJECT_ROOT / 'docs/VLM_FULL_TEACHER_KD_V1_PROTOCOL.md'
ROLES = ('hard', 'uniform', 'kd')
INTERNAL = ('native_hard', 'literal_validation', 'seen_validation', 'retention')


def prepare():
    import torch
    if (ROOT / 'STUDY_INPUT_LOCK.json').exists():
        raise FileExistsError('KD student study already frozen')
    signal = read(TEACHER / 'COMPLETION.json')
    if not signal['teacher_signal'] or not all(signal['gates'].values()):
        raise ValueError('No admitted full-teacher information signal')
    for name in ('MANIFEST.json', 'CPU_INPUTS.json', 'SCHEDULE.json'):
        destination = ROOT / 'data' / name; destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((PARENT / 'data' / name).read_bytes())
    data = read(ROOT / 'data/MANIFEST.json'); cpu = read(ROOT / 'data/CPU_INPUTS.json')
    schedule = read(ROOT / 'data/SCHEDULE.json')['updates']
    exposures = Counter(q for update in schedule for ids in update['micros'] for q in ids)
    if len(data['rows']) != 384 or len(schedule) != 32 or sum(exposures.values()) != 512:
        raise ValueError('Registered chosen exposure schedule differs')
    teacher_cpu = read(TEACHER / 'CPU_INPUTS.json'); valid = read(TEACHER / 'VALID_IDS.json'); cache = {}
    for item in read(TEACHER / 'DIAGNOSTICS.json'):
        t = item['teacher']; qid = item['question_id']
        if not (t['literal_correct'] and t['body_nll'] <= 1 and t['eos_nll'] <= .5):
            continue
        if exposures[qid] != 2:
            raise ValueError('Teacher auxiliary mask must remain in the twice-exposed native training pool')
        frame = teacher_cpu[qid]; expected = cpu[qid + ':chosen']
        if frame['input_sha256'] != expected['input_ids_sha256'] or frame['labels_sha256'] != expected['labels_sha256']:
            raise ValueError('Teacher cache and new student chosen frame differ')
        path = TEACHER / 'runs/teacher/logits' / (qid + '.pt')
        if sha_file(path) != t['logits_sha256']:
            raise ValueError('Admitted teacher cache changed')
        tensors = torch.load(path, map_location='cpu', weights_only=True)
        if tensors['logits'].shape != (len(tensors['labels']), len(valid)) or not torch.isfinite(tensors['logits']).all():
            raise ValueError('Invalid cached soft-target shape or values')
        if not set(tensors['labels'].tolist()) <= set(valid):
            raise ValueError('Cached teacher labels are outside actual shared IDs')
        cache[qid] = {'path': str(path), 'sha256': t['logits_sha256'], 'body_tokens': len(tensors['labels']),
                      'novel_correction': item['novel_teacher_correction']}
    if len(cache) != 103:
        raise ValueError('Fixed admitted teacher mask differs from preregistered103')
    write_json(ROOT / 'teacher/CACHE_INDEX.json', cache); write_json(ROOT / 'teacher/VALID_IDS.json', valid)
    write_json(ROOT / 'teacher/COMPLETE.json', {'status': 'completed', 'qa': len(cache), 'auxiliary_exposures_per_role': 206,
               'new_teacher_forwards': 0, 'new_generations': 0, 'optimizer_updates': 0})
    trained = read(PARENT / 'runs/ce/train/TRAIN_COMPLETE.json')
    config = {**read(PARENT / 'CONFIG.json'), 'initial_adapter': trained['adapter'], 'output_root': str(ROOT),
        'roles': list(ROLES), 'algorithm': 'joint_LoRA_CE_plus_shared_vocabulary_teacher_KL',
        'optimized_scope': 'all_existing_language_and_visual_LoRA', 'expected_trainable': 3686400,
        'lr': 1e-4, 'temperature': 2., 'auxiliary_weight': 1., 'training_cap_seconds': 1800,
        'evaluation_cap_seconds': 1500, 'automatic_more_training': False}
    for old in ('beta', 'dpo_weight'):
        config.pop(old, None)
    write_json(ROOT / 'CONFIG.json', config)
    reused = {}
    for name in INTERNAL + ('transfer_dev',):
        phase = 'transfer' if name == 'transfer_dev' else 'internal'
        original = PARENT / 'runs/ce' / phase / name / 'PREDICTIONS.json'
        destination = ROOT / 'baseline' / name / 'PREDICTIONS.json'; destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(original.read_bytes())
        reused[name] = {'source': str(original), 'sha256': sha_file(original), 'new_generations': 0}
    write_json(ROOT / 'BASELINE_REUSE.json', reused)
    (ROOT / 'DATA_RECEIPT.json').write_bytes((BASE / 'vlm_visual_preference_v1/DATA_RECEIPT.json').read_bytes())
    paths = [Path(__file__), PROTOCOL, ROOT / 'CONFIG.json', ROOT / 'BASELINE_REUSE.json', ROOT / 'DATA_RECEIPT.json',
        TEACHER / 'COMPLETION.json', TEACHER / 'DIAGNOSTICS.json', TEACHER / 'CPU_INPUTS.json',
        ROOT / 'teacher/CACHE_INDEX.json', ROOT / 'teacher/VALID_IDS.json', ROOT / 'teacher/COMPLETE.json',
        PROJECT_ROOT / 'src/vlm_full_teacher_kd_v1/loss.py', PROJECT_ROOT / 'src/vlm_full_teacher_kd_v1/__init__.py',
        PROJECT_ROOT / 'tests/test_full_teacher_kd_v1.py', ROOT / 'CPU_TESTS.json',
        PROJECT_ROOT / 'tools/run_visual_preference_v1.py', PROJECT_ROOT / 'tools/run_native_stable_kl_v1.py',
        PROJECT_ROOT / 'tools/run_native_correction_sft_dpo_v2.py', PROJECT_ROOT / 'src/vlm_object_train/train.py']
    paths += [ROOT / 'data' / n for n in ('MANIFEST.json', 'CPU_INPUTS.json', 'SCHEDULE.json')]
    paths += list(ROOT.glob('baseline/*/PREDICTIONS.json'))
    paths += [Path(trained['adapter']) / name for name in ('adapter_config.json', 'adapter_model.safetensors')]
    paths += [Path(item['path']) for item in cache.values()]
    used_images = {q['image_id'] for q in data['rows']}
    used_images.update(q['image_id'] for name in INTERNAL + ('transfer_dev',) for q in data['evaluations'][name])
    paths += [Path(data['images'][image_id]['path']) for image_id in used_images]
    paths += [PROJECT_ROOT / 'src' / name for name in ('vlm_native_scope_v1/model.py', 'vlm_batch_stable_v1/model.py',
        'vlm_fp32_readout_v1/model.py', 'vlm_grpo_iteration_v1/common.py', 'vlm_literal_preference_v1/common.py',
        'research_runtime/queue.py')]
    write_json(ROOT / 'STUDY_INPUT_LOCK.json', {'status': 'frozen', 'files': {str(p): sha_file(p) for p in paths},
        'initial_lora_sha256': trained['final_lora_sha256'], 'new_updates': 96, 'new_generations': 7731,
        'roles': list(ROLES), 'teacher_auxiliary_qa': 103, 'optimized_parameters': 3686400,
        'automatic_more_training': False})
    print('FULL_TEACHER_KD_PREPARED', {'roles': ROLES, 'updates_each': 32, 'teacher_mask': 103}, flush=True)


def verify():
    lock = read(ROOT / 'STUDY_INPUT_LOCK.json')
    for path, digest in lock['files'].items():
        if sha_file(path) != digest:
            raise ValueError('Frozen KD study input changed: ' + path)
    return lock


def train(role):
    import numpy as np
    import torch
    from safetensors.torch import load_file
    from vlm_native_scope_v1.model import load
    from vlm_human_sft.evaluate import lora_hash
    from vlm_full_teacher_kd_v1.loss import objectives
    from vlm_object_train.train import learning_rate, lock_json, save_checkpoint, resume_checkpoint
    from vlm_sft.common import to_device
    from run_native_stable_kl_v1 import batch_rows
    lock = verify(); config = read(ROOT / 'CONFIG.json'); data = read(ROOT / 'data/MANIFEST.json')
    cpu = read(ROOT / 'data/CPU_INPUTS.json'); updates = read(ROOT / 'data/SCHEDULE.json')['updates']
    directory = ROOT / 'runs' / role / 'train'; directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'TRAIN_COMPLETE.json').exists():
        return
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    model, processor, binding = load(config['initial_adapter'], 'visual', True)
    if lora_hash(model) != lock['initial_lora_sha256']:
        raise ValueError('Wrong stronger SFT warm start')
    selected = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_('lora_' in name)
        if parameter.requires_grad:
            selected.append((name, parameter))
    if len(selected) != 204 or sum(p.numel() for _, p in selected) != 3686400 or any(p.dtype != torch.float32 for _, p in selected):
        raise ValueError('Joint FP32 LoRA scope differs')
    params = [p for _, p in selected]; optimizer = torch.optim.AdamW(params, lr=config['lr'], weight_decay=0.)
    index = read(ROOT / 'teacher/CACHE_INDEX.json')
    caches = {qid: torch.load(meta['path'], map_location='cpu', weights_only=True) for qid, meta in index.items()}
    valid = torch.tensor(read(ROOT / 'teacher/VALID_IDS.json'), device='cuda:0')
    eos = processor.tokenizer.convert_tokens_to_ids('<|im_end|>')
    lineage = {'study_sha256': sha_file(ROOT / 'STUDY_INPUT_LOCK.json'), 'schedule_sha256': sha_file(ROOT / 'data/SCHEDULE.json'),
        'teacher_index_sha256': sha_file(ROOT / 'teacher/CACHE_INDEX.json'), 'role': role, 'gpu_uuid': os.environ['CUDA_VISIBLE_DEVICES']}
    lineage_sha = content_sha(lineage); lock_json(directory / 'INPUT_LOCK.json', lineage)
    stats = {'questions': 0, 'auxiliary_questions': 0, 'chosen_ce_sum': 0., 'auxiliary_sum': 0., 'loss_sum': 0.,
        'active_seconds': 0., 'trace_bytes': 0, 'peak_allocated_gib': 0., 'peak_reserved_gib': 0., 'source_counts': {}}
    completed = 0
    if (directory / 'LATEST.json').exists():
        completed, stats = resume_checkpoint(directory, model, optimizer, lineage_sha, lineage['schedule_sha256'], 32)
    curve_path = directory / 'CURVE.json'; curve = [q for q in read(curve_path) if q['step'] <= completed] if curve_path.exists() else []
    trace_path = directory / 'INPUT_TRACES.jsonl'; trace_path.touch(exist_ok=True)
    if trace_path.stat().st_size < stats['trace_bytes']:
        raise ValueError('Trace shorter than checkpoint')
    prior = stats['active_seconds']; begun = time.monotonic(); durations = []; torch.cuda.reset_peak_memory_stats()
    with trace_path.open('r+b') as trace:
        trace.truncate(stats['trace_bytes']); trace.seek(0, os.SEEK_END)
        def checkpoint(step):
            stats['active_seconds'] = prior + time.monotonic() - begun; stats['trace_bytes'] = trace.tell()
            trace.flush(); os.fsync(trace.fileno()); save_checkpoint(directory, model, optimizer, step, stats, lineage_sha, lineage['schedule_sha256'])
        for step in range(completed, 32):
            start = time.monotonic(); model.train(); model.get_base_model().visual.train(); model.config.use_cache = False
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group['lr'] = learning_rate(step, 32, config)
            ce_sum = aux_sum = total_sum = 0.; aux_count = 0
            for micro, ids in enumerate(updates[step]['micros']):
                batch, frame, rows, samples = batch_rows(processor, data, ids, cpu)
                batch = to_device(batch, torch.device('cuda:0')); keep = int(batch['labels'].ne(-100).sum(1).max()) + 1
                hook = model.get_output_embeddings().register_forward_pre_hook(lambda module, args: (args[0][:, -keep:, :],))
                try:
                    output = model(**{k: v for k, v in batch.items() if k != 'labels'}, use_cache=False)
                finally:
                    hook.remove()
                active_caches = [caches.get(q['question_id']) for q in rows]
                ce, aux = objectives(output.logits, batch['labels'], valid, active_caches, role, eos)
                losses = ce + aux
                if losses.dtype != torch.float32 or not torch.isfinite(losses).all():
                    raise FloatingPointError('Nonfinite KD student objective')
                (losses.sum() / 16).backward()
                ces, extras, totals = [x.detach().cpu().tolist() for x in (ce, aux, losses)]
                for q, sample, cv, av, lv, cache in zip(rows, samples, ces, extras, totals, active_caches):
                    entry = {'update': step + 1, 'microbatch': micro, 'question_id': q['question_id'], 'image_id': q['image_id'],
                        'source': q['study_source'], 'target': q['target'], 'chosen_ce': cv, 'auxiliary': av, 'objective': lv,
                        'auxiliary_active': cache is not None, 'input_ids': sample['input_ids'][0].tolist(), 'labels': sample['labels'][0].tolist(),
                        'actual_batch_frame': frame}
                    trace.write((json.dumps(entry, ensure_ascii=False, allow_nan=False) + '\n').encode())
                    stats['source_counts'][q['study_source']] = stats['source_counts'].get(q['study_source'], 0) + 1
                    aux_count += cache is not None
                ce_sum += sum(ces); aux_sum += sum(extras); total_sum += sum(totals)
                del output, batch, ce, aux, losses, samples, active_caches
            checks = {}
            if step == 0:
                for scope in ('language', 'visual'):
                    tensors = [(n, p) for n, p in selected if ('.visual.' in n) == (scope == 'visual')]
                    if any(p.grad is None or not torch.isfinite(p.grad).all() for _, p in tensors):
                        raise ValueError('Missing or nonfinite new joint-scope gradients')
                    norm = sum(float(p.grad.float().square().sum()) for _, p in tensors)**.5
                    if norm <= 0:
                        raise ValueError('No gradient in joint training scope: ' + scope)
                    checks[scope + '_gradient_norm'] = norm
                checks.update(trainable_factors=204, optimized_parameters=3686400)
            norm = float(torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)); peak = torch.cuda.max_memory_allocated() / 2**30
            if norm <= 0 or peak > 65:
                raise ValueError('KD student gradient/memory budget')
            optimizer.step(); torch.cuda.synchronize()
            stats['questions'] += 16; stats['auxiliary_questions'] += aux_count; stats['chosen_ce_sum'] += ce_sum
            stats['auxiliary_sum'] += aux_sum; stats['loss_sum'] += total_sum; stats['active_seconds'] = prior + time.monotonic() - begun
            stats['peak_allocated_gib'] = max(stats['peak_allocated_gib'], peak)
            stats['peak_reserved_gib'] = max(stats['peak_reserved_gib'], torch.cuda.max_memory_reserved() / 2**30)
            if step == 0 or (step + 1) % 8 == 0:
                checkpoint(step + 1)
            if step == 0:
                saved = load_file(str(directory / read(directory / 'LATEST.json')['directory'] / 'adapter/adapter_model.safetensors'))
                expected = {n.replace('.default.', '.'): p.detach().cpu() for n, p in selected}
                if set(saved) != set(expected) or any(not torch.equal(saved[n], expected[n]) for n in expected):
                    raise ValueError('New joint FP32 master save differs')
                checks['saved_master_values_exact'] = True
            durations.append(time.monotonic() - start); speed = sum(durations[-8:]) / len(durations[-8:])
            progress = {'status': 'training', 'role': role, 'completed_updates': step + 1, 'planned_updates': 32,
                'mean_chosen_ce_this_update': ce_sum / 16, 'mean_auxiliary_per_active_qa': aux_sum / aux_count if aux_count else 0.,
                'mean_objective_this_update': total_sum / 16, 'gradient_norm': norm, 'recent_seconds_per_update': speed,
                'remaining_training_seconds_estimate': speed * (31 - step), **stats}
            curve.append({'step': step + 1, 'ce': ce_sum / 16, 'kd': aux_sum / aux_count if aux_count else 0., 'objective': total_sum / 16})
            write_json(curve_path, curve); write_json(directory / 'PROGRESS.json', progress)
            if step == 0:
                write_json(directory / 'FIRST_UPDATE.json', {**progress, **checks, 'binding': binding})
            if stats['active_seconds'] > 1800:
                checkpoint(step + 1); raise RuntimeError('KD student training time cap')
    if stats['questions'] != 512 or stats['auxiliary_questions'] != 206 or stats['source_counts'] != {'common': 256, 'preference': 256}:
        raise ValueError('Chosen/auxiliary exposure budget differs')
    destination = directory / 'adapter'; model.save_pretrained(destination, safe_serialization=True)
    write_json(directory / 'TRAIN_COMPLETE.json', {'status': 'completed', 'role': role, 'updates': 32,
        'adapter': str(destination), 'final_lora_sha256': lora_hash(model), 'optimized_parameters': 3686400,
        'input_trace_sha256': sha_file(trace_path), 'frozen_parameters_not_optimized': True, **stats})


def evaluate(role, phase):
    from vlm_native_scope_v1.model import load
    verify(); trained = read(ROOT / 'runs' / role / 'train/TRAIN_COMPLETE.json')
    model, processor, _ = load(trained['adapter'], 'visual', False)
    parent = importlib.import_module('run_visual_preference_v1'); parent.ROOT = ROOT
    parent.evaluate(model, processor, role, INTERNAL if phase == 'internal' else ('transfer_dev',), phase)


def analyze():
    from vlm_grpo_iteration_v1.study import paired
    data = read(ROOT / 'data/MANIFEST.json'); names = INTERNAL + ('transfer_dev',)
    records = {'start': {n: read(ROOT / 'baseline' / n / 'PREDICTIONS.json') for n in names}}
    records.update({r: {n: read(ROOT / 'runs' / r / ('transfer' if n == 'transfer_dev' else 'internal') / n / 'PREDICTIONS.json') for n in names} for r in ROLES})
    for rec in records.values():
        for part in ('singleword', 'multiword', 'rare_1_5'):
            ids = {q['question_id'] for q in data['evaluations']['seen_validation'] if (1 <= q['answer_frequency'] <= 5 if part == 'rare_1_5' else q['stratum'] == part)}
            rec[part] = [q for q in rec['seen_validation'] if q['question_id'] in ids]
        rare = {q['question_id'] for q in data['evaluations']['transfer_dev'] if 1 <= q['answer_frequency'] <= 5}
        rec['full_rare342'] = [q for q in rec['transfer_dev'] if q['question_id'] in rare]
    def norm(rows, name):
        return [{**q, 'image_id': q['filename_family'], 'vqa_score': float(q['iiit_alnum_exact'])} for q in rows] if name == 'literal_validation' else rows
    metrics = {r: {n: 100 * sum(q['vqa_score'] for q in norm(rows, n)) / len(rows) for n, rows in rec.items()} for r, rec in records.items()}
    vs_start = {r: {n: paired(norm(records['start'][n], n), norm(rows, n)) for n, rows in records[r].items()} for r in ROLES}
    vs_controls = {r: {n: paired(norm(records[r][n], n), norm(rows, n)) for n, rows in records['kd'].items()} for r in ('hard', 'uniform')}
    old = read(BASE / 'vlm_visual_preference_v1/runs/ce_dpo/transfer/transfer_dev/PREDICTIONS.json')
    vs_old = paired(old, records['kd']['transfer_dev'])
    main = [vs_start['kd']['transfer_dev'], vs_controls['hard']['transfer_dev'], vs_controls['uniform']['transfer_dev'], vs_old]
    protections = ('literal_validation', 'seen_validation', 'singleword', 'multiword', 'rare_1_5', 'retention', 'full_rare342')
    gates = {'full1617_all_four_comparisons': all(q['delta_pp'] >= .5 and q['ci95'][0] > 0 for q in main),
        'start_point_protections': all(vs_start['kd'][n]['delta_pp'] >= -.5 for n in protections),
        'matched_control_protections': all(vs_controls[r][n]['delta_pp'] >= -.5 for r in ('hard', 'uniform') for n in ('literal_validation', 'seen_validation', 'rare_1_5', 'retention', 'full_rare342'))}
    result = {'status': 'completed', 'metrics': metrics, 'vs_start': vs_start, 'candidate_vs_controls': vs_controls,
        'candidate_vs_historical_strong_dpo': vs_old, 'gates': gates, 'student_signal': all(gates.values()),
        'new_updates': 96, 'new_generations': 7731, 'new_teacher_forwards': 0, 'goal_complete': False,
        'scope': 'Single-seed historically used internal evaluation; fixed Final32 and shared joint-LoRA controls; not external or seed-stability confirmation.'}
    write_json(ROOT / 'ANALYSIS.json', result)
    write_json(ROOT / 'DECISION.json', {'status': 'completed', 'decision': 'preregister_independent_seed_and_external_confirmation' if result['student_signal'] else 'stop_this_matched_KD_configuration_no_sweep', 'automatic_training': False})
    write_json(ROOT / 'COMPLETION.json', {**result, 'phase': 'internal_student_signal' if result['student_signal'] else 'completed_negative'})
    print('FULL_TEACHER_KD_COMPLETE', result, flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true'); parser.add_argument('--worker'); args = parser.parse_args()
    if args.prepare:
        prepare(); return
    if args.worker:
        phase, role = args.worker.split(':')
        if role not in ROLES or phase not in ('train', 'internal', 'transfer'):
            raise ValueError('Unregistered student role/phase')
        train(role) if phase == 'train' else evaluate(role, phase)
        return
    verify()
    from research_runtime.queue import run_queue
    entry = str(Path(__file__).resolve()); identity = {'study_sha256': sha_file(ROOT / 'STUDY_INPUT_LOCK.json')}
    run_queue(ROOT / 'training_queue', [{'key': role, 'command': [sys.executable, '-u', entry, '--worker', 'train:' + role],
        'completion_marker': str(ROOT / 'runs' / role / 'train/TRAIN_COMPLETE.json')} for role in ROLES], identity)
    run_queue(ROOT / 'evaluation_queue', [{'key': role + '_' + phase, 'command': [sys.executable, '-u', entry, '--worker', phase + ':' + role],
        'completion_marker': str(ROOT / 'runs' / role / (phase + '_COMPLETE.json'))} for role in ROLES for phase in ('internal', 'transfer')], identity)
    analyze()


if __name__ == '__main__':
    main()
