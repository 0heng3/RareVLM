"""Reuse the frozen KD trainer with a student-prefix auxiliary objective only."""
import argparse
from datetime import datetime
import importlib
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import content_sha, sha_file, write_json
from vlm_grpo_iteration_v1.common import read

BASE = PROJECT_ROOT.parents[1] / 'outputs'
ROOT = BASE / 'vlm_onpolicy_teacher_kd_v1'
OLD = BASE / 'vlm_full_teacher_kd_v1'
ASSETS = BASE / 'vlm_full_teacher_assets_v1'
PROTOCOL = PROJECT_ROOT / 'docs/VLM_ONPOLICY_KD_V1_PROTOCOL.md'
ROLE = 'op_kd'
INTERNAL = ('native_hard', 'literal_validation', 'seen_validation', 'retention')


def prepare():
    from vlm_onpolicy_kd_v1.objective import cpu_checks
    if (ROOT / 'STUDY_INPUT_LOCK.json').exists():
        raise FileExistsError('Student-prefix study already frozen')
    if not read(BASE / 'vlm_full_teacher_transfer_v1/COMPLETION.json')['full_task_teacher_opportunity']:
        raise ValueError('Require the complete-task teacher opportunity')
    write_json(ROOT / 'CPU_CHECKS.json', cpu_checks())
    mask = read(OLD / 'teacher/CACHE_INDEX.json')
    initial = {q['question_id']: q for q in read(OLD / 'baseline/native_hard/PREDICTIONS.json')}
    gold = {}
    for line in (OLD / 'runs/kd/train/INPUT_TRACES.jsonl').read_text().splitlines():
        row = __import__('json').loads(line)
        if row['question_id'] in mask:
            gold[row['question_id']] = [t for t in row['labels'] if t not in (-100, 151645)]
    support = []
    for qid in mask:
        body = []
        for token in initial[qid]['completion_ids']:
            if token in (151643, 151645):
                break
            body.append(token)
        off = sum(body[:j] != gold[qid][:j] for j in range(len(body)))
        support.append({'question_id': qid, 'body_tokens': len(body), 'off_GT_prefix_positions': off})
    admitted = sum(q['off_GT_prefix_positions'] > 0 for q in support)
    positions = sum(q['off_GT_prefix_positions'] for q in support)
    signal = {'status': 'passed' if admitted >= 16 and positions >= 32 else 'not_admitted',
        'qa_with_distinct_prefix': admitted, 'off_GT_prefix_positions': positions, 'eligible_qa': len(support),
        'rows': support, 'new_model_calls': 0, 'scope': 'Existing strong-start training trajectories only.'}
    write_json(ROOT / 'CPU_PREFIX_SIGNAL.json', signal)
    if signal['status'] != 'passed':
        write_json(ROOT / 'DECISION.json', {'status': 'completed', 'decision': 'stop_fixed_greedy_prefix_configuration_insufficient_distinct_states', 'automatic_training': False})
        raise ValueError('Insufficient distinct student-prefix support; no GPU training admitted')
    for folder, names in [('data', ('MANIFEST.json', 'CPU_INPUTS.json', 'SCHEDULE.json')),
                          ('teacher', ('CACHE_INDEX.json', 'VALID_IDS.json', 'COMPLETE.json'))]:
        for name in names:
            path = ROOT / folder / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes((OLD / folder / name).read_bytes())
    config = {**read(OLD / 'CONFIG.json'), 'output_root': str(ROOT), 'roles': [ROLE],
        'algorithm': 'GT_CE_plus_greedy_student_prefix_forward_KD',
        'prefix_source': 'current_student_raw_greedy32_stop_gradient', 'new_updates': 32,
        'new_evaluation_generations': 2577, 'maximum_training_generations': 206,
        'automatic_more_training': False, 'teacher_live_same_gpu': True}
    write_json(ROOT / 'CONFIG.json', config)
    paths = [Path(__file__), PROTOCOL, ROOT / 'CPU_CHECKS.json', ROOT / 'CPU_PREFIX_SIGNAL.json', ROOT / 'CONFIG.json',
        OLD / 'runs/kd/train/INPUT_TRACES.jsonl',
        PROJECT_ROOT / 'src/vlm_onpolicy_kd_v1/__init__.py', PROJECT_ROOT / 'src/vlm_onpolicy_kd_v1/objective.py',
        BASE / 'vlm_full_teacher_transfer_v1/COMPLETION.json', BASE / 'vlm_kd_component_restore_v1/COMPLETION.json',
        ASSETS / 'MODEL_SOURCE.json', ASSETS / 'TRANSPORT_COMPLETE.json',
        BASE / 'vlm_full_teacher_signal_v1/PLAN.json']
    paths += list((ROOT / 'data').glob('*.json')) + list((ROOT / 'teacher').glob('*.json'))
    for name in INTERNAL + ('transfer_dev',):
        source = OLD / 'baseline' / name / 'PREDICTIONS.json'; path = ROOT / 'baseline' / name / 'PREDICTIONS.json'
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(source.read_bytes()); paths.append(path)
    old_lock = read(OLD / 'STUDY_INPUT_LOCK.json')
    files = {**old_lock['files'], **{str(p): sha_file(p) for p in paths}}
    for role in ('hard', 'uniform', 'kd'):
        for name in INTERNAL + ('transfer_dev',):
            p = OLD / 'runs' / role / ('transfer' if name == 'transfer_dev' else 'internal') / name / 'PREDICTIONS.json'
            files[str(p)] = sha_file(p)
    old_dpo = BASE / 'vlm_visual_preference_v1/runs/ce_dpo/transfer/transfer_dev/PREDICTIONS.json'
    files[str(old_dpo)] = sha_file(old_dpo)
    write_json(ROOT / 'STUDY_INPUT_LOCK.json', {'status': 'frozen',
        'registered_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), 'files': files,
        'initial_lora_sha256': old_lock['initial_lora_sha256'], 'new_updates': 32,
        'new_evaluation_generations': 2577, 'maximum_training_generations': 206,
        'optimized_parameters': 3686400, 'auxiliary_admission_qa': 103,
        'automatic_more_training': False, 'old_config_stop_unchanged': True})
    print('STUDENT_PREFIX_KD_PREPARED', {'updates': 32, 'new_eval_generations': 2577, 'cpu_checks': 4}, flush=True)


def parent_module():
    parent = importlib.import_module('run_full_teacher_kd_v1'); parent.ROOT = ROOT
    return parent


def train():
    import torch
    import vlm_native_scope_v1.model as native
    import vlm_full_teacher_kd_v1.loss as loss_module
    import run_native_stable_kl_v1 as batch_module
    from transformers import Qwen2_5_VLForConditionalGeneration
    from vlm_fp32_readout_v1.model import install_fp32_readout
    from vlm_batch_stable_v1.model import install
    from vlm_fit_positive_control_v1.study import generated
    from vlm_probe.evaluate import inference_row
    from vlm_deep.common import collate, encode
    from vlm_sft.common import to_device
    from vlm_onpolicy_kd_v1.objective import attach_body, per_qa_kl
    parent = parent_module(); parent.verify()
    directory = ROOT / 'runs' / ROLE / 'train'
    if (directory / 'TRAIN_COMPLETE.json').exists():
        return
    ctx = {'data': read(ROOT / 'data/MANIFEST.json'), 'batch': 0}
    if (directory / 'LATEST.json').exists():
        ctx['batch'] = 4 * read(directory / 'LATEST.json')['completed_updates']
    old_load = native.load; old_batch = batch_module.batch_rows; old_objectives = loss_module.objectives
    valid_set = set(read(ROOT / 'teacher/VALID_IDS.json'))
    teacher_plan = read(BASE / 'vlm_full_teacher_signal_v1/PLAN.json')
    def load_with_teacher(adapter, role, training=False):
        model, processor, binding = old_load(adapter, role, training)
        ctx.update(model=model, processor=processor)
        started = time.monotonic()
        teacher = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(ASSETS / 'model'), local_files_only=True,
            torch_dtype=torch.bfloat16, attn_implementation='sdpa', low_cpu_mem_usage=True, device_map={'': 'cuda:0'})
        teacher.eval().requires_grad_(False)
        teacher_binding = {'readout': install_fp32_readout(teacher),
            'numerics': install(SimpleNamespace(get_base_model=lambda: teacher), 'imagewise_fp32_language')}
        teacher.eval().requires_grad_(False); teacher.config.use_cache = False; ctx['teacher'] = teacher
        if any(p.requires_grad for p in teacher.parameters()):
            raise ValueError('Teacher parameters must remain frozen')
        write_json(directory / 'LIVE_TEACHER.json', {'status': 'passed', 'load_seconds': time.monotonic() - started,
            'revision': teacher_plan['teacher_revision'], 'binding': teacher_binding, 'trainable_parameters': 0})
        return model, processor, binding
    def current_batch(processor, data, ids, cpu):
        result = old_batch(processor, data, ids, cpu); ctx['rows'] = result[2]; ctx['batch'] += 1
        return result
    def tail_forward(model, batch):
        keep = int(batch['labels'].ne(-100).sum(1).max()) + 1
        hook = model.get_output_embeddings().register_forward_pre_hook(lambda module, args: (args[0][:, -keep:, :],))
        try:
            return model(**{k: v for k, v in batch.items() if k != 'labels'}, use_cache=False).logits
        finally:
            hook.remove()
    def onpolicy_objectives(logits, labels, valid_ids, caches, role, eos, temperature=2.):
        if role != ROLE:
            raise ValueError('Unexpected student-prefix role')
        ce, zero = old_objectives(logits, labels, valid_ids, [None] * len(caches), 'kd', eos, temperature)
        slots = [i for i, cache in enumerate(caches) if cache is not None]
        if not slots:
            return ce, zero
        model = ctx['model']; processor = ctx['processor']; rows = [ctx['rows'][i] for i in slots]
        model.eval(); model.config.use_cache = True
        try:
            rollouts = generated(model, processor, rows, ctx['data'])
        finally:
            model.train(); model.get_base_model().visual.train(); model.config.use_cache = False
        terminal = model.generation_config.eos_token_id
        terminal = set(terminal if isinstance(terminal, (tuple, list)) else [terminal])
        samples = []; nonempty = []; trace = []
        for index, (row, rollout) in enumerate(zip(rows, rollouts)):
            prefix = encode(processor, inference_row(row), Path(ctx['data']['images'][row['image_id']]['path']), False)
            if content_sha(prefix['input_ids'].tolist()) != row['inference_input_sha256'] or rollout['input_sha256'] != row['inference_input_sha256']:
                raise ValueError('Student-prefix prompt changed')
            item = attach_body(prefix, rollout['completion_ids'], terminal)
            body = item['labels'][item['labels'].ne(-100)].tolist() if item is not None else []
            if not set(body) <= valid_set or item is not None and item['input_ids'].shape[1] > 2048:
                raise ValueError('Generated body outside actual vocabulary or sequence budget')
            if item is not None:
                samples.append(item); nonempty.append(index)
            gold = caches[slots[index]]['labels'].tolist()
            trace.append({'question_id': row['question_id'], 'image_id': row['image_id'],
                'prefix_sha256': rollout['input_sha256'], 'completion_ids': rollout['completion_ids'],
                'prediction': rollout['prediction'], 'terminated': rollout['terminated'],
                'body_ids': body, 'gt_target': row['target'], 'gt_body_ids': gold,
                'off_GT_prefix_positions': sum(body[:j] != gold[:j] for j in range(len(body))),
                'body_tokens': len(body), 'empty_body': not body})
        auxiliary = list(zero.unbind())
        if samples:
            batch = to_device(collate(samples, processor.tokenizer.pad_token_id), torch.device('cuda:0'))
            with torch.no_grad():
                teacher_logits = tail_forward(ctx['teacher'], batch)
            student_logits = tail_forward(model, batch)
            values = per_qa_kl(student_logits, teacher_logits, batch['labels'], valid_ids, temperature)
            if not torch.isfinite(values).all():
                raise FloatingPointError('Nonfinite student-prefix KL')
            for index, value in zip(nonempty, values.unbind()):
                auxiliary[slots[index]] = value; trace[index]['auxiliary'] = float(value.detach())
            del teacher_logits, student_logits, batch
        update = (ctx['batch'] - 1) // 4 + 1; micro = (ctx['batch'] - 1) % 4
        write_json(directory / 'rollouts' / f'update_{update:06d}_micro_{micro}.json',
            {'status': 'completed', 'update': update, 'micro': micro, 'role': ROLE,
             'teacher_student_input_shared': True, 'new_generations': len(rollouts),
             'teacher_forward_sequences': len(samples), 'rollouts': trace})
        return ce, torch.stack(auxiliary)
    native.load = load_with_teacher; batch_module.batch_rows = current_batch; loss_module.objectives = onpolicy_objectives
    try:
        parent.train(ROLE)
    finally:
        native.load = old_load; batch_module.batch_rows = old_batch; loss_module.objectives = old_objectives
    records = [read(p) for p in sorted((directory / 'rollouts').glob('update_*_micro_*.json'))]
    rollouts = [q for batch in records for q in batch['rollouts']]
    if len(rollouts) != 206:
        raise ValueError('Fixed auxiliary admission exposures differ')
    write_json(directory / 'ROLLOUT_SUMMARY.json', {'status': 'completed', 'new_generations': len(rollouts),
        'teacher_forward_sequences': sum(not q['empty_body'] for q in rollouts),
        'empty_bodies': sum(q['empty_body'] for q in rollouts), 'body_tokens': sum(q['body_tokens'] for q in rollouts),
        'length_censored': sum(not q['terminated'] for q in rollouts), 'raw_ids_used': True,
        'teacher_parameters_frozen': True, 'source': 'current student at each micro, never evaluation data'})


def analyze():
    from vlm_grpo_iteration_v1.study import paired
    data = read(ROOT / 'data/MANIFEST.json'); names = INTERNAL + ('transfer_dev',)
    records = {'start': {n: read(ROOT / 'baseline' / n / 'PREDICTIONS.json') for n in names}}
    for role in ('hard', 'uniform', 'kd', ROLE):
        root = ROOT if role == ROLE else OLD
        records[role] = {n: read(root / 'runs' / role / ('transfer' if n == 'transfer_dev' else 'internal') / n / 'PREDICTIONS.json') for n in names}
    for record in records.values():
        for part in ('singleword', 'multiword', 'rare_1_5'):
            ids = {q['question_id'] for q in data['evaluations']['seen_validation'] if (1 <= q['answer_frequency'] <= 5 if part == 'rare_1_5' else q['stratum'] == part)}
            record[part] = [q for q in record['seen_validation'] if q['question_id'] in ids]
        rare = {q['question_id'] for q in data['evaluations']['transfer_dev'] if 1 <= q['answer_frequency'] <= 5}
        record['full_rare342'] = [q for q in record['transfer_dev'] if q['question_id'] in rare]
    def norm(rows, name):
        return [{**q, 'image_id': q['filename_family'], 'vqa_score': float(q['iiit_alnum_exact'])} for q in rows] if name == 'literal_validation' else rows
    metrics = {r: {n: 100 * sum(q['vqa_score'] for q in norm(rows, n)) / len(rows) for n, rows in rec.items()} for r, rec in records.items()}
    comparisons = {r: {n: paired(norm(rows, n), norm(records[ROLE][n], n)) for n, rows in records[r].items()} for r in ('start', 'hard', 'uniform', 'kd')}
    old = read(BASE / 'vlm_visual_preference_v1/runs/ce_dpo/transfer/transfer_dev/PREDICTIONS.json')
    vs_old = paired(old, records[ROLE]['transfer_dev'])
    main = [comparisons[r]['transfer_dev'] for r in comparisons] + [vs_old]
    protections = ('literal_validation', 'seen_validation', 'singleword', 'multiword', 'rare_1_5', 'retention', 'full_rare342')
    gates = {'full1617_all_five_comparisons': all(q['delta_pp'] >= .5 and q['ci95'][0] > 0 for q in main),
        'start_point_protections': all(comparisons['start'][n]['delta_pp'] >= -.5 for n in protections),
        'control_point_protections': all(comparisons[r][n]['delta_pp'] >= -.5 for r in ('hard', 'uniform', 'kd') for n in ('literal_validation', 'seen_validation', 'rare_1_5', 'retention', 'full_rare342'))}
    result = {'status': 'completed', 'metrics': metrics, 'candidate_vs_controls': comparisons,
        'candidate_vs_historical_strong_dpo': vs_old, 'gates': gates, 'student_signal': all(gates.values()),
        'new_updates': 32, 'new_evaluation_generations': 2577,
        'rollouts': read(ROOT / 'runs' / ROLE / 'train/ROLLOUT_SUMMARY.json'), 'goal_complete': False,
        'scope': 'Deployment-greedy GKD adaptation, fixed seed42 Final32 on reused internal cohorts; extra forward cost differs; no external/seed confirmation.'}
    write_json(ROOT / 'ANALYSIS.json', result)
    write_json(ROOT / 'DECISION.json', {'status': 'completed', 'decision': 'preregister_seed_and_external_confirmation' if result['student_signal'] else 'stop_this_fixed_student_prefix_configuration_no_sweep', 'automatic_training': False})
    write_json(ROOT / 'COMPLETION.json', {**result, 'phase': 'internal_student_signal' if result['student_signal'] else 'completed_negative'})
    print('STUDENT_PREFIX_KD_COMPLETE', result, flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true'); parser.add_argument('--worker', choices=('train', 'internal', 'transfer')); args = parser.parse_args()
    if args.prepare:
        prepare(); return
    parent = parent_module(); parent.verify()
    if args.worker:
        train() if args.worker == 'train' else parent.evaluate(ROLE, args.worker)
        return
    from research_runtime.queue import run_queue
    entry = str(Path(__file__).resolve()); identity = {'study_sha256': sha_file(ROOT / 'STUDY_INPUT_LOCK.json')}
    run_queue(ROOT / 'training_queue', [{'key': ROLE, 'command': [sys.executable, '-u', entry, '--worker', 'train'],
        'completion_marker': str(ROOT / 'runs' / ROLE / 'train/TRAIN_COMPLETE.json')}], identity)
    run_queue(ROOT / 'evaluation_queue', [{'key': phase, 'command': [sys.executable, '-u', entry, '--worker', phase],
        'completion_marker': str(ROOT / 'runs' / ROLE / (phase + '_COMPLETE.json'))} for phase in ('internal', 'transfer')], identity)
    analyze()


if __name__ == '__main__': main()
