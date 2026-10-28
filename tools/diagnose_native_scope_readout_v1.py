"""Fixed native-QA answer likelihoods; no optimizer and no new generation."""
import argparse
import statistics
import sys
import time
from pathlib import Path

from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import collate, content_sha, encode, sha_file, write_json
from vlm_grpo_iteration_v1.common import read
from vlm_native_scope_v1.model import ROOT as PARENT, load

ROOT = PARENT.parent / 'vlm_native_scope_readout_v1'
PROTOCOL = PROJECT_ROOT / 'docs/VLM_NATIVE_SCOPE_READOUT_V1_PROTOCOL.md'
ROLES = ('start', 'visual')


def prepare():
    if (ROOT / 'PLAN.json').exists():
        raise FileExistsError('Diagnostic already frozen')
    data = read(PARENT / 'data/MANIFEST.json')
    rows = {q['question_id']: q for q in data['rows'] if q['study_source'] == 'preference'}
    predictions = read(PARENT / 'runs/visual/internal/native_hard/PREDICTIONS.json')
    if len(rows) != 128 or {q['question_id'] for q in predictions} != set(rows):
        raise ValueError('Require all original 128 training corrections')
    if any(not q['initial_prediction']['terminated'] for q in rows.values()) or any(not q['terminated'] for q in predictions):
        raise ValueError('Require observed complete answers, including EOS')
    groups = []
    seen = set()
    for update in read(PARENT / 'data/SCHEDULE.json')['updates']:
        for ids in update['micros']:
            if all(q in rows for q in ids) and tuple(ids) not in seen:
                seen.add(tuple(ids)); groups.append(ids)
    if len(groups) != 32 or sorted(q for g in groups for q in g) != sorted(rows):
        raise ValueError('Hard examples must occur in exactly 32 unique groups')
    paths = [Path(__file__), PROTOCOL, ROOT / 'CPU_READOUT_CHECK.json', PARENT / 'STUDY_INPUT_LOCK.json', PARENT / 'COMPLETION.json',
             PARENT / 'CONFIG.json', PARENT / 'data/MANIFEST.json', PARENT / 'data/CPU_INPUTS.json',
             PARENT / 'data/SCHEDULE.json', PARENT / 'runs/visual/train/TRAIN_COMPLETE.json',
             PARENT / 'baseline/native_hard/PREDICTIONS.json',
             PARENT / 'runs/visual/internal/native_hard/PREDICTIONS.json']
    for directory in [PARENT / 'initial_adapter', PARENT / 'runs/visual/train/adapter']:
        paths += [directory / 'adapter_config.json', directory / 'adapter_model.safetensors']
    paths += [PROJECT_ROOT / 'src' / name for name in ['vlm_native_scope_v1/model.py',
              'vlm_batch_stable_v1/model.py', 'vlm_literal_preference_v1/common.py',
              'vlm_sampled_signal_v1/trajectory.py']]
    write_json(ROOT / 'PLAN.json', {'status': 'frozen', 'files': {str(p): sha_file(p) for p in paths},
               'groups': groups, 'roles': list(ROLES), 'optimizer_updates': 0, 'new_generations': 0,
               'max_forward_sequences': 768, 'cap_seconds_per_role': 900, 'peak_allocated_cap_gib': 65})


def verify():
    plan = read(ROOT / 'PLAN.json')
    for path, digest in plan['files'].items():
        if sha_file(path) != digest:
            raise ValueError('Diagnostic input changed: ' + path)
    return plan


def likelihoods(model, batch, tokenizer):
    import torch
    from vlm_sampled_signal_v1.trajectory import readout_tail_length
    labels = batch['labels']; keep = readout_tail_length(labels)
    hook = model.get_output_embeddings().register_forward_pre_hook(lambda module, args: (args[0][:, -keep:, :],))
    try:
        logits = model(**{k: v for k, v in batch.items() if k != 'labels'}, use_cache=False).logits[:, :-1].float()
    finally:
        hook.remove()
    targets = labels[:, -logits.shape[1] - 1:][:, 1:]; valid = targets.ne(-100)
    if not torch.equal(valid.sum(1), labels.ne(-100).sum(1)):
        raise ValueError('Lost assistant actions')
    safe = targets.masked_fill(~valid, 0)
    logp = logits.log_softmax(-1).gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    top2, ids2 = logits.topk(2, dim=-1)
    gt = logits.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    competitor = torch.where(ids2[..., 0].eq(safe), top2[..., 1], top2[..., 0])
    margin = gt - competitor; result = []
    for i in range(len(labels)):
        token_ids = targets[i][valid[i]].tolist(); lp = logp[i][valid[i]].tolist()
        margins = margin[i][valid[i]].tolist(); top = ids2[i, :, 0][valid[i]].tolist()
        if not all(torch.isfinite(torch.tensor(lp))) or not token_ids:
            raise ValueError('Nonfinite or empty readout')
        wrong = [j for j, (a, b) in enumerate(zip(token_ids, top)) if a != b]
        result.append({'sum_logp': sum(lp), 'mean_logp': statistics.mean(lp), 'count': len(lp),
                       'body_nll': -statistics.mean(lp[:-1]) if len(lp) > 1 else None,
                       'last_action_nll': -lp[-1], 'all_gt_prefix_top1': not wrong,
                       'first_non_top1': wrong[0] if wrong else None,
                       'tokens': [{'id': a, 'token': tokenizer.convert_ids_to_tokens(a), 'logp': b,
                                   'top1_id': c, 'margin': d} for a, b, c, d in zip(token_ids, lp, top, margins)]})
    return result


def worker(role):
    import torch
    from vlm_human_sft.evaluate import lora_hash
    from vlm_literal_preference_v1.common import sample
    from vlm_probe.evaluate import inference_row
    from vlm_sampled_signal_v1.trajectory import attach_completion
    from vlm_sft.common import to_device
    plan = verify(); data = read(PARENT / 'data/MANIFEST.json'); cpu = read(PARENT / 'data/CPU_INPUTS.json')
    rows = {q['question_id']: q for q in data['rows']}
    final = {q['question_id']: q for q in read(PARENT / 'runs/visual/internal/native_hard/PREDICTIONS.json')}
    adapter = str(PARENT / ('initial_adapter' if role == 'start' else 'runs/visual/train/adapter'))
    model, processor, binding = load(adapter, 'visual', False); model.eval(); model.requires_grad_(False)
    before = lora_hash(model); directory = ROOT / role; done = read(directory / 'PARTIAL.json') if (directory / 'PARTIAL.json').exists() else []
    actual = 0; started = time.monotonic(); torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for index, group in enumerate(plan['groups']):
            if index < len(done):
                continue
            chunk = [rows[q] for q in group]; batches = {}
            for kind in ('chosen', 'old_negative', 'final_answer'):
                samples = []
                for q in chunk:
                    if kind == 'final_answer':
                        prefix = encode(processor, inference_row(q), Path(data['images'][q['image_id']]['path']), False)
                        if content_sha(prefix['input_ids'].tolist()) != q['inference_input_sha256']:
                            raise ValueError('Final answer prefix differs')
                        item = attach_completion(prefix, final[q['question_id']]['completion_ids'])
                    else:
                        item = sample(processor, data, q, rejected=kind == 'old_negative')
                        expected = cpu[q['question_id'] + (':chosen' if kind == 'chosen' else ':rejected')]
                        if any(content_sha(item[k].tolist()) != expected[k + '_sha256'] for k in ('input_ids', 'labels')):
                            raise ValueError('Frozen training actions differ')
                    samples.append(item)
                batch = collate(samples, processor.tokenizer.pad_token_id)
                batches[kind] = likelihoods(model, to_device(batch, next(model.parameters()).device), processor.tokenizer)
                actual += len(chunk)
                del samples, batch
            done.append([{'question_id': q['question_id'], 'image_id': q['image_id'],
                          'target': q['target'], 'old_answer': q['initial_prediction']['prediction'],
                          'final_answer': final[q['question_id']]['prediction'], 'final_vqa_score': final[q['question_id']]['vqa_score'],
                          **{kind: batches[kind][i] for kind in batches}} for i, q in enumerate(chunk)])
            write_json(directory / 'PARTIAL.json', done)
            elapsed = time.monotonic() - started; peak = torch.cuda.max_memory_allocated() / 2**30
            write_json(directory / 'PROGRESS.json', {'status': 'readout_diagnostic', 'role': role,
                       'completed_groups': len(done), 'planned_groups': 32, 'new_forward_sequences_this_process': actual,
                       'optimizer_updates': 0, 'new_generations': 0, 'active_seconds': elapsed, 'peak_allocated_gib': peak})
            if elapsed > 900 or peak > 65:
                raise RuntimeError('Readout diagnostic budget exceeded')
    if lora_hash(model) != before or any(p.grad is not None for p in model.parameters()):
        raise ValueError('Diagnostic changed model or accumulated gradients')
    write_json(directory / 'COMPLETE.json', {'status': 'completed', 'role': role, 'rows': [q for group in done for q in group],
               'optimizer_updates': 0, 'new_generations': 0, 'total_forward_sequences': 384,
               'model_hash_unchanged': True, 'lora_sha256': before, 'binding': binding,
               'active_seconds_this_process': time.monotonic() - started,
               'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30})


def analyze():
    summaries = {}
    for role in ROLES:
        rec = read(ROOT / role / 'COMPLETE.json'); rows = rec['rows']
        residual = [q for q in rows if q['final_vqa_score'] < 1 - 1e-9]
        summaries[role] = {'qa': len(rows), 'visual_final_residual_subset': len(residual),
                          'chosen_mean_nll': -statistics.mean(q['chosen']['mean_logp'] for q in rows),
                          'chosen_all_prefix_top1': sum(q['chosen']['all_gt_prefix_top1'] for q in rows),
                          'residual_chosen_all_prefix_top1': sum(q['chosen']['all_gt_prefix_top1'] for q in residual),
                          'residual_chosen_gt_old_negative_sum': sum(q['chosen']['sum_logp'] > q['old_negative']['sum_logp'] for q in residual),
                          'residual_chosen_gt_current_wrong_sum': sum(q['chosen']['sum_logp'] > q['final_answer']['sum_logp'] for q in residual),
                          'residual_chosen_gt_old_negative_mean': sum(q['chosen']['mean_logp'] > q['old_negative']['mean_logp'] for q in residual),
                          'residual_chosen_gt_current_wrong_mean': sum(q['chosen']['mean_logp'] > q['final_answer']['mean_logp'] for q in residual),
                          'peak_allocated_gib': rec['peak_allocated_gib'], 'active_seconds': rec['active_seconds_this_process']}
    write_json(ROOT / 'COMPLETION.json', {'status': 'completed', 'optimizer_updates': 0, 'new_generations': 0,
               'summaries': summaries, 'goal_complete': False,
               'scope': 'Posthoc train-only fixed-checkpoint readout; no capability gain or unique causal attribution.'})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true'); parser.add_argument('--worker', choices=ROLES)
    args = parser.parse_args()
    if args.prepare:
        prepare(); return
    if args.worker:
        worker(args.worker); return
    from research_runtime.queue import run_queue
    verify()
    run_queue(ROOT / 'queue', [{'key': r, 'command': [sys.executable, '-u', str(Path(__file__).resolve()), '--worker', r],
               'completion_marker': str(ROOT / r / 'COMPLETE.json')} for r in ROLES], {'plan_sha256': sha_file(ROOT / 'PLAN.json')})
    analyze()


if __name__ == '__main__':
    main()
