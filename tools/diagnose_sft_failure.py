"""Bounded code diagnosis: no optimizer, no training or benchmark selection."""
import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
import sys
import time

from rarevlm_paths import PROJECT_ROOT

OUT = Path('/opt/data/private/longtail_vlm/rarevlm_pilot/outputs/vlm_failure_code_review_v1')
CONFIG = PROJECT_ROOT / 'configs/vlm_sft_resolution_v1.json'


def read(p):
    return json.loads(Path(p).read_text())


def setup():
    import torch
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    c = read(CONFIG)
    m = read(c['manifest'])
    return torch, c, m


def weights(c, name):
    from safetensors.torch import load_file
    p = Path(c['initial_adapter']) if name == 'initial' else Path(c['output_root']) / name / 'adapter'
    return load_file(str(p / 'adapter_model.safetensors'))


def cpu():
    torch, c, m = setup()
    from vlm_deep.common import content_sha, write_json
    OUT.mkdir(parents=True, exist_ok=True)
    initial = {k: v.to(torch.bfloat16).float() for k, v in weights(c, 'initial').items()}
    records = {}
    for name in ['low256', 'high1024']:
        current = weights(c, name)
        update2 = rounding2 = after2 = 0.
        changed_fp32 = changed_bf16 = total = 0
        effective2 = rounded_effective2 = effective_error2 = 0.
        for k, x in current.items():
            x = x.float(); y = initial[k]; r = x.to(torch.bfloat16).float()
            update2 += float((x.double() - y.double()).square().sum())
            rounding2 += float((r.double() - x.double()).square().sum())
            after2 += float((r.double() - y.double()).square().sum())
            changed_fp32 += int(x.ne(y).sum()); changed_bf16 += int(r.ne(y).sum()); total += x.numel()
        def dense_norm(terms):
            # ||sum sign*B*A||_F using small rank Gram matrices, no dense W.
            b = torch.cat([sign * bb.double() for sign, bb, aa in terms], dim=1)
            a = torch.cat([aa.double() for sign, bb, aa in terms], dim=0)
            return float(((b.T @ b) * (a @ a.T)).sum().clamp_min(0))
        for k, a in current.items():
            if 'lora_A' not in k: continue
            bk = k.replace('lora_A', 'lora_B')
            b = current[bk]; ai = initial[k]; bi = initial[bk]
            ar = a.to(torch.bfloat16).float(); br = b.to(torch.bfloat16).float()
            effective2 += 4 * dense_norm([(1, b, a), (-1, bi, ai)])
            rounded_effective2 += 4 * dense_norm([(1, br, ar), (-1, bi, ai)])
            effective_error2 += 4 * dense_norm([(1, b, a), (-1, br, ar)])
        p = Path(c['output_root']) / name
        latest = read(p / 'LATEST.json')
        state = torch.load(p / latest['directory'] / 'state.pt', map_location='cpu', weights_only=False)
        optimizer_steps = Counter(int(s['step']) for s in state['optimizer']['state'].values())
        records[name] = {'parameters': total, 'changed_fp32': changed_fp32, 'changed_after_bf16': changed_bf16,
            'parameter_update_l2': update2 ** .5, 'eval_rounding_error_l2': rounding2 ** .5,
            'rounded_parameter_update_l2': after2 ** .5, 'effective_dense_update_l2': effective2 ** .5,
            'rounded_effective_update_l2': rounded_effective2 ** .5,
            'effective_rounding_error_vs_update': (effective_error2 / effective2) ** .5,
            'optimizer_steps': dict(optimizer_steps), 'final_learning_rate': state['optimizer']['param_groups'][0]['lr']}
        del state
    baseline = [json.loads(s) for s in Path(c['initial_predictions']).read_text().splitlines()]
    final = [json.loads(s) for s in (Path(c['output_root']) / 'high1024/dev/predictions.jsonl').read_text().splitlines()]
    changed_groups = {i // 4 for i, (a, b) in enumerate(zip(baseline, final)) if a['prediction'] != b['prediction']}
    groups = list(range((len(final) + 3) // 4))
    extra = sorted(set(groups) - changed_groups, key=lambda i: content_sha(['precision_unmodified_groups', i]))[:8]
    batches = sorted(changed_groups | set(extra))
    cohorts = {}
    for split in ['train_all', 'dev']:
        selected = sorted(range(len(m['splits'][split])), key=lambda i: content_sha(['fixed_nll_20261017', split, m['splits'][split][i]['question_id']]))[:128]
        cohorts[split] = selected
    plan = {'status': 'prepared', 'precision_original_batch_indices': batches, 'changed_original_batches': len(changed_groups),
            'unchanged_control_batches': len(extra), 'fixed_nll_indices': cohorts,
            'selection': 'NLL uses input-ID hash only; precision diagnostic includes ALL changed-answer original batches and8 hash-chosen unchanged batches; not independent performance evaluation',
            'new_optimizer_updates': 0, 'data_mutation': False}
    write_json(OUT / 'CPU_WEIGHTS.json', {'status': 'completed', 'conditions': records})
    write_json(OUT / 'PLAN.json', plan)
    print(json.dumps({'weights': records, 'precision_batches': len(batches), 'nll_qa_per_split': 128}), flush=True)


def assign(model, state, mode):
    import torch
    found = set()
    for name, p in model.named_parameters():
        if 'lora_' not in name: continue
        key = name.replace('.default.', '.')
        value = state[key]
        if mode == 'bf16': value = value.to(torch.bfloat16)
        elif mode == 'rounded_fp32': value = value.to(torch.bfloat16).float()
        elif mode == 'master_fp32': value = value.float()
        else: raise ValueError(mode)
        p.data = value.to(p.device).clone()
        found.add(key)
    if found != set(state): raise ValueError('LoRA keys do not match actual model')


def load(c):
    from vlm_precision.evaluate import load_path
    from vlm_probe.evaluate import processor_for
    import torch
    ref = read(c['reference_config'])
    model, _ = load_path(Path(c['model_root']), ref, torch.device('cuda:0'), 'original_bf16', Path(c['initial_adapter']))
    processor = processor_for(Path(c['model_root']), ref, 1024)
    return model, processor


def fit():
    torch, c, m = setup()
    import torch.nn.functional as F
    from vlm_deep.common import answer_loss, collate, encode, to_device, write_json
    plan = read(OUT / 'PLAN.json')
    model, processor = load(c)
    root = Path(c['manifest']).parent
    result = {'status': 'running', 'fixed_sample_loss': {}, 'optimizer_updates': 0}
    # Check slicing and causal shift with the real model, including gradients.
    assign(model, weights(c, 'initial'), 'rounded_fp32')
    params = [p for n, p in model.named_parameters() if 'lora_' in n]
    for p in params: p.requires_grad_(True)
    examples = [m['splits']['train_all'][i] for i in plan['fixed_nll_indices']['train_all'][:2]]
    batch = to_device(collate([encode(processor, r, root / m['images'][r['image_id']]['path'], True) for r in examples], processor.tokenizer.pad_token_id), torch.device('cuda:0'))
    losses, gradients = [], []
    for tail in [True, False]:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            values, _, _ = answer_loss(model, batch, tail=tail)
            loss = values.mean()
        gradients.append([g.detach().float().cpu() for g in torch.autograd.grad(loss, params)])
        losses.append(values.detach().float().cpu().tolist())
    grad_diff = sum(float((a.double()-b.double()).square().sum()) for a,b in zip(*gradients)) ** .5
    grad_size = sum(float(b.double().square().sum()) for b in gradients[1]) ** .5
    result['real_tail_vs_full'] = {'tail_per_question_ce': losses[0], 'full_per_question_ce': losses[1],
        'gradient_relative_l2_difference': grad_diff / grad_size,
        'max_loss_difference': max(abs(a-b) for a,b in zip(*losses))}
    del batch, gradients
    for p in params: p.requires_grad_(False)
    for name in ['initial', 'low256', 'high1024']:
        assign(model, weights(c, name), 'rounded_fp32' if name == 'initial' else 'master_fp32')
        model.eval()
        current = {}
        for split, indices in plan['fixed_nll_indices'].items():
            rows = m['splits'][split]
            records = []
            for begin in range(0, len(indices), 4):
                selected = [rows[i] for i in indices[begin:begin + 4]]
                batch = to_device(collate([encode(processor, r, root / m['images'][r['image_id']]['path'], True) for r in selected], processor.tokenizer.pad_token_id), torch.device('cuda:0'))
                # Match training AMP; force CE to FP32 under that context.
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    values, _, _ = answer_loss(model, batch, tail=True)
                records.extend({'question_id': r['question_id'], 'image_id': r['image_id'], 'target': r['target'], 'ce': float(v)} for r,v in zip(selected,values))
                del batch
            current[split] = {'qa': len(records), 'mean_per_question_ce': sum(v['ce'] for v in records) / len(records), 'records': records}
            print(name, split, current[split]['mean_per_question_ce'], flush=True)
        result['fixed_sample_loss'][name] = current
        write_json(OUT / 'FIT.json', result)
    result['status'] = 'completed'
    write_json(OUT / 'FIT.json', result)


def precision():
    torch, c, m = setup()
    from vlm_deep.common import consensus, write_json
    from vlm_probe.evaluate import evaluate_batch
    plan = read(OUT / 'PLAN.json')
    model, processor = load(c)
    dev = m['splits']['dev']; root = Path(c['manifest']).parent
    token_id = processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
    saved = [json.loads(s) for s in (Path(c['output_root']) / 'high1024/dev/predictions.jsonl').read_text().splitlines()]
    state = weights(c, 'high1024')
    variants = {}
    for mode in ['bf16', 'master_fp32', 'rounded_fp32']:
        assign(model, state, mode)
        records = []
        with torch.inference_mode():
            for group in plan['precision_original_batch_indices']:
                start = group * 4; chunk = dev[start:start + 4]
                answers, _ = evaluate_batch(model, processor, chunk, m['images'], root, torch.device('cuda:0'), 1024, token_id, 32)
                for row, answer, old in zip(chunk, answers, saved[start:start + 4]):
                    records.append({'question_id': row['question_id'], 'image_id': row['image_id'],
                        'prediction': answer['prediction'], 'score': consensus(answer['prediction'], row['answers']),
                        'old_prediction': old['prediction'], 'old_score': old['score']})
        variants[mode] = records
        write_json(OUT / ('PRECISION_' + mode + '.json'), {'status': 'completed', 'records': records})
        print(mode, len(records), 'changes_vs_saved', sum(r['prediction'] != r['old_prediction'] for r in records), flush=True)
    baseline = variants['bf16']
    result = {'status': 'completed', 'qa': len(baseline), 'comparison': {}, 'optimizer_updates': 0,
        'scope': plan['selection']}
    for mode, records in variants.items():
        result['comparison'][mode] = {'raw_changes_vs_bf16': sum(a['prediction'] != b['prediction'] for a,b in zip(baseline,records)),
            'sum_score_change_vs_bf16': sum(b['score'] - a['score'] for a,b in zip(baseline,records)),
            'score_up_vs_bf16': sum(b['score'] > a['score'] for a,b in zip(baseline,records)),
            'score_down_vs_bf16': sum(b['score'] < a['score'] for a,b in zip(baseline,records)),
            'raw_changes_vs_saved_original': sum(r['prediction'] != r['old_prediction'] for r in records)}
    write_json(OUT / 'PRECISION.json', result)


def queue():
    from research_runtime.queue import run_queue
    from vlm_deep.common import sha_file
    jobs = [{'key': name, 'command': [sys.executable, '-u', str(Path(__file__).resolve()), name],
             'completion_marker': str(OUT / (name.upper() + '.json'))} for name in ['fit', 'precision']]
    run_queue(OUT / 'queue', jobs, {'diagnostic_source_sha256': sha_file(Path(__file__)), 'plan_sha256': sha_file(OUT/'PLAN.json'), 'optimizer_updates': 0})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['cpu', 'fit', 'precision', 'queue'])
    args = p.parse_args()
    globals()[args.mode]()
