"""Six training QA, fixed initial/SFT/DPO readouts and initial local gradients; zero updates."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT
from vlm_deep.common import encode, collate, sha_file, write_json

ROOT = PROJECT_ROOT.parents[1] / 'outputs/vlm_region_decision_diagnosis_v1'
PARENT = ROOT.parent / 'vlm_region_preference_probe_v1'
QIDS = tuple(f'region-pref:{pair}:{side}' for pair in (10, 14, 9) for side in ('a', 'b'))


def read(path):
    return json.loads(Path(path).read_text())


def prepare():
    from vlm_probe.evaluate import processor_for
    c = read(PROJECT_ROOT / 'configs/vlm_region_preference_probe_v1.json')
    base = read(c['baseline_config'])
    data = read(PARENT / 'data/MANIFEST.json')
    predictions = {q['question_id']: q for q in read(PARENT / 'baseline/hard/PREDICTIONS.json')}
    rows = {q['question_id']: q for q in data['hard_eval']}
    processor = processor_for(Path(base['model_root']), read(base['reference_config']), 1024)
    selected = []
    for qid in QIDS:
        q = dict(rows[qid])
        initial = predictions[qid]
        actual_wrong = not initial['casefold_exact']
        q['contrast'] = initial['prediction'] if actual_wrong else q['rejected_target']
        q['contrast_source'] = 'actual_initial_wrong_text' if actual_wrong else 'registered_negative_for_initially_correct_QA'
        q['initial_prediction'] = initial['prediction']
        pair = [encode(processor, {**q, 'target': text}, Path(data['images'][q['image_id']]['path']), True)
                for text in (q['target'], q['contrast'])]
        ids = [s['labels'][0][s['labels'][0] != -100].tolist() for s in pair]
        divergence = next((i for i, (a, b) in enumerate(zip(*ids)) if a != b), None)
        if divergence is None:
            raise ValueError('No differing supervised token: ' + qid)
        starts = [int((s['labels'][0] != -100).nonzero()[0]) for s in pair]
        if not pair[0]['input_ids'][0, :starts[0] + divergence].equal(pair[1]['input_ids'][0, :starts[1] + divergence]):
            raise ValueError('Chosen/contrast prefixes differ before diagnostic token')
        q.update(divergence_index=divergence, correct_token=ids[0][divergence], contrast_token=ids[1][divergence],
                 supervised_tokens=ids, inputs=[{k: s[k].tolist() for k in ('input_ids', 'labels', 'image_grid_thw')} for s in pair])
        selected.append(q)
    adapters = {'initial': c['initial_adapter'], **{role: str(PARENT / 'runs' / role / 'train/adapter') for role in ('sft', 'dpo_sft')}}
    plan = {'rows': selected, 'images': {q['image_id']: data['images'][q['image_id']] for q in selected},
            'adapters': adapters, 'model_root': base['model_root'], 'reference_config': base['reference_config'],
            'scope': 'Training-only selected mechanism diagnosis; zero optimizer updates; no main benchmark.'}
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / 'INPUT_LOCK.json').exists():
        raise FileExistsError('Already prepared; use existing controller/results')
    write_json(ROOT / 'PLAN.json', plan)
    paths = [Path(__file__), ROOT / 'PLAN.json', PROJECT_ROOT / 'docs/VLM_REGION_DECISION_DIAGNOSIS_V1_PROTOCOL.md',
             PARENT / 'COMPLETION.json', PARENT / 'data/MANIFEST.json', PARENT / 'baseline/hard/PREDICTIONS.json',
             Path(c['baseline_config']), Path(base['reference_config'])]
    paths += [Path(a) / name for a in adapters.values() for name in ('adapter_model.safetensors', 'adapter_config.json')]
    paths += [SOURCE_ROOT / name for name in ('vlm_deep/common.py', 'vlm_alignment_signal/loss.py',
              'vlm_fp32_readout_v1/model.py', 'vlm_runtime_metadata_v1/model.py', 'vlm_fp32_adapter/model.py', 'research_runtime/queue.py')]
    write_json(ROOT / 'INPUT_LOCK.json', {'files': {str(p): sha_file(p) for p in paths},
              'images': {im['path']: im['sha256'] for im in plan['images'].values()},
              'note': 'Only new diagnosis dependencies and selected images; old completed audits reused.'})
    print(json.dumps({'status': 'prepared', 'qa': len(selected), 'real_cpu_encodings': len(selected) * 2, 'optimizer_updates': 0}))


def worker(role):
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file
    from vlm_alignment_signal.loss import completion_logps, dpo_loss
    from vlm_fp32_readout_v1.model import load
    from vlm_human_sft.evaluate import lora_hash
    from vlm_probe.evaluate import processor_for
    from vlm_sft.common import to_device
    lock = read(ROOT / 'INPUT_LOCK.json')
    for path, digest in {**lock['files'], **lock['images']}.items():
        if sha_file(path) != digest:
            raise ValueError('New diagnostic input changed: ' + path)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(42)
    plan = read(ROOT / 'PLAN.json')
    begun = time.monotonic()
    model, binding = load(plan['model_root'], plan['adapters'][role], torch.device('cuda:0'), training=True, dropout=0.)
    model.train()
    model.get_base_model().visual.eval()
    before = lora_hash(model)
    processor = processor_for(Path(plan['model_root']), read(plan['reference_config']), 1024)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    deltas = {}
    if role == 'initial':
        for candidate in ('sft', 'dpo_sft'):
            state = load_file(str(Path(plan['adapters'][candidate]) / 'adapter_model.safetensors'))
            deltas[candidate] = torch.cat([(state[n.replace('.default.', '.')] - p.detach().cpu()).reshape(-1) for n, p in named])
    results = []
    for q in plan['rows']:
        samples = [encode(processor, {**q, 'target': text}, Path(plan['images'][q['image_id']]['path']), True)
                   for text in (q['target'], q['contrast'])]
        for sample, expected in zip(samples, q['inputs']):
            if any(sample[k].tolist() != expected[k] for k in expected):
                raise ValueError('Actual diagnostic encoding differs from CPU inputs')
        batch = to_device(collate(samples, processor.tokenizer.pad_token_id), torch.device('cuda:0'))
        labels = batch['labels']
        keep = int((labels != -100).sum(1).max()) + 1
        visual = []
        def capture(module, args, output):
            leaf = output.detach().requires_grad_(True)
            visual.append(leaf)
            return leaf
        vh = model.get_base_model().visual.register_forward_hook(capture) if role == 'initial' else None
        hh = model.get_output_embeddings().register_forward_pre_hook(lambda module, args: (args[0][:, -keep:, :],))
        try:
            with torch.set_grad_enabled(role == 'initial'), torch.autocast('cuda', enabled=False):
                output = model(**{k: v for k, v in batch.items() if k != 'labels'}, use_cache=False)
                logp, counts = completion_logps(output.logits, labels)
                pos = int((labels[0] != -100).nonzero()[0]) + q['divergence_index'] - 1 - (labels.shape[1] - output.logits.shape[1])
                logits = output.logits[0, pos].float()
                margin = logits[q['correct_token']] - logits[q['contrast_token']]
                ce = -logp[0] / counts[0]
                decision_ce = -F.log_softmax(logits, -1)[q['correct_token']]
                if role == 'initial':
                    ref = logp.detach()
                else:
                    ref = torch.tensor(read(ROOT / 'initial/COMPLETE.json')['references'][q['question_id']], device=logp.device)
                dpo = dpo_loss(logp[0:1], logp[1:2], ref[0:1], ref[1:2], .1).mean()
                top = logits.topk(5)
                rec = {'qid': q['question_id'], 'target': q['target'], 'contrast': q['contrast'], 'contrast_source': q['contrast_source'],
                       'divergence_index': q['divergence_index'], 'correct_token_id': q['correct_token'], 'contrast_token_id': q['contrast_token'],
                       'correct_token_text': processor.tokenizer.decode([q['correct_token']]), 'contrast_token_text': processor.tokenizer.decode([q['contrast_token']]),
                       'correct_minus_contrast_logit': float(margin.detach()), 'correct_token_rank': int((logits > logits[q['correct_token']]).sum()) + 1,
                       'chosen_ce': float(ce.detach()), 'decision_ce': float(decision_ce.detach()), 'dpo': float(dpo.detach()),
                       'sequence_logps': logp.detach().cpu().tolist(), 'assistant_token_count': int(counts[0]),
                       'top5_token_text': [processor.tokenizer.decode([int(i)]) for i in top.indices],
                       'top5_logits': top.values.detach().cpu().tolist()}
                if role == 'initial':
                    grads = {}
                    for name, loss in (('margin', margin), ('full_ce', ce), ('decision_ce', decision_ce), ('dpo', dpo)):
                        g = torch.autograd.grad(loss, params + visual, retain_graph=True, allow_unused=True)
                        grads[name] = torch.cat([(torch.zeros_like(p) if v is None else v).detach().float().cpu().reshape(-1) for p, v in zip(params, g)])
                        rec.setdefault('gradient_norms', {})[name] = float(grads[name].norm())
                        rec.setdefault('visual_output_gradient_norms', {})[name] = float(g[-1].float().norm()) if g[-1] is not None else 0.
                    def cosine(a, b):
                        return float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-30))
                    rec['full_ce_vs_decision_ce_gradient_cosine'] = cosine(grads['full_ce'], grads['decision_ce'])
                    rec['dpo_vs_decision_ce_gradient_cosine'] = cosine(grads['dpo'], grads['decision_ce'])
                    rec['first_order_margin_from_actual_final_delta'] = {k: float(torch.dot(grads['margin'], delta)) for k, delta in deltas.items()}
                    rec['gradient_scope_note'] = 'q/v local initial gradients; visual OUTPUT sensitivity, not merger parameter gradients/attention. Delta projection is first-order, not actual Adam trajectory.'
                    del grads, g
        finally:
            hh.remove()
            if vh is not None:
                vh.remove()
        if not all(math.isfinite(rec[k]) for k in ('correct_minus_contrast_logit', 'chosen_ce', 'decision_ce', 'dpo')):
            raise FloatingPointError('Nonfinite diagnostic readout')
        results.append(rec)
        write_json(ROOT / role / 'PROGRESS.json', {'status': 'diagnosing', 'completed_qa': len(results), 'expected_qa': len(plan['rows']), 'active_seconds': time.monotonic() - begun})
        del output, logits, margin, ce, decision_ce, dpo, batch, visual, logp
        if time.monotonic() - begun > 300:
            raise TimeoutError('Read-only diagnostic exceeded five-minute worker budget')
    if lora_hash(model) != before:
        raise ValueError('Read-only diagnosis changed adapter')
    write_json(ROOT / role / 'COMPLETE.json', {'status': 'completed', 'role': role, 'optimizer_updates': 0,
               'teacher_forced_sequences': len(results) * 2, 'adapter_unchanged': True, 'lora_sha256': before,
               'results': results, 'references': {x['qid']: x['sequence_logps'] for x in results},
               'active_seconds': time.monotonic() - begun, 'model_binding': binding})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--role', choices=('initial', 'sft', 'dpo_sft'))
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.role:
        worker(args.role)
    else:
        from research_runtime.queue import run_queue
        identity = {'diagnostic_input_sha256': sha_file(ROOT / 'INPUT_LOCK.json')}
        def job(role):
            return {'key': role, 'command': [sys.executable, '-u', str(Path(__file__)), '--role', role],
                    'completion_marker': str(ROOT / role / 'COMPLETE.json')}
        run_queue(ROOT / 'initial_queue', [job('initial')], identity)
        run_queue(ROOT / 'final_queue', [job('sft'), job('dpo_sft')], identity)
        write_json(ROOT / 'COMPLETION.json', {'status': 'completed', 'scope': read(ROOT / 'PLAN.json')['scope'],
                   'optimizer_updates': 0, 'teacher_forced_sequences': 36,
                   'roles': {role: read(ROOT / role / 'COMPLETE.json') for role in ('initial', 'sft', 'dpo_sft')}})


if __name__ == '__main__':
    main()
