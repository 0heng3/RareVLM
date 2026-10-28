from rarevlm_paths import PROJECT_ROOT
from vlm_grpo_iteration_v1.common import actor, chosen_batch, read

PARENT = PROJECT_ROOT.parents[1] / 'outputs/vlm_reference_replay_v1'
ROOT = PARENT.parent / 'vlm_replay_projection_v1'
PROTOCOL = PROJECT_ROOT / 'docs/VLM_REPLAY_PROJECTION_V1_PROTOCOL.md'
RESULT = PROJECT_ROOT / 'docs/VLM_REPLAY_PROJECTION_V1_RESULTS.md'
PAGE = PROJECT_ROOT / 'docs/learning/回放约束Adam_纠错与保持.html'


def project(delta, gradients):
    # Reuse the already tested active-set solver; zero margin is always feasible.
    from vlm_adam_conflict_v1.projection import project_displacement
    return project_displacement(delta, gradients, minimum_descent_cosine=0.)


def flat(parameters):
    import torch
    return torch.cat([p.detach().reshape(-1) for p in parameters])


def gradients(model, processor, block, data, cpu, trace=None, update=None):
    """Four forwards/backwards, identical CE and ordering in both fresh arms.

    Source gradients use the same /16 normalization as total accumulated CE.
    There is no extra backward pass and no replay/KL coefficient selection.
    """
    import torch
    from vlm_sft.common import to_device
    from vlm_reference_replay_v1.loss import forward_logits, loss_values
    lookup = {q['question_id']: q for q in data['training_rows']}
    parameters = [p for p in model.parameters() if p.requires_grad]
    source_grad = {s: [torch.zeros_like(p) for p in parameters] for s in ('hard', 'replay')}
    totals = {'hard': 0., 'replay': 0.}
    records = []
    model.train(); model.get_base_model().visual.eval(); model.config.use_cache = False
    for micro, ids in enumerate(block['micros']):
        source = 'replay' if micro in (0, 2) else 'hard'
        rows = [lookup[qid] for qid in ids]
        if any(q['training_source'] != source for q in rows):
            raise ValueError('Unexpected source/order')
        batch = chosen_batch(processor, rows, data['images'], cpu)
        gpu = to_device(batch, next(model.parameters()).device)
        logits = forward_logits(model, gpu)
        ce, _ = loss_values(logits, gpu['labels'])
        if not bool(torch.isfinite(ce).all()):
            raise FloatingPointError('Nonfinite CE')
        grads = torch.autograd.grad(ce.sum()/16, parameters)
        for p, g, target in zip(parameters, grads, source_grad[source]):
            if p.grad is None:
                p.grad = g.detach().clone()
            else:
                p.grad.add_(g.detach())
            target.add_(g.detach())
        totals[source] += float(ce.detach().sum())
        for q, value, tokens in zip(rows, ce.detach().cpu().tolist(), gpu['labels'].ne(-100).sum(1).tolist()):
            records.append({'update': update, 'micro': micro, 'question_id': q['question_id'], 'source': source,
                            'chosen_ce': value, 'input_sha256': cpu[q['question_id']]['input_ids'],
                            'labels_sha256': cpu[q['question_id']]['labels'], 'assistant_tokens': int(tokens),
                            'visual_tokens': cpu[q['question_id']]['visual_tokens']})
        del batch, gpu, logits, ce, grads
    vectors = {s: torch.cat([g.flatten() for g in values]) for s, values in source_grad.items()}
    return parameters, vectors, totals, records
