"""Assistant-only per-QA CE with disjoint, weight-neutral token diagnostics.

The FP32 log-softmax/gather and causal tail slicing match the frozen
vlm_alignment_signal.loss contract. Diagnostics reuse this very forward.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'
import torch
import torch.nn.functional as F

from vlm_object_train.loss import normalized_qa_loss

ROLE_NAMES = {1: 'first_content', 2: 'later_content', 3: 'actual_eos', 4: 'template_suffix'}


def completion_ce(logits, labels, answer_roles):
    """Return per-QA CE/counts and detached role sums from identical token logps."""
    if (labels.ndim != 2 or answer_roles.shape != labels.shape
            or labels.dtype != torch.long or answer_roles.dtype != torch.long):
        raise ValueError('labels/answer_roles must be matching int64 [batch,sequence]')
    valid_full = labels != -100
    if (not bool(((answer_roles > 0) == valid_full).all())
            or not bool(((answer_roles >= 0) & (answer_roles <= 4)).all())):
        raise ValueError('roles must exclusively and completely cover assistant labels')
    if bool(((answer_roles == 1).sum(1) > 1).any()):
        raise ValueError('at most one first-content role per question')
    targets = labels[:, -logits.shape[1]:][:, 1:]
    roles = answer_roles[:, -logits.shape[1]:][:, 1:]
    shifted = logits[:, :-1, :].float()
    valid = targets != -100
    safe = targets.masked_fill(~valid, 0)
    logp = F.log_softmax(shifted, dim=-1).gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    counts = valid.sum(1)
    if (not bool((counts > 0).all())
            or not bool((counts == valid_full.sum(1)).all())):
        raise ValueError('causal tail slicing lost supervised assistant tokens')
    # Preserve the old reduction/sign/order, rather than averaging flattened tokens.
    ce = -(logp * valid).sum(1) / counts
    diagnostics = {}
    detached = -logp.detach()
    for role, name in ROLE_NAMES.items():
        mask = roles == role
        diagnostics[name] = {'token_count': int(mask.sum()),
                             'ce_sum': float((detached * mask).sum()),
                             'questions_with_role': int(mask.any(1).sum())}
    if sum(value['token_count'] for value in diagnostics.values()) != int(counts.sum()):
        raise ValueError('role diagnostics lost or double-counted assistant tokens')
    return ce, counts, diagnostics


def sequence_ce(model, batch):
    """Exactly one actor forward; answer_roles never enters the model kwargs."""
    labels, roles = batch['labels'], batch['answer_roles']
    keep = int((labels != -100).sum(1).max()) + 1
    hook = model.get_output_embeddings().register_forward_pre_hook(
        lambda module, args: (args[0][:, -keep:, :],))
    try:
        output = model(**{key: value for key, value in batch.items()
                          if key not in ('labels', 'answer_roles')}, use_cache=False)
    finally:
        hook.remove()
    return completion_ce(output.logits, labels, roles)


def empty_role_statistics():
    return {name: {'token_count': 0, 'ce_sum': 0., 'questions_with_role': 0}
            for name in ROLE_NAMES.values()}


def accumulate_roles(total, new):
    if set(total) != set(new) or set(total) != set(ROLE_NAMES.values()):
        raise ValueError('role statistics have an unknown or missing role')
    for name in ROLE_NAMES.values():
        for key in ('token_count', 'ce_sum', 'questions_with_role'):
            total[name][key] += new[name][key]


def summarize_roles(total):
    return {name: {**value, 'mean_token_ce': value['ce_sum']/value['token_count']
                            if value['token_count'] else None}
            for name, value in total.items()}
