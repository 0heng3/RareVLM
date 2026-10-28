"""Generated-prefix full-vocabulary forward KL and unchanged object CE.

Object CE is an explicit copy of the frozen behavior-replay numerical path.
KL uses the same assistant tail and causal mask; diagnostics never backpropagate.
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


def assistant_logprobs(model, batch, expected_vocabulary=None):
    """One sparse-head forward, retaining the full vocabulary at assistant rows."""
    labels, roles = batch['labels'], batch['answer_roles']
    if (labels.ndim != 2 or roles.shape != labels.shape or labels.dtype != torch.long
            or roles.dtype != torch.long or not torch.equal(roles > 0, labels != -100)
            or not bool(((roles >= 0) & (roles <= 4)).all())
            or bool(((roles == 1).sum(1) > 1).any())):
        raise ValueError('assistant labels/roles must have matching exclusive int64 masks')
    keep = int((labels != -100).sum(1).max()) + 1
    if keep <= 1:
        raise ValueError('assistant tokens are required')
    hook = model.get_output_embeddings().register_forward_pre_hook(
        lambda module, args: (args[0][:, -keep:, :],))
    try:
        output = model(**{key:value for key,value in batch.items()
                          if key not in ('labels','answer_roles')}, use_cache=False)
    finally:
        hook.remove()
    logits = output.logits
    if expected_vocabulary is not None and logits.shape[-1] != expected_vocabulary:
        raise ValueError('model head full vocabulary differs from locked reference')
    targets = labels[:, -logits.shape[1]:][:, 1:]
    shifted_roles = roles[:, -logits.shape[1]:][:, 1:]
    valid = targets != -100
    counts = valid.sum(1)
    if (not bool((counts > 0).all())
            or not torch.equal(counts, (labels != -100).sum(1))
            or not torch.equal(shifted_roles > 0, valid)):
        raise ValueError('causal tail slicing lost assistant/EOS/template tokens')
    rows = F.log_softmax(logits[:, :-1, :][valid].float(), dim=-1)
    if not bool(torch.isfinite(rows).all()):
        raise FloatingPointError('nonfinite full-vocabulary FP32 assistant log probabilities')
    return rows, counts, targets[valid], shifted_roles[valid]


def per_question_kl(actor_logprobs, teacher_logprobs, counts):
    """KL(p_initial || p_actor), vocab SUM then token mean inside each QA."""
    if (actor_logprobs.ndim != 2 or teacher_logprobs.shape != actor_logprobs.shape
            or actor_logprobs.dtype != torch.float32 or teacher_logprobs.dtype != torch.float32
            or counts.ndim != 1 or counts.dtype != torch.long or counts.numel() == 0
            or not bool((counts > 0).all()) or int(counts.sum()) != actor_logprobs.shape[0]
            or not bool(torch.isfinite(actor_logprobs).all())
            or not bool(torch.isfinite(teacher_logprobs).all())):
        raise ValueError('full finite FP32 vocabulary rows and positive int64 QA counts required')
    teacher = teacher_logprobs.detach()
    token_kl = (teacher.exp() * (teacher-actor_logprobs)).sum(-1)
    return torch.stack([piece.mean() for piece in token_kl.split(counts.tolist())]), token_kl


def replay_diagnostics(actor_logprobs, targets, counts, roles, token_kl):
    """Detached hard CE and disjoint role CE/KL, from this same actor forward."""
    if targets.shape != (actor_logprobs.shape[0],) or roles.shape != targets.shape or token_kl.shape != targets.shape:
        raise ValueError('assistant token rows/targets/roles differ')
    token_ce = -actor_logprobs.detach().gather(1,targets[:,None]).squeeze(1)
    ce = torch.stack([piece.mean() for piece in token_ce.split(counts.tolist())])
    ce_roles, kl_roles = empty_role_statistics(), empty_kl_role_statistics()
    boundaries = list(roles.split(counts.tolist()))
    for role,name in ROLE_NAMES.items():
        selected = roles == role
        questions = sum(bool((part == role).any()) for part in boundaries)
        ce_roles[name] = {'token_count':int(selected.sum()),'ce_sum':float(token_ce[selected].sum()),
                          'questions_with_role':questions}
        kl_roles[name] = {'token_count':int(selected.sum()),'kl_sum':float(token_kl.detach()[selected].sum()),
                          'questions_with_role':questions}
    if sum(value['token_count'] for value in kl_roles.values()) != int(counts.sum()):
        raise ValueError('role KL diagnostics lost or duplicated assistant tokens')
    return ce, ce_roles, kl_roles


def empty_kl_role_statistics():
    return {name:{'token_count':0,'kl_sum':0.,'questions_with_role':0} for name in ROLE_NAMES.values()}


def accumulate_kl_roles(total, new):
    if set(total) != set(new) or set(total) != set(ROLE_NAMES.values()):
        raise ValueError('unknown/missing role KL statistics')
    for name in ROLE_NAMES.values():
        for key in ('token_count','kl_sum','questions_with_role'):
            total[name][key] += new[name][key]


def summarize_kl_roles(total):
    return {name:{**value,'mean_token_kl':value['kl_sum']/value['token_count']
                           if value['token_count'] else None} for name,value in total.items()}
