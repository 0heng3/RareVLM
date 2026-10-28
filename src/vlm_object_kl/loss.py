"""Assistant-only causal forward KL, averaged over tokens inside each QA.

Teacher forcing uses the original train_all target prefixes. This is fixed
teacher distillation, not on-policy GKD, preference optimization, or RL.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def assistant_logprobs(model, batch):
    """Return row-major assistant prediction rows and their per-QA counts.

    Reuse the frozen tail-head optimization: labels at position t are predicted
    by logits t-1. Prompt, padding and all other labels=-100 are excluded. The
    complete vocabulary is retained in FP32 after BF16 model/head computation.
    """
    labels = batch['labels']
    keep = int((labels != -100).sum(1).max()) + 1
    hook = model.get_output_embeddings().register_forward_pre_hook(
        lambda module, args: (args[0][:, -keep:, :],))
    try:
        output = model(**{key: value for key, value in batch.items() if key != 'labels'}, use_cache=False)
    finally:
        hook.remove()
    targets = labels[:, -output.logits.shape[1]:][:, 1:]
    valid = targets != -100
    counts = valid.sum(1)
    if not bool((counts > 0).all()) or not torch.equal(counts, (labels != -100).sum(1)):
        raise ValueError('lost assistant/end tokens in causal shift')
    rows = output.logits[:, :-1, :][valid].float()
    logprobs = F.log_softmax(rows, dim=-1)
    return logprobs, counts, targets[valid]


def per_question_kl(actor_logprobs, teacher_logprobs, counts):
    """KL(p_initial || p_actor): sum vocabulary, mean answer tokens per QA.

    Detachment is enforced even if the caller accidentally supplies a teacher
    graph. The full FP32 teacher distribution is never truncated/quantized.
    """
    if (actor_logprobs.ndim != 2 or teacher_logprobs.shape != actor_logprobs.shape
            or actor_logprobs.dtype != torch.float32 or teacher_logprobs.dtype != torch.float32
            or counts.ndim != 1 or counts.numel() == 0 or not bool((counts > 0).all())
            or int(counts.sum()) != actor_logprobs.shape[0]):
        raise ValueError('full FP32 vocabulary rows and positive per-QA token counts required')
    teacher = teacher_logprobs.detach()
    token_kl = (teacher.exp() * (teacher - actor_logprobs)).sum(-1)
    return torch.stack([piece.mean() for piece in token_kl.split(counts.tolist())])


def per_question_hard_ce(logprobs, targets, counts):
    if targets.shape != (logprobs.shape[0],):
        raise ValueError('assistant targets differ from full probability rows')
    token_ce = -logprobs.gather(1, targets[:, None]).squeeze(1)
    return torch.stack([piece.mean() for piece in token_ce.split(counts.tolist())])
