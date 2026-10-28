"""Target-free first-token diagnostics for fixed checkpoints.

This module receives prompt-only, left-padded inputs. It does not append an
answer, compute a supervised loss, change model dtypes, or enable autocast.
Returned probabilities describe the next-token distribution, not generation
accuracy. Canonical and alias-set readouts deliberately remain separate.
The default preserves the dense-head path used by the installed Qwen forward.
Optional BF16 head slicing can alter GEMM rounding; these diagnostic logits do
not replace saved greedy-generation decisions.
"""

from math import isfinite
from numbers import Integral


_REFERENCE_KEYS = frozenset({
    'labels', 'label', 'target', 'targets', 'answer', 'answers',
    'canonical_answer', 'reference', 'references', 'ground_truth',
})


def _token_sets(yes_ids, no_ids, canonical_yes_id, canonical_no_id):
    def checked(values, name):
        values = tuple(values)
        if not values or any(isinstance(x, bool) or not isinstance(x, Integral)
                             or x < 0 for x in values):
            raise ValueError(f'{name} must contain nonnegative integer token IDs')
        values = tuple(int(x) for x in values)
        if len(set(values)) != len(values):
            raise ValueError(f'{name} must be unique; resolve aliases before readout')
        return values

    yes_ids, no_ids = checked(yes_ids, 'yes_ids'), checked(no_ids, 'no_ids')
    if set(yes_ids) & set(no_ids):
        raise ValueError('YES and NO alias sets must be disjoint')
    if (isinstance(canonical_yes_id, bool) or isinstance(canonical_no_id, bool)
            or not isinstance(canonical_yes_id, Integral)
            or not isinstance(canonical_no_id, Integral)
            or canonical_yes_id not in yes_ids or canonical_no_id not in no_ids):
        raise ValueError('Canonical IDs must belong to their respective alias sets')
    return yes_ids, no_ids, int(canonical_yes_id), int(canonical_no_id)


def forward_first_readout(model, batch, yes_ids, no_ids,
                          canonical_yes_id, canonical_no_id, *, slice_head=False):
    """Read the first generated position using exactly one no-label forward.

    The temporary output-head hook verifies the full prompt alignment and, only
    when slice_head=True, retains the final hidden state. Dense is the default
    to preserve the installed Qwen head's GEMM rounding near the zero boundary.
    The hook is removed even when the model raises. The caller owns model loading,
    tokenizer alias resolution, prompt construction, and post-readout scoring.
    A model in training mode or a right-padded batch is rejected rather than
    silently changing the checkpoint's evaluation contract.
    """
    import torch

    yes_ids, no_ids, canonical_yes_id, canonical_no_id = _token_sets(
        yes_ids, no_ids, canonical_yes_id, canonical_no_id)
    forbidden = _REFERENCE_KEYS.intersection(batch)
    if forbidden:
        raise ValueError(f'Prompt-only readout rejects reference fields: {sorted(forbidden)}')
    if 'use_cache' in batch:
        raise ValueError('The readout controls use_cache=False')
    input_ids, mask = batch.get('input_ids'), batch.get('attention_mask')
    if (not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2
            or min(input_ids.shape) == 0):
        raise ValueError('input_ids must be a nonempty [batch, prompt] tensor')
    if not isinstance(mask, torch.Tensor) or mask.shape != input_ids.shape:
        raise ValueError('An aligned attention_mask is required')
    if not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError('attention_mask must be binary')
    lengths = mask.sum(1)
    expected = torch.arange(mask.shape[1], device=mask.device)[None, :] >= (
        mask.shape[1] - lengths[:, None])
    if not bool(lengths.gt(0).all()) or not torch.equal(mask.bool(), expected):
        raise ValueError('Readout requires left padding and a valid final prompt token')
    if model.training:
        raise ValueError('Fixed-checkpoint readout requires model.eval()')
    underlying = model.module if hasattr(model, 'module') else model
    head = underlying.get_output_embeddings()
    head_calls = 0

    def last_prompt_only(module, args):
        nonlocal head_calls
        head_calls += 1
        if head_calls != 1:
            raise ValueError('Expected one output-head call per forward')
        if (not args or args[0].ndim != 3
                or args[0].shape[:2] != input_ids.shape):
            raise ValueError('Output head must receive the full aligned prompt states')
        return (args[0][:, -1:, :], *args[1:]) if slice_head else args

    handle = head.register_forward_pre_hook(last_prompt_only)
    try:
        with torch.inference_mode():
            output = model(**batch, use_cache=False)
    finally:
        handle.remove()
    expected_positions = 1 if slice_head else input_ids.shape[1]
    if head_calls != 1 or output.logits.ndim != 3 or output.logits.shape[:2] != (
            input_ids.shape[0], expected_positions):
        raise ValueError('Output logits must match the declared dense/sliced prompt path')
    logits = output.logits[:, -1, :].float()
    if not bool(torch.isfinite(logits).all()):
        raise ValueError('Next-token logits must be finite')
    if max(*yes_ids, *no_ids) >= logits.shape[-1]:
        raise ValueError('Alias token ID is outside the model vocabulary')
    # Explicit FP32 reductions; base/head and LoRA parameter storage are untouched.
    normalizer = torch.logsumexp(logits, dim=-1)
    yes_mass = torch.logsumexp(logits[:, list(yes_ids)], dim=-1)
    no_mass = torch.logsumexp(logits[:, list(no_ids)], dim=-1)
    alias_margin = yes_mass - no_mass
    canonical_margin = logits[:, canonical_yes_id] - logits[:, canonical_no_id]
    top1_logit, top1_id = logits.max(-1)
    values = {
        'canonical_margin': canonical_margin,
        'canonical_yes_log_probability': logits[:, canonical_yes_id] - normalizer,
        'canonical_no_log_probability': logits[:, canonical_no_id] - normalizer,
        'canonical_conditional_yes_probability': canonical_margin.sigmoid(),
        'alias_margin': alias_margin,
        'conditional_yes_probability': alias_margin.sigmoid(),
        'yes_alias_log_probability': yes_mass - normalizer,
        'no_alias_log_probability': no_mass - normalizer,
        'top1_token_id': top1_id,
        'top1_logit': top1_logit,
    }
    cpu = {key: value.detach().cpu().tolist() for key, value in values.items()}
    return [{key: value[row] for key, value in cpu.items()}
            for row in range(input_ids.shape[0])]


def classify_pair(m_pos, m_neg):
    """Separate cross-image ordering from the zero decision boundary.

    Positive/negative denote the known post-readout labels of the two images.
    Ties are explicit and are not counted as strictly correct decisions.
    """
    m_pos, m_neg = float(m_pos), float(m_neg)
    if not isfinite(m_pos) or not isfinite(m_neg):
        raise ValueError('Pair margins must be finite')
    correct = m_pos > 0 and m_neg < 0
    return {'gap': m_pos - m_neg, 'center': (m_pos + m_neg) / 2,
            'ordered': m_pos > m_neg, 'tie': m_pos == m_neg,
            'threshold_fail': not correct, 'both_binary_correct': correct,
            'boundary_tie': m_pos == 0 or m_neg == 0}
