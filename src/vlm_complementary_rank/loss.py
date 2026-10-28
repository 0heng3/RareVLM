"""FP32 per-QA CE and a same-question, opposite-image ranking term.

The caller owns update-level normalization: ``qa_ce.sum() / effective_batch``
and ``lambda_rank * pair_rank.sum() / update_pair_count``. A microbatch must
contain both members of every pair listed in ``pair_indices``. No rejected
completion, reference-model forward, or generated answer is needed.
"""


def _assistant_counts(labels):
    import torch

    if labels.ndim != 2 or labels.dtype != torch.long:
        raise ValueError('labels must be a two-dimensional torch.long tensor')
    counts = labels.ne(-100).sum(1)
    if not bool(counts.gt(0).all()):
        raise ValueError('Every question needs an assistant target')
    return counts


def loss_from_logits(logits, labels, pair_indices, yes_token_id, no_token_id,
                     *, logits_start=0):
    """Return unaggregated losses for full or explicitly aligned tail logits.

    ``logits[:, j]`` predicts the token following the original sequence position
    ``logits_start + j``. For tail logits of length K from a sequence of length
    T, the caller must pass ``logits_start=T-K``. The first supervised label is
    therefore read from the *preceding* logit, never its own input position.

    Only paired rows must start with the canonical single-token ``yes``/``no``
    targets. Empty pairs permit ordinary TextVQA microbatches with other targets.
    IDs must be resolved with the actual local tokenizer, not hard-coded here.
    """
    import operator

    import torch
    import torch.nn.functional as F

    counts = _assistant_counts(labels)
    if (logits.ndim != 3 or not logits.is_floating_point()
            or logits.shape[0] != labels.shape[0]
            or logits.device != labels.device):
        raise ValueError('logits must be floating [batch, sequence, vocabulary] on the labels device')
    try:
        logits_start = operator.index(logits_start)
        yes_token_id, no_token_id = operator.index(yes_token_id), operator.index(no_token_id)
    except TypeError as error:
        raise ValueError('logits_start and answer token IDs must be integers') from error
    if (logits.shape[1] < 2 or logits_start < 0
            or logits_start + logits.shape[1] != labels.shape[1]):
        raise ValueError('logits must cover the full sequence or an explicitly aligned suffix')
    if (not 0 <= yes_token_id < logits.shape[-1]
            or not 0 <= no_token_id < logits.shape[-1]
            or yes_token_id == no_token_id):
        raise ValueError('yes/no token IDs must be distinct vocabulary entries')

    # This is exactly the frozen FP32 CE shift and per-QA reduction. Explicit
    # alignment above prevents a shorter tensor from being mistaken for dense.
    targets = labels[:, logits_start:][:, 1:]
    valid = targets.ne(-100)
    if not torch.equal(valid.sum(1), counts):
        raise ValueError('Causal shift/head slicing lost assistant targets')
    shifted = logits[:, :-1, :].float()
    if not bool(torch.isfinite(shifted).all()):
        raise ValueError('Non-finite FP32 prediction logits')
    values = F.cross_entropy(shifted.reshape(-1, shifted.shape[-1]), targets.reshape(-1),
                             ignore_index=-100, reduction='none').reshape(targets.shape)
    qa_ce = values.sum(1) / counts

    first = valid.to(torch.long).argmax(1)
    rows = torch.arange(labels.shape[0], device=labels.device)
    first_logits = shifted[rows, first]
    margins = first_logits[:, yes_token_id] - first_logits[:, no_token_id]
    first_ids = targets[rows, first]

    pairs = torch.as_tensor(pair_indices, device=labels.device)
    if pairs.numel() == 0 and (pairs.shape == (0,) or pairs.shape == (0, 2)):
        pairs = torch.empty((0, 2), dtype=torch.long, device=labels.device)
    elif pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype != torch.long:
        raise ValueError('pair_indices must be integer [yes_row, no_row] pairs')
    if pairs.numel():
        if not bool(((pairs >= 0) & (pairs < labels.shape[0])).all()):
            raise ValueError('Pair row index is outside this microbatch')
        if (not bool(first_ids[pairs[:, 0]].eq(yes_token_id).all())
                or not bool(first_ids[pairs[:, 1]].eq(no_token_id).all())):
            raise ValueError('Pair first assistant targets must be canonical yes, then no')
    pair_rank = F.softplus(-(margins[pairs[:, 0]] - margins[pairs[:, 1]]))
    if (not bool(torch.isfinite(qa_ce).all())
            or not bool(torch.isfinite(margins).all())
            or not bool(torch.isfinite(pair_rank).all())):
        raise ValueError('Non-finite CE, first-token margin, or pair ranking loss')
    return {
        'qa_ce': qa_ce,
        'pair_rank': pair_rank,
        'first_token_margin': margins,
        'token_ce_sum': values.sum(),
        'supervised_token_count': int(counts.sum()),
        'first_target_token_ids': first_ids,
    }


def forward_pair_loss(model, batch, pair_indices, yes_token_id, no_token_id,
                      *, tail=True):
    """One actor forward, retaining the frozen CE backend's tail-head strategy."""
    import torch

    labels = batch['labels']
    counts = _assistant_counts(labels)
    positions = torch.arange(labels.shape[1], device=labels.device)[None, :]
    expected = positions >= labels.shape[1] - counts[:, None]
    if tail and not torch.equal(labels.ne(-100), expected):
        raise ValueError('Tail loss requires contiguous assistant labels at the sequence end')
    handle = None
    if tail:
        underlying = model.module if hasattr(model, 'module') else model
        keep = int(counts.max()) + 1
        handle = underlying.get_output_embeddings().register_forward_pre_hook(
            lambda module, args: (args[0][:, -keep:, :],))
    try:
        output = model(**{key: value for key, value in batch.items() if key != 'labels'},
                       use_cache=False)
    finally:
        if handle is not None:
            handle.remove()
    return loss_from_logits(
        output.logits, labels, pair_indices, yes_token_id, no_token_id,
        logits_start=labels.shape[1] - output.logits.shape[1])
