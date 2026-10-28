"""Keep auxiliary text gradients unchanged; remove only EOS and later tail losses."""


def reduce_auxiliary(logits, targets, eos_token_id):
    import torch
    import torch.nn.functional as F
    valid = targets.ne(-100)
    counts = valid.sum(1)
    eos = targets.eq(eos_token_id) & valid
    if not bool(counts.gt(0).all()) or not bool(eos.sum(1).eq(1).all()):
        raise ValueError('Each auxiliary QA needs nonempty labels and exactly one EOS')
    text = valid & eos.cumsum(1).eq(0)
    if not bool(text.sum(1).gt(0).all()):
        raise ValueError('Each auxiliary QA needs actual text before EOS')
    values = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1),
                             ignore_index=-100, reduction='none').reshape(targets.shape)
    # Deliberately keep the ORIGINAL denominator. Renormalizing over text would
    # also increase word-token gradients and cease to isolate termination loss.
    return (values * text).sum(1) / counts, values.sum(1) / counts, text.sum(1)


def answer_loss(model, batch, auxiliary, eos_token_id):
    if not auxiliary:
        from vlm_fp32_adapter.loss import answer_loss as original
        losses, _, _ = original(model, batch)
        return losses, losses, batch['labels'].ne(-100).sum(1)
    import torch
    labels = batch['labels']; counts = labels.ne(-100).sum(1)
    positions = torch.arange(labels.shape[1], device=labels.device)[None, :]
    if not torch.equal(labels.ne(-100), positions >= labels.shape[1] - counts[:, None]):
        raise ValueError('Expected the original contiguous assistant-tail labels')
    keep = int(counts.max()) + 1
    hook = model.get_output_embeddings().register_forward_pre_hook(lambda module, args: (args[0][:, -keep:, :],))
    try:
        output = model(**{k: v for k, v in batch.items() if k != 'labels'}, use_cache=False)
    finally:
        hook.remove()
    targets = labels[:, -output.logits.shape[1]:][:, 1:]
    if not torch.equal(targets.ne(-100).sum(1), counts):
        raise ValueError('Causal shifting lost assistant labels')
    return reduce_auxiliary(output.logits[:, :-1], targets, eos_token_id)
