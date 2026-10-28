"""Per-question assistant CE with an explicit FP32 reduction, with or without AMP."""


def answer_loss(model, batch, tail=True):
    import torch
    import torch.nn.functional as F

    labels = batch['labels']
    counts = labels.ne(-100).sum(1)
    if not bool(counts.gt(0).all()):
        raise ValueError('Every question needs an assistant target')
    # The shared collator left-pads. Check the contiguous answer-tail contract
    # before slicing; do not silently drop labels for another collator/layout.
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
    targets = labels[:, -output.logits.shape[1]:][:, 1:]
    if not torch.equal(targets.ne(-100).sum(1), counts):
        raise ValueError('Causal shift/head slicing lost assistant targets')
    # BF16 CE outside autocast otherwise rounds loss/gradients in BF16. The cast
    # remains differentiable; it does not turn the model/head itself into FP32.
    logits = output.logits[:, :-1, :].float()
    values = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                             ignore_index=-100, reduction='none').reshape(targets.shape)
    return values.sum(1) / counts, values.sum(), int(counts.sum())
