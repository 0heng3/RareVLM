"""Read the supervised first-answer loss alongside the historical per-QA CE."""


def components_from_logits(logits, labels):
    import torch
    import torch.nn.functional as F

    counts = labels.ne(-100).sum(1)
    if logits.ndim != 3 or not bool(counts.gt(0).all()):
        raise ValueError('Need nonempty assistant targets and batch/sequence/vocab logits')
    targets = labels[:, -logits.shape[1]:][:, 1:]
    valid = targets.ne(-100)
    if not torch.equal(valid.sum(1), counts):
        raise ValueError('Shift lost assistant targets')
    shifted = logits[:, :-1].float()
    values = F.cross_entropy(shifted.reshape(-1, shifted.shape[-1]), targets.reshape(-1),
                             ignore_index=-100, reduction='none').reshape(targets.shape)
    first = valid.long().argmax(1)
    row = torch.arange(labels.shape[0], device=labels.device)
    return {'qa_ce': values.sum(1) / counts, 'first_answer_ce': values[row, first],
            'counts': counts, 'first_prediction_positions_in_tail': first}


def forward_components(model, batch):
    import torch

    labels = batch['labels']
    counts = labels.ne(-100).sum(1)
    positions = torch.arange(labels.shape[1], device=labels.device)[None]
    if not torch.equal(labels.ne(-100), positions >= labels.shape[1] - counts[:, None]):
        raise ValueError('Expected the historical contiguous left-padded assistant tail')
    keep = int(counts.max()) + 1
    hook = model.get_output_embeddings().register_forward_pre_hook(
        lambda module, args: (args[0][:, -keep:],))
    try:
        result = model(**{k: v for k, v in batch.items() if k != 'labels'}, use_cache=False)
    finally:
        hook.remove()
    return components_from_logits(result.logits, labels)
