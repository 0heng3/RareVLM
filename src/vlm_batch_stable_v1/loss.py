"""Per-question CE and reference-to-actor KL over actual assistant tokens."""
import torch


def assistant_logps(model, batch):
    labels = batch['labels']; counts = labels.ne(-100).sum(1); keep = int(counts.max()) + 1
    handle = model.get_output_embeddings().register_forward_pre_hook(lambda mod, args: (args[0][:, -keep:, :],))
    try: output = model(**{k: v for k, v in batch.items() if k != 'labels'}, use_cache=False)
    finally: handle.remove()
    targets = labels[:, -output.logits.shape[1]:][:, 1:]; valid = targets.ne(-100)
    if not torch.equal(valid.sum(1), counts) or not bool(counts.gt(0).all()): raise ValueError('Lost assistant tokens')
    logp = output.logits[:, :-1].float().log_softmax(-1)[valid]
    owners = torch.arange(labels.shape[0], device=labels.device)[:, None].expand_as(valid)[valid]
    token_ce = -logp.gather(1, targets[valid].unsqueeze(1)).squeeze(1)
    ce = torch.zeros(labels.shape[0], device=logp.device).scatter_add(0, owners, token_ce) / counts
    return logp, owners, counts, ce


def replay_kl(logp, reference, owners, counts):
    if logp.shape != reference.shape or logp.dtype != torch.float32 or reference.dtype != torch.float32: raise ValueError('KL token frame or precision mismatch')
    reference = reference.detach()
    token_kl = (reference.exp() * (reference - logp)).sum(-1)
    return torch.zeros(len(counts), device=logp.device).scatter_add(0, owners, token_kl) / counts
