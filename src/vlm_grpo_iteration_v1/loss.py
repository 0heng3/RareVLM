import torch
import torch.nn.functional as F
from vlm_sampled_signal_v1.trajectory import readout_tail_length


def token_logps(model, batch):
    labels = batch['labels']
    keep = readout_tail_length(labels)
    handle = model.get_output_embeddings().register_forward_pre_hook(lambda m, args: (args[0][:, -keep:, :],))
    try:
        out = model(**{k: v for k, v in batch.items() if k != 'labels'}, use_cache=False)
    finally:
        handle.remove()
    targets = labels[:, -out.logits.shape[1]:][:, 1:]
    valid = targets.ne(-100)
    if not bool(valid.sum(1).eq(labels.ne(-100).sum(1)).all()):
        raise ValueError('Lost sampled action positions')
    values = F.log_softmax(out.logits[:, :-1].float(), dim=-1).gather(
        -1, targets.masked_fill(~valid, 0).unsqueeze(-1)).squeeze(-1)
    return values.masked_fill(~valid, 0), valid


def behavior_logps(samples, current, valid):
    old = torch.zeros_like(current)
    for index, sample in enumerate(samples):
        values = sample['sampling_token_logps']
        if len(values) != int(valid[index].sum()):
            raise ValueError('Behavior token probability/action length differs')
        old[index, valid[index]] = torch.tensor(values, device=current.device, dtype=current.dtype)
    return old.detach()


def clipped_group_loss(current, old, valid, advantages, epsilon=.2):
    """Original per-sequence-normalized token-ratio GRPO surrogate, beta(KL)=0.

    Each behavior action appears in only one minibatch during one rollout epoch.
    Fixed behavior probabilities are the actual generation probabilities.
    """
    if current.shape != old.shape or current.shape != valid.shape or not bool(valid.any(1).all()):
        raise ValueError('Invalid policy probability/mask shapes')
    advantage = advantages.detach().unsqueeze(1)
    log_ratio = current - old.detach()
    ratio = torch.exp(log_ratio)
    objective = torch.minimum(ratio * advantage, ratio.clamp(1-epsilon, 1+epsilon) * advantage)
    per_sequence = -(objective * valid).sum(1) / valid.sum(1)
    loss = per_sequence.mean()
    active = valid & advantage.ne(0)
    values = ratio[active]
    return loss, {'policy_loss': float(loss.detach()),
        'ratio_clip_fraction': float(((values < 1-epsilon) | (values > 1+epsilon)).float().mean()) if values.numel() else 0.,
        'max_absolute_log_ratio': float(log_ratio[valid].detach().abs().max()),
        'active_actions': int(active.sum())}
