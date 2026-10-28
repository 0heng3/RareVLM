"""Keep the sampled policy action intact; decoded text is a scoring view only."""
import math
import statistics


def stop_at_eos(token_ids, eos_ids, max_new_tokens):
    ids = [int(x) for x in token_ids]
    eos = {int(x) for x in eos_ids}
    if not ids or not eos or max_new_tokens < 1 or len(ids) > max_new_tokens:
        raise ValueError('Invalid generated completion')
    for index, token in enumerate(ids):
        if token in eos:
            return ids[:index + 1], True
    if len(ids) != max_new_tokens:
        raise ValueError('Completion ended without EOS or the declared length limit')
    return ids, False


def attach_completion(prefix, completion_ids):
    """One unpadded image/prompt plus actual actions, including an observed EOS.

    Do not decode/re-encode, append an EOS, or add a chat-template newline.
    Caller must record whether the completion actually terminated.
    """
    import torch
    ids = prefix['input_ids']
    if ids.ndim != 2 or ids.shape[0] != 1 or not completion_ids:
        raise ValueError('Require one unpadded prefix and a nonempty completion')
    if not bool(prefix['attention_mask'].eq(1).all()):
        raise ValueError('Attach before batching/left padding')
    tail = torch.tensor([completion_ids], device=ids.device, dtype=ids.dtype)
    full = torch.cat((ids, tail), dim=1)
    return {**prefix, 'input_ids': full, 'attention_mask': torch.ones_like(full),
            'labels': torch.cat((torch.full_like(ids, -100), tail.clone()), dim=1)}


def readout_tail_length(labels):
    """Retain the predictor before the earliest valid label, including masked tail.

    Count-based cropping is unsafe when there is an ignored token after EOS.
    This changes output-head readout only; full transformer inputs stay intact.
    """
    valid = labels.ne(-100)
    if labels.ndim != 2 or not bool(valid.any(dim=1).all()):
        raise ValueError('Every row needs supervised actions')
    first = valid.long().argmax(dim=1)
    if int(first.min()) < 1:
        raise ValueError('An action needs a preceding predictor token')
    return labels.shape[1] - int(first.min()) + 1


def completion_logps(logits, labels):
    import torch.nn.functional as F
    targets = labels[:, -logits.shape[1]:][:, 1:]
    valid = targets.ne(-100)
    counts = valid.sum(dim=1)
    if not bool(counts.eq(labels.ne(-100).sum(dim=1)).all()) or not bool(counts.gt(0).all()):
        raise ValueError('Readout lost sampled actions')
    values = F.log_softmax(logits[:, :-1].float(), dim=-1).gather(
        -1, targets.masked_fill(~valid, 0).unsqueeze(-1)).squeeze(-1)
    return values.masked_fill(~valid, 0).sum(dim=1), counts


def sequence_logps(model, batch):
    labels = batch['labels']
    keep = readout_tail_length(labels)
    handle = model.get_output_embeddings().register_forward_pre_hook(
        lambda module, args: (args[0][:, -keep:, :],))
    try:
        output = model(**{key: value for key, value in batch.items() if key != 'labels'}, use_cache=False)
    finally:
        handle.remove()
    return completion_logps(output.logits, labels)


def group_signal(rewards, terminated):
    """Sample-std normalization for diagnostic purposes, not an RL update.

    Length-censored answers have no advantage and cannot establish a clear pair.
    Reward is the unchanged VQA consensus proxy, not a semantic truth label.
    """
    if len(rewards) != len(terminated) or len(rewards) < 2:
        raise ValueError('Require a same-question group')
    if any(not math.isfinite(r) or not 0 <= r <= 1 for r in rewards):
        raise ValueError('Rewards must be finite VQA scores')
    usable = [r for r, ok in zip(rewards, terminated) if ok]
    mean = statistics.mean(usable) if usable else None
    std = statistics.stdev(usable) if len(usable) > 1 else 0.
    advantages = [(r - mean) / (std + 1e-4) if ok else None
                  for r, ok in zip(rewards, terminated)]
    return {'terminated_samples': len(usable), 'reward_mean': mean,
            'reward_sample_std': std, 'advantages': advantages,
            'nonzero_reward_variance': bool(std > 1e-12),
            'has_full_positive': any(r >= .999999 for r in usable),
            'has_low_negative': any(r <= .3 + 1e-12 for r in usable),
            'clear_reward_pair': any(r >= .999999 for r in usable) and any(r <= .3 + 1e-12 for r in usable)}
