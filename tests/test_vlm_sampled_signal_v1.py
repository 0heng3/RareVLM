import pytest
import torch
from vlm_sampled_signal_v1.trajectory import (
    attach_completion, completion_logps, group_signal, readout_tail_length, stop_at_eos)


def test_real_alternative_tokenization_is_not_reencoded():
    prefix = {'input_ids': torch.tensor([[17, 21]]), 'attention_mask': torch.ones(1, 2, dtype=torch.long)}
    # Actual Qwen FGHIJ actions: fg + hi + j + im_end, not f + gh + ij.
    actions, done = stop_at_eos([4817, 6023, 73, 151645, 151643], [151645, 151643], 32)
    batch = attach_completion(prefix, actions)
    assert done and batch['input_ids'][0].tolist() == [17, 21, 4817, 6023, 73, 151645]
    assert batch['labels'][0].tolist() == [-100, -100, 4817, 6023, 73, 151645]
    assert prefix['input_ids'][0].tolist() == [17, 21]


def test_no_fabricated_eos_at_length_cap():
    actions, done = stop_at_eos([3, 4, 5], [9], 3)
    assert actions == [3, 4, 5] and not done
    with pytest.raises(ValueError):
        stop_at_eos([3], [9], 3)


def test_masked_tail_readout_matches_full_values_and_gradients():
    # Ignored post-EOS suffix and unequal answer lengths exposed the old count assumption.
    labels = torch.tensor([[-100, -100, -100, 2, 3, -100, -100],
                           [-100, -100, -100, -100, 4, 5, -100]])
    logits = torch.randn(2, 7, 8, generator=torch.Generator().manual_seed(42), requires_grad=True)
    full, counts = completion_logps(logits, labels)
    tail, tail_counts = completion_logps(logits[:, -readout_tail_length(labels):], labels)
    assert counts.tolist() == tail_counts.tolist() == [2, 2]
    torch.testing.assert_close(full, tail, rtol=0, atol=0)
    full_grad = torch.autograd.grad(full.sum(), logits, retain_graph=True)[0]
    tail_grad = torch.autograd.grad(tail.sum(), logits)[0]
    torch.testing.assert_close(full_grad, tail_grad, rtol=0, atol=0)
    with pytest.raises(ValueError, match='lost sampled actions'):
        completion_logps(logits[:, -(int(counts.max()) + 1):], labels)


def test_group_variance_and_censoring():
    equal = group_signal([0.] * 8, [True] * 8)
    assert equal['advantages'] == [0.] * 8 and not equal['clear_reward_pair']
    mixed = group_signal([1., 0., .3, 1.], [True, True, True, False])
    assert mixed['clear_reward_pair'] and mixed['nonzero_reward_variance']
    assert mixed['advantages'][-1] is None
    assert abs(sum(x for x in mixed['advantages'] if x is not None)) < 1e-12
    assert not group_signal([1., 0.], [False, True])['clear_reward_pair']
