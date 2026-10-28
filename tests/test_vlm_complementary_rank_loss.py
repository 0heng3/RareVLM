"""Necessary CPU mathematics for the new pair objective; no model/GPU fixtures."""

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vlm_complementary_rank.loss import forward_pair_loss, loss_from_logits
from vlm_fp32_adapter.loss import answer_loss


# Tiny synthetic vocabulary IDs; production IDs come from the local tokenizer.
YES, NO, EOS, VOCAB = 2, 3, 9, 11


class TinyActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(VOCAB, 7)
        self.head = torch.nn.Linear(7, VOCAB, bias=False)
        self.forward_calls = 0

    def get_output_embeddings(self):
        return self.head

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        assert use_cache is False
        self.forward_calls += 1
        return SimpleNamespace(logits=self.head(self.embedding(input_ids)))


def paired_labels():
    # Different prefix/answer lengths also exercise left padding and tail masks.
    return torch.tensor([[-100, -100, -100, YES, EOS],
                         [-100, -100, NO, 7, EOS]])


def example_batch():
    return {'input_ids': torch.tensor([[0, 4, 6, YES, EOS], [5, 6, NO, 7, EOS]]),
            'attention_mask': torch.tensor([[0, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
            'labels': paired_labels()}


def test_causal_shift_mask_and_explicit_tail_alignment():
    torch.manual_seed(1)
    labels = paired_labels()
    logits = torch.randn(2, 5, VOCAB, requires_grad=True)
    dense = loss_from_logits(logits, labels, [(0, 1)], YES, NO)
    tail = loss_from_logits(logits[:, 1:], labels, [(0, 1)], YES, NO, logits_start=1)
    manual = torch.stack([
        F.cross_entropy(logits[0, 2:4].float(), labels[0, 3:5]),
        F.cross_entropy(logits[1, 1:4].float(), labels[1, 2:5]),
    ])
    torch.testing.assert_close(dense['qa_ce'], manual, rtol=0, atol=0)
    for name in ['qa_ce', 'pair_rank', 'first_token_margin']:
        torch.testing.assert_close(dense[name], tail[name], rtol=0, atol=0)
    # A flat FP32 reduction including different numbers of ignored zero terms
    # can differ by one ULP; each per-QA reduction above remains exactly equal.
    torch.testing.assert_close(dense['token_ce_sum'], tail['token_ce_sum'], rtol=1e-7, atol=0)
    expected_margins = torch.stack([logits[0, 2, YES] - logits[0, 2, NO],
                                    logits[1, 1, YES] - logits[1, 1, NO]])
    torch.testing.assert_close(dense['first_token_margin'], expected_margins)
    assert dense['supervised_token_count'] == 5
    with pytest.raises(ValueError, match='explicitly aligned suffix'):
        loss_from_logits(logits[:, 1:], labels, [(0, 1)], YES, NO)


@pytest.mark.parametrize('tail', [False, True])
def test_lambda_zero_matches_frozen_ce_and_parameter_gradients_with_one_forward(tail):
    torch.manual_seed(2)
    original = TinyActor()
    candidate = copy.deepcopy(original)
    batch = example_batch()
    ce, token_sum, count = answer_loss(original, batch, tail=tail)
    result = forward_pair_loss(candidate, batch, [(0, 1)], YES, NO, tail=tail)
    torch.testing.assert_close(result['qa_ce'], ce, rtol=0, atol=0)
    torch.testing.assert_close(result['token_ce_sum'], token_sum, rtol=0, atol=0)
    assert result['supervised_token_count'] == count
    ce.mean().backward()
    (result['qa_ce'].mean() + 0.0 * result['pair_rank'].mean()).backward()
    for expected, actual in zip(original.parameters(), candidate.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)
    assert original.forward_calls == candidate.forward_calls == 1
    assert not candidate.head._forward_pre_hooks


def test_pair_loss_and_its_bias_gradient_are_invariant_to_a_common_yes_bias():
    torch.manual_seed(3)
    labels = paired_labels()
    logits = torch.randn(2, 5, VOCAB)
    base = loss_from_logits(logits, labels, [(0, 1)], YES, NO)['pair_rank']
    common_bias = torch.tensor(7.0, requires_grad=True)
    yes_axis = F.one_hot(torch.tensor(YES), VOCAB).to(logits.dtype)
    shifted = loss_from_logits(logits + common_bias * yes_axis, labels, [(0, 1)], YES, NO)
    torch.testing.assert_close(shifted['pair_rank'], base, rtol=1e-6, atol=1e-6)
    shifted['pair_rank'].sum().backward()
    torch.testing.assert_close(common_bias.grad, torch.tensor(0.0), rtol=0, atol=0)


def test_rank_gradient_pushes_opposite_images_in_opposite_directions():
    labels = paired_labels()
    logits = torch.zeros(2, 5, VOCAB, requires_grad=True)
    result = loss_from_logits(logits, labels, [(0, 1)], YES, NO)
    result['pair_rank'].sum().backward()
    expected = torch.zeros_like(logits)
    expected[0, 2, YES], expected[0, 2, NO] = -0.5, 0.5
    expected[1, 1, YES], expected[1, 1, NO] = 0.5, -0.5
    torch.testing.assert_close(logits.grad, expected, rtol=0, atol=0)


def test_empty_pairs_support_nonbinary_text_targets_and_zero_rank_gradient():
    labels = torch.tensor([[-100, -100, 6, EOS], [-100, -100, 8, EOS]])
    logits = torch.randn(2, 4, VOCAB, requires_grad=True)
    result = loss_from_logits(logits, labels, [], YES, NO)
    assert result['qa_ce'].shape == (2,)
    assert result['pair_rank'].shape == (0,)
    assert result['first_target_token_ids'].tolist() == [6, 8]
    result['pair_rank'].sum().backward(retain_graph=True)
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits), rtol=0, atol=0)
    logits.grad = None
    result['qa_ce'].mean().backward()
    assert bool(torch.isfinite(logits.grad).all())
    assert bool(logits.grad.ne(0).any())


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_prediction_logits_fail(bad):
    logits = torch.zeros(2, 5, VOCAB)
    logits[0, 2, YES] = bad
    with pytest.raises(ValueError, match='Non-finite'):
        loss_from_logits(logits, paired_labels(), [(0, 1)], YES, NO)


@pytest.mark.parametrize('pairs', [[(1, 0)], [(0, 0)], [(0, 2)], [(0.0, 1.0)]])
def test_invalid_pair_order_bounds_or_type_fail(pairs):
    with pytest.raises(ValueError):
        loss_from_logits(torch.zeros(2, 5, VOCAB), paired_labels(), pairs, YES, NO)


def test_tail_requires_contiguous_assistant_targets_before_forward():
    actor = TinyActor()
    batch = example_batch()
    batch['labels'][0] = torch.tensor([-100, YES, -100, -100, EOS])
    with pytest.raises(ValueError, match='contiguous assistant'):
        forward_pair_loss(actor, batch, [(0, 1)], YES, NO)
    assert actor.forward_calls == 0


def test_lost_causal_target_and_all_ignored_row_fail():
    logits = torch.zeros(2, 5, VOCAB)
    labels = paired_labels()
    labels[0, 0] = YES
    with pytest.raises(ValueError, match='lost assistant targets'):
        loss_from_logits(logits, labels, [(0, 1)], YES, NO)
    labels[0] = -100
    with pytest.raises(ValueError, match='Every question'):
        loss_from_logits(logits, labels, [], YES, NO)
