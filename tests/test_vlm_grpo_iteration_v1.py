import math
import torch
from vlm_grpo_iteration_v1.loss import clipped_group_loss


def test_advantage_updates_actual_action_in_correct_direction():
    logp = torch.zeros(2, 3, requires_grad=True)
    valid = torch.tensor([[1,1,0],[1,1,1]], dtype=torch.bool)
    loss, _ = clipped_group_loss(logp, torch.zeros_like(logp), valid, torch.tensor([1., -1.]))
    loss.backward()
    assert bool((logp.grad[0,:2] < 0).all()) and bool((logp.grad[1] > 0).all())
    assert logp.grad[0,2] == 0


def test_clipping_has_correct_sign_for_positive_and_negative_rewards():
    logp = torch.tensor([[math.log(1.3)], [math.log(.7)]], requires_grad=True)
    loss, stats = clipped_group_loss(logp, torch.zeros_like(logp), torch.ones_like(logp,dtype=torch.bool), torch.tensor([1., -1.]))
    loss.backward()
    assert bool(logp.grad.eq(0).all()) and stats['ratio_clip_fraction'] == 1.


def test_zero_advantage_does_not_train_and_reference_has_no_gradient():
    logp = torch.zeros(2, 2, requires_grad=True)
    old = torch.zeros_like(logp, requires_grad=True)
    loss, _ = clipped_group_loss(logp, old, torch.ones_like(logp,dtype=torch.bool), torch.zeros(2))
    loss.backward()
    assert bool(logp.grad.eq(0).all()) and old.grad is None


def test_sequence_normalization_does_not_reward_longer_answers():
    logp = torch.zeros(2, 4, requires_grad=True)
    valid = torch.tensor([[1,0,0,0],[1,1,1,1]],dtype=torch.bool)
    loss, _ = clipped_group_loss(logp, torch.zeros_like(logp), valid, torch.ones(2))
    loss.backward()
    torch.testing.assert_close(logp.grad[0].sum(),logp.grad[1].sum())
