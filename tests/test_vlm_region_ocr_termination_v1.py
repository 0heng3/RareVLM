import pytest
import torch
import torch.nn.functional as F
from vlm_region_ocr_termination_v1.loss import reduce_auxiliary


def test_text_gradients_exactly_match_original_denominator_and_tail_has_zero_gradient():
    targets = torch.tensor([[4, 5, 2, 3], [-100, 1, 2, 3]])
    logits = torch.arange(56, dtype=torch.float64).reshape(2, 4, 7).div(30).requires_grad_()
    candidate, plain, kept = reduce_auxiliary(logits, targets, 2)
    expected = F.cross_entropy(logits.float().reshape(-1, 7), targets.flatten(),
                              ignore_index=-100, reduction='none').reshape(2, 4)
    assert torch.equal(plain, expected.sum(1) / torch.tensor([4, 3]))
    assert kept.tolist() == [2, 1]
    full_grad = torch.autograd.grad(plain.sum(), logits, retain_graph=True)[0]
    new_grad = torch.autograd.grad(candidate.sum(), logits)[0]
    text = torch.tensor([[True, True, False, False], [False, True, False, False]])
    assert torch.equal(new_grad[text], full_grad[text])
    assert torch.count_nonzero(new_grad[~text]) == 0


@pytest.mark.parametrize('targets', [[[1, 3, 4]], [[1, 2, 2]], [[2, 3, 4]]])
def test_reject_missing_repeated_or_empty_text_eos_contract(targets):
    with pytest.raises(ValueError):
        reduce_auxiliary(torch.zeros(1, 3, 7), torch.tensor(targets), 2)
