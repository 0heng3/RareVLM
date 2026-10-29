import pytest
import torch
import torch.nn.functional as F
from vlm_full_teacher_kd_v1.loss import objectives


def inputs():
    torch.manual_seed(42)
    logits = torch.randn(1, 4, 6, requires_grad=True)
    labels = torch.tensor([[-100, 4, 2, 5]])
    ids = torch.tensor([0, 2, 4])
    cache = {'labels': torch.tensor([4, 2]), 'logits': torch.tensor([[.2, -.3, 1.], [.8, .3, -.4]])}
    return logits, labels, ids, cache


def test_real_KD_gradient_matches_temperature_formula_and_excludes_padding_and_EOS():
    logits, labels, ids, cache = inputs()
    _, aux = objectives(logits, labels, ids, [cache], 'kd', 5)
    gradient = torch.autograd.grad(aux.sum(), logits)[0]
    ps = (logits.detach()[0, :2].index_select(-1, ids) / 2).softmax(-1)
    pt = (cache['logits'] / 2).softmax(-1)
    assert torch.allclose(gradient[0, :2].index_select(-1, ids), 2 * (ps - pt) / 2, atol=1e-6)
    assert torch.count_nonzero(gradient[0, :, [1, 3, 5]]) == 0
    assert torch.count_nonzero(gradient[0, 2:]) == 0


def test_base_CE_keeps_real_EOS_and_all_valid_assistant_labels():
    logits, labels, ids, cache = inputs()
    ce, aux = objectives(logits, labels, ids, [None], 'hard', 5)
    expected = F.cross_entropy(logits[0, :3], labels[0, 1:])
    assert torch.allclose(ce[0], expected) and aux[0] == 0


def test_uniform_tail_keeps_teacher_GT_mass_but_changes_non_GT_structure():
    logits, labels, ids, cache = inputs()
    _, actual = objectives(logits, labels, ids, [cache], 'uniform', 5)
    pt = (cache['logits'] / 2).softmax(-1)
    index = torch.tensor([2, 1]); gt = pt.gather(1, index[:, None])
    q = ((1 - gt) / 2).expand_as(pt).clone(); q.scatter_(1, index[:, None], gt)
    expected = 4 * F.kl_div((logits[0, :2].index_select(-1, ids) / 2).log_softmax(-1), q, reduction='batchmean')
    assert torch.allclose(actual[0], expected)
    assert torch.equal(q.gather(1, index[:, None]), gt) and torch.allclose(q.sum(-1), torch.ones(2))


def test_teacher_cache_label_mismatch_is_rejected_before_backward():
    logits, labels, ids, cache = inputs(); cache['labels'] = cache['labels'].flip(0)
    with pytest.raises(ValueError, match='GT-token frame'):
        objectives(logits, labels, ids, [cache], 'kd', 5)
