import copy
import numpy as np
import torch
import torch.nn.functional as F
from clip_retrieval.train import DualAdapter, contrastive_loss
from clip_retrieval_distill.loss import contrastive_distill_loss
from clip_retrieval_shuffle.loss import negative_permutation, teacher_targets, target_kl, contrastive_shuffle_loss


def test_negative_bijection_positive_fixed_and_entropy_preserved():
    positive = torch.tensor([[True, False, True, False, False], [False, True, False, False, True]])
    teacher = torch.tensor([[2., 1., 0., -1., -2.], [-1., 2., 0., 3., -2.]], dtype=torch.float64, requires_grad=True)
    rng = torch.Generator().manual_seed(42)
    shuffled, mapping = teacher_targets(teacher, positive, rng)
    original = teacher.detach().log_softmax(1)
    columns = torch.arange(5).expand(2, -1)
    assert torch.equal(mapping[positive], columns[positive])
    assert torch.equal(mapping.sort(1).values, columns)
    assert torch.equal(shuffled[positive], original[positive])
    assert torch.equal(shuffled.sort(1).values, original.sort(1).values)
    assert torch.allclose(shuffled.exp().sum(1), torch.ones(2, dtype=torch.float64))
    def entropy(p): return -(p * p.clamp_min(1e-300).log()).sum(1)
    gt = positive.double() / positive.sum(1)[:, None]
    assert torch.allclose(entropy(original.exp()), entropy(shuffled.exp()), atol=1e-12)
    assert torch.allclose(entropy((gt+original.exp())/2), entropy((gt+shuffled.exp())/2), atol=1e-12)
    assert not torch.equal(shuffled, original)


def test_rng_is_independent_reproducible_and_edge_masks_supported():
    torch.manual_seed(123)
    before = torch.get_rng_state().clone()
    mask = torch.tensor([[True, True, True], [False, True, True], [False, False, False]])
    a = negative_permutation(mask, torch.Generator().manual_seed(55))
    b = negative_permutation(mask, torch.Generator().manual_seed(55))
    assert torch.equal(a, b)
    assert torch.equal(torch.get_rng_state(), before)
    assert torch.equal(a[:2], torch.arange(3).expand(2, -1))


def inputs():
    torch.manual_seed(7)
    model = DualAdapter().double()
    with torch.no_grad():
        model.image.fc2.weight.normal_(0, .01); model.text.fc2.weight.normal_(0, .01)
    images = F.normalize(torch.randn(8, 512, dtype=torch.float64), dim=-1)
    texts = F.normalize(torch.randn(40, 512, dtype=torch.float64), dim=-1)
    return model, images, texts, np.array([0, 1]), np.arange(2, 8), np.arange(8), .07


def test_no_shuffle_recovers_previous_distillation_and_zero_weight_original_gradients():
    model, *args = inputs()
    for weight, shuffle in [(1., False), (0., True)]:
        a = copy.deepcopy(model); b = copy.deepcopy(model)
        old = contrastive_distill_loss(a, *args, 1.) if weight else contrastive_loss(a, *args)
        new = contrastive_shuffle_loss(b, *args, weight, permutation_seed=42, step=7, shuffle=shuffle)
        assert torch.allclose(old, new, atol=1e-12)
        ga = torch.autograd.grad(old, tuple(a.parameters())); gb = torch.autograd.grad(new, tuple(b.parameters()))
        assert all(torch.allclose(x, y, atol=1e-10, rtol=1e-10) for x, y in zip(ga, gb))


def test_stopped_teacher_and_gradient_uses_shuffled_target():
    teacher = torch.tensor([[3., 2., 1., 0., -1.]], dtype=torch.float64, requires_grad=True)
    student = torch.tensor([[1., 2., 3., 4., 5.]], dtype=torch.float64, requires_grad=True)
    target, _ = teacher_targets(teacher, torch.tensor([[True, False, False, False, False]]), torch.Generator().manual_seed(5))
    grad_s, grad_t = torch.autograd.grad(target_kl(student, target).sum(), (student, teacher), allow_unused=True)
    assert grad_t is None
    assert torch.allclose(grad_s, student.detach().softmax(1)-target.exp(), atol=1e-12)
