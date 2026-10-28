"""CPU vectors for the source-group diagnostic, without model loading."""

import json

import pytest
import torch

from vlm_gradient_mechanism_v1.metrics import (
    GradientSnapshot, capture_gradients, compare_gradients, compose_gradients,
    qa_mean_weights, summarize_gradients,
)


def snapshot(values):
    parameter = torch.nn.Parameter(torch.zeros(len(values), dtype=torch.float32))
    return capture_gradients([('lora', parameter)], [torch.tensor(values, dtype=torch.float32)])


def test_snapshots_preserve_real_parameter_grad_storage_values_and_missing_coordinates():
    a = torch.nn.Parameter(torch.tensor([1., 2.]))
    b = torch.nn.Parameter(torch.tensor([3.]))
    frozen = torch.nn.Parameter(torch.tensor([4.]), requires_grad=False)
    a.grad = torch.tensor([3., -4.])
    original_ptr = a.grad.data_ptr()
    original = a.grad.clone()
    actual = capture_gradients([('a', a), ('frozen', frozen), ('b', b)])
    assert actual.parameter_names == ('a', 'b')
    assert actual.parameter_shapes == ((2,), (1,))
    assert actual.missing_gradients == ('b',)
    assert actual.values.dtype == torch.float64 and actual.values.device.type == 'cpu'
    assert actual.values.tolist() == [3., -4., 0.]
    actual.values.add_(100)
    assert a.grad.data_ptr() == original_ptr and torch.equal(a.grad, original)
    assert b.grad is None and frozen.grad is None


def test_external_autograd_gradients_do_not_assign_parameter_grads():
    p = torch.nn.Parameter(torch.tensor([2., -3.]))
    gradients = torch.autograd.grad(p.square().sum(), [p])
    actual = capture_gradients([('p', p)], gradients)
    assert actual.values.tolist() == [4., -6.]
    assert p.grad is None
    assert torch.equal(p.detach(), torch.tensor([2., -3.]))


def test_fp64_accumulation_prevents_fp32_squared_norm_overflow():
    a = snapshot([2e20, 2e20])
    result = compare_gradients(a, a)
    assert result['dot'] == pytest.approx(8e40, rel=1e-6)
    assert result['cosine'] == pytest.approx(1.)


def test_conflict_and_zero_norm_are_not_inferred_from_gradient_magnitude():
    small, large, opposed, zero = snapshot([1., 0.]), snapshot([100., 0.]), snapshot([-2., 0.]), snapshot([0., 0.])
    assert not compare_gradients(small, large)['negative_dot_conflict']
    assert compare_gradients(small, opposed)['cosine'] == -1
    assert compare_gradients(small, opposed)['negative_dot_conflict']
    result = compare_gradients(small, zero)
    assert result['cosine'] is None and result['dot'] == 0
    assert not result['negative_dot_conflict']
    json.dumps(result, allow_nan=False)


def test_actual_qa_weights_and_answer_format_coefficients_are_not_renormalized():
    answer, endings = snapshot([6., 0.]), snapshot([0., 3.])
    human_qa = compose_gradients({'answer': answer, 'format_mean': endings},
                                 {'answer': 1/3, 'format_mean': 2/3})
    assert human_qa.values.tolist() == [2., 2.]
    # One such QA in a 16-QA update has first-answer coefficient 1/48.
    update = compose_gradients({'human': human_qa}, qa_mean_weights({'human': 1}, 16))
    assert update.values.tolist() == [.125, .125]
    assert qa_mean_weights({'no': 4, 'yes': 4, 'other': 0, 'replay': 8}, 16) == {
        'no': .25, 'yes': .25, 'other': 0., 'replay': .5}


def test_pairwise_conflict_does_not_imply_net_objective_harms_the_group():
    groups = {'no': snapshot([2., 0.]), 'replay': snapshot([-1., 0.])}
    results = summarize_gradients(groups, {'no': .5, 'replay': .5})
    assert results['pairwise']['no']['replay']['negative_dot_conflict']
    no = results['groups']['no']
    assert no['weighted_source_dots'] == {'no': 2., 'replay': -1.}
    assert no['dot_with_objective'] == 1.
    assert no['first_order_loss_change_per_unit_lr'] == -1.
    assert not no['negative_dot_conflict_with_objective']
    json.dumps(results, allow_nan=False)


def test_first_order_sign_matches_a_real_tiny_sgd_direction():
    p = torch.tensor([.3, -.2], dtype=torch.float64)
    target = torch.tensor([1., .5], dtype=torch.float64)
    left = snapshot((p-target).tolist())
    right = snapshot([.4, -.9])
    result = compare_gradients(left, right)
    lr = 1e-6
    old = .5*(p-target).square().sum()
    new = .5*(p-lr*right.values-target).square().sum()
    assert float((new-old)/lr) == pytest.approx(
        result['first_order_left_loss_change_for_right_descent_per_unit_lr'], abs=1e-6)


@pytest.mark.parametrize('bad', [float('inf'), float('nan')])
def test_nonfinite_gradients_rejected(bad):
    with pytest.raises(ValueError, match='finite'):
        snapshot([bad])


def test_rejects_parameter_dtype_and_gradient_coordinate_mismatches():
    p = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match='FP32'):
        capture_gradients([('p', p)])
    p = torch.nn.Parameter(torch.ones(2))
    with pytest.raises(ValueError, match='shape'):
        capture_gradients([('p', p)], [torch.ones(1)])
    with pytest.raises(ValueError, match='count'):
        capture_gradients([('p', p)], [])
    a = snapshot([1., 2.])
    b = GradientSnapshot(('different_parameter',), ((2,),), a.values.clone())
    with pytest.raises(ValueError, match='coordinates'):
        compare_gradients(a, b)


@pytest.mark.parametrize('weights', [{'a': -1.}, {'a': float('nan')}, {'wrong': 1.}])
def test_rejects_invalid_objective_coefficients(weights):
    with pytest.raises(ValueError):
        compose_gradients({'a': snapshot([1.])}, weights)


@pytest.mark.parametrize('counts,total', [({'a': 17}, 16), ({'a': -1}, 16), ({'a': 1.5}, 16), ({'a': 1}, 0)])
def test_rejects_invalid_qa_denominator_or_counts(counts, total):
    with pytest.raises(ValueError):
        qa_mean_weights(counts, total)
