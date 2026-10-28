import pytest
import torch

from vlm_adam_conflict_v1.projection import project_displacement


def test_projection_is_closest_feasible_point_in_known_two_dimensional_case():
    delta = torch.tensor([1., -2.], dtype=torch.float64)
    g = {'a': torch.tensor([1., 0.]), 'b': torch.tensor([0., 1.])}
    result, report = project_displacement(delta, g, .05)
    expected = torch.tensor([-.05 * 5**.5, -2.], dtype=torch.float64)
    torch.testing.assert_close(result, expected)
    assert report['active_constraints'] == ['a']


def test_normalized_constraints_are_invariant_to_positive_loss_rescaling():
    delta = torch.tensor([1., 2., -1.])
    a, _ = project_displacement(delta, {'a': torch.tensor([1., 0., 0.]),
                                         'b': torch.tensor([0., 1., 1.])})
    b, _ = project_displacement(delta, {'a': torch.tensor([100., 0., 0.]),
                                         'b': torch.tensor([0., .01, .01])})
    torch.testing.assert_close(a, b)


def test_opposite_gradients_cannot_both_have_strict_descent():
    with pytest.raises(ValueError, match='No feasible'):
        project_displacement(torch.tensor([1., 2.]),
                             {'a': torch.tensor([1., 0.]), 'b': torch.tensor([-1., 0.])})


def test_already_feasible_direction_is_unchanged_and_inputs_are_not_modified():
    delta = torch.tensor([-1., -2.]); original = delta.clone()
    result, report = project_displacement(delta, {'a': torch.tensor([1., 0.])})
    torch.testing.assert_close(result, delta.double())
    assert torch.equal(delta, original) and report['active_constraints'] == []
