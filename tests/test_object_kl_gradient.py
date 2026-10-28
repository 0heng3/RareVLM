"""Small CPU contracts for independent sources, numerical floors and Adam."""
from copy import deepcopy
from types import SimpleNamespace
import math
import random

import numpy as np
import pytest
import torch

from vlm_object_kl_gradient.math import (adam_shadow_summary, gradient_geometry,
    optimizer_vectors, shadow_direction)
from vlm_object_kl_gradient.probe import capture_gradient, validate_config
from vlm_object_kl.train import rng_hash, rng_state, trainable_hash
from vlm_object_train.loss import normalized_qa_loss


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen = torch.nn.Parameter(torch.tensor([.7, -.2]), requires_grad=False)
        self.lora = torch.nn.Parameter(torch.tensor([.1, .3]))
        self.register_buffer('temporary', torch.zeros(1))
        self.rope_deltas = None


def test_separate_gradients_equal_joint_QA_objective_and_restore_all_state():
    """Variable answer lengths and tail8 catch token/microbatch denominators."""
    torch.manual_seed(17); random.seed(17); np.random.seed(17)
    model = Tiny()
    initial = deepcopy(model.state_dict())
    before, random_before = trainable_hash(model), rng_hash(rng_state())
    object_targets = torch.linspace(-.5, .7, 8)
    replay_targets = torch.linspace(.4, -.3, 8)
    def objective(parameter, targets):
        values = []
        for index, target in enumerate(targets):
            tokens = torch.stack([(parameter[0] + parameter[1] * j - target) ** 2 for j in range(index % 3 + 1)])
            values.append(tokens.mean())
        return torch.stack(values)
    def callback(targets, source):
        def run():
            model.temporary.add_(1)
            model.rope_deltas = torch.tensor([42])
            torch.rand(2); random.random(); np.random.random()
            per_qa = objective(model.lora, targets)
            for start in (0, 4):
                normalized_qa_loss(per_qa[start:start+4], source, 8, 8).backward(retain_graph=start == 0)
            return {'questions': 8}
        return run
    go, _, _ = capture_gradient(model, callback(object_targets, 'object'), torch.device('cpu'))
    gk, _, _ = capture_gradient(model, callback(replay_targets, 'replay'), torch.device('cpu'))
    reference = model.lora.detach().clone().requires_grad_()
    combined = objective(reference, object_targets).mean() + objective(reference, replay_targets).mean()
    combined.backward()
    assert torch.allclose(go + gk, reference.grad, atol=2e-7)
    assert trainable_hash(model) == before and rng_hash(rng_state()) == random_before
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in initial.items())
    assert model.rope_deltas is None and all(p.grad is None for p in model.parameters())


def test_zero_and_near_floor_cosines_are_not_interpreted_but_raw_values_retained():
    go = torch.tensor([1., 0.])
    floor = torch.tensor([.01, 0.])
    close = gradient_geometry(go, torch.tensor([-.0101, 0.]), floor)
    assert close['raw_cosine_if_nonzero'] == pytest.approx(-1.)
    assert close['interpretable_cosine'] is None
    assert close['cosine_floor_guard_l2'] == pytest.approx(.1)
    below_absolute = gradient_geometry(go, torch.tensor([0., 5e-11]), torch.zeros(2))
    assert below_absolute['interpretable_cosine'] is None
    assert below_absolute['raw_cosine_if_nonzero'] == pytest.approx(0.)
    zero = gradient_geometry(torch.zeros(2), torch.tensor([1., 0.]), floor)
    assert zero['raw_cosine_if_nonzero'] is None and zero['KL_to_object_norm_ratio'] is None
    strong = gradient_geometry(go, torch.tensor([-1., 0.]), floor)
    assert strong['interpretable_cosine'] == pytest.approx(-1.)


def test_Adam_shadow_is_nonmutating_and_raw_conflict_does_not_determine_direction():
    state = {'completed_updates': 25, 'trainable_parameters': {'lora': torch.zeros(2)},
        'optimizer': {'param_groups': [{'params': [0], 'betas': (.9, .999), 'eps': 1e-8,
            'weight_decay': 0., 'amsgrad': False, 'maximize': False}],
            'state': {0: {'step': torch.tensor(25.), 'exp_avg': torch.tensor([-3., 0.]),
                          'exp_avg_sq': torch.tensor([1., 1.])}}}}
    before = deepcopy(state)
    go, gk = torch.tensor([1., 0.]), torch.tensor([-.1, 0.])
    result = adam_shadow_summary(go, gk, state, ['lora'], 2.5e-6)
    assert gradient_geometry(go, gk, torch.zeros(2))['raw_cosine_if_nonzero'] == pytest.approx(-1.)
    # Historical negative first moment makes the actual approximate direction
    # positive, so KL's first-order change is negative despite raw conflict.
    assert result['with_current_KL']['object_gradient_dot_delta_theta'] > 0
    assert result['with_current_KL']['KL_gradient_dot_delta_theta'] < 0
    first, second, step, group = optimizer_vectors(state, ['lora'])
    direction, clip = shadow_direction(torch.tensor([3., 0.]), first, second, step, group, 2.5e-6)
    expected_clip = 1. / (3. + 1e-6)
    expected_g = 3. * expected_clip
    expected_m = .9 * (-3.) + .1 * expected_g
    expected_v = .999 * 1. + .001 * expected_g ** 2
    expected_delta = -2.5e-6 * (expected_m / (1.-.9**26)) / (math.sqrt(expected_v / (1.-.999**26)) + 1e-8)
    assert clip == pytest.approx(expected_clip)
    assert float(direction[0]) == pytest.approx(expected_delta, rel=2e-6, abs=1e-12)
    for key in ('step', 'exp_avg', 'exp_avg_sq'):
        assert torch.equal(state['optimizer']['state'][0][key], before['optimizer']['state'][0][key])
    assert torch.equal(state['trainable_parameters']['lora'], before['trainable_parameters']['lora'])
    assert state['optimizer']['param_groups'] == before['optimizer']['param_groups']
    with pytest.raises(ValueError, match='layout'):
        optimizer_vectors(state, ['wrong_coordinate_order'])
