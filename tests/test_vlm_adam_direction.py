from copy import deepcopy

import torch

from vlm_gradient_mechanism_v1.adam import flatten_parameters, moment_delta, projection, shadow_delta


def state_after_one_step():
    p = torch.nn.Parameter(torch.tensor([.25, -.4, .1]))
    before = p.detach().clone()
    optimizer = torch.optim.AdamW([p], lr=.001, weight_decay=0., foreach=False)
    p.grad = torch.tensor([.2, -.3, .05])
    optimizer.step()
    return {'trainable_parameters': {'p': p.detach().clone()},
            'optimizer': deepcopy(optimizer.state_dict())}, before


def test_post_moments_reconstruct_recorded_step_up_to_fp32_parameter_rounding():
    state, before = state_after_one_step()
    predicted = moment_delta(state)
    observed = flatten_parameters(state) - before.double()
    torch.testing.assert_close(predicted, observed, atol=2e-8, rtol=2e-5)


def test_cpu_shadow_does_not_modify_original_parameters_or_moments():
    state, _ = state_after_one_step()
    before = deepcopy(state)
    delta, receipt = shadow_delta(state, torch.tensor([-.2, .7, -.1], dtype=torch.float64))
    torch.testing.assert_close(state['trainable_parameters']['p'], before['trainable_parameters']['p'], rtol=0, atol=0)
    for key in ('step', 'exp_avg', 'exp_avg_sq'):
        torch.testing.assert_close(state['optimizer']['state'][0][key], before['optimizer']['state'][0][key], rtol=0, atol=0)
    assert delta.numel() == 3 and receipt['trained_model_updates'] == 0


def test_momentum_can_reverse_a_raw_gradient_descent_interpretation():
    p = torch.nn.Parameter(torch.tensor([.1]))
    optimizer = torch.optim.AdamW([p], lr=.001, weight_decay=0.)
    p.grad = torch.tensor([10.])
    optimizer.step()
    state = {'trainable_parameters': {'p': p.detach().clone()}, 'optimizer': deepcopy(optimizer.state_dict())}
    # Probe grad +1; raw next gradient -1 predicts damage under SGD, but old
    # positive momentum still moves the copied parameter down under Adam.
    g = torch.tensor([-1.], dtype=torch.float64)
    assert projection(torch.ones(1), -g)['local_first_order'] == 'loss_up'
    delta, _ = shadow_delta(state, g)
    assert projection(torch.ones(1), delta)['local_first_order'] == 'loss_down'
