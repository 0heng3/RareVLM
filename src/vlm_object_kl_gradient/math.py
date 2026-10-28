"""CPU gradient geometry and nonmutating, conditional Adam direction estimates.

These are offline local approximations in the fixed LoRA parameterization.
They are not executed optimizer steps or counterfactual training trajectories.
"""
from __future__ import annotations

import math
import torch


def vector(value, name='gradient'):
    if (not isinstance(value, torch.Tensor) or value.ndim != 1 or value.numel() == 0
            or value.dtype != torch.float32 or value.device.type != 'cpu'
            or value.requires_grad or not bool(torch.isfinite(value).all())):
        raise ValueError(f'{name} must be a detached finite CPU FP32 vector')
    return value


def norm(value):
    return float(torch.linalg.vector_norm(value.double()))


def dot(first, second):
    return float(torch.dot(first.double(), second.double()))


def gradient_geometry(object_gradient, kl_gradient, initial_kl_gradient):
    values = [vector(value) for value in (object_gradient, kl_gradient, initial_kl_gradient)]
    if len({value.shape for value in values}) != 1:
        raise ValueError('gradient vectors must use the same parameter layout')
    object_gradient, kl_gradient, initial_kl_gradient = values
    object_norm, kl_norm, floor = map(norm, values)
    guard = max(10. * floor, 1e-10)
    product = dot(object_gradient, kl_gradient)
    raw_cosine = product / (object_norm * kl_norm) if object_norm and kl_norm else None
    if not object_norm or not kl_norm:
        cosine, reason = None, 'zero_source_gradient_norm'
    elif kl_norm <= guard:
        cosine, reason = None, 'KL_gradient_at_or_below_preregistered_numerical_floor_guard'
    else:
        cosine, reason = raw_cosine, 'above_measured_initial_floor; local_fixed_LoRA_geometry_only'
    return {'object_gradient_l2': object_norm, 'KL_gradient_l2': kl_norm,
        'initial_KL_gradient_floor_l2': floor, 'object_KL_inner_product': product,
        'cosine_floor_guard_l2': guard, 'cosine_floor_guard_rule': 'max(10 * measured_initial_KL_gradient_l2, 1e-10)',
        'KL_to_object_norm_ratio': kl_norm / object_norm if object_norm else None,
        'KL_to_initial_floor_norm_ratio': kl_norm / floor if floor else None,
        'raw_cosine_if_nonzero': raw_cosine, 'interpretable_cosine': cosine,
        'cosine_interpretation': reason,
        'limitations': 'Small loss is not small gradient. Negative cosine is not forgetting causality; a near-floor direction remains numerically uncertain.'}


def optimizer_vectors(state, parameter_names):
    """Map saved optimizer IDs to the checkpoint's original parameter order."""
    names = list(state['trainable_parameters'])
    groups = state['optimizer']['param_groups']
    if names != list(parameter_names) or len(groups) != 1:
        raise ValueError('checkpoint optimizer/parameter layout differs')
    group = groups[0]
    ids = group['params']
    if len(ids) != len(names) or len(set(ids)) != len(ids):
        raise ValueError('checkpoint optimizer parameter ID count differs')
    if (group.get('weight_decay') != 0. or group.get('amsgrad', False)
            or group.get('maximize', False) or group.get('differentiable', False)):
        raise ValueError('shadow supports the fixed weight_decay0 ordinary Adam contract only')
    first, second, steps = [], [], []
    for name, index in zip(names, ids):
        parameter = state['trainable_parameters'][name]
        item = state['optimizer']['state'][index]
        for field, result in (('exp_avg', first), ('exp_avg_sq', second)):
            value = item[field]
            if value.dtype != torch.float32 or value.shape != parameter.shape or not bool(torch.isfinite(value).all()):
                raise ValueError('checkpoint Adam moments must be finite FP32 and match parameter shapes')
            result.append(value.detach().cpu().reshape(-1))
        step = float(item['step'])
        if not math.isfinite(step) or int(step) != step:
            raise ValueError('checkpoint Adam step must be an integer')
        steps.append(int(step))
    if not steps or len(set(steps)) != 1 or steps[0] != state['completed_updates']:
        raise ValueError('Adam steps differ from checkpoint completed updates')
    return torch.cat(first).clone(), torch.cat(second).clone(), steps[0], group


def shadow_direction(gradient, first_moment, second_moment, completed_steps, group, learning_rate, clip=1.):
    """Copy FP32 moments and approximate next Adam direction on CPU, no writes.

    CPU float64 global norm is used for joint clipping (including the PyTorch
    clip denominator epsilon1e-6), followed by FP32 moment arithmetic. GPU
    foreach/reduction rounding can differ: this is not a bitwise step replay.
    """
    gradient, first_moment, second_moment = [vector(v) for v in (gradient, first_moment, second_moment)]
    if len({v.shape for v in (gradient, first_moment, second_moment)}) != 1 or bool((second_moment < 0).any()):
        raise ValueError('invalid Adam vector shapes or second moment')
    if not math.isfinite(learning_rate) or learning_rate <= 0 or clip != 1. or completed_steps < 0:
        raise ValueError('fixed positive next LR, step and joint clip1 required')
    beta1, beta2 = group['betas']
    epsilon = group['eps']
    if not (0 <= beta1 < 1 and 0 <= beta2 < 1 and math.isfinite(epsilon) and epsilon > 0):
        raise ValueError('invalid saved Adam hyperparameters')
    coefficient = min(1., clip / (norm(gradient) + 1e-6))
    clipped = gradient.clone() * coefficient
    first = first_moment.clone() * beta1 + clipped * (1. - beta1)
    second = second_moment.clone() * beta2 + clipped.square() * (1. - beta2)
    step = completed_steps + 1
    direction = -learning_rate * (first / (1. - beta1 ** step)) / ((second / (1. - beta2 ** step)).sqrt() + epsilon)
    if not bool(torch.isfinite(direction).all()):
        raise ValueError('nonfinite shadow Adam direction')
    return direction, coefficient


def adam_shadow_summary(object_gradient, kl_gradient, state, parameter_names, learning_rate):
    object_gradient, kl_gradient = vector(object_gradient), vector(kl_gradient)
    first, second, step, group = optimizer_vectors(state, parameter_names)
    if first.shape != object_gradient.shape or kl_gradient.shape != object_gradient.shape:
        raise ValueError('gradient and saved Adam coordinate counts differ')
    joint, joint_clip = shadow_direction(object_gradient + kl_gradient, first, second, step, group, learning_rate)
    only, only_clip = shadow_direction(object_gradient, first, second, step, group, learning_rate)
    def summary(direction, coefficient):
        return {'delta_theta_l2': norm(direction), 'joint_clip_coefficient': coefficient,
            'object_gradient_dot_delta_theta': dot(object_gradient, direction),
            'KL_gradient_dot_delta_theta': dot(kl_gradient, direction)}
    difference = norm(joint - only)
    return {'checkpoint_completed_updates': step, 'next_update_one_based': step + 1,
        'next_learning_rate': learning_rate, 'saved_betas': list(group['betas']), 'saved_eps': group['eps'],
        'weight_decay': group['weight_decay'], 'clip_norm_cap': 1.,
        'with_current_KL': summary(joint, joint_clip), 'without_current_KL_same_history': summary(only, only_clip),
        'current_KL_marginal_direction_difference_l2': difference,
        'current_KL_marginal_difference_over_joint_norm': difference / norm(joint) if norm(joint) else None,
        'method': 'CPU FP32 copied moments, float64 clipping norm; conditional local approximate direction, no parameter writes',
        'limitations': 'Both directions condition on the actual same historical m/v. First-order g·delta is not an executed post-update loss, full counterfactual training or forgetting causality.'}
