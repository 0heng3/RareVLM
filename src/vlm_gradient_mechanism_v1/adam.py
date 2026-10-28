"""CPU-only analysis of immutable optimizer states; no live model is updated."""

import copy

import torch


def flatten_parameters(state):
    return torch.cat([p.detach().cpu().double().reshape(-1)
                      for p in state['trainable_parameters'].values()])


def moment_delta(state):
    """FP64 ideal AdamW increment derived from post-step moments, not FP32 rounding."""
    groups = state['optimizer']['param_groups']
    if len(groups) != 1:
        raise ValueError('This diagnostic is scoped to one historical optimizer group')
    group = groups[0]
    if group['amsgrad'] or group['maximize'] or group['weight_decay'] != 0:
        raise ValueError('Historical study used non-AMSGrad, non-maximize, zero-decay AdamW')
    values = list(state['trainable_parameters'].values())
    if len(group['params']) != len(values):
        raise ValueError('Optimizer coordinate count differs from saved named parameters')
    beta1, beta2 = group['betas']
    parts = []
    for param, pid in zip(values, group['params']):
        item = state['optimizer']['state'][pid]
        if item['exp_avg'].shape != param.shape or item['exp_avg_sq'].shape != param.shape:
            raise ValueError('Optimizer parameter shape/order differs')
        step = float(item['step'])
        if step <= 0:
            raise ValueError('Need a post-update moment state')
        m = item['exp_avg'].double() / (1 - beta1 ** step)
        v = item['exp_avg_sq'].double() / (1 - beta2 ** step)
        delta = -group['lr'] * (m / (v.sqrt() + group['eps'])
                                + group['weight_decay'] * param.double())
        parts.append(delta.reshape(-1))
    result = torch.cat(parts)
    if not bool(torch.isfinite(result).all()):
        raise ValueError('Non-finite moment-derived direction')
    return result


def shadow_delta(state, gradient, clip_norm=1.):
    """One counterfactual installed-AdamW step on independent CPU copies only.

    Uses the saved last LR as a declared comparison scale, not a new schedule.
    CPU kernels need not be bit-identical to historical GPU foreach kernels.
    """
    original = state['trainable_parameters']
    params = [torch.nn.Parameter(p.detach().cpu().clone()) for p in original.values()]
    if gradient.numel() != sum(p.numel() for p in params):
        raise ValueError('Gradient coordinates differ')
    if any(p.dtype != torch.float32 for p in params):
        raise ValueError('Historical trainable masters must be FP32')
    optimizer = torch.optim.AdamW(params)
    optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
    if any(g['capturable'] or g['differentiable'] or g.get('fused') for g in optimizer.param_groups):
        raise ValueError('CPU shadow is scoped to the recorded noncapturable optimizer')
    offset = 0
    for p in params:
        p.grad = gradient[offset:offset + p.numel()].reshape(p.shape).to(dtype=p.dtype).clone()
        offset += p.numel()
    before_norm = float(torch.nn.utils.clip_grad_norm_(params, clip_norm))
    optimizer.step()
    delta = torch.cat([(p.detach().double() - before.double()).reshape(-1)
                       for p, before in zip(params, original.values())])
    return delta, {'copied_cpu_optimizer_steps': 1, 'trained_model_updates': 0,
                   'gradient_norm_before_clip': before_norm,
                   'clip_coefficient': min(1., clip_norm / (before_norm + 1e-6)),
                   'lr': [g['lr'] for g in optimizer.param_groups],
                   'changed_master_coordinates': int(delta.ne(0).sum()),
                   'coordinates': delta.numel()}


def projection(gradient, displacement):
    g, d = gradient.double(), displacement.double()
    if g.ndim != 1 or g.shape != d.shape or not bool(torch.isfinite(g).all() and torch.isfinite(d).all()):
        raise ValueError('Need matching finite flat gradient and displacement')
    dot = float(torch.dot(g, d))
    norm_g, norm_d = float(g.norm()), float(d.norm())
    return {'gradient_dot_parameter_displacement': dot,
            'first_order_loss_change': dot,
            'cosine_with_parameter_displacement': dot / norm_g / norm_d if norm_g and norm_d else None,
            'displacement_norm': norm_d,
            'local_first_order': 'loss_down' if dot < 0 else 'loss_up' if dot > 0 else 'stationary'}
