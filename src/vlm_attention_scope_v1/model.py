"""Expand an existing q/v adapter with zero-output k/o factors, preserving q/v."""

from copy import deepcopy
import math
import re


ROLES = ('qv', 'qkvo')


def expand_qv(state, config, seed=42):
    import torch

    if (set(config['target_modules']) != {'q_proj', 'v_proj'}
            or config.get('rank_pattern') or config.get('alpha_pattern')
            or config.get('use_dora') or config.get('use_rslora')
            or config.get('bias', 'none') != 'none'):
        raise ValueError('This initializer is scoped to ordinary, uniform-rank q/v LoRA')
    pattern = re.compile(r'(.+\.layers\.\d+\.self_attn\.)(q_proj|v_proj)\.(lora_[AB])\.weight$')
    parsed = [pattern.fullmatch(name) for name in state]
    if any(match is None for match in parsed) or any(v.dtype != torch.float32 for v in state.values()):
        raise ValueError('Expected canonical FP32 language q/v adapter tensors')
    prefixes = sorted({match.group(1) for match in parsed})
    expanded = {name: value.clone() for name, value in state.items()}
    rng = torch.Generator(device='cpu').manual_seed(seed)
    added = []
    for prefix in prefixes:
        qa, qb, va, vb = [state[prefix + projection + '.lora_' + factor + '.weight']
                          for projection, factor in [('q_proj', 'A'), ('q_proj', 'B'), ('v_proj', 'A'), ('v_proj', 'B')]]
        rank = config['r']
        if (qa.shape[0] != rank or va.shape != qa.shape
                or qb.shape[1] != rank or vb.shape[1] != rank):
            raise ValueError('Projection dimensions differ from declared rank')
        for projection, a_shape, b_shape in [
            ('k_proj', tuple(va.shape), tuple(vb.shape)),
            ('o_proj', (rank, qb.shape[0]), (qa.shape[1], rank)),
        ]:
            ak, bk = [prefix + projection + '.lora_' + factor + '.weight' for factor in ('A', 'B')]
            expanded[ak] = torch.empty(a_shape, dtype=torch.float32).uniform_(
                -1 / math.sqrt(a_shape[1]), 1 / math.sqrt(a_shape[1]), generator=rng)
            expanded[bk] = torch.zeros(b_shape, dtype=torch.float32)
            added.extend([ak, bk])
    new_config = {**deepcopy(config), 'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                  'lora_dropout': 0., 'inference_mode': True}
    receipt = {'original_tensors': len(state), 'added_tensors': len(added),
               'original_parameters': sum(v.numel() for v in state.values()),
               'total_adapter_parameters': sum(v.numel() for v in expanded.values()),
               'new_factor_seed': seed, 'zero_output_initialization': True,
               'new_A_initialization': 'CPU seeded uniform[-1/sqrt(fan_in),1/sqrt(fan_in)] as kaiming_uniform(a=sqrt(5))',
               'qv_values_preserved': all(torch.equal(state[name], expanded[name]) for name in state),
               'new_B_all_zero': all(not bool(expanded[name].ne(0).any()) for name in added if '.lora_B.' in name)}
    return expanded, new_config, receipt


def apply_scope(model, role):
    if role not in ROLES:
        raise ValueError('Unknown attention trainable scope')
    allowed = {'q_proj', 'v_proj'} if role == 'qv' else {'q_proj', 'k_proj', 'v_proj', 'o_proj'}
    counts, tensors = {}, {}
    for name, parameter in model.named_parameters():
        if 'lora_' not in name:
            if parameter.requires_grad:
                raise ValueError('Base weights must remain frozen')
            continue
        matches = [p for p in ('q_proj', 'k_proj', 'v_proj', 'o_proj') if '.' + p + '.' in name]
        if len(matches) != 1 or 'visual' in name:
            raise ValueError('Unexpected adapter placement')
        projection = matches[0]
        parameter.requires_grad_(projection in allowed)
        if parameter.requires_grad:
            counts[projection] = counts.get(projection, 0) + parameter.numel()
            tensors[projection] = tensors.get(projection, 0) + 1
    return {'role': role, 'trainable_parameters_by_projection': counts,
            'trainable_tensors_by_projection': tensors,
            'trainable_parameters': sum(counts.values()), 'trainable_tensors': sum(tensors.values()),
            'common_graph_targets': ['q_proj', 'k_proj', 'v_proj', 'o_proj']}
