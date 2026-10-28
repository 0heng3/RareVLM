"""Remove companion-dependent vision batching without changing learned values."""
from types import MethodType


def install(model, policy):
    import torch
    if policy not in ('imagewise', 'imagewise_fp32_language'): raise ValueError('Unknown numerical policy')
    base = model.get_base_model()
    if getattr(base.visual, '_rarevlm_imagewise', False): raise ValueError('Already installed')
    original = base.visual.forward

    def imagewise_forward(self, hidden_states, grid_thw):
        if grid_thw.ndim != 2 or grid_thw.shape[1] != 3: raise ValueError('Expected ordered image grids')
        lengths = grid_thw.prod(-1).tolist()
        if sum(lengths) != hidden_states.shape[0]: raise ValueError('Image/patch boundary mismatch')
        parts = hidden_states.split(lengths, dim=0)
        # Use the original HF implementation independently, including window
        # reordering, all transformer blocks, merger and inverse permutation.
        return torch.cat([original(part, grid_thw[i:i+1]) for i, part in enumerate(parts)], dim=0)

    base.visual.forward = MethodType(imagewise_forward, base.visual)
    base.visual._rarevlm_imagewise = True
    if policy == 'imagewise_fp32_language': base.model.float()
    return {'policy': policy, 'visual_execution': 'one original HF forward per image grid',
            'visual_storage_and_compute': 'bfloat16', 'language_storage_and_compute': 'float32' if policy.endswith('language') else 'bfloat16',
            'lora_master': 'float32', 'head_compute': 'float32', 'tf32': False,
            'learned_values_changed': False, 'note': 'FP32 language option upcasts the already loaded BF16 base values; no claim of restoring source FP32 tails.'}


def load(model_root, adapter_dir, device, *, policy, training=False, dropout=0., gradient_checkpointing=True):
    from vlm_fp32_readout_v1.model import load as original_load
    model, binding = original_load(model_root, adapter_dir, device, training=training, dropout=dropout,
                                   gradient_checkpointing=gradient_checkpointing)
    return model, {**binding, 'batch_numerical_policy': install(model, policy)}
