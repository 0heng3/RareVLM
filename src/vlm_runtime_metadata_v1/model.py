"""Synchronize new training adapters' dropout configuration with runtime behavior.

The historical loader and checkpoints remain unchanged.  ``load`` delegates all
weight loading, precision, freezing and checkpointing behavior to the validated
FP32 loader.  In training mode only, it repairs the in-memory PEFT configuration
which ordinary ``save_pretrained`` will serialize into a *new* adapter directory.
This is a reproducibility repair, not evidence of model capability improvement.
"""

import math
from numbers import Real


def _probability(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError('dropout must be a finite real number in [0, 1)')
    probability = float(value)
    if not math.isfinite(probability) or not 0 <= probability < 1:
        raise ValueError('dropout must be a finite real number in [0, 1)')
    return probability


def sync_training_dropout(model, dropout):
    """Align every loaded LoRA adapter's config and dropout, without touching weights.

    Call only on an already training-mode PEFT LoRA model.  Non-LoRA dropout is
    deliberately outside this helper: the delegated historical loader retains
    its explicit global Dropout override.  Every config and module is validated
    before any mutation, including adapters that are currently inactive.
    """
    import torch
    from peft.tuners.lora.layer import LoraLayer

    probability = _probability(dropout)
    if not model.training:
        raise ValueError('sync_training_dropout requires model.training=True')
    configs = getattr(model, 'peft_config', None)
    if not isinstance(configs, dict) or not configs:
        raise ValueError('expected a PEFT model with loaded LoRA configurations')
    if any(not hasattr(config, 'lora_dropout') for config in configs.values()):
        raise ValueError('all loaded adapters must expose LoRA dropout configuration')

    layers = []
    counts = {name: 0 for name in configs}
    for path, layer in model.named_modules():
        if not isinstance(layer, LoraLayer):
            continue
        for adapter, module in layer.lora_dropout.items():
            if adapter not in configs:
                raise ValueError(f'LoRA module has no PEFT configuration: {adapter}')
            if not isinstance(module, (torch.nn.Dropout, torch.nn.Identity)):
                raise ValueError(f'unsupported LoRA dropout module at {path}/{adapter}')
            layers.append((path, layer, adapter, module))
            counts[adapter] += 1
    if any(count == 0 for count in counts.values()):
        raise ValueError('every loaded adapter must have an actual LoRA dropout module')

    before_configs = {name: float(config.lora_dropout) for name, config in configs.items()}
    changes = []
    for path, layer, adapter, module in layers:
        before = {'type': type(module).__name__,
                  'p': float(module.p) if isinstance(module, torch.nn.Dropout) else 0.,
                  'training': bool(module.training)}
        if isinstance(module, torch.nn.Dropout):
            module.p = probability
        elif probability > 0:
            replacement = torch.nn.Dropout(p=probability)
            replacement.train(module.training)
            layer.lora_dropout[adapter] = replacement
            module = replacement
        after = {'type': type(module).__name__,
                 'p': float(module.p) if isinstance(module, torch.nn.Dropout) else 0.,
                 'training': bool(module.training)}
        changes.append({'module': path, 'adapter': adapter,
                        'before': before, 'after': after})
    for config in configs.values():
        config.lora_dropout = probability

    return {'schema_version': 1, 'training': True, 'requested_dropout': probability,
            'weights_modified': False, 'optimizer_updates': 0,
            'scope': 'all loaded LoRA adapters; non-LoRA dropout unchanged by helper',
            'config_before': before_configs,
            'config_after': {name: float(config.lora_dropout)
                             for name, config in configs.items()},
            'module_count_by_adapter': counts, 'modules': changes}


def load(model_root, adapter_dir, device, *, training=False, dropout=0.,
         gradient_checkpointing=True):
    """Compatible independent loader for new training, preserving historical files."""
    from vlm_fp32_adapter.model import load as load_fp32

    # Validate before loading the large model.  An eval load has no dropout
    # override; preserve its serialized training configuration without mutation.
    if training:
        dropout = _probability(dropout)
    model, binding = load_fp32(
        model_root, adapter_dir, device, training=training, dropout=dropout,
        gradient_checkpointing=gradient_checkpointing)
    if training:
        binding = {**binding, 'dropout_runtime_metadata': sync_training_dropout(model, dropout)}
    else:
        binding = {**binding, 'dropout_runtime_metadata': {
            'schema_version': 1, 'training': False, 'configuration_mutated': False,
            'reason': 'evaluation preserves serialized training dropout configuration'}}
    return model, binding
