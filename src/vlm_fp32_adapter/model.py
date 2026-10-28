"""Keep trainable checkpoint values separate from base-model storage precision.

Historical all-BF16 studies remain valid under their declared numerical policy.
This independent loader must be used for BOTH a new baseline and its candidate;
it must not silently replace a frozen study's loader or numerical contract.
"""
from pathlib import Path


def restore_fp32_adapter(model, adapter_dir):
    """Restore exact LoRA masters, without a lossy BF16 intermediate copy."""
    import torch
    from safetensors.torch import load_file
    state = load_file(str(Path(adapter_dir) / 'adapter_model.safetensors'))
    parameters = {name.replace('.default.', '.'): parameter
                  for name, parameter in model.named_parameters() if 'lora_' in name}
    if set(parameters) != set(state):
        raise ValueError('adapter checkpoint keys differ from actual LoRA parameters')
    if any(parameters[key].shape != value.shape or not value.is_floating_point()
           for key, value in state.items()):
        raise ValueError('adapter checkpoint shapes/dtypes differ')
    with torch.no_grad():
        for key, value in state.items():
            parameter = parameters[key]
            parameter.data = value.to(device=parameter.device, dtype=torch.float32).clone()
    return {'adapter_tensors': len(parameters), 'adapter_parameters': sum(p.numel() for p in parameters.values()),
            'adapter_precision': 'float32', 'lossy_intermediate_cast': False}


def load(model_root, adapter_dir, device, *, training=False, dropout=0., gradient_checkpointing=True):
    """Load a BF16 base and preserve the adapter's FP32 checkpoint values.

    No global autocast is enabled here. Callers must declare forward-compute
    precision as well as storage precision, and compute CE in FP32. A training
    driver may deliberately use BF16 AMP; that is a separate explicit choice.
    """
    import torch
    from peft import PeftModel
    from transformers import Qwen2_5_VLForConditionalGeneration
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_root), local_files_only=True, torch_dtype=torch.bfloat16,
        quantization_config=None, attn_implementation='sdpa', low_cpu_mem_usage=True,
        device_map={'': str(device)})
    model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=training,
                                      autocast_adapter_dtype=True)
    binding = restore_fp32_adapter(model, adapter_dir)
    model.requires_grad_(False)
    if training:
        for name, parameter in model.named_parameters():
            if 'lora_' in name:
                parameter.requires_grad_(True)
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = dropout
        if gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            model.enable_input_require_grads()
        model.train()
        model.get_base_model().visual.eval()
    else:
        model.eval()
    model.config.use_cache = not training
    model.generation_config.temperature = 1.0
    model.generation_config.top_p = 1.0
    model.generation_config.top_k = 50
    return model, {**binding, 'base_precision': 'bfloat16', 'training': training,
                   'forward_autocast_enabled_by_loader': False}
