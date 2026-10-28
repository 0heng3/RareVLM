"""Avoid quantizing answer logits BEFORE the explicitly FP32 CE reduction.

BF16 LoRA/decoder outputs are unchanged. The head receives FP32 copies of its
hidden states and computes with FP32 copies of the same frozen weight values.
An independent head is essential when input/output embeddings share weights:
calling old_head.float() could otherwise silently change the input embedding.
New studies need matched initial/control policies; this is not a gain claim.
"""


def install_fp32_readout(model):
    import torch

    class FrozenFP32Head(torch.nn.Module):
        def __init__(self, previous):
            super().__init__()
            self.weight = torch.nn.Parameter(previous.weight.detach().to(dtype=torch.float32, copy=True),
                                             requires_grad=False)
            self.bias = (torch.nn.Parameter(previous.bias.detach().to(dtype=torch.float32, copy=True),
                                            requires_grad=False)
                         if previous.bias is not None else None)
            self.in_features, self.out_features = previous.in_features, previous.out_features

        def forward(self, hidden):
            with torch.autocast(hidden.device.type, enabled=False):
                return torch.nn.functional.linear(hidden.float(), self.weight, self.bias)

    old = model.get_output_embeddings()
    embedding = model.get_input_embeddings()
    if not isinstance(old, torch.nn.Linear) or old.weight.requires_grad:
        raise ValueError('This policy requires an ordinary frozen Linear output head')
    before_pointer, before_dtype = embedding.weight.data_ptr(), embedding.weight.dtype
    tied = old.weight.data_ptr() == before_pointer
    head = FrozenFP32Head(old)
    head.train(old.training)
    model.set_output_embeddings(head)
    if embedding.weight.data_ptr() != before_pointer or embedding.weight.dtype != before_dtype:
        raise RuntimeError('FP32 readout must not modify input embeddings')
    return {'base_and_input_embeddings_unchanged': True, 'old_head_dtype': str(old.weight.dtype),
            'new_head_dtype': str(head.weight.dtype), 'output_dtype': 'torch.float32',
            'input_output_weights_were_tied': tied, 'head_copy_is_frozen': True,
            'fp32_head_allocated_parameter_bytes': head.weight.numel() * head.weight.element_size()
                                      + (head.bias.numel()*head.bias.element_size() if head.bias is not None else 0),
            'note': 'Independent numerical policy; same-checkpoint matched baseline required'}


def load(model_root, adapter_dir, device, *, training=False, dropout=0., gradient_checkpointing=True):
    from vlm_runtime_metadata_v1.model import load as load_metadata

    model, binding = load_metadata(model_root, adapter_dir, device, training=training,
                                   dropout=dropout, gradient_checkpointing=gradient_checkpointing)
    return model, {**binding, 'frozen_readout': install_fp32_readout(model)}
