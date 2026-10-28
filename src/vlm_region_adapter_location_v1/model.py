"""Zero-output FP32 bottleneck, with exact ordinary-SFT initialization and explicit saving."""
from pathlib import Path
import math
import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from vlm_deep.common import write_json
from vlm_object_train.train import tensor_hash
from vlm_fp32_readout_v1.model import load as load_base


class Residual(nn.Module):
    def __init__(self, dimension=2048, bottleneck=32):
        super().__init__()
        self.down = nn.Parameter(torch.empty(bottleneck, dimension, dtype=torch.float32))
        self.up = nn.Parameter(torch.zeros(dimension, bottleneck, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.down, a=math.sqrt(5), generator=torch.Generator(device='cpu').manual_seed(42))

    def forward(self, value):
        with torch.autocast(value.device.type, enabled=False):
            delta = F.linear(F.gelu(F.linear(value.float(), self.down)), self.up)
            # One cast after the addition; no unnecessary separate BF16 delta rounding.
            return (value.float() + delta).to(value.dtype)


def residual_hash(model):
    return tensor_hash(model.region_residual.named_parameters())


def save_residual(model, directory, role):
    directory = Path(directory)
    save_file({k: v.detach().cpu().contiguous() for k, v in model.region_residual.state_dict().items()},
              str(directory / 'residual.safetensors'))
    saved = load_file(str(directory / 'residual.safetensors'))
    if any(not torch.equal(saved[k], v.detach().cpu()) for k, v in model.region_residual.state_dict().items()):
        raise ValueError('Saved residual differs from FP32 live parameters')
    metadata = {'role': role, 'dimension': 2048, 'bottleneck': 32, 'parameters': 131072,
                'dtype': 'float32', 'output_dtype': 'same as input', 'residual_sha256': residual_hash(model)}
    write_json(directory / 'residual_config.json', metadata)
    return metadata


def load(model_root, adapter_dir, device, *, role, training=False, dropout=0.):
    if role not in ('readout', 'visual'):
        raise ValueError('Unknown registered location')
    model, binding = load_base(model_root, adapter_dir, device, training=training, dropout=dropout)
    if model.config.hidden_size != 2048:
        raise ValueError('Matched2048-dimensional locations required')
    residual = Residual().to(device=device)
    residual.requires_grad_(training)
    model.add_module('region_residual', residual)
    if role == 'visual':
        model.get_base_model().visual.register_forward_hook(lambda module, args, output: residual(output))
    else:
        model.get_output_embeddings().register_forward_pre_hook(lambda module, args: (residual(args[0]),))
    path = Path(adapter_dir) / 'residual.safetensors'
    if path.exists():
        import json
        metadata = json.loads((Path(adapter_dir) / 'residual_config.json').read_text())
        if metadata['role'] != role:
            raise ValueError('Saved residual placement differs')
        residual.load_state_dict(load_file(str(path)))
        if residual_hash(model) != metadata['residual_sha256']:
            raise ValueError('Saved residual identity differs')
    model.train(training)
    model.get_base_model().visual.eval()
    return model, {**binding, 'residual_role': role, 'residual_parameters': 131072,
                   'residual_sha256': residual_hash(model), 'residual_restored': path.exists()}
