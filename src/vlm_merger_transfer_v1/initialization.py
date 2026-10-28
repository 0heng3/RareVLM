import math
from pathlib import Path
from vlm_deep.common import sha_file, write_json

TARGETS = ('visual.merger.mlp.0', 'visual.merger.mlp.2')
EXTRA_NAMES = tuple('base_model.model.' + name + '.lora_' + factor + '.weight' for name in TARGETS for factor in ('A', 'B'))


def build(root, original_adapter, model_root):
    """Export a CPU composite; existing FP32 language values never round to BF16."""
    import torch
    from safetensors.torch import load_file, save_file
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, Qwen2_5_VLForConditionalGeneration
    from vlm_fp32_adapter.model import restore_fp32_adapter
    from vlm_human_sft.evaluate import lora_hash
    import json
    original_adapter=Path(original_adapter);destination=root/'initial_adapter'
    if destination.exists():raise FileExistsError('Composite initializer already exists')
    destination.mkdir(parents=True)
    raw=load_file(str(original_adapter/'adapter_model.safetensors'))
    if len(raw)!=144 or sum(x.numel() for x in raw.values())!=1843200 or any(x.dtype!=torch.float32 for x in raw.values()):
        raise ValueError('Need exact original144 FP32 language masters')
    state={k:v.clone() for k,v in raw.items()}
    generator=torch.Generator(device='cpu').manual_seed(42)
    for name,inp,out in [(TARGETS[0],5120,5120),(TARGETS[1],5120,2048)]:
        a=torch.empty((8,inp),dtype=torch.float32);torch.nn.init.kaiming_uniform_(a,a=math.sqrt(5),generator=generator)
        b=torch.zeros((out,8),dtype=torch.float32)
        state['base_model.model.'+name+'.lora_A.weight']=a
        state['base_model.model.'+name+'.lora_B.weight']=b
    config=json.loads((original_adapter/'adapter_config.json').read_text())
    config.update(lora_dropout=0.,target_modules=['q_proj','v_proj',*TARGETS])
    write_json(destination/'adapter_config.json',config)
    save_file(state,str(destination/'adapter_model.safetensors'),metadata={'format':'pt'})
    # Real architecture and PEFT injection, with the large frozen base on meta.
    with torch.device('meta'):
        base=Qwen2_5_VLForConditionalGeneration(AutoConfig.from_pretrained(str(model_root),local_files_only=True))
        model=get_peft_model(base,LoraConfig.from_pretrained(str(destination)))
    model.requires_grad_(False)
    found={}
    for name,parameter in list(model.named_parameters()):
        if 'lora_' not in name:continue
        key=name.replace('.default.','.')
        if key not in state or tuple(parameter.shape)!=tuple(state[key].shape):raise ValueError('Injected composite key/shape differs')
        parent,leaf=name.rsplit('.',1)
        model.get_submodule(parent).register_parameter(leaf,torch.nn.Parameter(state[key].clone(),requires_grad=True))
        found[key]=name
    if set(found)!=set(state) or len(found)!=148 or sum(x.numel() for x in state.values())!=1982464:
        raise ValueError('Composite must contain148 factors/1982464 parameters')
    actual_targets={name for name,module in model.named_modules() if hasattr(module,'lora_A') and len(module.lora_A)}
    if len(actual_targets)!=74 or not all(any(name.endswith(t) for name in actual_targets) for t in TARGETS):
        raise ValueError('Wrong actual74 target scopes')
    initial_hash=lora_hash(model)
    model.save_pretrained(root/'cpu_roundtrip',safe_serialization=True)
    roundtrip=load_file(str(root/'cpu_roundtrip/adapter_model.safetensors'))
    if set(roundtrip)!=set(state) or any(not torch.equal(v,roundtrip[k]) for k,v in state.items()):
        raise ValueError('Real PEFT save changes composite FP32 values')
    restored=restore_fp32_adapter(model,root/'cpu_roundtrip')
    if lora_hash(model)!=initial_hash or any(not torch.equal(raw[k],roundtrip[k]) for k in raw):
        raise ValueError('Generic restore changes language/composite values')
    receipt={'status':'passed','real_architecture_targets':len(actual_targets),'factors':148,'language_parameters':1843200,
             'new_visual_parameters':139264,'total_trainable_parameters':1982464,'original_language_values_exact':True,
             'new_visual_B_all_zero':all(bool(state[k].eq(0).all()) for k in EXTRA_NAMES if '.lora_B.' in k),
             'real_peft_save_load_exact':True,'generic_fp32_restore':restored,'lora_sha256':initial_hash,
             'initial_adapter_files':{name:sha_file(destination/name) for name in ('adapter_config.json','adapter_model.safetensors')}}
    write_json(root/'CPU_INITIALIZER_REVIEW.json',receipt)
    return receipt
