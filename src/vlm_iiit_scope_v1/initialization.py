"""Keep original language LoRA and add zero-output vision qkv adapters."""
import json
import math
from pathlib import Path

from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import sha_file, write_json

ROOT=PROJECT_ROOT.parents[1]/'outputs/vlm_iiit_scope_v1'
PARAMETERS=1843200


def build():
    import torch
    from safetensors.torch import load_file,save_file
    from transformers import AutoConfig,Qwen2_5_VLForConditionalGeneration
    from peft import LoraConfig,get_peft_model
    from vlm_human_sft.evaluate import lora_hash
    from vlm_object_train.train import tensor_hash
    source=json.loads((ROOT.parent/'vlm_grpo_iteration_v1/PLAN.json').read_text())
    original=Path(source['adapter']);destination=ROOT/'initial_adapter'
    if destination.exists():raise FileExistsError('Initializer already exists')
    raw=load_file(str(original/'adapter_model.safetensors'))
    if len(raw)!=144 or sum(v.numel() for v in raw.values())!=PARAMETERS or any(v.dtype!=torch.float32 for v in raw.values()):
        raise ValueError('Need original144 FP32 language factors')
    config=AutoConfig.from_pretrained(source['model_root'],local_files_only=True)
    if (config.hidden_size,config.num_hidden_layers,config.vision_config.hidden_size,config.vision_config.depth)!=(2048,36,1280,32):
        raise ValueError('Unexpected real architecture')
    targets=[f'visual.blocks.{i}.attn.qkv' for i in range(2,32)]
    state={k:v.clone() for k,v in raw.items()};generator=torch.Generator(device='cpu').manual_seed(42)
    for target in targets:
        a=torch.empty((12,1280),dtype=torch.float32);torch.nn.init.kaiming_uniform_(a,a=math.sqrt(5),generator=generator)
        state['base_model.model.'+target+'.lora_A.weight']=a
        state['base_model.model.'+target+'.lora_B.weight']=torch.zeros((3840,12),dtype=torch.float32)
    cfg=json.loads((original/'adapter_config.json').read_text());cfg.update(target_modules=['q_proj','v_proj',*targets],lora_dropout=0.,inference_mode=False,
        rank_pattern={t:12 for t in targets},alpha_pattern={t:24 for t in targets})
    destination.mkdir(parents=True);write_json(destination/'adapter_config.json',cfg);save_file(state,str(destination/'adapter_model.safetensors'),metadata={'format':'pt'})
    with torch.device('meta'):
        model=get_peft_model(Qwen2_5_VLForConditionalGeneration(config),LoraConfig.from_pretrained(str(destination)))
    found={}
    for name,p in list(model.named_parameters()):
        if 'lora_' not in name:continue
        key=name.replace('.default.','.')
        if key not in state or p.shape!=state[key].shape:raise ValueError('PEFT target/rank shape differs: '+name)
        parent,leaf=name.rsplit('.',1);model.get_submodule(parent).register_parameter(leaf,torch.nn.Parameter(state[key].clone(),requires_grad=True));found[key]=name
    if set(found)!=set(state) or len(found)!=204:raise ValueError('Expected144 language plus60 visual factors')
    language={n:p for n,p in model.named_parameters() if 'lora_' in n and '.visual.' not in n}
    vision={n:p for n,p in model.named_parameters() if 'lora_' in n and '.visual.' in n}
    if sum(p.numel() for p in language.values())!=PARAMETERS or sum(p.numel() for p in vision.values())!=PARAMETERS:
        raise ValueError('Optimized parameter budgets must match')
    if tensor_hash(language.items())!=source['lora_sha256']:raise ValueError('Original language values/order changed')
    for name,m in model.named_modules():
        if hasattr(m,'lora_A') and len(m.lora_A):
            if m.r['default']!=(12 if '.visual.' in name else 8) or m.scaling['default']!=2.:raise ValueError('Wrong rank/scaling')
    initial_hash=lora_hash(model);model.save_pretrained(ROOT/'cpu_roundtrip',safe_serialization=True)
    saved=load_file(str(ROOT/'cpu_roundtrip/adapter_model.safetensors'))
    if set(saved)!=set(state) or any(not torch.equal(saved[k],v) for k,v in state.items()):raise ValueError('PEFT save altered factors')
    result={'status':'passed','language_factors':144,'visual_factors':60,'stored_adapter_parameters':2*PARAMETERS,'optimized_parameters_per_arm':PARAMETERS,
            'original_language_lora_sha256':source['lora_sha256'],'composite_lora_sha256':initial_hash,'original_language_values_exact':True,
            'new_visual_B_zero':True,'visual_targets':targets,'rank_language':8,'rank_vision':12,'scaling':2.,'cpu_peft_injection_and_roundtrip_exact':True,
            'model_root':source['model_root'],'original_adapter':str(original),'adapter':str(destination),
            'files_sha256':{str(destination/name):sha_file(destination/name) for name in ['adapter_config.json','adapter_model.safetensors']}}
    write_json(ROOT/'INITIALIZATION.json',result);print('INITIALIZER_PASSED',result,flush=True)
    return result


if __name__=='__main__':build()
