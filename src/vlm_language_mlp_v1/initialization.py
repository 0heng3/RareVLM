import json
import math
from pathlib import Path
from vlm_deep.common import sha_file,write_json
from .common import ROOT,NEW_PARAMETERS,TOTAL_PARAMETERS


def build(original_adapter,model_root):
    import torch
    from safetensors.torch import load_file,save_file
    from peft import LoraConfig,get_peft_model
    from transformers import AutoConfig,Qwen2_5_VLForConditionalGeneration
    from vlm_fp32_adapter.model import restore_fp32_adapter
    from vlm_human_sft.evaluate import lora_hash
    destination=ROOT/'initial_adapter'
    if destination.exists():raise FileExistsError('Initializer already exists')
    original_adapter=Path(original_adapter);raw=load_file(str(original_adapter/'adapter_model.safetensors'))
    if len(raw)!=144 or sum(v.numel() for v in raw.values())!=1843200 or any(v.dtype!=torch.float32 for v in raw.values()):raise ValueError('Need original FP32 q/v checkpoint')
    config=AutoConfig.from_pretrained(str(model_root),local_files_only=True)
    if (config.hidden_size,config.intermediate_size,config.num_hidden_layers)!=(2048,11008,36):raise ValueError('Unexpected architecture')
    targets=[f'model.layers.{layer}.mlp.{projection}' for layer in range(36) for projection in ('gate_proj','up_proj','down_proj')]
    state={k:v.clone() for k,v in raw.items()};generator=torch.Generator(device='cpu').manual_seed(42)
    for target in targets:
        inp,out=(11008,2048) if target.endswith('down_proj') else (2048,11008)
        a=torch.empty((2,inp),dtype=torch.float32);torch.nn.init.kaiming_uniform_(a,a=math.sqrt(5),generator=generator)
        state['base_model.model.'+target+'.lora_A.weight']=a
        state['base_model.model.'+target+'.lora_B.weight']=torch.zeros((out,2),dtype=torch.float32)
    cfg=json.loads((original_adapter/'adapter_config.json').read_text())
    cfg.update(target_modules=['q_proj','v_proj',*targets],lora_dropout=0.,rank_pattern={t:2 for t in targets},alpha_pattern={t:4 for t in targets})
    destination.mkdir(parents=True);write_json(destination/'adapter_config.json',cfg)
    save_file(state,str(destination/'adapter_model.safetensors'),metadata={'format':'pt'})
    with torch.device('meta'):
        base=Qwen2_5_VLForConditionalGeneration(config)
        model=get_peft_model(base,LoraConfig.from_pretrained(str(destination)))
    model.requires_grad_(False);found={}
    for name,p in list(model.named_parameters()):
        if 'lora_' not in name:continue
        key=name.replace('.default.','.')
        if key not in state or p.shape!=state[key].shape:raise ValueError('Real PEFT target/rank shape mismatch: '+name)
        parent,leaf=name.rsplit('.',1);model.get_submodule(parent).register_parameter(leaf,torch.nn.Parameter(state[key].clone(),requires_grad=True));found[key]=name
    if set(found)!=set(state) or len(found)!=360 or sum(p.numel() for p in state.values())!=TOTAL_PARAMETERS:raise ValueError('Expected360 factors/4663296 parameters')
    actual={n:m for n,m in model.named_modules() if hasattr(m,'lora_A') and len(m.lora_A)}
    if len(actual)!=180:raise ValueError('Expected180 scoped linear modules')
    for name,module in actual.items():
        mlp='.mlp.' in name
        if module.r['default']!=(2 if mlp else 8) or module.scaling['default']!=2.:raise ValueError('Actual mixed-rank scale mismatch')
    initial_hash=lora_hash(model);model.save_pretrained(ROOT/'cpu_roundtrip',safe_serialization=True)
    restored=load_file(str(ROOT/'cpu_roundtrip/adapter_model.safetensors'))
    if set(restored)!=set(state) or any(not torch.equal(restored[k],v) for k,v in state.items()):raise ValueError('Real PEFT roundtrip changed values')
    if any(not torch.equal(restored[k],v) for k,v in raw.items()):raise ValueError('Original language factors changed')
    restore_fp32_adapter(model,ROOT/'cpu_roundtrip')
    if lora_hash(model)!=initial_hash:raise ValueError('Generic FP32 restore failed')
    receipt={'status':'passed','factors':360,'target_modules':180,'old_qv_parameters':1843200,'new_mlp_parameters':NEW_PARAMETERS,
             'total_trainable_parameters':TOTAL_PARAMETERS,'old_qv_values_exact':True,'new_B_zero':True,'rank_qv':8,'rank_mlp':2,'scaling_both':2.,
             'real_architecture_and_peft_injection':True,'real_save_restore_exact':True,'lora_sha256':initial_hash,
             'initializer_files':{name:sha_file(destination/name) for name in ('adapter_config.json','adapter_model.safetensors')}}
    write_json(ROOT/'CPU_INITIALIZER_REVIEW.json',receipt);return receipt
