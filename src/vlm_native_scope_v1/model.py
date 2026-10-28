"""Compose existing exact language factors and previously validated zero vision factors."""
from pathlib import Path
import json

from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import sha_file,write_json

BASE=PROJECT_ROOT.parents[1]/'outputs'
ROOT=BASE/'vlm_native_scope_v1'
PARAMETERS=1843200


def build():
    import torch
    from safetensors.torch import load_file,save_file
    from peft import LoraConfig,get_peft_model
    from transformers import AutoConfig,Qwen2_5_VLForConditionalGeneration
    from vlm_human_sft.evaluate import lora_hash
    read=lambda p:json.loads(Path(p).read_text())
    source=Path(read(BASE/'vlm_native_stable_kl_v1/CONFIG.json')['initial_adapter'])
    old=read(BASE/'vlm_iiit_scope_v1/INITIALIZATION.json');visual_source=Path(old['adapter'])
    for path,digest in old['files_sha256'].items():
        if sha_file(path)!=digest:raise ValueError('Existing zero-vision initializer changed')
    language=load_file(str(source/'adapter_model.safetensors'));visual={k:v for k,v in load_file(str(visual_source/'adapter_model.safetensors')).items() if '.visual.' in k}
    if len(language)!=144 or len(visual)!=60 or any(v.dtype!=torch.float32 for v in [*language.values(),*visual.values()]):raise ValueError('Expected exact FP32 factors')
    if sum(v.numel() for v in language.values())!=PARAMETERS or sum(v.numel() for v in visual.values())!=PARAMETERS:raise ValueError('Unequal scope budgets')
    if any(torch.count_nonzero(v) for k,v in visual.items() if '.lora_B.' in k):raise ValueError('Visual B must be exactly zero')
    state={**language,**visual};destination=ROOT/'initial_adapter'
    if destination.exists():raise FileExistsError('Initializer exists')
    cfg=read(visual_source/'adapter_config.json');cfg.update(inference_mode=False,lora_dropout=0.)
    destination.mkdir(parents=True);write_json(destination/'adapter_config.json',cfg);save_file(state,str(destination/'adapter_model.safetensors'),metadata={'format':'pt'})
    base_config=read(read(BASE/'vlm_native_stable_kl_v1/CONFIG.json')['baseline_config']);config=AutoConfig.from_pretrained(base_config['model_root'],local_files_only=True)
    with torch.device('meta'):model=get_peft_model(Qwen2_5_VLForConditionalGeneration(config),LoraConfig.from_pretrained(str(destination)))
    found=set()
    for name,parameter in list(model.named_parameters()):
        if 'lora_' not in name:continue
        key=name.replace('.default.','.')
        if key not in state or parameter.shape!=state[key].shape:raise ValueError('Actual PEFT injection shape mismatch: '+name)
        path,leaf=name.rsplit('.',1);model.get_submodule(path).register_parameter(leaf,torch.nn.Parameter(state[key].clone()));found.add(key)
    if found!=set(state):raise ValueError('Composite factor inventory differs')
    for name,module in model.named_modules():
        if hasattr(module,'lora_A') and len(module.lora_A):
            if module.r['default']!=(12 if '.visual.' in name else 8) or module.scaling['default']!=2.:raise ValueError('Rank/scaling differs')
    digest=lora_hash(model);model.save_pretrained(ROOT/'cpu_roundtrip',safe_serialization=True);saved=load_file(str(ROOT/'cpu_roundtrip/adapter_model.safetensors'))
    if set(saved)!=set(state) or any(not torch.equal(v,saved[k]) for k,v in state.items()):raise ValueError('CPU save roundtrip changed factors')
    result={'status':'passed','adapter':str(destination),'model_root':base_config['model_root'],'lora_sha256':digest,'language_factors':144,'visual_factors':60,'stored_parameters':2*PARAMETERS,'optimized_parameters_per_role':PARAMETERS,'language_values_exact':True,'all_visual_B_zero':True,'cpu_injection_roundtrip_exact':True,'language_source':str(source),'visual_zero_source':str(visual_source),'source_files_sha256':{str(path):sha_file(path) for directory in [source,visual_source] for path in [directory/'adapter_model.safetensors',directory/'adapter_config.json']},'files_sha256':{str(destination/name):sha_file(destination/name) for name in ['adapter_model.safetensors','adapter_config.json']}}
    write_json(ROOT/'INITIALIZATION.json',result);return result


def load(adapter,role,training=False):
    import torch
    from vlm_batch_stable_v1.model import load as stable_load
    from vlm_probe.evaluate import processor_for
    read=lambda p:json.loads(Path(p).read_text());c=read(ROOT/'CONFIG.json');base=read(c['baseline_config'])
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,binding=stable_load(base['model_root'],adapter,torch.device('cuda:0'),policy='imagewise_fp32_language',training=training,dropout=0.)
    if training:
        model.requires_grad_(False)
        for name,parameter in model.named_parameters():
            if 'lora_' in name and (('.visual.' in name)==(role=='visual')):parameter.requires_grad_(True)
        model.train();model.get_base_model().visual.train(role=='visual');model.config.use_cache=False
        selected=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        if len(selected)!=(60 if role=='visual' else 144) or sum(p.numel() for n,p in selected)!=PARAMETERS or any(p.dtype!=torch.float32 for n,p in selected):raise ValueError('Wrong optimized scope')
    processor=processor_for(Path(base['model_root']),read(base['reference_config']),1024)
    return model,processor,{**binding,'optimized_scope':role,'optimized_parameters':PARAMETERS,'visual_checkpointing':getattr(model.get_base_model().visual,'gradient_checkpointing',False)}
