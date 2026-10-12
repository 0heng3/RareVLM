"""Explicit, counted LoRA placement and matched FP32 numerical paths."""
from __future__ import annotations


def load(model_dir,config,device,method=None,adapter_dir=None,training=False,
         quantization='nf4_language_only',device_map=None):
    import torch
    from transformers import AutoProcessor,BitsAndBytesConfig,Qwen2_5_VLForConditionalGeneration
    quant=None
    if quantization=='nf4_language_only':
        quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float32,bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=['visual','lm_head'])
    elif quantization!='original':raise ValueError('unknown weight path')
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(str(model_dir),
        local_files_only=True,torch_dtype=torch.float32,quantization_config=quant,
        attn_implementation='sdpa',low_cpu_mem_usage=True,
        device_map=device_map or {'':str(device)})
    model.requires_grad_(False)
    if quant:
        from bitsandbytes.nn import Linear4bit
        if any(isinstance(m,Linear4bit) for m in model.visual.modules()):raise ValueError('quantized vision')
        if not any(isinstance(m,Linear4bit) for m in model.model.modules()):raise ValueError('language not quantized')
    processor=AutoProcessor.from_pretrained(str(model_dir),local_files_only=True,
        min_pixels=config['min_pixels'],max_pixels=config['max_pixels'],use_fast=False)
    processor.tokenizer.padding_side='left'
    if training:
        from peft import LoraConfig,get_peft_model
        spec=config['methods'][method]
        language=[name for name,module in model.named_modules()
                  if '.layers.' in name and 'visual' not in name
                  and name.endswith(tuple(spec['projections']))]
        merger=['visual.merger.'+name for name in spec.get('merger_ranks',{})]
        if len(language)!=36*len(spec['projections']):raise ValueError('wrong language target count')
        targets=language+merger
        named=dict(model.named_modules())
        if any(not isinstance(named[name],torch.nn.Linear) for name in merger):raise ValueError('merger target is not linear')
        ranks={'visual.merger.'+name:r for name,r in spec.get('merger_ranks',{}).items()}
        alphas={name:2*r for name,r in ranks.items()}
        model=get_peft_model(model,LoraConfig(r=spec['rank'],lora_alpha=spec['alpha'],
            lora_dropout=config['lora_dropout'],target_modules=targets,
            rank_pattern=ranks,alpha_pattern=alphas,bias='none',task_type='CAUSAL_LM'))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.enable_input_require_grads();model.config.use_cache=False
        base=model.get_base_model();base.visual.gradient_checkpointing=False
        for name,p in model.named_parameters():
            if p.requires_grad:
                if 'lora_' not in name:raise ValueError('non-LoRA trainable weight')
                if 'visual' in name and (not merger or '.visual.merger.' not in name):raise ValueError('vision trunk became trainable')
                p.data=p.data.float()
        actual=sum(p.numel() for p in model.parameters() if p.requires_grad)
        if actual!=spec['expected_trainable']:raise ValueError(f'parameter budget: {method}: {actual} != {spec["expected_trainable"]}')
        set_training(model,method,config)
    elif adapter_dir:
        from peft import PeftModel
        model=PeftModel.from_pretrained(model,str(adapter_dir),is_trainable=False)
        model.eval()
    else:model.eval()
    if any(p.dtype!=torch.float32 for p in model.visual.parameters()
           if p.__class__.__name__!='Params4bit'):raise ValueError('vision precision is not FP32')
    return model,processor


def set_training(model,method,config):
    model.train()
    base=model.get_base_model()
    base.visual.eval()
    if config['methods'][method].get('merger_ranks'):base.visual.merger.train()


def parameter_groups(model):
    out={'language':0,'merger':0}
    for name,p in model.named_parameters():
        if p.requires_grad:out['merger' if '.visual.merger.' in name else 'language']+=p.numel()
    return out
