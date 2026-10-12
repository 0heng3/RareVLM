"""Frozen Qwen vision/LLM backbone with optional language-attention LoRA."""
from .common import sha_file


def load_stack(model_dir,config,device,adapter_dir=None,training=False):
    import torch
    from transformers import AutoProcessor,BitsAndBytesConfig,Qwen2_5_VLForConditionalGeneration
    quantization=None
    if config.get('quantization')=='nf4_language_only':
        quantization=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float32,bnb_4bit_use_double_quant=config['double_quantization'],
            llm_int8_skip_modules=['visual','lm_head'])
    dtype=torch.float32 if config['precision']=='fp32' else torch.float16
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(str(model_dir),local_files_only=True,
            torch_dtype=dtype,quantization_config=quantization,attn_implementation=config['attention_implementation'],
            low_cpu_mem_usage=True,device_map={'':str(device)})
    if quantization:
        from bitsandbytes.nn import Linear4bit
        if any(isinstance(m,Linear4bit) for m in model.visual.modules()):raise ValueError('vision encoder unexpectedly quantized')
        if not any(isinstance(m,Linear4bit) for m in model.model.modules()):raise ValueError('language backbone was not quantized')
        if any(p.dtype!=torch.float32 for p in model.visual.parameters()):raise ValueError('vision encoder must remain FP32')
    processor=AutoProcessor.from_pretrained(str(model_dir),local_files_only=True,
            min_pixels=config['min_pixels'],max_pixels=config['max_pixels'],use_fast=False)
    if training:
        from peft import LoraConfig,get_peft_model
        targets=[name for name,module in model.named_modules() if '.layers.' in name
                 and name.endswith(tuple(config['target_projections'])) and 'visual' not in name]
        if not targets:raise ValueError('no language-attention LoRA target modules found')
        model=get_peft_model(model,LoraConfig(r=config['lora_rank'],lora_alpha=config['lora_alpha'],
            lora_dropout=config['lora_dropout'],target_modules=targets,bias='none',task_type='CAUSAL_LM'))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.enable_input_require_grads()
        model.config.use_cache=False
        base=model.get_base_model()
        if hasattr(base.visual,'gradient_checkpointing'):base.visual.gradient_checkpointing=False
        trainable=[name for name,p in model.named_parameters() if p.requires_grad]
        if not trainable or any('lora_' not in name or 'visual' in name for name in trainable):raise ValueError('parameters outside language LoRA became trainable')
        for p in model.parameters():
            if p.requires_grad:p.data=p.data.float()
    elif adapter_dir:
        from peft import PeftModel
        model=PeftModel.from_pretrained(model,str(adapter_dir),is_trainable=False)
    model.train(training)
    if training:model.get_base_model().visual.eval()
    return model,processor
