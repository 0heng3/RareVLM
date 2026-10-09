"""Real training-image forward/backward contract check; no final labels used."""
import argparse
import json
from pathlib import Path
from .common import encode_record,frozen_parameter_hash,read_manifest,to_device,write_json
from .model import load_stack


def main():
    import torch
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--asset-root',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'configs/vlm_sft_textvqa.json')
    args=p.parse_args();config=json.loads(args.config.read_text());torch.set_num_threads(4)
    manifest_path=args.asset_root/'prepared/manifest.json';m=read_manifest(manifest_path,verify_images=True)
    model,processor=load_stack(args.asset_root/'model',config,'cuda:0',training=True)
    before=frozen_parameter_hash(model)
    batch=to_device(encode_record(processor,m['splits']['train'][0],manifest_path.parent,True),'cuda:0')
    n=batch['input_ids'].shape[1]
    if n>config['max_sequence_length']:raise ValueError('sample exceeds the fixed sequence budget')
    trainable=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(trainable,lr=config['lr'])
    use_amp=config['precision']=='fp16'
    scaler=torch.cuda.amp.GradScaler(enabled=use_amp,init_scale=config['grad_scaler_initial_scale']);torch.cuda.reset_peak_memory_stats()
    with torch.autocast('cuda',dtype=torch.float16,enabled=use_amp):loss=model(**batch).loss
    if not torch.isfinite(loss):raise FloatingPointError('invalid real-sample loss')
    scaler.scale(loss).backward();scaler.unscale_(optimizer)
    if not any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in trainable):raise ValueError('no LoRA gradient')
    if any(p.grad is not None for p in model.parameters() if not p.requires_grad):raise ValueError('frozen parameter gradient leaked')
    finite=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in trainable)
    if not finite:raise FloatingPointError('FP16 produced nonfinite LoRA gradients')
    scaler.step(optimizer);scaler.update()
    after=frozen_parameter_hash(model)
    if before!=after:raise ValueError('frozen base weights changed')
    result={'passed':True,'training_image_only':True,'sequence_tokens':n,'supervised_answer_tokens':int((batch['labels']!=-100).sum()),
            'loss':float(loss),'trainable_parameters':sum(p.numel() for p in trainable),
            'frozen_parameters_unchanged':True,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
            'gpu':torch.cuda.get_device_name(),'precision':config['precision'],'quantization':config.get('quantization')}
    write_json(args.asset_root/'PREFLIGHT.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
