"""One fixed language LoRA SFT epoch; torchrun DDP, assistant-only loss."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT

from .common import content_sha,encode_record,frozen_parameter_hash,read_manifest,sha_file,to_device,write_json
from .model import load_stack


def main():
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data.distributed import DistributedSampler
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--asset-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--config',type=Path,default=PROJECT_ROOT/'configs/vlm_sft_textvqa.json')
    args=p.parse_args();config=json.loads(args.config.read_text())
    rank=int(os.environ.get('RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'));local=int(os.environ.get('LOCAL_RANK','0'))
    if world!=config['expected_world_size']:raise ValueError('launch the frozen world size with torchrun')
    torch.cuda.set_device(local);device=torch.device('cuda',local);torch.set_num_threads(2)
    if world>1:dist.init_process_group('nccl')
    manifest_path=args.asset_root/'prepared/manifest.json';m=read_manifest(manifest_path)
    if {k:len(v) for k,v in m['splits'].items()}!={'train':config['training_images'],'validation':config['internal_validation_images'],'final':config['final_evaluation_images']}:raise ValueError('prepared sizes differ from fixed config')
    seed=config['train_seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    # Container RAM can be much smaller than host RAM. Avoid four concurrent
    # CPU staging copies while preserving the fixed DDP training world size.
    for initialization_rank in range(world):
        if rank==initialization_rank:
            model,processor=load_stack(args.asset_root/'model',config,device,training=True)
            print('model initialized on rank',rank,flush=True)
        if world>1:dist.barrier(device_ids=[local])
    trainable={n:p.numel() for n,p in model.named_parameters() if p.requires_grad}
    base_hash=frozen_parameter_hash(model) if rank==0 else None
    lock={'manifest_sha256':m['manifest_sha256'],'config_sha256':content_sha(config),
          'model_provenance_sha256':sha_file(args.asset_root/'model_provenance.json'),
          'trainable_parameters':sum(trainable.values()),'total_parameters':model.num_parameters(),
          'trainable_names':trainable,'world_size':world,'training_precision':config['precision'],
          'quantization':config.get('quantization'),'vision_precision':'fp32' if config['precision']=='fp32' else 'fp16',
          'frozen_parameter_sha256_before':base_hash,
          'model_loading':'direct placement on local GPU; serialized rank initialization to respect 32-GiB container RAM',
          'source_sha256':{**{f.name:sha_file(f) for f in Path(__file__).parent.glob('*.py')},
              'rarevlm_paths.py':sha_file(SOURCE_ROOT/'rarevlm_paths.py')}}
    if rank==0:write_json(args.output_dir/'TRAIN_INPUT_LOCK.json',lock)
    if world>1:model=DistributedDataParallel(model,device_ids=[local],find_unused_parameters=False)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config['lr']/max(config['warmup_steps'],1),weight_decay=config['weight_decay'])
    use_amp=config['precision']=='fp16'
    scaler=torch.cuda.amp.GradScaler(enabled=use_amp,init_scale=config['grad_scaler_initial_scale'])
    rows=m['splits']['train'];acc=config['gradient_accumulation']
    if len(rows)%(world*acc):raise ValueError('sample count must divide the fixed effective batch')
    sampler=DistributedSampler(rows,num_replicas=world,rank=rank,shuffle=True,seed=seed,drop_last=True)
    total_steps=len(rows)//(world*acc)*config['epochs']
    started=time.monotonic();history=[];updates=0;skipped=0
    optimizer.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(config['epochs']):
        sampler.set_epoch(epoch);local_loss=0.;seen=0
        order=list(iter(sampler))
        for i,index in enumerate(order):
            batch=encode_record(processor,rows[index],manifest_path.parent,supervised=True)
            if batch['input_ids'].shape[1]>config['max_sequence_length']:raise ValueError('sequence exceeds predeclared budget; no silent truncation')
            batch=to_device(batch,device);should_step=(i+1)%acc==0
            context=model.no_sync() if world>1 and not should_step else nullcontext()
            with context,torch.autocast('cuda',dtype=torch.float16,enabled=use_amp):
                loss=model(**batch).loss
                if not torch.isfinite(loss):raise FloatingPointError('non-finite supervised loss')
                scaler.scale(loss/acc).backward()
            local_loss+=float(loss.detach());seen+=1
            if should_step:
                scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
                scale_before=scaler.get_scale();scaler.step(optimizer);scaler.update()
                if scaler.get_scale()<scale_before:skipped+=1
                else:updates+=1
                optimizer.zero_grad(set_to_none=True)
                step=(i+1)//acc+epoch*len(order)//acc
                next_step=step+1;warmup=config['warmup_steps']
                factor=next_step/max(warmup,1) if next_step<=warmup else max((total_steps-next_step+1)/max(total_steps-warmup,1),0.)
                for group in optimizer.param_groups:group['lr']=config['lr']*factor
                if rank==0 and (step==1 or step%8==0):print(f'epoch={epoch+1} step={step}/{total_steps} loss={local_loss/seen:.4f} scaler={scaler.get_scale():g}',flush=True)
        base=model.module if world>1 else model;base.eval()
        val_loss=0.;val_tokens=0
        with torch.inference_mode():
            for row in m['splits']['validation'][rank::world]:
                batch=to_device(encode_record(processor,row,manifest_path.parent,True),device)
                count=int((batch['labels']!=-100).sum())
                with torch.autocast('cuda',dtype=torch.float16,enabled=use_amp):value=base(**batch).loss
                if not torch.isfinite(value):raise FloatingPointError('nonfinite validation loss')
                val_loss+=float(value)*count;val_tokens+=count
        totals=torch.tensor([local_loss,seen,val_loss,val_tokens,skipped,updates],dtype=torch.float64,device=device)
        if world>1:dist.all_reduce(totals)
        record={'epoch':epoch+1,'train_mean_sample_loss':float(totals[0]/totals[1]),
                'internal_validation_token_nll':float(totals[2]/totals[3]),
                'validation_answer_tokens':int(totals[3]),'optimizer_steps_per_rank':updates,
                'skipped_updates_across_ranks':int(totals[4]),'local_order_sha256':content_sha(order)}
        history.append(record)
        if rank==0:print('epoch complete',json.dumps(record),flush=True)
    base=model.module if world>1 else model
    peaks=torch.tensor([torch.cuda.max_memory_allocated(device),torch.cuda.max_memory_reserved(device)],device=device,dtype=torch.float64)
    if world>1:dist.all_reduce(peaks,op=dist.ReduceOp.MAX)
    if rank==0:
        final_hash=frozen_parameter_hash(base)
        if final_hash!=base_hash:raise ValueError('frozen base parameters changed')
        if history[-1]['skipped_updates_across_ranks']:raise ValueError('FP16 skipped optimizer updates; report failure rather than an incomplete epoch')
        adapter_dir=args.output_dir/'adapter';base.save_pretrained(adapter_dir,safe_serialization=True)
        result={**lock,'frozen_parameter_sha256_after':final_hash,'frozen_parameters_unchanged':True,
            'history':history,'epochs_completed':config['epochs'],'optimizer_steps_per_rank':updates,
            'effective_batch_size':world*acc,'training_images':len(rows),'duration_seconds':time.monotonic()-started,
            'peak_gpu_allocated_gib':float(peaks[0]/2**30),'peak_gpu_reserved_gib':float(peaks[1]/2**30),
            'gpu_name':torch.cuda.get_device_name(device),'checkpoint_selection':config['checkpoint_selection'],
            'adapter_files_sha256':{f.name:sha_file(f) for f in adapter_dir.iterdir() if f.is_file()}}
        write_json(args.output_dir/'TRAIN_RESULT.json',result)
        print('SFT complete',result['trainable_parameters'],'trainable parameters; base unchanged',flush=True)
    if world>1:dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
