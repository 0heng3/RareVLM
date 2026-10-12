"""Paired full-question SFT with counted LoRA placement and step-boundary resume."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import time
from pathlib import Path

from .common import answer_loss,collate,content_sha,encode,frozen_parameter_hash,read_manifest,sha_file,to_device,write_json
from .model import load,parameter_groups,set_training


class EncodedQuestions:
    def __init__(self,rows,images,image_root,processor,max_length):
        self.rows=rows;self.images=images;self.image_root=Path(image_root)
        self.processor=processor;self.max_length=max_length

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        if isinstance(index,tuple):index,valid=index
        else:valid=True
        row=self.rows[index]
        item=encode(self.processor,row,self.image_root/self.images[row['image_id']]['path'],True)
        if item['input_ids'].shape[1]>self.max_length:raise ValueError(f'question exceeds frozen sequence budget: {row["question_id"]}')
        return item,valid,row['question_id']


class BatchCollator:
    def __init__(self,pad,numpy_transport=False):self.pad=pad;self.numpy_transport=numpy_transport
    def __call__(self,items):
        import torch
        data=collate([x[0] for x in items],self.pad)
        valid=torch.tensor([x[1] for x in items],dtype=torch.bool)
        if self.numpy_transport:return {k:v.numpy() for k,v in data.items()},valid.numpy(),[x[2] for x in items]
        return data,valid,[x[2] for x in items]


def schedule(n,seed,world,micro,acc):
    """All real questions once; sentinel computation carries zero loss weight."""
    import torch
    effective=world*micro*acc
    order=torch.randperm(n,generator=torch.Generator().manual_seed(seed)).tolist()
    padded=order+[-1]*((-n)%effective)
    return order,padded


def optimizer_lr(step,total,config):
    warm=max(math.ceil(total*config['warmup_ratio']),1)
    factor=(step+1)/warm if step<warm else (total-step)/max(total-warm,1)
    return config['lr']*factor


def cpu_state(value):
    import torch
    if isinstance(value,torch.Tensor):return value.detach().cpu()
    if isinstance(value,dict):return {k:cpu_state(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_state(x) for x in value]
    if isinstance(value,tuple):return tuple(cpu_state(x) for x in value)
    return value


def rng_state():
    import numpy as np
    import torch
    return {'python':random.getstate(),'numpy':np.random.get_state(),
            'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state()}


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state['python']);np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu']);torch.cuda.set_rng_state(state['cuda'])


def main():
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader
    from safetensors.torch import load_file
    from peft import set_peft_model_state_dict
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--asset-root',type=Path,required=True);p.add_argument('--model-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--method',required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--split',choices=['train_10k','train_all'],required=True)
    p.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'configs/vlm_deep_v2.json')
    p.add_argument('--workers',type=int,default=2)
    args=p.parse_args();c=json.loads(args.config.read_text())
    rank=int(os.environ.get('RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'));local=int(os.environ.get('LOCAL_RANK','0'))
    if world!=c['world_size']:raise ValueError('wrong training world')
    torch.cuda.set_device(local);device=torch.device('cuda',local);torch.set_num_threads(2)
    if world>1:dist.init_process_group('nccl')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    path=args.asset_root/'prepared/manifest.json';m=read_manifest(path);rows=m['splits'][args.split]
    micro=c['training_microbatch'];acc=c['training_accumulation'];effective=world*micro*acc
    if effective!=c['effective_batch']:raise ValueError('effective batch changed')
    if acc!=1:raise ValueError('this validated runner currently requires accumulation1; lower-batch repair needs its own verified implementation')
    order,padded=schedule(len(rows),args.seed,world,micro,acc);steps=len(padded)//effective
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for active_rank in range(world):
        if rank==active_rank:model,processor=load(args.model_root,c,device,args.method,training=True)
        if world>1:dist.barrier(device_ids=[local])
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=optimizer_lr(0,steps,c),weight_decay=c['weight_decay'])
    source={str(f.relative_to(Path(__file__).parent)):sha_file(f) for f in Path(__file__).parent.rglob('*.py')}
    frozen_before=frozen_parameter_hash(model) if rank==0 else None
    lock={'schema_version':'rarevlm.vlm_train.v2','config_sha256':content_sha(c),'manifest_sha256':m['manifest_sha256'],
        'method':args.method,'seed':args.seed,'split':args.split,'question_count':len(rows),
        'unique_training_images':len({r['image_id'] for r in rows}),'world_size':world,
        'effective_batch':effective,'microbatch':micro,'accumulation':acc,'planned_updates':steps,
        'global_question_order_sha256':content_sha([rows[i]['question_id'] for i in order]),
        'trainable_names':{n:p.numel() for n,p in model.named_parameters() if p.requires_grad},
        'parameter_groups':parameter_groups(model),'trainable_parameters':sum(p.numel() for p in params),
        'total_parameters':model.num_parameters(),'frozen_parameter_sha256_before':frozen_before,
        'source_sha256':source,'tail_head_optimization':'temporary head-input slicing, equivalence preflight required',
        'loss':c['loss'],'model_provenance_sha256':sha_file(args.model_root.parent/'model_provenance.json'),
        'checkpoint_selection':c['checkpoint_selection'],'padding_policy':'sentinel computations get zero loss weight; no real question dropped/repeated'}
    if rank==0:
        existing=args.output_dir/'TRAIN_INPUT_LOCK.json'
        if existing.exists() and json.loads(existing.read_text())!=lock:raise ValueError('training input/code lock changed')
        write_json(existing,lock)
    if world>1:dist.barrier(device_ids=[local])
    resume=None;latest=args.output_dir/'LATEST.json'
    if latest.exists():
        item=json.loads(latest.read_text());checkpoint=args.output_dir/item['directory']
        if sha_file(checkpoint/'state.pt')!=item['state_sha256']:raise ValueError('resume state changed')
        for name,digest in item['adapter_files_sha256'].items():
            if sha_file(checkpoint/'adapter'/name)!=digest:raise ValueError('resume adapter changed')
        state=torch.load(checkpoint/'state.pt',map_location='cpu',weights_only=False)
        if state['config_sha256']!=content_sha(c) or state['manifest_sha256']!=m['manifest_sha256'] or state['order_sha256']!=lock['global_question_order_sha256']:raise ValueError('resume protocol changed')
        set_peft_model_state_dict(model,load_file(str(checkpoint/'adapter/adapter_model.safetensors')))
        optimizer.load_state_dict(state['optimizer']);resume=state
    start_step=resume['completed_updates'] if resume else 0
    if world>1:model=DistributedDataParallel(model,device_ids=[local],find_unused_parameters=False)
    base=model.module if world>1 else model
    batches=[]
    for step in range(start_step,steps):
        block=padded[step*effective:(step+1)*effective];fallback=next(x for x in block if x>=0)
        local_indices=block[rank*micro:(rank+1)*micro]
        batches.append([(x if x>=0 else fallback,x>=0) for x in local_indices])
    dataset=EncodedQuestions(rows,m['images'],path.parent,processor,c['max_sequence_length'])
    loader=DataLoader(dataset,batch_sampler=batches,num_workers=args.workers,
        collate_fn=BatchCollator(processor.tokenizer.pad_token_id,numpy_transport=True),pin_memory=False,
        multiprocessing_context='spawn' if args.workers else None,
        persistent_workers=bool(args.workers))
    iterator=iter(loader)
    local_stats=resume['rank_states'][rank]['stats'] if resume else {'sample_loss_sum':0.,'questions':0,'tokens':0,'answer_tokens':0,'active_seconds':0.}
    if resume:restore_rng(resume['rank_states'][rank]['rng'])
    begun=time.monotonic();torch.cuda.reset_peak_memory_stats(device);last_save=begun
    def save_resume(completed):
        nonlocal last_save
        now=time.monotonic();local_stats['active_seconds']+=now-last_save;last_save=now
        record={'rng':rng_state(),'stats':dict(local_stats)}
        gathered=[None]*world if rank==0 else None
        if world>1:dist.gather_object(record,gathered,dst=0)
        else:gathered=[record]
        if rank==0:
            destination=args.output_dir/f'resume_{completed:06d}';destination.mkdir(exist_ok=True)
            base.save_pretrained(destination/'adapter',safe_serialization=True)
            state={'optimizer':cpu_state(optimizer.state_dict()),'completed_updates':completed,
                'rank_states':gathered,'config_sha256':content_sha(c),'manifest_sha256':m['manifest_sha256'],
                'order_sha256':lock['global_question_order_sha256']}
            tmp=destination/'state.tmp';torch.save(state,tmp);tmp.replace(destination/'state.pt')
            write_json(latest,{'directory':destination.name,'state_sha256':sha_file(destination/'state.pt'),
                'adapter_files_sha256':{f.name:sha_file(f) for f in (destination/'adapter').iterdir() if f.is_file()}})
        if world>1:dist.barrier(device_ids=[local])
    for offset,(cpu_batch,valid_cpu,question_ids) in enumerate(iterator):
        step=start_step+offset
        cpu_batch={k:torch.from_numpy(v).pin_memory() for k,v in cpu_batch.items()}
        valid_cpu=torch.from_numpy(valid_cpu)
        batch=to_device(cpu_batch,device);valid=valid_cpu.to(device)
        actual=min(effective,len(rows)-step*effective)
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:group['lr']=optimizer_lr(step,steps,c)
        losses,_,answer_tokens=answer_loss(model,batch,tail=True)
        if not torch.isfinite(losses).all():raise FloatingPointError('nonfinite assistant loss')
        loss=(losses*valid).sum()*world/actual
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,c['gradient_clip_norm'],error_if_nonfinite=True)
        optimizer.step()
        local_stats['sample_loss_sum']+=float((losses.detach()*valid).sum())
        local_stats['questions']+=int(valid.sum())
        local_stats['tokens']+=int((batch['attention_mask']*valid[:,None]).sum())
        local_stats['answer_tokens']+=int(((batch['labels']!=-100)*valid[:,None]).sum())
        if rank==0 and ((step+1)%25==0 or step==0 or step+1==steps):
            print(json.dumps({'step':step+1,'total_steps':steps,'mean_question_loss':local_stats['sample_loss_sum']/max(local_stats['questions'],1),
                'grad_norm':float(norm),'lr':optimizer.param_groups[0]['lr'],'elapsed_active_seconds':local_stats['active_seconds']+time.monotonic()-last_save}),flush=True)
        if (step+1)%50==0 or step+1==steps:
            save_resume(step+1)
            if rank==0:write_json(args.output_dir/'PROGRESS.json',{'completed_updates':step+1,'planned_updates':steps,'status':'running'})
    del iterator,loader;gc.collect()
    base.eval();val_sum=0.;val_count=0
    with torch.inference_mode():
        for row in m['splits']['dev'][rank::world]:
            batch=to_device(collate([encode(processor,row,path.parent/m['images'][row['image_id']]['path'],True)],processor.tokenizer.pad_token_id),device)
            losses,nll,count=answer_loss(base,batch,tail=True)
            if not torch.isfinite(losses).all():raise FloatingPointError('nonfinite dev NLL')
            val_sum+=float(nll);val_count+=count
    values=torch.tensor([local_stats['sample_loss_sum'],local_stats['questions'],local_stats['tokens'],local_stats['answer_tokens'],val_sum,val_count],device=device,dtype=torch.float64)
    if world>1:dist.all_reduce(values)
    peak=torch.tensor([torch.cuda.max_memory_allocated(device),torch.cuda.max_memory_reserved(device)],device=device,dtype=torch.float64)
    if world>1:dist.all_reduce(peak,op=dist.ReduceOp.MAX)
    duration=local_stats['active_seconds']+time.monotonic()-last_save
    duration_tensor=torch.tensor(duration,device=device,dtype=torch.float64)
    if world>1:dist.all_reduce(duration_tensor,op=dist.ReduceOp.MAX)
    if rank==0:
        if int(values[1])!=len(rows):raise ValueError('not every real training question consumed once')
        after=frozen_parameter_hash(base)
        if after!=frozen_before:raise ValueError('frozen weights changed')
        for name,p in base.named_parameters():
            if p.requires_grad and not torch.isfinite(p).all():raise ValueError('nonfinite final LoRA weight')
        directory=args.output_dir/'adapter';base.save_pretrained(directory,safe_serialization=True)
        result={**lock,'completed_updates':steps,'epochs_completed':1,'actual_questions_seen':int(values[1]),
            'skipped_updates':0,'mean_question_loss':float(values[0]/values[1]),
            'internal_dev_token_nll':float(values[4]/values[5]),'dev_answer_tokens':int(values[5]),
            'training_input_tokens':int(values[2]),'training_answer_tokens':int(values[3]),
            'active_seconds':float(duration_tensor),'input_tokens_per_second':float(values[2]/duration_tensor),
            'peak_gpu_allocated_gib':float(peak[0]/2**30),'peak_gpu_reserved_gib':float(peak[1]/2**30),
            'frozen_parameter_sha256_after':after,'frozen_parameters_unchanged':True,
            'gpu_name':torch.cuda.get_device_name(device),'resumed_at_update':start_step,
            'adapter_files_sha256':{f.name:sha_file(f) for f in directory.iterdir() if f.is_file()}}
        write_json(args.output_dir/'TRAIN_RESULT.json',result)
        write_json(args.output_dir/'PROGRESS.json',{'completed_updates':steps,'planned_updates':steps,'status':'complete'})
        print('TRAINING COMPLETE',json.dumps({k:result[k] for k in ['method','seed','split','completed_updates','mean_question_loss','internal_dev_token_nll','active_seconds','peak_gpu_allocated_gib']}),flush=True)
    if world>1:dist.barrier(device_ids=[local]);dist.destroy_process_group()


if __name__=='__main__':main()
