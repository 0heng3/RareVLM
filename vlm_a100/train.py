"""Two-A100 runtime for frozen VLM deep v2; same scientific batch/order/loss."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import time
from pathlib import Path

from vlm_deep.common import answer_loss,collate,content_sha,encode,frozen_parameter_hash,read_manifest,sha_file,to_device,write_json
from vlm_deep.model import load,parameter_groups,set_training


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


def runtime_layout(world,micro,acc,config):
    """Separate physical execution layout from the untouched scientific config."""
    if (world,micro,acc)!=(2,8,1):raise ValueError('validated A100 runtime requires world2/micro8/accumulation1')
    if world*micro*acc!=config['effective_batch']:raise ValueError('effective batch changed')
    if config['precision']!='fp32' or config['training_quantization']!='nf4_language_only':
        raise ValueError('this migration is validated only for frozen NF4-language/FP32 compute')
    return {'world_size':world,'microbatch':micro,'accumulation':acc,
            'effective_batch':world*micro*acc,'origin':'explicit A100 runtime CLI',
            'scientific_config_layout':{'world_size':config['world_size'],
                'microbatch':config['training_microbatch'],'accumulation':config['training_accumulation'],
                'effective_batch':config['effective_batch']}}


def source_fingerprints():
    root=Path(__file__).resolve().parents[1]
    original={str(f.relative_to(root/'vlm_deep')):sha_file(f)
              for f in sorted((root/'vlm_deep').rglob('*.py'))}
    adapted={name:sha_file(root/'vlm_a100'/name) for name in ('__init__.py','train.py')}
    return original,adapted


def local_block(padded,step,effective,rank,micro):
    """Same global block as four-card runner, repartitioned into two ranks."""
    block=padded[step*effective:(step+1)*effective]
    fallback=next(x for x in block if x>=0)
    return [(x if x>=0 else fallback,x>=0) for x in block[rank*micro:(rank+1)*micro]]


def gradient_group_contract(model,method,config):
    """Check group signal, allowing the initially zero LoRA-A gradients."""
    import torch
    groups={name:{'trainable_tensors':0,'gradient_tensors':0,
                  'finite_gradient_tensors':0,'nonzero_gradient_tensors':0,
                  'gradient_l2_squared':0.} for name in ('language','merger')}
    for name,parameter in model.named_parameters():
        if not parameter.requires_grad:continue
        group=groups['merger' if '.visual.merger.' in name else 'language']
        group['trainable_tensors']+=1
        if parameter.grad is None:continue
        gradient=parameter.grad.detach()
        group['gradient_tensors']+=1
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError(f'nonfinite first-update gradient: {name}')
        group['finite_gradient_tensors']+=1
        group['nonzero_gradient_tensors']+=int(bool(gradient.ne(0).any()))
        group['gradient_l2_squared']+=float(gradient.double().square().sum())
    required=['language']+(['merger'] if config['methods'][method].get('merger_ranks') else [])
    for name,group in groups.items():
        group['gradient_l2_norm']=math.sqrt(group.pop('gradient_l2_squared'))
        group['required']=name in required
        if group['required'] and (group['finite_gradient_tensors']<1 or group['nonzero_gradient_tensors']<1):
            raise ValueError(f'first-update LoRA group has no finite nonzero gradient: {name}')
    return {'checked_at_update':1,'position':'after backward, before gradient clipping',
            'criterion':'at least one finite nonzero gradient tensor in each required group',
            'groups':groups,'passed':True}


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
    p.add_argument('--runtime-world-size',type=int,default=2)
    p.add_argument('--runtime-microbatch',type=int,default=8)
    p.add_argument('--runtime-accumulation',type=int,default=1)
    p.add_argument('--contract-questions',type=int,choices=[33],default=None,
                   help='nonresearch check only: first33 questions,3 updates including1-question tail')
    p.add_argument('--contract-stop-after-updates',type=int,default=None,
                   help='contract-only checkpoint/pause control; omitted on checkpoint restore')
    args=p.parse_args();c=json.loads(args.config.read_text())
    rank=int(os.environ.get('RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'));local=int(os.environ.get('LOCAL_RANK','0'))
    if world!=args.runtime_world_size:raise ValueError('torchrun world differs from explicit runtime world')
    layout=runtime_layout(world,args.runtime_microbatch,args.runtime_accumulation,c)
    if args.contract_stop_after_updates is not None and args.contract_questions is None:
        raise ValueError('manual stop is restricted to the separate nonresearch contract')
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise ValueError('TF32 was not disabled')
    torch.cuda.set_device(local);device=torch.device('cuda',local);torch.set_num_threads(2)
    if world>1:dist.init_process_group('nccl')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    path=args.asset_root/'prepared/manifest.json';m=read_manifest(path);rows=m['splits'][args.split]
    full_question_count=len(rows)
    if args.contract_questions is not None:
        if len(rows)<args.contract_questions:raise ValueError('contract subset exceeds available rows')
        rows=rows[:args.contract_questions]
    micro=layout['microbatch'];acc=layout['accumulation'];effective=layout['effective_batch']
    if effective!=c['effective_batch']:raise ValueError('effective batch changed')
    if acc!=1:raise ValueError('this validated runner currently requires accumulation1; lower-batch repair needs its own verified implementation')
    order,padded=schedule(len(rows),args.seed,world,micro,acc);steps=len(padded)//effective
    if args.contract_stop_after_updates is not None and not 1<=args.contract_stop_after_updates<=steps:
        raise ValueError('contract stop must fall inside planned updates')
    if args.contract_questions is not None and 'runs' in args.output_dir.parts:
        raise ValueError('contract checks must be outside research runs tables')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for active_rank in range(world):
        if rank==active_rank:model,processor=load(args.model_root,c,device,args.method,training=True)
        if world>1:dist.barrier(device_ids=[local])
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=optimizer_lr(0,steps,c),weight_decay=c['weight_decay'])
    source,runtime_source=source_fingerprints()
    hardware={'devices':[{'visible_index':i,'name':torch.cuda.get_device_name(i),
                         'total_memory_bytes':torch.cuda.get_device_properties(i).total_memory,
                         'compute_capability':list(torch.cuda.get_device_capability(i))} for i in range(world)],
              'torch_version':torch.__version__,'cuda_version':torch.version.cuda}
    compute={'precision':'fp32','training_quantization':'nf4_language_only',
             'vision_precision':'fp32','float32_matmul_precision':torch.get_float32_matmul_precision(),
             'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
             'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32}
    dev_rows=m['splits']['dev'][:3] if args.contract_questions is not None else m['splits']['dev']
    frozen_before=frozen_parameter_hash(model) if rank==0 else None
    if world>1:
        canonical_frozen=[frozen_before]
        dist.broadcast_object_list(canonical_frozen,src=0)
        frozen_before=canonical_frozen[0]
    lock={'schema_version':'rarevlm.vlm_train.v2','config_sha256':content_sha(c),'manifest_sha256':m['manifest_sha256'],
        'method':args.method,'seed':args.seed,'split':args.split,'question_count':len(rows),
        'unique_training_images':len({r['image_id'] for r in rows}),'world_size':world,
        'effective_batch':effective,'microbatch':micro,'accumulation':acc,'planned_updates':steps,
        'global_question_order_sha256':content_sha([rows[i]['question_id'] for i in order]),
        'trainable_names':{n:p.numel() for n,p in model.named_parameters() if p.requires_grad},
        'parameter_groups':parameter_groups(model),'trainable_parameters':sum(p.numel() for p in params),
        'total_parameters':model.num_parameters(),'frozen_parameter_sha256_before':frozen_before,
        'source_sha256':source,'runtime_source_sha256':runtime_source,
        'runtime_source_scope':['vlm_a100/__init__.py','vlm_a100/train.py'],
        'runtime_schema_version':'rarevlm.a100_train_runtime.v1','runtime_layout':layout,
        'hardware':hardware,'compute':compute,
        'scientific_config_origin':{'path':str(args.config.resolve()),'sha256':content_sha(c),
            'config_file_sha256':sha_file(args.config),'config_is_unmodified':True},
        'run_purpose':'nonresearch_runtime_contract' if args.contract_questions is not None else 'research',
        'sample_subset':{'selection':'first33 from unchanged manifest split' if args.contract_questions is not None else 'all questions in manifest split',
            'manifest_split_question_count':full_question_count,'expected_count':len(rows),
            'selected_question_ids_sha256':content_sha([r['question_id'] for r in rows]),
            'internal_dev_question_count':len(dev_rows)},
        'runtime_comparability':'same scientific config and global sample/update schedule; GPU/layout changed and stochastic trajectories may differ',
        'tail_head_optimization':'temporary head-input slicing, equivalence preflight required',
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
        if state.get('training_lock_sha256')!=content_sha(lock):raise ValueError('resume input/runtime/source lock changed')
        if state.get('runtime_layout')!=layout or state.get('source_sha256')!=source or state.get('runtime_source_sha256')!=runtime_source:
            raise ValueError('resume world/layout/code changed')
        if len(state['rank_states'])!=world:raise ValueError('cannot restore a checkpoint from another rank layout')
        if not 0<=state['completed_updates']<=steps:raise ValueError('checkpoint completed updates outside schedule')
        set_peft_model_state_dict(model,load_file(str(checkpoint/'adapter/adapter_model.safetensors')))
        optimizer.load_state_dict(state['optimizer']);resume=state
    start_step=resume['completed_updates'] if resume else 0
    if world>1:model=DistributedDataParallel(model,device_ids=[local],find_unused_parameters=False)
    base=model.module if world>1 else model
    batches=[]
    for step in range(start_step,steps):
        batches.append(local_block(padded,step,effective,rank,micro))
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
                'order_sha256':lock['global_question_order_sha256'],
                'training_lock_sha256':content_sha(lock),'runtime_layout':layout,
                'source_sha256':source,'runtime_source_sha256':runtime_source}
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
        loss.backward()
        if step==0:
            local_stats['first_update_gradient_groups']=gradient_group_contract(base,args.method,c)
            if rank==0:
                print('FIRST UPDATE GRADIENT GROUPS',json.dumps(local_stats['first_update_gradient_groups']),flush=True)
        norm=torch.nn.utils.clip_grad_norm_(params,c['gradient_clip_norm'],error_if_nonfinite=True)
        optimizer.step()
        local_stats['sample_loss_sum']+=float((losses.detach()*valid).sum())
        local_stats['questions']+=int(valid.sum())
        local_stats['tokens']+=int((batch['attention_mask']*valid[:,None]).sum())
        local_stats['answer_tokens']+=int(((batch['labels']!=-100)*valid[:,None]).sum())
        if rank==0 and (args.contract_questions is not None or (step+1)%25==0 or step==0 or step+1==steps):
            print(json.dumps({'step':step+1,'total_steps':steps,'mean_question_loss':local_stats['sample_loss_sum']/max(local_stats['questions'],1),
                'grad_norm':float(norm),'lr':optimizer.param_groups[0]['lr'],'elapsed_active_seconds':local_stats['active_seconds']+time.monotonic()-last_save}),flush=True)
        if args.contract_questions is not None:
            print(json.dumps({'contract':True,'rank':rank,'step':step+1,'actual_global_questions':actual,
                  'local_valid_questions':int(valid.sum()),'ddp_local_loss_scale':world/actual}),flush=True)
        stop_here=args.contract_stop_after_updates is not None and step+1==args.contract_stop_after_updates
        if (step+1)%50==0 or step+1==steps or stop_here:
            save_resume(step+1)
            if rank==0:write_json(args.output_dir/'PROGRESS.json',{'completed_updates':step+1,'planned_updates':steps,'status':'running'})
        if stop_here:
            if rank==0:
                write_json(args.output_dir/'PROGRESS.json',{'completed_updates':step+1,'planned_updates':steps,'status':'paused_contract'})
                print('CONTRACT CHECKPOINT PAUSED',json.dumps({'completed_updates':step+1,'planned_updates':steps}),flush=True)
            del iterator,loader;gc.collect()
            if world>1:dist.barrier(device_ids=[local]);dist.destroy_process_group()
            return
    del iterator,loader;gc.collect()
    base.eval();val_sum=0.;val_count=0
    with torch.inference_mode():
        for row in dev_rows[rank::world]:
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
            'first_update_gradient_groups':local_stats['first_update_gradient_groups'],
            'adapter_files_sha256':{f.name:sha_file(f) for f in directory.iterdir() if f.is_file()}}
        write_json(args.output_dir/'TRAIN_RESULT.json',result)
        write_json(args.output_dir/'PROGRESS.json',{'completed_updates':steps,'planned_updates':steps,'status':'complete'})
        print('TRAINING COMPLETE',json.dumps({k:result[k] for k in ['method','seed','split','completed_updates','mean_question_loss','internal_dev_token_nll','active_seconds','peak_gpu_allocated_gib']}),flush=True)
    if world>1:dist.barrier(device_ids=[local]);dist.destroy_process_group()


if __name__=='__main__':main()
