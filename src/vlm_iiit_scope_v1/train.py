"""Same mean-CE/source weights; vary OCR coverage or optimized adapter scope."""
import json
import os
import random
import statistics
import time

from vlm_deep.common import content_sha,write_json,collate
from vlm_grpo_iteration_v1.common import actor,chosen_sample,read
from .initialization import ROOT,PARAMETERS
from .data import canonical


def train(role,plan):
    import numpy as np
    import torch
    from vlm_human_sft.evaluate import lora_hash
    from vlm_object_train.train import learning_rate,lock_json,save_checkpoint,resume_checkpoint,tensor_hash
    from vlm_sampled_signal_v1.trajectory import sequence_logps
    from vlm_sft.common import to_device
    directory=ROOT/'runs'/role/'train';directory.mkdir(parents=True,exist_ok=True);closed=directory/'TRAIN_COMPLETE.json'
    if closed.exists():
        result=read(closed);model,processor,_=actor(adapter=result['adapter']);return model,processor,result
    data=read(ROOT/'DATA.json');cpu=read(ROOT/'CPU_INPUTS.json');updates=read(ROOT/'SCHEDULE.json')[role];rows={q['question_id']:q for q in data['training_rows']}
    initial=read(ROOT/'INITIALIZATION.json');random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    model,processor,_=actor(training=True,adapter=initial['adapter'])
    if lora_hash(model)!=plan['initial_lora_sha256']:raise ValueError('Wrong common initialization')
    model.requires_grad_(False);visual_role=role=='dense_visual'
    for name,p in model.named_parameters():
        if 'lora_' in name and (('.visual.' in name)==visual_role):p.requires_grad_(True)
    trainable={n:p for n,p in model.named_parameters() if p.requires_grad}
    frozen={n:p for n,p in model.named_parameters() if 'lora_' in n and not p.requires_grad}
    if sum(p.numel() for p in trainable.values())!=PARAMETERS or len(trainable)!=(60 if visual_role else 144):raise ValueError('Wrong optimized scope')
    if any(p.dtype!=torch.float32 for p in trainable.values()):raise ValueError('Need original FP32 masters')
    frozen_hash=tensor_hash(frozen.items());parameters=list(trainable.values())
    lineage={'role':role,'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES'],'plan_sha256':__import__('hashlib').sha256((ROOT/'PLAN.json').read_bytes()).hexdigest(),
             'schedule_sha256':__import__('hashlib').sha256((ROOT/'SCHEDULE.json').read_bytes()).hexdigest()}
    lock_json(directory/'INPUT_LOCK.json',lineage);lineage_sha=content_sha(lineage);optimizer=torch.optim.AdamW(parameters,lr=2e-4,weight_decay=0.)
    stats={'active_seconds':0.,'trace_bytes':0,'sources':{'native':0,'ocr':0},'assistant_tokens':0,'per_question':{}};completed=0
    if (directory/'LATEST.json').exists():completed,stats=resume_checkpoint(directory,model,optimizer,lineage_sha,lineage['schedule_sha256'],64)
    path=directory/'INPUT_TRACES.jsonl';path.touch(exist_ok=True);prior=stats['active_seconds'];started=time.monotonic();durations=[]
    torch.cuda.reset_peak_memory_stats()
    with path.open('r+b') as trace:
        if path.stat().st_size<stats['trace_bytes']:raise ValueError('Saved trace truncated')
        trace.truncate(stats['trace_bytes']);trace.seek(0,os.SEEK_END)
        def checkpoint(step):
            stats.update(active_seconds=prior+time.monotonic()-started,trace_bytes=trace.tell());trace.flush();os.fsync(trace.fileno())
            save_checkpoint(directory,model,optimizer,step,stats,lineage_sha,lineage['schedule_sha256'])
        for step in range(completed,64):
            if prior+time.monotonic()-started>=1800:checkpoint(step);raise TimeoutError('Training cap1800sec')
            begin=time.monotonic();model.train();model.get_base_model().visual.train(visual_role);model.config.use_cache=False;optimizer.zero_grad(set_to_none=True)
            lr=learning_rate(step,64,{'lr':2e-4,'warmup_ratio':8/64})
            for group in optimizer.param_groups:group['lr']=lr
            totals={'native':0.,'ocr':0.}
            for micro,ids in enumerate(updates[step]['micros']):
                chunk=[rows[qid] for qid in ids];domain=chunk[0]['training_domain'];samples=[]
                if any(q['training_domain']!=domain for q in chunk):raise ValueError('Mixed-source micro')
                for q in chunk:
                    row={**q,'target':canonical(q['target'])} if domain=='ocr' else q
                    item=chosen_sample(processor,row,data['images'])
                    if any(content_sha(item[k].tolist())!=cpu[q['question_id']][k] for k in ['input_ids','labels']):raise ValueError('Actual input/labels differ')
                    samples.append(item)
                batch=to_device(collate(samples,processor.tokenizer.pad_token_id),next(model.parameters()).device);logps,counts=sequence_logps(model,batch);ce=-logps/counts
                if not bool(torch.isfinite(ce).all()):raise FloatingPointError('Nonfinite CE')
                (ce.sum()/(32 if domain=='ocr' else 16)).backward()
                for q,value,count in zip(chunk,ce.detach().cpu().tolist(),counts.tolist()):
                    totals[domain]+=value;stats['sources'][domain]+=1;stats['assistant_tokens']+=int(count);stats['per_question'][q['question_id']]=stats['per_question'].get(q['question_id'],0)+1
                    trace.write((json.dumps({'update':step+1,'micro':micro,'question_id':q['question_id'],'domain':domain,'chosen_ce':value,'assistant_tokens':int(count),
                        'input_sha256':cpu[q['question_id']]['input_ids'],'labels_sha256':cpu[q['question_id']]['labels']})+'\n').encode())
                del ce,logps,counts,batch,samples
            checks={}
            if step==0:
                if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in trainable.values()):raise ValueError('Missing/nonfinite selected gradient')
                if any(p.grad is not None for p in frozen.values()):raise ValueError('Frozen adapter received gradient')
                if visual_role:
                    a=[float(p.grad.norm()) for n,p in trainable.items() if '.lora_A.' in n];b=[float(p.grad.norm()) for n,p in trainable.items() if '.lora_B.' in n]
                    if len(a)!=30 or len(b)!=30 or any(v!=0 for v in a) or any(v<=0 for v in b):raise ValueError('Zero-B visual gradient path failed')
                    checks={'visual_A_initial_gradient_all_zero':True,'visual_B_nonzero_finite':30,'visual_B_gradient_min':min(b),'visual_B_gradient_max':max(b)}
            norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True))
            if norm<=0:raise ValueError('No learning gradient')
            peak=torch.cuda.max_memory_allocated()/2**30
            if peak>65:raise RuntimeError('Memory cap65GiB before optimizer')
            optimizer.step();torch.cuda.synchronize();durations.append(time.monotonic()-begin)
            if step==0 or (step+1)%8==0:checkpoint(step+1)
            progress={'status':'training','role':role,'completed_updates':step+1,'planned_updates':64,'native_ce':totals['native']/8,'ocr_ce':totals['ocr']/16,'gradient_norm':norm,'lr':lr,
                'active_seconds':prior+time.monotonic()-started,'recent_seconds_per_update':statistics.mean(durations[-8:]),'peak_allocated_gib':peak,'optimized_parameters':PARAMETERS,
                'optimized_factors':len(trainable),'visual_gradient_checkpointing':getattr(model.get_base_model().visual,'gradient_checkpointing',None)}
            write_json(directory/'PROGRESS.json',progress)
            if step==0:write_json(directory/'FIRST_UPDATE.json',{**progress,**checks,'all_actual_cpu_labels_matched':True,'frozen_adapter_gradients_none':True})
    actual_ocr=set(data['sparse_question_ids']) if role=='sparse_language' else{q['question_id'] for q in data['training_rows'] if q['training_domain']=='ocr'}
    if stats['sources']!={'native':512,'ocr':1024}:raise ValueError('Source exposure budget differs')
    if any(stats['per_question'].get(qid)!=(8 if role=='sparse_language' else 2) for qid in actual_ocr):raise ValueError('OCR exposures differ')
    if tensor_hash(frozen.items())!=frozen_hash:raise ValueError('Frozen adapter values changed')
    destination=directory/'adapter';model.save_pretrained(destination,safe_serialization=True)
    result={'status':'completed','role':role,'updates':64,'adapter':str(destination),'final_lora_sha256':lora_hash(model),'frozen_adapter_hash_unchanged':True,
            'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,**stats}
    write_json(closed,result);write_json(directory/'PROGRESS.json',result);return model,processor,result
