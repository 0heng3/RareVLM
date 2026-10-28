import json
import os
import random
import time
import numpy as np
import torch
from vlm_deep.common import content_sha, sha_file, write_json
from vlm_human_sft.evaluate import lora_hash
from vlm_object_train.train import learning_rate, lock_json, save_checkpoint, resume_checkpoint
from .common import ROOT, PARENT, read, actor, gradients, flat, project


def train(role):
    plan=read(ROOT/'PLAN.json'); admission=read(ROOT/'DIAGNOSTIC.json')
    if not admission['training_allowed'] or admission['plan_sha256']!=sha_file(ROOT/'PLAN.json'):
        raise ValueError('Train-only mechanism gate did not pass')
    data=read(PARENT/'DATA.json');cpu=read(PARENT/'CPU_INPUTS.json');schedule=read(PARENT/'SCHEDULE.json')
    directory=ROOT/'runs'/role/'train';directory.mkdir(parents=True,exist_ok=True)
    lineage={'plan_sha256':sha_file(ROOT/'PLAN.json'),'schedule_sha256':sha_file(PARENT/'SCHEDULE.json'),
             'admission_sha256':sha_file(ROOT/'DIAGNOSTIC.json'),'role':role,'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES']}
    lock_json(directory/'INPUT_LOCK.json',lineage)
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    model,processor,_=actor(training=True)
    if lora_hash(model)!=plan['initial_lora_sha256']:raise ValueError('Must start from original SFT')
    parameters=[p for p in model.parameters() if p.requires_grad]
    if sum(p.numel() for p in parameters)!=1843200 or any(p.dtype!=torch.float32 for p in parameters):raise ValueError('FP32 qv trainables differ')
    optimizer=torch.optim.AdamW(parameters,lr=2e-4,weight_decay=0.)
    stats={'active_seconds':0.,'trace_bytes':0,'metrics_bytes':0,'sources':{'hard':0,'replay':0},'per_question':{},'assistant_tokens':0,'visual_tokens':0,'projected_updates':0}
    completed=0
    if (directory/'LATEST.json').exists():completed,stats=resume_checkpoint(directory,model,optimizer,content_sha(lineage),lineage['schedule_sha256'],64)
    if (directory/'TRAIN_COMPLETE.json').exists():
        result=read(directory/'TRAIN_COMPLETE.json')
        if completed!=64 or lora_hash(model)!=result['final_lora_sha256']:raise ValueError('Completed checkpoint differs')
        return model,processor,result
    old_first=read(PARENT/'FIRST_UPDATE_REFERENCE.json');traces=directory/'INPUT_TRACES.jsonl';metrics=directory/'STEP_METRICS.jsonl'
    traces.touch(exist_ok=True);metrics.touch(exist_ok=True);prior=stats['active_seconds'];begin=time.monotonic();durations=[]
    torch.cuda.reset_peak_memory_stats()
    with traces.open('r+b') as trace,metrics.open('r+b') as metric:
        for stream,key in [(trace,'trace_bytes'),(metric,'metrics_bytes')]:
            if os.fstat(stream.fileno()).st_size<stats[key]:raise ValueError('Truncated checkpoint log')
            stream.truncate(stats[key]);stream.seek(0,os.SEEK_END)
        def checkpoint(step):
            stats.update(active_seconds=prior+time.monotonic()-begin,trace_bytes=trace.tell(),metrics_bytes=metric.tell())
            trace.flush();metric.flush();os.fsync(trace.fileno());os.fsync(metric.fileno())
            save_checkpoint(directory,model,optimizer,step,stats,content_sha(lineage),lineage['schedule_sha256'])
        for step in range(completed,64):
            if prior+time.monotonic()-begin>1800:checkpoint(step);raise TimeoutError('1800sec training budget')
            start=time.monotonic();optimizer.zero_grad(set_to_none=True)
            lr=learning_rate(step,64,{'lr':2e-4,'warmup_ratio':.125})
            for group in optimizer.param_groups:group['lr']=lr
            _,vectors,totals,records=gradients(model,processor,schedule[step],data,cpu,update=step+1)
            if step==0:
                if len(records)!=16 or any(any(a[k]!=b[k] for k in ('question_id','input_sha256','labels_sha256')) for a,b in zip(records,old_first)):
                    raise ValueError('First16 inputs/labels differ')
                difference=max(abs(a['chosen_ce']-b['chosen_ce']) for a,b in zip(records,old_first))
                if difference>1e-5:raise ValueError('Initial baseCE mismatch')
                write_json(directory/'FIRST_UPDATE_CHECK.json',{'status':'passed','base_ce_max_difference':difference,'inputs_labels_exact':True,'completed_optimizer_updates':0})
            before=flat(parameters).clone()
            norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True));optimizer.step()
            proposed=flat(parameters)-before;projected,detail=project(proposed,vectors)
            if role=='projected' and detail['active_constraints']:
                offset=0
                with torch.no_grad():
                    corrected=projected.to(device=before.device,dtype=torch.float32)
                    for p in parameters:
                        count=p.numel();p.copy_((before[offset:offset+count]+corrected[offset:offset+count]).reshape_as(p));offset+=count
                stats['projected_updates']+=1
            actual=flat(parameters)-before
            detail['actual_normalized_projections']={s:float(torch.dot(g.double(),actual.double())/g.double().norm()) for s,g in vectors.items()}
            detail['actual_displacement_norm']=float(actual.double().norm())
            for record in records:
                trace.write((json.dumps(record)+'\n').encode());qid=record['question_id'];source=record['source']
                stats['per_question'][qid]=stats['per_question'].get(qid,0)+1;stats['sources'][source]+=1
                stats['assistant_tokens']+=record['assistant_tokens'];stats['visual_tokens']+=record['visual_tokens']
            torch.cuda.synchronize();seconds=time.monotonic()-start;durations.append(seconds);speed=sum(durations[-8:])/len(durations[-8:])
            progress={'status':'training','role':role,'completed_updates':step+1,'planned_updates':64,'mean_hard_ce':totals['hard']/8,
                      'mean_replay_ce':totals['replay']/8,'gradient_norm':norm,'lr':lr,'step_seconds':seconds,'recent_seconds_per_update':speed,
                      'remaining_training_seconds_estimate':speed*(63-step),'active_seconds':prior+time.monotonic()-begin,
                      'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'sources':dict(stats['sources']),
                      'projection_applied':role=='projected' and bool(detail['active_constraints']),'geometry':detail}
            metric.write((json.dumps(progress)+'\n').encode());metric.flush();trace.flush()
            del before,proposed,projected,actual,vectors
            if step==0 or (step+1)%8==0:checkpoint(step+1)
            write_json(directory/'PROGRESS.json',progress)
            if progress['peak_allocated_gib']>65:checkpoint(step+1);raise RuntimeError('65GiB allocation budget')
    if stats['sources']!={'hard':512,'replay':512} or any(stats['per_question'][q['question_id']]!=(4 if q['training_source']=='hard' else 1) for q in data['training_rows']):
        raise ValueError('Formal exposure counts differ')
    destination=directory/'adapter';model.save_pretrained(destination,safe_serialization=True)
    result={'status':'completed','role':role,'updates':64,'adapter':str(destination),'final_lora_sha256':lora_hash(model),**stats}
    write_json(directory/'TRAIN_COMPLETE.json',result);return model,processor,result
