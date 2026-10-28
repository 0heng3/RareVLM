import json
import os
import random
import time
import numpy as np
import torch
from vlm_deep.common import content_sha,sha_file,write_json
from vlm_sft.common import to_device
from vlm_human_sft.evaluate import lora_hash
from vlm_object_train.train import learning_rate,lock_json,save_checkpoint,resume_checkpoint
from .common import ROOT,actor,chosen_batch,read
from .loss import forward_logits,loss_values
from .reference import reference_packet


def train(role,model=None,processor=None):
    plan=read(ROOT/'PLAN.json');cal=read(ROOT/'CALIBRATION.json');data=read(ROOT/'DATA.json');cpu=read(ROOT/'CPU_INPUTS.json');schedule=read(ROOT/'SCHEDULE.json')
    if not cal['training_allowed'] or cal['plan_sha256']!=sha_file(ROOT/'PLAN.json'):raise ValueError('Calibration does not admit training')
    beta=cal['beta'];lookup={q['question_id']:q for q in data['training_rows']};directory=ROOT/'runs'/role/'train';directory.mkdir(parents=True,exist_ok=True)
    lineage={'plan_sha256':sha_file(ROOT/'PLAN.json'),'schedule_sha256':sha_file(ROOT/'SCHEDULE.json'),'calibration_sha256':sha_file(ROOT/'CALIBRATION.json'),
             'role':role,'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES']}
    lock_json(directory/'INPUT_LOCK.json',lineage)
    if role=='soft_kl' and read(ROOT/'CACHE_INPUT_LOCK.json')['producer_uuid']!=lineage['gpu_uuid']:
        raise ValueError('Soft actor must retain teacher producer GPU affinity')
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    if model is None:model,processor,_=actor(training=True)
    if lora_hash(model)!=plan['initial_lora_sha256']:raise ValueError('Training must start from original SFT,not calibration checkpoint')
    parameters=[p for p in model.parameters() if p.requires_grad]
    if sum(p.numel() for p in parameters)!=1843200 or any(p.dtype!=torch.float32 for p in parameters):raise ValueError('Wrong FP32 qv trainables')
    optimizer=torch.optim.AdamW(parameters,lr=2e-4,weight_decay=0.)
    stats={'active_seconds':0.,'trace_bytes':0,'metrics_bytes':0,'sources':{'hard':0,'replay':0},'per_question':{},'assistant_tokens':0,'visual_tokens':0}
    completed=0
    if (directory/'LATEST.json').exists():completed,stats=resume_checkpoint(directory,model,optimizer,content_sha(lineage),lineage['schedule_sha256'],64)
    if (directory/'TRAIN_COMPLETE.json').exists():
        result=read(directory/'TRAIN_COMPLETE.json')
        if completed!=64 or lora_hash(model)!=result['final_lora_sha256']:raise ValueError('Completed checkpoint differs')
        return model,processor,result
    old_first=read(ROOT/'FIRST_UPDATE_REFERENCE.json');traces=directory/'INPUT_TRACES.jsonl';metrics=directory/'STEP_METRICS.jsonl';traces.touch(exist_ok=True);metrics.touch(exist_ok=True)
    prior=stats['active_seconds'];begun=time.monotonic();durations=[];torch.cuda.reset_peak_memory_stats()
    with traces.open('r+b') as trace,metrics.open('r+b') as metric:
        if traces.stat().st_size<stats['trace_bytes'] or metrics.stat().st_size<stats['metrics_bytes']:raise ValueError('Checkpoint log truncated')
        trace.truncate(stats['trace_bytes']);trace.seek(0,os.SEEK_END);metric.truncate(stats['metrics_bytes']);metric.seek(0,os.SEEK_END)
        def checkpoint(step):
            stats.update(active_seconds=prior+time.monotonic()-begun,trace_bytes=trace.tell(),metrics_bytes=metric.tell())
            trace.flush();metric.flush();os.fsync(trace.fileno());os.fsync(metric.fileno())
            save_checkpoint(directory,model,optimizer,step,stats,content_sha(lineage),lineage['schedule_sha256'])
        for step in range(completed,64):
            if prior+time.monotonic()-begun>1800:checkpoint(step);raise TimeoutError('1800sec training budget')
            start=time.monotonic();model.train();model.get_base_model().visual.eval();model.config.use_cache=False;optimizer.zero_grad(set_to_none=True)
            lr=learning_rate(step,64,{'lr':2e-4,'warmup_ratio':.125})
            for group in optimizer.param_groups:group['lr']=lr
            ce_totals={'hard':0.,'replay':0.};aux_total=0.;first=[];initial_kl=[]
            for micro,ids in enumerate(schedule[step]['micros']):
                rows=[lookup[qid] for qid in ids];source='replay' if micro in (0,2) else 'hard'
                if any(q['training_source']!=source for q in rows):raise ValueError('Schedule source differs')
                batch=chosen_batch(processor,rows,data['images'],cpu)
                refs=reference_packet(step,micro,batch,ids) if role=='soft_kl' and source=='replay' else None
                gpu=to_device(batch,next(model.parameters()).device);logits=forward_logits(model,gpu);ce,kl=loss_values(logits,gpu['labels'],refs)
                auxiliary=kl if refs is not None else ce if source=='replay' else None
                if not bool(torch.isfinite(ce).all()) or (auxiliary is not None and not bool(torch.isfinite(auxiliary).all())):
                    raise FloatingPointError('Nonfinite CE/regularizer')
                if kl is not None and bool((kl< -1e-4).any()):raise ValueError('KL below numerical tolerance')
                loss=ce.sum()/16
                if auxiliary is not None:loss=loss+beta*auxiliary.sum()/8
                loss.backward();ce_values=ce.detach().cpu().tolist();aux_values=auxiliary.detach().cpu().tolist() if auxiliary is not None else [None]*4
                token_counts=gpu['labels'].ne(-100).sum(1).tolist()
                ce_totals[source]+=sum(ce_values);aux_total+=sum(x for x in aux_values if x is not None)
                if step==0 and kl is not None:initial_kl+=aux_values
                for q,value,aux,count in zip(rows,ce_values,aux_values,token_counts):
                    record={'update':step+1,'micro':micro,'question_id':q['question_id'],'source':source,'chosen_ce':value,'regularizer':aux,
                            'input_sha256':cpu[q['question_id']]['input_ids'],'labels_sha256':cpu[q['question_id']]['labels'],
                            'assistant_tokens':int(count),'visual_tokens':cpu[q['question_id']]['visual_tokens']}
                    trace.write((json.dumps(record)+'\n').encode());first.append(record)
                    stats['per_question'][q['question_id']]=stats['per_question'].get(q['question_id'],0)+1
                    stats['assistant_tokens']+=int(count);stats['visual_tokens']+=record['visual_tokens']
                stats['sources'][source]+=4
                del batch,gpu,logits,ce,kl,auxiliary,refs,loss
            if step==0:
                if any(any(a[k]!=b[k] for k in ('question_id','input_sha256','labels_sha256')) for a,b in zip(first,old_first)) or len(first)!=16:
                    raise ValueError('First16 model input/labels differ')
                difference=max(abs(a['chosen_ce']-b['chosen_ce']) for a,b in zip(first,old_first))
                if difference>1e-5:raise ValueError('Initial baseCE differs beyond1e-5')
                if role=='soft_kl' and (len(initial_kl)!=8 or max(abs(x) for x in initial_kl)>1e-5):raise ValueError('Initial teacher KL parity failed')
                write_json(directory/'FIRST_UPDATE_CHECK.json',{'status':'passed','base_ce_max_difference':difference,'inputs_labels_exact':True,
                           'initial_regularizer_mean':aux_total/8,'initial_kl':initial_kl if role=='soft_kl' else None,
                           'hard_regularizer_nonzero_is_expected':role=='hard_replay','beta':beta,'completed_optimizer_updates':0})
            gradient=float(torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True))
            optimizer.step();torch.cuda.synchronize();seconds=time.monotonic()-start;durations.append(seconds);speed=sum(durations[-8:])/len(durations[-8:])
            peak=torch.cuda.max_memory_allocated()/2**30
            progress={'status':'training','role':role,'completed_updates':step+1,'planned_updates':64,'mean_hard_ce':ce_totals['hard']/8,
                      'mean_replay_ce':ce_totals['replay']/8,'regularizer_mean':aux_total/8,'beta':beta,'gradient_norm':gradient,'lr':lr,
                      'step_seconds':seconds,'recent_seconds_per_update':speed,'remaining_training_seconds_estimate':speed*(63-step),
                      'active_seconds':prior+time.monotonic()-begun,'peak_allocated_gib':peak,'sources':dict(stats['sources'])}
            metric.write((json.dumps(progress)+'\n').encode());metric.flush();trace.flush()
            if step==0 or (step+1)%8==0:checkpoint(step+1)
            write_json(directory/'PROGRESS.json',progress)
            if peak>65:checkpoint(step+1);raise RuntimeError('65GiB allocation budget')
    if stats['sources']!={'hard':512,'replay':512} or any(stats['per_question'][q['question_id']]!=(4 if q['training_source']=='hard' else 1) for q in data['training_rows']):
        raise ValueError('Formal exposure counts differ')
    destination=directory/'adapter';model.save_pretrained(destination,safe_serialization=True)
    result={'status':'completed','role':role,'updates':64,'adapter':str(destination),'final_lora_sha256':lora_hash(model),'beta':beta,
            'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,**stats}
    write_json(directory/'TRAIN_COMPLETE.json',result);return model,processor,result
