import json
import os
import random
import time
import numpy as np
import torch
from vlm_deep.common import content_sha,sha_file,write_json
from vlm_sft.common import to_device
from vlm_sampled_signal_v1.trajectory import sequence_logps
from vlm_object_train.train import learning_rate,lock_json,save_checkpoint,resume_checkpoint
from vlm_human_sft.evaluate import lora_hash
from .common import ROOT,actor,chosen_batch,read


def train(role):
    plan=read(ROOT/'PLAN.json');data=read(ROOT/'DATA.json');cpu=read(ROOT/'CPU_INPUTS.json')
    updates=read(ROOT/'SCHEDULE.json')['roles'][role];lookup={q['question_id']:q for q in data['training_rows']}
    directory=ROOT/'runs'/role/'train';directory.mkdir(parents=True,exist_ok=True)
    lineage={'plan_sha256':sha_file(ROOT/'PLAN.json'),'schedule_sha256':sha_file(ROOT/'SCHEDULE.json'),
        'role':role,'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES']}
    lock_json(directory/'INPUT_LOCK.json',lineage);lineage_sha=content_sha(lineage)
    if (directory/'TRAIN_COMPLETE.json').exists():
        result=read(directory/'TRAIN_COMPLETE.json');model,processor,_=actor(adapter=result['adapter'])
        if lora_hash(model)!=result['final_lora_sha256']:raise ValueError('Reloaded Final differs')
        return model,processor,result
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    model,processor,binding=actor(training=True)
    if lora_hash(model)!=plan['lora_sha256']:raise ValueError('Initial model differs')
    parameters=[p for p in model.parameters() if p.requires_grad]
    if sum(p.numel() for p in parameters)!=1982464 or any(p.dtype!=torch.float32 for p in parameters):raise ValueError('Wrong FP32 joint qv/merger trainables')
    optimizer=torch.optim.AdamW(parameters,lr=2e-4,weight_decay=0.)
    stats={'active_seconds':0.,'trace_bytes':0,'metrics_bytes':0,'sources':{'hard':0,'replay':0},'per_question':{},'visual_tokens':0,'assistant_tokens':0}
    completed=0
    if (directory/'LATEST.json').exists():completed,stats=resume_checkpoint(directory,model,optimizer,lineage_sha,lineage['schedule_sha256'],64)
    traces=directory/'INPUT_TRACES.jsonl';metrics=directory/'STEP_METRICS.jsonl'
    traces.touch(exist_ok=True);metrics.touch(exist_ok=True);prior=stats['active_seconds'];begun=time.monotonic();durations=[]
    torch.cuda.reset_peak_memory_stats()
    with traces.open('r+b') as trace,metrics.open('r+b') as metric:
        if traces.stat().st_size<stats['trace_bytes'] or metrics.stat().st_size<stats['metrics_bytes']:raise ValueError('Checkpoint logs truncated')
        trace.truncate(stats['trace_bytes']);trace.seek(0,os.SEEK_END);metric.truncate(stats['metrics_bytes']);metric.seek(0,os.SEEK_END)
        def checkpoint(step):
            stats.update(active_seconds=prior+time.monotonic()-begun,trace_bytes=trace.tell(),metrics_bytes=metric.tell())
            trace.flush();metric.flush();os.fsync(trace.fileno());os.fsync(metric.fileno())
            save_checkpoint(directory,model,optimizer,step,stats,lineage_sha,lineage['schedule_sha256'])
        for step in range(completed,64):
            if prior+time.monotonic()-begun>=1800:checkpoint(step);raise TimeoutError('1800sec fixed training budget')
            started=time.monotonic();model.train();model.get_base_model().visual.eval();model.config.use_cache=False
            optimizer.zero_grad(set_to_none=True);lr=learning_rate(step,64,{'lr':2e-4,'warmup_ratio':.125})
            for group in optimizer.param_groups:group['lr']=lr
            totals={'hard':0.,'replay':0.};first_replay=[];first_actual=[]
            for micro,ids in enumerate(updates[step]['micros']):
                chunk=[lookup[qid] for qid in ids];source='replay' if micro in (0,2) else 'hard'
                if any(q['training_source']!=source for q in chunk):raise ValueError('Training source order changed')
                batch=to_device(chosen_batch(processor,chunk,data['images'],cpu),next(model.parameters()).device)
                logps,counts=sequence_logps(model,batch);ces=-logps/counts
                if not bool(torch.isfinite(ces).all()):raise FloatingPointError('Nonfinite CE')
                (ces.sum()/16).backward();values=ces.detach().cpu().tolist()
                for q,ce,count in zip(chunk,values,counts.tolist()):
                    entry={'update':step+1,'micro':micro,'question_id':q['question_id'],'image_id':q['image_id'],'source':source,
                        'chosen_ce':ce,'input_sha256':cpu[q['question_id']]['input_ids'],'labels_sha256':cpu[q['question_id']]['labels'],
                        'assistant_tokens':int(count),'visual_tokens':cpu[q['question_id']]['visual_tokens']}
                    first_actual.append(entry) if step==0 else None
                    trace.write((json.dumps(entry)+'\n').encode());stats['per_question'][q['question_id']]=stats['per_question'].get(q['question_id'],0)+1
                    stats['visual_tokens']+=entry['visual_tokens'];stats['assistant_tokens']+=entry['assistant_tokens']
                    if step==0 and source=='replay':first_replay.append({'question_id':q['question_id'],'ce':ce,'input_sha256':entry['input_sha256']})
                totals[source]+=sum(values);stats['sources'][source]+=4
                del batch,logps,counts,ces
            visual_gradients={name:float(p.grad.norm()) if p.grad is not None else None for name,p in model.named_parameters() if 'lora_' in name and '.visual.merger.' in name}
            if step==0:
                expected=read(ROOT/'FIRST_UPDATE_REFERENCE.json')
                if len(first_actual)!=16 or any(any(a[k]!=b[k] for k in ('question_id','input_sha256','labels_sha256')) for a,b in zip(first_actual,expected)):
                    raise ValueError('First-step actual input/labels differ from fixed qv control')
                max_ce_delta=max(abs(a['chosen_ce']-b['chosen_ce']) for a,b in zip(first_actual,expected))
                new_b=[v for name,v in visual_gradients.items() if '.lora_B.' in name]
                if len(visual_gradients)!=4 or len(new_b)!=2 or any(v is None or not np.isfinite(v) or v<=0 for v in new_b):
                    raise ValueError('New merger B has missing/nonfinite/zero gradient before optimizer')
                if max_ce_delta>1e-5:raise ValueError('Zero-output composite initial CE differs beyond preregistered1e-5')
                write_json(directory/'FIRST_SCOPE_CHECK.json',{'status':'passed','actual_first16_inputs_labels_exact':True,'max_initial_ce_delta':max_ce_delta,
                    'new_visual_gradients_before_clip':visual_gradients,'completed_optimizer_updates':0,'initial_lora_sha256':plan['lora_sha256']})
            norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True))
            if norm<=0:raise ValueError('No training gradient')
            optimizer.step();torch.cuda.synchronize()
            elapsed=time.monotonic()-started;durations.append(elapsed);speed=sum(durations[-8:])/len(durations[-8:])
            stats['active_seconds']=prior+time.monotonic()-begun;peak=torch.cuda.max_memory_allocated()/2**30
            progress={'status':'training','role':role,'completed_updates':step+1,'planned_updates':64,'mean_hard_ce':totals['hard']/8,
                'mean_replay_ce':totals['replay']/8,'gradient_norm':norm,'visual_gradients_before_clip':visual_gradients,'lr':lr,'step_seconds':elapsed,'recent_seconds_per_update':speed,
                'remaining_training_seconds_estimate':speed*(63-step),'peak_allocated_gib':peak,'active_seconds':stats['active_seconds'],
                'sources':dict(stats['sources']),'assistant_tokens':stats['assistant_tokens'],'visual_tokens':stats['visual_tokens']}
            metric.write((json.dumps(progress)+'\n').encode());metric.flush();trace.flush()
            if step==0 or (step+1)%8==0:checkpoint(step+1)
            if step==0:write_json(directory/'FIRST_UPDATE.json',{'status':'completed','shared_replay':first_replay,'gradient_norm':norm,'lr':lr})
            write_json(directory/'PROGRESS.json',progress)
            if peak>65:checkpoint(step+1);raise RuntimeError('65GiB allocated budget exceeded')
    if stats['sources']!={'hard':512,'replay':512}:raise ValueError('Source exposures differ')
    hard_counts=[v for qid,v in stats['per_question'].items() if lookup[qid]['training_source']=='hard']
    if len(hard_counts)!=128 or set(hard_counts)!={4}:raise ValueError('Coverage exposure contract differs')
    destination=directory/'adapter';model.save_pretrained(destination,safe_serialization=True)
    result={'status':'completed','role':role,'updates':64,'adapter':str(destination),'final_lora_sha256':lora_hash(model),
        'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,**stats}
    write_json(directory/'TRAIN_COMPLETE.json',result);return model,processor,result
