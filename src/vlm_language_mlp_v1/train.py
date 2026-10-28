import json
import os
import random
import time
import numpy as np
import torch
from vlm_deep.common import content_sha,sha_file,write_json
from vlm_sft.common import to_device
from vlm_reference_replay_v1.loss import forward_logits,loss_values
from vlm_human_sft.evaluate import lora_hash
from vlm_object_train.train import learning_rate,lock_json,save_checkpoint,resume_checkpoint
from .common import ROOT,PARENT,TOTAL_PARAMETERS,actor,chosen_batch,read,is_mlp


def train():
    plan=read(ROOT/'PLAN.json');data=read(PARENT/'DATA.json');cpu=read(PARENT/'CPU_INPUTS.json');schedule=read(PARENT/'SCHEDULE.json')
    lookup={q['question_id']:q for q in data['training_rows']};directory=ROOT/'runs/candidate/train';directory.mkdir(parents=True,exist_ok=True)
    lineage={'plan_sha256':sha_file(ROOT/'PLAN.json'),'schedule_sha256':sha_file(PARENT/'SCHEDULE.json'),'role':'candidate','gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES']}
    lock_json(directory/'INPUT_LOCK.json',lineage)
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    model,processor,_=actor(training=True)
    if lora_hash(model)!=plan['initial_lora_sha256']:raise ValueError('Wrong composite initial adapter')
    parameters=[p for p in model.parameters() if p.requires_grad]
    if len(parameters)!=360 or sum(p.numel() for p in parameters)!=TOTAL_PARAMETERS or any(p.dtype!=torch.float32 for p in parameters):raise ValueError('Wrong trainable inventory')
    if any(p.requires_grad and 'lora_' not in n for n,p in model.named_parameters()):raise ValueError('Non-LoRA parameters unfrozen')
    optimizer=torch.optim.AdamW(parameters,lr=2e-4,weight_decay=0.)
    stats={'active_seconds':0.,'trace_bytes':0,'metrics_bytes':0,'sources':{'hard':0,'replay':0},'per_question':{},'visual_tokens':0,'assistant_tokens':0};completed=0
    if (directory/'LATEST.json').exists():completed,stats=resume_checkpoint(directory,model,optimizer,content_sha(lineage),lineage['schedule_sha256'],64)
    if (directory/'TRAIN_COMPLETE.json').exists():
        result=read(directory/'TRAIN_COMPLETE.json')
        if completed!=64 or lora_hash(model)!=result['final_lora_sha256']:raise ValueError('Completed checkpoint differs')
        return model,processor,result
    old_first=read(PARENT/'FIRST_UPDATE_REFERENCE.json');traces=directory/'INPUT_TRACES.jsonl';metrics=directory/'STEP_METRICS.jsonl';traces.touch(exist_ok=True);metrics.touch(exist_ok=True)
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
            totals={'hard':0.,'replay':0.};first=[]
            for micro,ids in enumerate(schedule[step]['micros']):
                rows=[lookup[qid] for qid in ids];source='replay' if micro in (0,2) else 'hard'
                if any(q['training_source']!=source for q in rows):raise ValueError('Source/order differs')
                gpu=to_device(chosen_batch(processor,rows,data['images'],cpu),next(model.parameters()).device)
                logits=forward_logits(model,gpu);ce,_=loss_values(logits,gpu['labels'])
                if not bool(torch.isfinite(ce).all()):raise FloatingPointError('Nonfinite CE')
                (ce.sum()/16).backward();values=ce.detach().cpu().tolist();totals[source]+=sum(values)
                for q,value,tokens in zip(rows,values,gpu['labels'].ne(-100).sum(1).tolist()):
                    entry={'update':step+1,'micro':micro,'question_id':q['question_id'],'source':source,'chosen_ce':value,
                           'input_sha256':cpu[q['question_id']]['input_ids'],'labels_sha256':cpu[q['question_id']]['labels'],
                           'assistant_tokens':int(tokens),'visual_tokens':cpu[q['question_id']]['visual_tokens']}
                    trace.write((json.dumps(entry)+'\n').encode());first.append(entry)
                    qid=q['question_id'];stats['per_question'][qid]=stats['per_question'].get(qid,0)+1
                    stats['assistant_tokens']+=int(tokens);stats['visual_tokens']+=entry['visual_tokens']
                stats['sources'][source]+=4
                del gpu,logits,ce
            if step==0:
                if len(first)!=16 or any(any(a[k]!=b[k] for k in ('question_id','input_sha256','labels_sha256')) for a,b in zip(first,old_first)):raise ValueError('Initial16 input/labels mismatch')
                difference=max(abs(a['chosen_ce']-b['chosen_ce']) for a,b in zip(first,old_first))
                if difference>1e-5:raise ValueError('Zero-output MLP changed initial CE beyond1e-5')
                bgrad={n:float(p.grad.norm()) if p.grad is not None else None for n,p in model.named_parameters() if is_mlp(n) and '.lora_B.' in n}
                agrad={n:float(p.grad.norm()) if p.grad is not None else None for n,p in model.named_parameters() if is_mlp(n) and '.lora_A.' in n}
                if len(bgrad)!=108 or any(v is None or not np.isfinite(v) or v<=0 for v in bgrad.values()):raise ValueError('New MLP B has absent/nonfinite/zero gradients')
                if len(agrad)!=108 or any(v is None or v!=0 for v in agrad.values()):raise ValueError('Zero-B first-step A gradient must be zero')
                write_json(directory/'FIRST_SCOPE_CHECK.json',{'status':'passed','original16_inputs_labels_exact':True,'initial_ce_max_difference':difference,
                           'new_B_gradients':bgrad,'new_A_gradients':agrad,'completed_optimizer_updates':0,'trainable_parameters':TOTAL_PARAMETERS})
            scope_grad={}
            for label,predicate in [('qv',lambda n:not is_mlp(n)),('mlp_A',lambda n:is_mlp(n) and '.lora_A.' in n),('mlp_B',lambda n:is_mlp(n) and '.lora_B.' in n)]:
                values=[p.grad.detach().float().square().sum() for n,p in model.named_parameters() if p.requires_grad and predicate(n) and p.grad is not None]
                scope_grad[label]=float(torch.stack(values).sum().sqrt())
            norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True));optimizer.step();torch.cuda.synchronize()
            seconds=time.monotonic()-start;durations.append(seconds);speed=sum(durations[-8:])/len(durations[-8:]);peak=torch.cuda.max_memory_allocated()/2**30
            progress={'status':'training','completed_updates':step+1,'planned_updates':64,'mean_hard_ce':totals['hard']/8,'mean_replay_ce':totals['replay']/8,
                      'gradient_norm':norm,'scope_gradient_norms':scope_grad,'lr':lr,'step_seconds':seconds,'recent_seconds_per_update':speed,
                      'remaining_training_seconds_estimate':speed*(63-step),'active_seconds':prior+time.monotonic()-begun,'peak_allocated_gib':peak,'sources':dict(stats['sources'])}
            metric.write((json.dumps(progress)+'\n').encode());metric.flush();trace.flush()
            if step==0 or (step+1)%8==0:checkpoint(step+1)
            write_json(directory/'PROGRESS.json',progress)
            if peak>65:checkpoint(step+1);raise RuntimeError('65GiB allocation budget')
    if stats['sources']!={'hard':512,'replay':512} or any(stats['per_question'][q['question_id']]!=(4 if q['training_source']=='hard' else 1) for q in data['training_rows']):raise ValueError('Exposure totals mismatch')
    destination=directory/'adapter';model.save_pretrained(destination,safe_serialization=True)
    result={'status':'completed','updates':64,'adapter':str(destination),'final_lora_sha256':lora_hash(model),'trainable_parameters':TOTAL_PARAMETERS,**stats}
    write_json(directory/'TRAIN_COMPLETE.json',result);return model,processor,result
