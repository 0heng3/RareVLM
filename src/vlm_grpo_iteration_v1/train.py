import os
import random
import time
import numpy as np
import torch
from vlm_deep.common import content_sha, sha_file, write_json
from vlm_sft.common import to_device
from vlm_human_sft.evaluate import lora_hash
from vlm_object_train.train import learning_rate, lock_json, save_checkpoint, resume_checkpoint
from vlm_sampled_signal_v1.trajectory import sequence_logps
from .common import ROOT, actor, chosen_batch, read, rollout_batch
from .loss import token_logps, behavior_logps, clipped_group_loss


def train(role):
    if not read(ROOT / 'REFERENCE_COMPLETE.json')['passed']:
        raise ValueError('Initial behavior/replay numerical gate failed; no training')
    plan = read(ROOT / 'PLAN.json')
    data = read(ROOT / 'DATA.json')
    cpu = read(ROOT / 'CPU_INPUTS.json')
    schedule = read(ROOT / 'SCHEDULE.json')['updates']
    rows = {q['question_id']: q for q in data['rows']}
    groups = {g['question_id']: g for g in read(ROOT / 'REVIEWED_ROLLOUTS.json')}
    directory = ROOT / 'runs' / role / 'train'
    directory.mkdir(parents=True, exist_ok=True)
    lineage = {'plan_sha256': sha_file(ROOT / 'PLAN.json'), 'role': role,
        'gpu_uuid': os.environ['CUDA_VISIBLE_DEVICES'], 'schedule_sha256': sha_file(ROOT / 'SCHEDULE.json')}
    lock_json(directory / 'INPUT_LOCK.json', lineage)
    lineage_sha = content_sha(lineage)
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    if (directory / 'TRAIN_COMPLETE.json').exists():
        result = read(directory / 'TRAIN_COMPLETE.json')
        model, processor, _ = actor(adapter=result['adapter'])
        return model, processor, result
    model, processor, binding = actor(training=True)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if sum(p.numel() for p in parameters) != 1843200 or any(p.dtype != torch.float32 for p in parameters):
        raise ValueError('Wrong trainable FP32 q/v parameters')
    optimizer = torch.optim.AdamW(parameters, lr=2e-5, weight_decay=0.)
    stats = {'active_seconds':0., 'chosen_exposures':0, 'rollout_sequences':0, 'rollout_groups':[], 'trace_bytes':0}
    completed = 0
    if (directory / 'LATEST.json').exists():
        completed, stats = resume_checkpoint(directory, model, optimizer, lineage_sha, lineage['schedule_sha256'], 32)
    path = directory / 'INPUT_TRACES.jsonl'
    path.touch(exist_ok=True)
    prior = stats['active_seconds'];begun = time.monotonic();durations=[]
    torch.cuda.reset_peak_memory_stats()
    with path.open('r+b') as trace:
        if path.stat().st_size < stats['trace_bytes']:
            raise ValueError('Training trace shorter than checkpoint')
        trace.truncate(stats['trace_bytes']);trace.seek(0,os.SEEK_END)
        def checkpoint(step):
            stats['active_seconds'] = prior + time.monotonic() - begun
            stats['trace_bytes'] = trace.tell()
            trace.flush();os.fsync(trace.fileno())
            save_checkpoint(directory,model,optimizer,step,stats,lineage_sha,lineage['schedule_sha256'])
        for step in range(completed,32):
            if prior+time.monotonic()-begun >= 900:
                checkpoint(step);raise TimeoutError('Fixed900sec training budget reached')
            started=time.monotonic();model.train();model.get_base_model().visual.eval();model.config.use_cache=False
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group['lr']=learning_rate(step,32,{'lr':2e-5,'warmup_ratio':.125})
            ce_sum=0.;policy_stats={'policy_loss':0.,'ratio_clip_fraction':0.,'active_actions':0}
            for micro,ids in enumerate(schedule[step]['micros']):
                chunk=[rows[q] for q in ids]
                batch=to_device(chosen_batch(processor,chunk,data['images'],cpu),next(model.parameters()).device)
                logps,counts=sequence_logps(model,batch)
                ce=-logps/counts
                (ce.sum()/16).backward()
                ce_sum+=float(ce.detach().sum())
                for q,cv in zip(chunk,ce.detach().cpu().tolist()):
                    trace.write((__import__('json').dumps({'update':step+1,'micro':micro,'question_id':q['question_id'],
                        'chosen_ce':cv,'input_sha256':cpu[q['question_id']]['input_ids'],'labels_sha256':cpu[q['question_id']]['labels']})+'\n').encode())
                del batch,logps,counts,ce
            qid=data['hard_eval'][step]['question_id']
            if role=='ce_grpo' and qid in groups:
                g=groups[qid]
                batch=to_device(rollout_batch(processor,g,rows[qid],data['images']),next(model.parameters()).device)
                current,valid=token_logps(model,batch)
                old=behavior_logps(g['samples'],current,valid)
                advantage=torch.tensor(g['advantages'],device=current.device,dtype=current.dtype)
                loss,policy_stats=clipped_group_loss(current,old,valid,advantage,.2)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Nonfinite GRPO loss')
                loss.backward()
                stats['rollout_sequences']+=8;stats['rollout_groups'].append(qid)
                del batch,current,valid,old,advantage,loss
            norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True))
            if norm<=0:
                raise ValueError('Zero training gradient')
            optimizer.step();torch.cuda.synchronize()
            stats['chosen_exposures']+=16
            stats['active_seconds']=prior+time.monotonic()-begun
            peak=torch.cuda.max_memory_allocated()/2**30
            if peak>65:
                checkpoint(step+1);raise RuntimeError('65GiB memory budget exceeded')
            if step==0 or (step+1)%8==0:checkpoint(step+1)
            durations.append(time.monotonic()-started)
            seconds=sum(durations[-8:])/len(durations[-8:])
            write_json(directory/'PROGRESS.json',{'status':'training','role':role,'completed_updates':step+1,
                'planned_updates':32,'mean_ce':ce_sum/16,'gradient_norm':norm,'recent_seconds_per_update':seconds,
                'remaining_training_seconds_estimate':seconds*(31-step),'peak_allocated_gib':peak,**policy_stats,**stats})
    if stats['chosen_exposures']!=512 or stats['rollout_sequences']!=(136 if role=='ce_grpo' else 0):
        raise ValueError('Training exposure budget differs')
    if len(stats['rollout_groups'])!=len(set(stats['rollout_groups'])):
        raise ValueError('Behavior rollout used in more than one update')
    destination=directory/'adapter';model.save_pretrained(destination,safe_serialization=True)
    result={'status':'completed','role':role,'updates':32,'adapter':str(destination),
        'final_lora_sha256':lora_hash(model),'binding':binding,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,**stats}
    write_json(directory/'TRAIN_COMPLETE.json',result)
    return model,processor,result
