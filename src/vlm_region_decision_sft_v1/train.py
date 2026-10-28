"""Derived matched32-step trainer: QA CE weighting versus first-divergence-token CE. No DPO updates."""
from pathlib import Path
import json,os,random,time
import numpy as np
import torch
from safetensors.torch import load_file
from vlm_deep.common import content_sha,sha_file,write_json
from vlm_sft.common import to_device
from vlm_fp32_readout_v1.model import load
from vlm_alignment_signal.loss import sequence_logps,dpo_loss
from vlm_human_sft.evaluate import lora_hash
from vlm_probe.evaluate import processor_for
from vlm_object_train.train import learning_rate,lock_json,save_checkpoint,resume_checkpoint
from .common import ROOT,DATA_ROOT,CONFIG,read,make_batch
from .loss import forward


def train(role):
    c=read(CONFIG);decisions=read(ROOT/'DECISIONS.json');data=read(DATA_ROOT/'data/MANIFEST.json');rows={q['question_id']:q for q in data['rows']}
    cpu=read(DATA_ROOT/'data/CPU_INPUTS.json');updates=read(DATA_ROOT/'data/SCHEDULE.json')['updates']
    refs=read(DATA_ROOT/'reference/CACHE.json')['batches'];directory=ROOT/'runs'/role/'train';directory.mkdir(parents=True,exist_ok=True)
    if len(updates)!=32:raise ValueError('Fixed32-step positive-control schedule required')
    lock=read(ROOT/'STUDY_INPUT_LOCK.json')
    lineage={'study_sha256':sha_file(ROOT/'STUDY_INPUT_LOCK.json'),'schedule_sha256':sha_file(DATA_ROOT/'data/SCHEDULE.json'),
        'source_sha256':sha_file(Path(__file__)),'config_sha256':content_sha(c),'role':role,'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES']}
    lock_json(directory/'INPUT_LOCK.json',lineage);lineage_sha=content_sha(lineage)
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=read(c['baseline_config']);completed_path=directory/'TRAIN_COMPLETE.json'
    adapter=read(completed_path)['adapter'] if completed_path.exists() else c['initial_adapter']
    model,binding=load(base['model_root'],adapter,torch.device('cuda:0'),training=not completed_path.exists(),dropout=0.)
    processor=processor_for(Path(base['model_root']),read(base['reference_config']),1024)
    if completed_path.exists():return model,processor,read(completed_path)
    if lora_hash(model)!=lock['initial_lora_sha256']:raise ValueError('Unexpected initial SFT checkpoint')
    parameters=[v for v in model.parameters() if v.requires_grad]
    if sum(v.numel() for v in parameters)!=c['expected_trainable'] or any(v.dtype!=torch.float32 for v in parameters):
        raise ValueError('FP32 q/v trainable contract differs')
    optimizer=torch.optim.AdamW(parameters,lr=c['lr'],weight_decay=0.)
    stats={'questions':0,'loss_sum':0.,'chosen_ce_sum':0.,'preference_dpo_sum':0.,
        'negative_forward_sequences':0,'source_counts':{},'active_seconds':0.,'trace_bytes':0,'extra_ce_sum':0.,
        'peak_allocated_gib':0.,'peak_reserved_gib':0.}
    completed=0
    if (directory/'LATEST.json').exists():
        completed,stats=resume_checkpoint(directory,model,optimizer,lineage_sha,lineage['schedule_sha256'],32)
    path=directory/'INPUT_TRACES.jsonl';path.touch(exist_ok=True)
    if path.stat().st_size<stats['trace_bytes']:raise ValueError('Trace shorter than saved optimizer boundary')
    prior=stats['active_seconds'];begun=time.monotonic();durations=[]
    torch.cuda.reset_peak_memory_stats()
    with path.open('r+b') as trace:
        trace.truncate(stats['trace_bytes']);trace.seek(0,os.SEEK_END)
        def checkpoint(step):
            stats['active_seconds']=prior+time.monotonic()-begun;stats['trace_bytes']=trace.tell()
            trace.flush();os.fsync(trace.fileno())
            save_checkpoint(directory,model,optimizer,step,stats,lineage_sha,lineage['schedule_sha256'])
        for step in range(completed,32):
            if prior+time.monotonic()-begun>=c['training_cap_seconds']:
                checkpoint(step);raise TimeoutError('Positive-control cumulative active budget reached')
            start=time.monotonic();model.train();model.get_base_model().visual.eval();model.config.use_cache=False
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:group['lr']=learning_rate(step,32,c)
            ce_total=0.;objective_total=0.;dpo_total=0.;extra_total=0.
            for micro,ids in enumerate(updates[step]['micros']):
                chunk=[rows[q] for q in ids]
                if len(chunk)!=4:raise ValueError('Registered chosen microbatch must have4 QA')
                batch,key,chosen,rejected=make_batch(processor,data,chunk,cpu)
                pref=bool(rejected);ref=None
                if pref:
                    ref=refs[f'{step+1}:{micro}']
                    if key!=ref['binding']:raise ValueError('Reference/actor padded inputs differ')
                gpu=to_device(batch,next(model.parameters()).device)
                with torch.autocast('cuda',enabled=False):
                    logp,counts,decision_ce,decision_active=forward(model,gpu,chunk,decisions)
                    ce=-logp[:4]/counts[:4]
                    losses=ce;dpo=torch.zeros_like(ce)
                    if pref:
                        refc=torch.tensor(ref['chosen_logps'],device=logp.device,dtype=torch.float32)
                        refr=torch.tensor(ref['rejected_logps'],device=logp.device,dtype=torch.float32)
                        relative=(logp[:4]-logp[4:])-(refc-refr)
                        if step==0:
                            value=float(relative.detach().abs().max())
                            write_json(directory/f'INITIAL_REFERENCE_MARGIN_MICRO_{micro}.json',
                                {'max_absolute_relative_logp_margin':value,'tolerance':.05,
                                 'note':'Same checkpoint and padded batch; float kernel differences allowed within .005 DPO-logit at beta.1'})
                            if value>.05:raise ValueError('Initial actor/reference margin inconsistent')
                        dpo=dpo_loss(logp[:4],logp[4:],refc,refr,c['beta'])
                        losses=ce+c['dpo_weight'][role]*dpo
                    extra_ce=ce*decision_active if role=='qa_weighted' else decision_ce
                    losses=losses+c['auxiliary_weight']*extra_ce
                    if losses.dtype!=torch.float32 or not bool(torch.isfinite(losses).all()):
                        raise FloatingPointError('Nonfinite or non-FP32 chosen objective')
                    loss=losses.sum()/16
                loss.backward()
                ces=ce.detach().cpu().tolist();ds=dpo.detach().cpu().tolist();ls=losses.detach().cpu().tolist();es=extra_ce.detach().cpu().tolist();decvals=decision_ce.detach().cpu().tolist()
                for index,(q,item,cv,dv,lv,ev) in enumerate(zip(chunk,chosen,ces,ds,ls,es)):
                    entry={'update':step+1,'microbatch':micro,'question_id':q['question_id'],'image_id':q['image_id'],
                        'source':q['study_source'],'target':q['target'],'chosen_ce':cv,'dpo_loss':dv,
                        'dpo_weight':c['dpo_weight'][role] if pref else 0.,'qa_objective':lv,'extra_ce':ev,
                        'decision_ce':decvals[index],'decision':decisions.get(q['question_id']),'auxiliary_kind':role,
                        'input_ids':item['input_ids'][0].tolist(),'labels':item['labels'][0].tolist()}
                    if pref:
                        entry.update(rejected_target=q['rejected_target'],rejected_input_ids=rejected[index]['input_ids'][0].tolist(),
                            rejected_labels=rejected[index]['labels'][0].tolist(),reference_batch_key=f'{step+1}:{micro}',
                            actual_padded_batch_binding=key)
                    trace.write((json.dumps(entry,ensure_ascii=False,allow_nan=False)+'\n').encode())
                    stats['source_counts'][q['study_source']]=stats['source_counts'].get(q['study_source'],0)+1
                ce_total+=sum(ces);dpo_total+=sum(ds);objective_total+=sum(ls);extra_total+=sum(es)
                if pref:stats['negative_forward_sequences']+=4
                del gpu,batch,logp,counts,ce,dpo,losses,loss,chosen,rejected,decision_ce,decision_active,extra_ce
            norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True))
            if norm<=0:raise ValueError('Zero optimizer gradient')
            optimizer.step();torch.cuda.synchronize()
            stats['questions']+=16;stats['loss_sum']+=objective_total;stats['chosen_ce_sum']+=ce_total
            stats['preference_dpo_sum']+=dpo_total;stats['extra_ce_sum']+=extra_total
            stats['active_seconds']=prior+time.monotonic()-begun
            stats['peak_allocated_gib']=max(stats['peak_allocated_gib'],torch.cuda.max_memory_allocated()/2**30)
            stats['peak_reserved_gib']=max(stats['peak_reserved_gib'],torch.cuda.max_memory_reserved()/2**30)
            if step==0 or (step+1)%8==0:checkpoint(step+1)
            if step==0:
                latest=read(directory/'LATEST.json');saved=load_file(str(directory/latest['directory']/'adapter/adapter_model.safetensors'))
                expected={n.replace('.default.','.'):v for n,v in model.named_parameters() if 'lora_' in n}
                if set(saved)!=set(expected) or any(not torch.equal(saved[n],v.detach().cpu()) for n,v in expected.items()):
                    raise ValueError('Saved adapter differs from FP32 masters')
            durations.append(time.monotonic()-start);speed=sum(durations[-8:])/len(durations[-8:])
            progress={'status':'training','role':role,'completed_updates':step+1,'planned_updates':32,
                'mean_chosen_ce_this_update':ce_total/16,'mean_objective_this_update':objective_total/16,'mean_extra_ce_this_update':extra_total/16,
                'mean_dpo_per_preference_qa':dpo_total/8,'gradient_norm':norm,'recent_seconds_per_update':speed,
                'remaining_training_seconds_estimate':speed*(31-step),**stats}
            write_json(directory/'PROGRESS.json',progress)
            if step==0:write_json(directory/'FIRST_UPDATE.json',{**progress,'saved_adapter_master_values_verified':True,
                'discarded_preflight_updates':0,'binding':binding})
            if stats['active_seconds']>=c['training_cap_seconds'] and step<31:
                checkpoint(step+1);raise TimeoutError('Positive-control cap reached at saved boundary')
    if stats['questions']!=512 or stats['negative_forward_sequences']!=256 or stats['source_counts']!={'common':256,'preference':256}:
        raise ValueError('Actual chosen/rejected/source exposure counts differ')
    destination=directory/'adapter';model.save_pretrained(destination,safe_serialization=True)
    result={'status':'completed','role':role,'updates':32,'adapter':str(destination),
        'final_lora_sha256':lora_hash(model),'input_trace_sha256':sha_file(path),
        'scope':'Short supervised objective comparison; not benchmark gain',**stats}
    write_json(completed_path,result);write_json(directory/'PROGRESS.json',result)
    return model,processor,result
