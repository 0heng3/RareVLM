"""Native-QA CE with matched trainable budgets in language or vision blocks."""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import sys
import time

from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import sha_file,write_json,content_sha
from vlm_grpo_iteration_v1.common import read
from vlm_native_scope_v1.model import ROOT,PARAMETERS,build,load
import run_native_stable_kl_v1 as common

BASE=PROJECT_ROOT.parents[1]/'outputs'
PARENT=BASE/'vlm_native_stable_kl_v1'
PROTOCOL=PROJECT_ROOT/'docs/VLM_NATIVE_SCOPE_V1_PROTOCOL.md'
ROLES=('language','visual')
INTERNAL=('native_hard','literal_validation','seen_validation','retention')
common.ROOT=ROOT  # Bind only the reused evaluation helper; historical files stay unchanged.


def prepare():
    if (ROOT/'STUDY_INPUT_LOCK.json').exists():raise FileExistsError('Already frozen')
    initial=build();data=copy.deepcopy(read(PARENT/'data/MANIFEST.json'))
    for name in ('replay_ocr','replay_native'):data['evaluations'].pop(name,None)
    write_json(ROOT/'data/MANIFEST.json',data)
    for name in ('CPU_INPUTS.json','SCHEDULE.json'):(ROOT/'data'/name).write_bytes((PARENT/'data'/name).read_bytes())
    receipt={**read(PARENT/'DATA_RECEIPT.json'),'source_data_sha256':sha_file(PARENT/'data/MANIFEST.json'),'new_labels_or_questions':0,'identical_chosen_schedule':True}
    write_json(ROOT/'DATA_RECEIPT.json',receipt)
    c={**read(PARENT/'CONFIG.json'),'output_root':str(ROOT),'initial_adapter':initial['adapter'],'roles':list(ROLES),'training_cap_seconds':2400,'algorithm':'per_QA_CE_only','scope_budget':PARAMETERS,'shared_composite_parameters':2*PARAMETERS}
    for key in ('kl_weight','reference_cap_seconds','maximum_reference_seconds'):c.pop(key,None)
    write_json(ROOT/'CONFIG.json',c);parent=read(PARENT/'STUDY_INPUT_LOCK.json');files=dict(parent['files'])
    paths=[Path(__file__),PROTOCOL,PROJECT_ROOT/'src/vlm_native_scope_v1/model.py',PROJECT_ROOT/'src/vlm_native_scope_v1/__init__.py',ROOT/'INITIALIZATION.json',ROOT/'CONFIG.json',ROOT/'DATA_RECEIPT.json',ROOT/'data/MANIFEST.json',ROOT/'data/CPU_INPUTS.json',ROOT/'data/SCHEDULE.json',PARENT/'baseline/native_hard/PREDICTIONS.json']
    files.update(initial['source_files_sha256']);files.update(initial['files_sha256']);files.update({str(p):sha_file(p) for p in paths})
    write_json(ROOT/'STUDY_INPUT_LOCK.json',{**parent,'files':files,'initial_lora_sha256':initial['lora_sha256'],'roles':list(ROLES),'new_updates':64,'new_baseline_generations':960,'new_generations_phase1':2896,'preflight_generations':16,'conditional_native1617_generations':4851,'conditional_repeated_test_generations':0,'negative_forward_sequences':0,'optimized_parameters_per_role':PARAMETERS,'scope_factor_counts':{'language':144,'visual':60}})
    print('NATIVE_SCOPE_PREPARED',{'roles':ROLES,'updates_per_role':32,'trainable_per_role':PARAMETERS,'new_generations_phase1':2896},flush=True)


def verify():
    from vlm_object_train.train import package_lock
    lock=read(ROOT/'STUDY_INPUT_LOCK.json')
    for path,digest in {**lock['files'],**lock['images']}.items():
        if sha_file(path)!=digest:raise ValueError('Frozen scope input changed: '+path)
    if package_lock()!=lock['packages']:raise ValueError('Package lock differs')
    for path,meta in lock['base_weight_files'].items():
        stat=Path(path).stat()
        if stat.st_size!=meta['bytes'] or stat.st_mtime_ns!=meta['mtime_ns']:raise ValueError('Base changed')
    return lock


def mode(model,role):
    model.train();model.get_base_model().visual.train(role=='visual');model.config.use_cache=False


def gradient_check(model,role):
    import torch
    selected={n:p for n,p in model.named_parameters() if p.requires_grad};frozen={n:p for n,p in model.named_parameters() if 'lora_' in n and not p.requires_grad}
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in selected.values()):raise ValueError('Missing/nonfinite selected gradient')
    if any(p.grad is not None for p in frozen.values()):raise ValueError('Frozen factor received gradient')
    result={'selected_factors':len(selected),'frozen_factors':len(frozen),'frozen_factor_gradients_none':True}
    if role=='visual':
        a=[float(p.grad.norm()) for n,p in selected.items() if '.lora_A.' in n];b=[float(p.grad.norm()) for n,p in selected.items() if '.lora_B.' in n]
        if len(a)!=30 or len(b)!=30 or any(a) or any(x<=0 for x in b):raise ValueError('Zero-B visual gradient condition failed')
        if not model.get_base_model().visual.gradient_checkpointing or not model.get_base_model().visual.training:raise ValueError('Visual checkpoint not active')
        result.update(visual_A_initial_gradients_zero=True,visual_B_nonzero_finite=30,visual_B_gradient_min=min(b),visual_B_gradient_max=max(b),actual_visual_checkpointing=True)
    return result


def preflight(role):
    import torch
    from vlm_alignment_signal.loss import sequence_logps
    from vlm_fit_positive_control_v1.study import generated
    from vlm_human_sft.evaluate import lora_hash
    from vlm_sft.common import to_device
    lock=verify();data=read(ROOT/'data/MANIFEST.json');cpu=read(ROOT/'data/CPU_INPUTS.json');model,processor,binding=load(read(ROOT/'CONFIG.json')['initial_adapter'],role,True)
    before=lora_hash(model)
    if before!=lock['initial_lora_sha256']:raise ValueError('Wrong composite initial')
    started=time.monotonic();model.eval();rows=data['evaluations']['native_hard'][:8];predictions=generated(model,processor,rows,data);expected=read(PARENT/'baseline/native_hard/PREDICTIONS.json')[:8]
    if any(q['question_id']!=z['question_id'] or q['completion_ids']!=z['completion_ids'] or q['input_sha256']!=z['input_sha256'] for q,z in zip(predictions,expected)):raise ValueError('New zero-B composite 8-QA generation parity failed')
    directory=ROOT/'preflight'/role;write_json(directory/'SPOT_PREDICTIONS.json',predictions);mode(model,role);model.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();total=0.
    for ids in read(ROOT/'data/SCHEDULE.json')['updates'][0]['micros']:
        batch,_,_,_=common.batch_rows(processor,data,ids,cpu);gpu=to_device(batch,next(model.parameters()).device);logp,counts=sequence_logps(model,gpu);ce=-logp/counts
        if not bool(torch.isfinite(ce).all()):raise FloatingPointError('Nonfinite preflight CE')
        (ce.sum()/16).backward();total+=float(ce.detach().sum());del batch,gpu,logp,counts,ce
    checks=gradient_check(model,role);params=[p for p in model.parameters() if p.requires_grad];norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True));peak=torch.cuda.max_memory_allocated()/2**30
    if norm<=0 or peak>65 or lora_hash(model)!=before or time.monotonic()-started>300:raise ValueError('Preflight resource/identity condition failed')
    result={'status':'completed','role':role,'optimizer_updates':0,'new_generations':8,'actual_gradient_qa':16,'initial_ce':total/16,'gradient_norm':norm,'peak_allocated_gib':peak,'active_seconds':time.monotonic()-started,'zero_B_generation_spot_parity':True,**checks,'binding':binding}
    write_json(directory/'COMPLETE.json',result);print('SCOPE_PREFLIGHT_COMPLETE',result,flush=True)


def train(role):
    import numpy as np
    import torch
    from safetensors.torch import load_file
    from vlm_alignment_signal.loss import sequence_logps
    from vlm_human_sft.evaluate import lora_hash
    from vlm_object_train.train import learning_rate,lock_json,save_checkpoint,resume_checkpoint,tensor_hash
    from vlm_sft.common import to_device
    lock=verify();c=read(ROOT/'CONFIG.json');data=read(ROOT/'data/MANIFEST.json');cpu=read(ROOT/'data/CPU_INPUTS.json');updates=read(ROOT/'data/SCHEDULE.json')['updates'];directory=ROOT/'runs'/role/'train';directory.mkdir(parents=True,exist_ok=True);closed=directory/'TRAIN_COMPLETE.json'
    if closed.exists():
        result=read(closed);model,processor,_=load(result['adapter'],role,False);return model,processor,result
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);model,processor,binding=load(c['initial_adapter'],role,True)
    if lora_hash(model)!=lock['initial_lora_sha256']:raise ValueError('Unexpected initial factors')
    frozen={n:p for n,p in model.named_parameters() if 'lora_' in n and not p.requires_grad};frozen_hash=tensor_hash(frozen.items());parameters=[p for p in model.parameters() if p.requires_grad]
    lineage={'study_sha256':sha_file(ROOT/'STUDY_INPUT_LOCK.json'),'schedule_sha256':sha_file(ROOT/'data/SCHEDULE.json'),'role':role,'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES']};lineage_sha=content_sha(lineage);lock_json(directory/'INPUT_LOCK.json',lineage);optimizer=torch.optim.AdamW(parameters,lr=c['lr'],weight_decay=0.)
    stats={'questions':0,'chosen_ce_sum':0.,'loss_sum':0.,'trace_bytes':0,'active_seconds':0.,'source_counts':{},'peak_allocated_gib':0.,'peak_reserved_gib':0.};completed=0
    if (directory/'LATEST.json').exists():completed,stats=resume_checkpoint(directory,model,optimizer,lineage_sha,lineage['schedule_sha256'],32)
    path=directory/'INPUT_TRACES.jsonl';path.touch(exist_ok=True)
    if path.stat().st_size<stats['trace_bytes']:raise ValueError('Truncated trace')
    prior=stats['active_seconds'];begun=time.monotonic();durations=[];torch.cuda.reset_peak_memory_stats()
    with path.open('r+b') as trace:
        trace.truncate(stats['trace_bytes']);trace.seek(0,os.SEEK_END)
        def checkpoint(step):
            stats['active_seconds']=prior+time.monotonic()-begun;stats['trace_bytes']=trace.tell();trace.flush();os.fsync(trace.fileno());save_checkpoint(directory,model,optimizer,step,stats,lineage_sha,lineage['schedule_sha256'])
        for step in range(completed,32):
            begin=time.monotonic();mode(model,role);optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:group['lr']=learning_rate(step,32,c)
            total=0.
            for micro,ids in enumerate(updates[step]['micros']):
                batch,_,chunk,items=common.batch_rows(processor,data,ids,cpu);gpu=to_device(batch,next(model.parameters()).device);logp,counts=sequence_logps(model,gpu);ce=-logp/counts
                if ce.dtype!=torch.float32 or not bool(torch.isfinite(ce).all()):raise FloatingPointError('Invalid actual CE')
                (ce.sum()/16).backward();values=ce.detach().cpu().tolist();total+=sum(values)
                for q,item,value in zip(chunk,items,values):
                    trace.write((json.dumps({'update':step+1,'microbatch':micro,'question_id':q['question_id'],'image_id':q['image_id'],'source':q['study_source'],'target':q['target'],'chosen_ce':value,'qa_objective':value,'dpo_loss':0.,'input_ids':item['input_ids'][0].tolist(),'labels':item['labels'][0].tolist()},ensure_ascii=False)+'\n').encode());stats['source_counts'][q['study_source']]=stats['source_counts'].get(q['study_source'],0)+1
                del batch,gpu,logp,counts,ce
            checks=gradient_check(model,role) if step==0 else {};norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True));peak=torch.cuda.max_memory_allocated()/2**30
            if norm<=0 or peak>65:raise ValueError('Gradient/memory condition before optimizer')
            optimizer.step();torch.cuda.synchronize();stats['questions']+=16;stats['chosen_ce_sum']+=total;stats['loss_sum']+=total;stats['active_seconds']=prior+time.monotonic()-begun;stats['peak_allocated_gib']=max(stats['peak_allocated_gib'],peak);stats['peak_reserved_gib']=torch.cuda.max_memory_reserved()/2**30
            if step==0 or (step+1)%8==0:checkpoint(step+1)
            if step==0:
                saved=load_file(str(directory/read(directory/'LATEST.json')['directory']/'adapter/adapter_model.safetensors'));expected={n.replace('.default.','.'):v for n,v in model.named_parameters() if 'lora_' in n}
                if set(saved)!=set(expected) or any(not torch.equal(saved[n],v.detach().cpu()) for n,v in expected.items()):raise ValueError('Saved factors differ from masters')
            durations.append(time.monotonic()-begin);speed=sum(durations[-8:])/len(durations[-8:]);progress={'status':'training','role':role,'completed_updates':step+1,'planned_updates':32,'mean_chosen_ce_this_update':total/16,'gradient_norm':norm,'recent_seconds_per_update':speed,'remaining_training_seconds_estimate':speed*(31-step),**stats};write_json(directory/'PROGRESS.json',progress)
            if step==0:write_json(directory/'FIRST_UPDATE.json',{**progress,**checks,'binding':binding,'saved_master_values_exact':True})
            if stats['active_seconds']>2400:
                checkpoint(step+1);raise RuntimeError('Training cap at saved boundary')
    if stats['questions']!=512 or stats['source_counts']!={'common':256,'preference':256} or tensor_hash(frozen.items())!=frozen_hash:raise ValueError('Exposures/frozen factor identity differ')
    destination=directory/'adapter';model.save_pretrained(destination,safe_serialization=True);result={'status':'completed','role':role,'updates':32,'adapter':str(destination),'final_lora_sha256':lora_hash(model),'input_trace_sha256':sha_file(path),'frozen_factors_unchanged':True,'optimized_parameters':PARAMETERS,**stats};write_json(closed,result);write_json(directory/'PROGRESS.json',result);return model,processor,result


def analyze(transfer=False):
    from vlm_grpo_iteration_v1.study import paired
    data=read(ROOT/'data/MANIFEST.json');names=list(INTERNAL)+(['transfer_dev'] if transfer else []);records={'start':{n:read(ROOT/'baseline'/n/'PREDICTIONS.json') for n in names}}
    for role in ROLES:records[role]={n:read(ROOT/'runs'/role/('transfer' if n=='transfer_dev' else 'internal')/n/'PREDICTIONS.json') for n in names}
    for rec in records.values():
        for group in ('singleword','multiword','rare_1_5'):
            ids={q['question_id'] for q in data['evaluations']['seen_validation'] if (q['stratum']==group if group!='rare_1_5' else 1<=q['answer_frequency']<=5)};rec[group]=[q for q in rec['seen_validation'] if q['question_id'] in ids]
    def norm(rows,name):return [{**q,'image_id':q['filename_family'],'vqa_score':float(q['iiit_alnum_exact'])} for q in rows] if name=='literal_validation' else rows
    metrics={r:{n:100*sum(q['vqa_score'] for q in norm(rows,n))/len(rows) for n,rows in rec.items()} for r,rec in records.items()};comparisons={r:{n:paired(norm(records['start'][n],n),norm(rows,n)) for n,rows in records[r].items()} for r in ROLES};checks={}
    for role in ROLES:
        d=comparisons[role];checks[role]={'hard_learning':sum(q['vqa_score']>=.999999 for q in records[role]['native_hard'])>=32 and d['native_hard']['delta_pp']>=20,'ocr_point_protection':d['literal_validation']['delta_pp']>=-.5,'native_point_protections':all(d[n]['delta_pp']>=-.5 for n in ('seen_validation','retention','singleword','multiword','rare_1_5'))}
    advantage={n:paired(norm(records['language'][n],n),norm(records['visual'][n],n)) for n in records['language']};passed={r:all(v.values()) for r,v in checks.items()}
    result={'status':'completed','metrics':metrics,'vs_start':comparisons,'screen_checks':checks,'screen_passed':passed,'same_data_visual_vs_language':advantage,'expand_transfer':passed['visual'] and comparisons['visual']['seen_validation']['delta_pp']>=.5 and advantage['seen_validation']['delta_pp']>=.25,'goal_complete':False,'scope':'Single-seed native QA adaptation-scope/rank/projection combination; historically used internal evaluation, not external or pure position causality.'}
    if transfer:result['native_positive_signal']={'visual':passed['visual'] and all(x['delta_pp']>=.5 and x['ci95'][0]>0 for x in [comparisons['visual']['transfer_dev'],advantage['transfer_dev']])}
    write_json(ROOT/('TRANSFER_ANALYSIS.json' if transfer else 'SCREEN_ANALYSIS.json'),result);return result


def worker(arg):
    verify();kind,role=arg.split(':');data=read(ROOT/'data/MANIFEST.json')
    if kind=='preflight':preflight(role);return
    if kind=='internal':model,processor,result=train(role)
    else:model,processor,_=load(read(ROOT/'CONFIG.json')['initial_adapter'] if kind=='baseline' or role=='start' else read(ROOT/'runs'/role/'train/TRAIN_COMPLETE.json')['adapter'],role,False)
    if kind=='baseline':
        names=('native_hard','literal_validation','retention') if role=='a' else ('seen_validation',);directory=ROOT/('baseline_'+role)
    else:names=('transfer_dev',) if kind=='transfer' else INTERNAL;directory=ROOT/'baseline' if role=='start' else ROOT/'runs'/role/kind
    common.evaluate(model,processor,directory,names,data)
    marker=directory/'COMPLETE.json' if kind=='baseline' else ROOT/'baseline_TRANSFER_COMPLETE.json' if role=='start' else ROOT/'runs'/role/(kind+'_COMPLETE.json');write_json(marker,{'status':'completed','role':role,'cohorts':list(names)})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--worker');args=parser.parse_args()
    if args.prepare:prepare();return
    if args.worker:worker(args.worker);return
    from research_runtime.queue import run_queue
    verify();identity={'study_sha256':sha_file(ROOT/'STUDY_INPUT_LOCK.json')};entry=str(Path(__file__).resolve())
    def job(kind,role,marker):return {'key':role,'command':[sys.executable,'-u',entry,'--worker',kind+':'+role],'completion_marker':str(marker)}
    run_queue(ROOT/'preflight_queue',[job('preflight',r,ROOT/'preflight'/r/'COMPLETE.json') for r in ROLES],identity)
    first={r:read(ROOT/'preflight'/r/'COMPLETE.json') for r in ROLES};difference=abs(first['language']['initial_ce']-first['visual']['initial_ce']);write_json(ROOT/'PREFLIGHT_ANALYSIS.json',{'status':'completed','initial_ce_difference':difference,'tolerance':1e-5,'passed':difference<=1e-5,'optimizer_updates':0})
    if difference>1e-5:
        write_json(ROOT/'COMPLETION.json',{'status':'completed','phase':'preflight_negative_no_training','new_updates':0,'new_generations':16,'goal_complete':False});return
    run_queue(ROOT/'baseline_queue',[job('baseline',r,ROOT/('baseline_'+r)/'COMPLETE.json') for r in ('a','b')],identity)
    for part in ('a','b'):
        for name in read(ROOT/('baseline_'+part)/'COMPLETE.json')['cohorts']:
            source=ROOT/('baseline_'+part)/name/'PREDICTIONS.json';target=ROOT/'baseline'/name/'PREDICTIONS.json';target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes())
    write_json(ROOT/'baseline/COMPLETE.json',{'status':'completed','new_generations':960})
    run_queue(ROOT/'training_queue',[job('internal',r,ROOT/'runs'/r/'internal_COMPLETE.json') for r in ROLES],identity);result=analyze()
    if not result['expand_transfer']:
        write_json(ROOT/'COMPLETION.json',{'status':'completed','phase':'screen_negative_no_expansion','new_updates':64,'new_generations':2896,'goal_complete':False});return
    run_queue(ROOT/'transfer_queue',[job('transfer',r,ROOT/'baseline_TRANSFER_COMPLETE.json' if r=='start' else ROOT/'runs'/r/'transfer_COMPLETE.json') for r in ('start',)+ROLES],identity);result=analyze(True);write_json(ROOT/'COMPLETION.json',{**result,'phase':'pilot_complete','new_updates':64,'new_generations':7747,'automatic_more_training':False})


if __name__=='__main__':main()
