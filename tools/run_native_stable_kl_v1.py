"""Matched corrective CE versus CE + admitted replay KL, after numerical repair."""
import argparse
from collections import Counter
import copy
import os
from pathlib import Path
import random
import sys
import time

from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import content_sha, sha_file, write_json, collate
from vlm_grpo_iteration_v1.common import read

BASE = PROJECT_ROOT.parents[1] / 'outputs'
OLD = BASE / 'vlm_native_correction_sft_dpo_v2'
REPAIR = BASE / 'vlm_batch_numeric_repair_v1'
ROOT = BASE / 'vlm_native_stable_kl_v1'
PROTOCOL = PROJECT_ROOT / 'docs/VLM_NATIVE_STABLE_KL_V1_PROTOCOL.md'
ROLES = ('sft', 'sft_kl')
INTERNAL = ('native_hard', 'literal_validation', 'seen_validation', 'retention')


def prepare():
    import torch
    from vlm_batch_stable_v1.loss import replay_kl
    if (ROOT / 'STUDY_INPUT_LOCK.json').exists(): raise FileExistsError('Already frozen')
    repair = read(REPAIR / 'COMPLETION.json')
    if not repair['repair_passed']: raise ValueError('Numerical repair not admitted')
    # Closed-form KL check uses an independently calculable two-outcome case.
    z = torch.tensor([[.25, .75]], dtype=torch.float32, requires_grad=True); p = torch.tensor([[.5, .5]])
    value = replay_kl(z.log(), p.log(), torch.tensor([0]), torch.tensor([1]))
    if abs(float(value)-.14384103622589045)>1e-6: raise ValueError('Closed-form KL mismatch')
    logits = torch.tensor([[.3, -.4]], requires_grad=True); logp = logits.log_softmax(-1)
    zero = replay_kl(logp, logp.detach(), torch.tensor([0]), torch.tensor([1])); grad = torch.autograd.grad(zero.sum(), logits)[0]
    if abs(float(zero))>1e-7 or float(grad.abs().max())>1e-6: raise ValueError('Identity KL mismatch')
    write_json(ROOT / 'CPU_LOSS_CHECK.json', {'status':'passed','closed_form_kl':float(value.detach()),'identity_gradient_max':float(grad.abs().max())})
    data = copy.deepcopy(read(OLD / 'data/MANIFEST.json'))
    data['evaluations']['replay_ocr'] = [q for q in data['rows'] if q.get('training_domain') == 'ocr']
    data['evaluations']['replay_native'] = [q for q in data['rows'] if q.get('training_domain') == 'native']
    if any(len(data['evaluations'][name]) != 128 for name in ('replay_ocr','replay_native')): raise ValueError('Replay domain mismatch')
    write_json(ROOT / 'data/MANIFEST.json', data)
    for name in ['CPU_INPUTS.json','SCHEDULE.json']: (ROOT / 'data'/name).write_bytes((OLD/'data'/name).read_bytes())
    (ROOT/'DATA_RECEIPT.json').write_bytes((OLD/'DATA_RECEIPT.json').read_bytes())
    c = {**read(OLD/'CONFIG.json'), 'output_root':str(ROOT), 'policy':repair['selected_policy'], 'roles':list(ROLES), 'kl_weight':{'sft':0.,'sft_kl':1.}, 'reference_cap_seconds':3600, 'maximum_reference_seconds':3600, 'evaluation_cap_seconds':1800}
    for unused in ('beta','dpo_weight'):c.pop(unused,None)
    write_json(ROOT/'CONFIG.json', c)
    parent = read(OLD/'STUDY_INPUT_LOCK.json'); files = dict(parent['files'])
    paths = [Path(__file__),PROTOCOL,ROOT/'CONFIG.json',ROOT/'data/MANIFEST.json',ROOT/'data/CPU_INPUTS.json',ROOT/'data/SCHEDULE.json',ROOT/'CPU_LOSS_CHECK.json',ROOT/'CPU_TOKEN_FRAME_CHECK.json',REPAIR/'PLAN.json',REPAIR/'COMPLETION.json',PROJECT_ROOT/'src/vlm_batch_stable_v1/model.py',PROJECT_ROOT/'src/vlm_batch_stable_v1/loss.py']
    files.update(read(REPAIR/'PLAN.json')['files']); files.update({str(p):sha_file(p) for p in paths})
    write_json(ROOT/'STUDY_INPUT_LOCK.json', {**parent,'files':files,'numerical_policy':c['policy'],'new_updates':64,'roles':list(ROLES),'new_baseline_generations':1216,'new_generations_phase1':3136,'conditional_native1617_generations':4851,'conditional_repeated_test_generations':0,'replay_kl_lambda':1.,'negative_forward_sequences':0})
    print('STABLE_KL_PREPARED',{'policy':c['policy'],'updates':64,'phase1_generations':3136},flush=True)


def verify():
    from vlm_object_train.train import package_lock
    lock = read(ROOT/'STUDY_INPUT_LOCK.json')
    for path,digest in {**lock['files'],**lock['images']}.items():
        if sha_file(path)!=digest: raise ValueError('Frozen input changed: '+path)
    if package_lock()!=lock['packages']: raise ValueError('Package mismatch')
    for path,meta in lock['base_weight_files'].items():
        stat=Path(path).stat()
        if stat.st_size!=meta['bytes'] or stat.st_mtime_ns!=meta['mtime_ns']: raise ValueError('Base model changed')
    return lock


def load(adapter=None, training=False):
    import torch
    from vlm_batch_stable_v1.model import load as stable_load
    from vlm_probe.evaluate import processor_for
    c=read(ROOT/'CONFIG.json'); base=read(c['baseline_config'])
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,binding=stable_load(base['model_root'],adapter or c['initial_adapter'],torch.device('cuda:0'),policy=c['policy'],training=training,dropout=0.)
    processor=processor_for(Path(base['model_root']),read(base['reference_config']),1024)
    return model,processor,binding


def batch_rows(processor,data,ids,cpu):
    from vlm_literal_preference_v1.common import sample
    rows={q['question_id']:q for q in data['rows']}; chunk=[rows[q] for q in ids]; chosen=[sample(processor,data,q) for q in chunk]
    for q,item in zip(chunk,chosen):
        if any(content_sha(item[k].tolist())!=cpu[q['question_id']+':chosen'][k+'_sha256'] for k in ('input_ids','labels')): raise ValueError('Actual chosen frame mismatch')
    batch=collate(chosen,processor.tokenizer.pad_token_id)
    binding={'question_ids':ids,'padded_input_ids_sha256':content_sha(batch['input_ids'].tolist()),'padded_labels_sha256':content_sha(batch['labels'].tolist()),'grid_sha256':content_sha(batch['image_grid_thw'].tolist())}
    return batch,binding,chunk,chosen


def evaluate(model,processor,directory,names,data):
    import torch
    from vlm_fit_positive_control_v1.study import generated
    from run_literal_tile_v1 import generate
    from run_iiit_scope_v1 import alphanumeric
    from vlm_object_train.train import lock_json
    from vlm_human_sft.evaluate import lora_hash
    before=lora_hash(model);model.eval();model.config.use_cache=True
    lock_json(directory/'INPUT_LOCK.json',{'study':sha_file(ROOT/'STUDY_INPUT_LOCK.json'),'model':before,'names':list(names)})
    started=time.monotonic(); cap=2400 if 'transfer_dev' in names else 1800
    for name in names:
        rows=data['evaluations'][name];path=directory/name/'PREDICTIONS.json';preds=read(path) if path.exists() else []
        if [q['question_id'] for q in preds]!=[q['question_id'] for q in rows[:len(preds)]] or len(preds)>len(rows) or (len(preds)%4 and len(preds)!=len(rows)):raise ValueError('Resume not at valid batch')
        for start in range(len(preds),len(rows),4):
            chunk=rows[start:start+4];ocr=name in ('literal_validation','replay_ocr')
            answers=(generate if ocr else generated)(model,processor,chunk,data)
            for q,a in zip(chunk,answers):
                if a['input_sha256']!=q['inference_input_sha256']:raise ValueError('Actual inference prefix changed')
                if ocr:a.update(iiit_alnum_exact=alphanumeric(a['prediction'])==alphanumeric(q['target']),filename_family=q.get('filename_family',q['image_id']))
            preds.extend(answers);temporary=path.with_suffix('.tmp');write_json(temporary,preds);temporary.replace(path)
            write_json(directory/'PROGRESS.json',{'status':'evaluating','cohort':name,'completed':len(preds),'expected':len(rows),'active_seconds':time.monotonic()-started})
            if time.monotonic()-started>cap or torch.cuda.max_memory_allocated()/2**30>65:raise RuntimeError('Evaluation budget reached')
    if lora_hash(model)!=before:raise ValueError('Evaluation changed weights')


def baseline(part):
    verify(); data=read(ROOT/'data/MANIFEST.json'); model,processor,_=load()
    names=('native_hard','literal_validation','retention') if part=='a' else ('seen_validation','replay_ocr','replay_native')
    directory=ROOT/('baseline_'+part);evaluate(model,processor,directory,names,data)
    write_json(directory/'COMPLETE.json',{'status':'completed','cohorts':list(names),'new_generations':sum(len(data['evaluations'][n]) for n in names),'optimizer_updates':0})


def reference():
    import torch
    from safetensors.torch import save_file
    from vlm_batch_stable_v1.loss import assistant_logps
    from vlm_sft.common import to_device
    from vlm_human_sft.evaluate import lora_hash
    lock=verify();data=read(ROOT/'data/MANIFEST.json');cpu=read(ROOT/'data/CPU_INPUTS.json');model,processor,binding=load(training=True)
    if lora_hash(model)!=lock['initial_lora_sha256']:raise ValueError('Wrong reference')
    begun=time.monotonic();merge={}
    for part in ('a','b'):
        directory=ROOT/('baseline_'+part);complete=read(directory/'COMPLETE.json')
        if complete['status']!='completed':raise ValueError('Fresh baseline missing')
        for name in complete['cohorts']:
            source=directory/name/'PREDICTIONS.json';destination=ROOT/'baseline'/name/'PREDICTIONS.json';destination.parent.mkdir(exist_ok=True,parents=True);destination.write_bytes(source.read_bytes());merge[str(source)]={'destination':str(destination),'sha256':sha_file(source)}
    write_json(ROOT/'reference/BASELINE_MERGE.json',merge)
    admitted={};counts={}
    for name in ('replay_ocr','replay_native'):
        preds=read(ROOT/'baseline'/name/'PREDICTIONS.json');flags={q['question_id']:bool(q['iiit_alnum_exact'] if name=='replay_ocr' else q['vqa_score']>=.999999) for q in preds};admitted.update(flags);counts[name]=sum(flags.values())
    passed=sum(counts.values())>=224 and min(counts.values())>=112
    write_json(ROOT/'reference/REPLAY_ADMISSION.json',{'status':'passed' if passed else 'negative','counts':counts,'admitted':admitted,'training_evaluation_only':True})
    if not passed:
        write_json(ROOT/'reference/COMPLETE.json',{'status':'completed','admitted':False,'optimizer_updates':0});return
    cache={};memo={};model.train();model.get_base_model().visual.eval();model.config.use_cache=False
    # Match the actor's grad-enabled checkpointed execution when building the
    # teacher. No backward or optimizer is run for this fixed reference.
    with torch.enable_grad():
        for step,update in enumerate(read(ROOT/'data/SCHEDULE.json')['updates']):
            for micro,ids in enumerate(update['micros']):
                if micro not in (0,2):continue
                batch,key,chunk,_=batch_rows(processor,data,ids,cpu);digest=content_sha(key)
                if digest not in memo:
                    flat,owners,n,ce=assistant_logps(model,to_device(batch,next(model.parameters()).device))
                    path=ROOT/'reference/tensors'/(digest+'.safetensors');path.parent.mkdir(exist_ok=True,parents=True);save_file({'log_probs':flat.detach().cpu().contiguous()},str(path))
                    memo[digest]={'binding':key,'path':str(path),'sha256':sha_file(path),'counts':n.cpu().tolist(),'admitted':[admitted[q] for q in ids]}
                    del flat,owners,n,ce
                cache[f'{step+1}:{micro}']=memo[digest]
                write_json(ROOT/'reference/PROGRESS.json',{'status':'reference_cache','completed_batches':len(cache),'expected':64,'active_seconds':time.monotonic()-begun})
                if time.monotonic()-begun>3600:raise RuntimeError('Reference budget')
    if len(cache)!=64 or lora_hash(model)!=lock['initial_lora_sha256']:raise ValueError('Reference mismatch')
    write_json(ROOT/'reference/CACHE.json',{'batches':cache,'lora_sha256':lock['initial_lora_sha256']})
    reference_paths=[ROOT/'reference/CACHE.json',ROOT/'reference/REPLAY_ADMISSION.json',ROOT/'reference/BASELINE_MERGE.json']+[Path(m['destination']) for m in merge.values()]
    write_json(ROOT/'reference/INPUT_LOCK.json',{str(x):sha_file(x) for x in reference_paths})
    write_json(ROOT/'reference/COMPLETE.json',{'status':'completed','admitted':True,'forward_batches':64,'new_generations':1216,'optimizer_updates':0,'binding':binding})


def train(role):
    import json
    import numpy as np
    import torch
    from safetensors.torch import load_file
    from vlm_batch_stable_v1.loss import assistant_logps,replay_kl
    from vlm_human_sft.evaluate import lora_hash
    from vlm_sft.common import to_device
    from vlm_object_train.train import save_checkpoint,resume_checkpoint,learning_rate,lock_json
    lock=verify();c=read(ROOT/'CONFIG.json');data=read(ROOT/'data/MANIFEST.json');cpu=read(ROOT/'data/CPU_INPUTS.json');updates=read(ROOT/'data/SCHEDULE.json')['updates'];refs=read(ROOT/'reference/CACHE.json')['batches']
    for path,digest in read(ROOT/'reference/INPUT_LOCK.json').items():
        if sha_file(path)!=digest:raise ValueError('Reference metadata changed')
    directory=ROOT/'runs'/role/'train';directory.mkdir(parents=True,exist_ok=True)
    lineage={'study_sha256':sha_file(ROOT/'STUDY_INPUT_LOCK.json'),'schedule_sha256':sha_file(ROOT/'data/SCHEDULE.json'),'role':role,'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES']};lineage_sha=content_sha(lineage);lock_json(directory/'INPUT_LOCK.json',lineage)
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    complete=directory/'TRAIN_COMPLETE.json';model,processor,binding=load(read(complete)['adapter'] if complete.exists() else None,training=not complete.exists())
    if complete.exists():return model,processor,read(complete)
    if lora_hash(model)!=lock['initial_lora_sha256']:raise ValueError('Wrong starting adapter')
    parameters=[p for p in model.parameters() if p.requires_grad]
    if sum(p.numel() for p in parameters)!=1843200 or any(p.dtype!=torch.float32 for p in parameters):raise ValueError('Trainable contract mismatch')
    optimizer=torch.optim.AdamW(parameters,lr=c['lr'],weight_decay=0.)
    stats={'questions':0,'chosen_ce_sum':0.,'replay_kl_sum':0.,'loss_sum':0.,'trace_bytes':0,'active_seconds':0.,'source_counts':{},'peak_allocated_gib':0.,'peak_reserved_gib':0.};completed=0
    if (directory/'LATEST.json').exists():completed,stats=resume_checkpoint(directory,model,optimizer,lineage_sha,lineage['schedule_sha256'],32)
    path=directory/'INPUT_TRACES.jsonl';path.touch(exist_ok=True)
    if path.stat().st_size<stats['trace_bytes']:raise ValueError('Trace shorter than checkpoint')
    prior=stats['active_seconds'];begun=time.monotonic();durations=[];torch.cuda.reset_peak_memory_stats()
    with path.open('r+b') as trace:
        trace.truncate(stats['trace_bytes']);trace.seek(0,os.SEEK_END)
        def checkpoint(step):
            stats['active_seconds']=prior+time.monotonic()-begun;stats['trace_bytes']=trace.tell();trace.flush();os.fsync(trace.fileno());save_checkpoint(directory,model,optimizer,step,stats,lineage_sha,lineage['schedule_sha256'])
        for step in range(completed,32):
            start=time.monotonic();model.train();model.get_base_model().visual.eval();model.config.use_cache=False;optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:group['lr']=learning_rate(step,32,c)
            ce_sum=kl_sum=objective_sum=0.
            for micro,ids in enumerate(updates[step]['micros']):
                batch,key,chunk,items=batch_rows(processor,data,ids,cpu);gpu=to_device(batch,next(model.parameters()).device)
                logp,owners,counts,ce=assistant_logps(model,gpu);kl=torch.zeros_like(ce);flags=torch.zeros_like(ce);raw_difference=None
                if micro in (0,2):
                    ref=refs[f'{step+1}:{micro}']
                    if key!=ref['binding'] or counts.cpu().tolist()!=ref['counts'] or sha_file(ref['path'])!=ref['sha256']:raise ValueError('Reference frame changed')
                    teacher=load_file(ref['path'])['log_probs'].to(logp.device);kl=replay_kl(logp,teacher,owners,counts);flags=torch.tensor(ref['admitted'],device=logp.device,dtype=torch.float32)
                    if step==0 and micro==0:
                        raw_difference=float((logp.detach()-teacher).abs().max());kl_grads=torch.autograd.grad((kl*flags).sum()/16,parameters,retain_graph=True,allow_unused=True);gn=float(sum((g.detach().float().square().sum() for g in kl_grads if g is not None),torch.tensor(0.,device=logp.device)).sqrt())
                        preflight={'actor_reference_max_logp_difference':raw_difference,'initial_max_abs_kl':float(kl.detach().abs().max()),'kl_only_gradient_norm':gn,'optimizer_updates':0,'extra_gradient_passes':1}
                        write_json(directory/'INITIAL_KL_CHECK.json',preflight)
                        if raw_difference>1e-6 or preflight['initial_max_abs_kl']>1e-6 or gn>1e-5:raise ValueError('Initial KL identity failed')
                        del kl_grads
                losses=ce+c['kl_weight'][role]*flags*kl;loss=losses.sum()/16
                if not bool(torch.isfinite(losses).all()) or losses.dtype!=torch.float32:raise FloatingPointError('Invalid objective')
                loss.backward();ces=ce.detach().cpu().tolist();ks=kl.detach().cpu().tolist();fs=flags.cpu().tolist();ls=losses.detach().cpu().tolist()
                for q,item,cv,kv,f,lv in zip(chunk,items,ces,ks,fs,ls):
                    record={'update':step+1,'microbatch':micro,'question_id':q['question_id'],'image_id':q['image_id'],'source':q['study_source'],'target':q['target'],'chosen_ce':cv,'kl_loss':kv,'kl_admitted':bool(f),'kl_weight':c['kl_weight'][role]*f,'qa_objective':lv,'dpo_loss':0.,'dpo_weight':0.,'input_ids':item['input_ids'][0].tolist(),'labels':item['labels'][0].tolist()}
                    trace.write((json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n').encode());stats['source_counts'][q['study_source']]=stats['source_counts'].get(q['study_source'],0)+1
                ce_sum+=sum(ces);kl_sum+=sum(k*f for k,f in zip(ks,fs));objective_sum+=sum(ls)
                del logp,owners,counts,ce,kl,flags,losses,loss,gpu,batch
                if micro in (0,2):del teacher
            norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True))
            if norm<=0:raise ValueError('Zero training gradient')
            optimizer.step();torch.cuda.synchronize();stats['questions']+=16;stats['chosen_ce_sum']+=ce_sum;stats['replay_kl_sum']+=kl_sum;stats['loss_sum']+=objective_sum
            stats['active_seconds']=prior+time.monotonic()-begun;stats['peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30;stats['peak_reserved_gib']=torch.cuda.max_memory_reserved()/2**30
            if step==0 or (step+1)%8==0:checkpoint(step+1)
            durations.append(time.monotonic()-start);speed=sum(durations[-8:])/len(durations[-8:])
            progress={'status':'training','role':role,'completed_updates':step+1,'planned_updates':32,'mean_chosen_ce_this_update':ce_sum/16,'mean_kl_per_replay_qa':kl_sum/8,'gradient_norm':norm,'recent_seconds_per_update':speed,'remaining_training_seconds_estimate':speed*(31-step),**stats};write_json(directory/'PROGRESS.json',progress)
            if step==0:write_json(directory/'FIRST_UPDATE.json',{**progress,'binding':binding,'extra_reference_identity_backward':1})
            if stats['active_seconds']>1800 or stats['peak_allocated_gib']>65:
                checkpoint(step+1);raise RuntimeError('Training budget reached at saved boundary')
    if stats['questions']!=512 or stats['source_counts']!={'common':256,'preference':256}:raise ValueError('Actual exposures differ')
    adapter=directory/'adapter';model.save_pretrained(adapter,safe_serialization=True)
    result={'status':'completed','role':role,'updates':32,'adapter':str(adapter),'final_lora_sha256':lora_hash(model),'input_trace_sha256':sha_file(path),'negative_forward_sequences':0,**stats};write_json(complete,result);write_json(directory/'PROGRESS.json',result)
    return model,processor,result


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
    advantage={n:paired(norm(records['sft'][n],n),norm(records['sft_kl'][n],n)) for n in records['sft']}
    result={'status':'completed','metrics':metrics,'vs_start':comparisons,'screen_checks':checks,'screen_passed':{r:all(v.values()) for r,v in checks.items()},'same_data_kl_vs_sft':advantage,'goal_complete':False,'scope':'Single-seed pilot; fresh common numerical policy, historically used internal evaluation; not global project success.'}
    result['expand_transfer']=result['screen_passed']['sft_kl'] and comparisons['sft_kl']['seen_validation']['delta_pp']>=.5 and advantage['seen_validation']['delta_pp']>=.25
    if transfer:result['native_positive_signal']={'sft_kl':result['screen_passed']['sft_kl'] and all(x['delta_pp']>=.5 and x['ci95'][0]>0 for x in [comparisons['sft_kl']['transfer_dev'],advantage['transfer_dev']])}
    write_json(ROOT/('TRANSFER_ANALYSIS.json' if transfer else 'SCREEN_ANALYSIS.json'),result);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--worker');args=parser.parse_args()
    if args.prepare:prepare();return
    if args.worker:
        if args.worker.startswith('baseline:'):baseline(args.worker.split(':')[1]);return
        if args.worker=='reference':reference();return
        mode,role=args.worker.split(':');verify();data=read(ROOT/'data/MANIFEST.json')
        if mode=='internal':model,processor,result=train(role)
        else:
            adapter=read(ROOT/'runs'/role/'train/TRAIN_COMPLETE.json')['adapter'] if role!='start' else None;model,processor,_=load(adapter)
        evaluate(model,processor,ROOT/'baseline' if role=='start' else ROOT/'runs'/role/mode,('transfer_dev',) if mode=='transfer' else INTERNAL,data)
        write_json(ROOT/('baseline_TRANSFER_COMPLETE.json' if role=='start' else f'runs/{role}/{mode}_COMPLETE.json'),{'status':'completed','role':role,'mode':mode});return
    from research_runtime.queue import run_queue
    verify();identity={'study_sha256':sha_file(ROOT/'STUDY_INPUT_LOCK.json')};entry=str(Path(__file__).resolve())
    run_queue(ROOT/'baseline_queue',[{'key':'baseline_'+part,'command':[sys.executable,'-u',entry,'--worker','baseline:'+part],'completion_marker':str(ROOT/('baseline_'+part)/'COMPLETE.json')} for part in ('a','b')],identity)
    run_queue(ROOT/'reference_queue',[{'key':'reference','command':[sys.executable,'-u',entry,'--worker','reference'],'completion_marker':str(ROOT/'reference/COMPLETE.json')}],identity)
    if not read(ROOT/'reference/COMPLETE.json')['admitted']:
        write_json(ROOT/'COMPLETION.json',{'status':'completed','phase':'replay_admission_negative_no_training','goal_complete':False,'new_updates':0,'new_generations':1216});return
    def job(mode,role):return {'key':role,'command':[sys.executable,'-u',entry,'--worker',mode+':'+role],'completion_marker':str(ROOT/('baseline_TRANSFER_COMPLETE.json' if role=='start' else f'runs/{role}/{mode}_COMPLETE.json'))}
    run_queue(ROOT/'training_queue',[job('internal',r) for r in ROLES],identity);result=analyze()
    if not result['expand_transfer']:
        write_json(ROOT/'COMPLETION.json',{'status':'completed','phase':'screen_negative_no_expansion','new_updates':64,'new_generations':3136,'goal_complete':False});return
    run_queue(ROOT/'transfer_queue',[job('transfer',r) for r in ('start',)+ROLES],identity);result=analyze(True)
    write_json(ROOT/'COMPLETION.json',{**result,'phase':'pilot_complete','new_updates':64,'new_generations':7987,'automatic_more_training':False})


if __name__=='__main__':main()
