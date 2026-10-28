"""Train-only source gradients and actual visual-Adam displacement; no pilot."""
import argparse
from pathlib import Path
import sys
import time

from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import sha_file,write_json
from vlm_grpo_iteration_v1.common import read

BASE=PROJECT_ROOT.parents[1]/'outputs'
PARENT=BASE/'vlm_visual_preference_v1'
ROOT=BASE/'vlm_visual_task_geometry_v1'
PROTOCOL=PROJECT_ROOT/'docs/VLM_VISUAL_TASK_GEOMETRY_V1_PROTOCOL.md'
POINTS=(8,16,24)


def cosine(a,b):
    a=a.double(); b=b.double()
    denom=a.norm()*b.norm()
    return float((a@b)/denom) if denom>0 else None


def classify(rows):
    if all(q['study_source']=='preference' for q in rows): return 'hard'
    if not all(q['study_source']=='common' for q in rows): raise ValueError('Mixed source')
    if all(q['image_id'].startswith('iiit5k:') for q in rows): return 'ocr_replay'
    if any(q['image_id'].startswith('iiit5k:') for q in rows): raise ValueError('Mixed OCR/native')
    return 'native_replay'


def mechanism(point):
    a=point['cos_hard_actual_delta']; b=point['cos_replay_actual_delta']
    return (a is not None and b is not None and a<-.01 and b>.01
            and point['finite_hard_objective_change']<=-.001
            and point['finite_replay_ce_change']>=.001)


def prepare():
    import json
    if (ROOT/'PLAN.json').exists(): raise FileExistsError('Geometry plan frozen')
    parent=read(PARENT/'STUDY_INPUT_LOCK.json')
    paths=[Path(__file__),PROTOCOL,PARENT/'STUDY_INPUT_LOCK.json',PARENT/'CONFIG.json',PARENT/'reference/CACHE.json',
           PARENT/'reference/COMPLETE.json',PARENT/'runs/ce_dpo/train/INPUT_TRACES.jsonl']
    paths += [PARENT/'data'/n for n in ('MANIFEST.json','CPU_INPUTS.json','SCHEDULE.json')]
    checkpoints=[]
    for step in POINTS:
        candidates=list((PARENT/'runs/ce_dpo/train').glob(f'checkpoint_{step:06d}_*'))
        if len(candidates)!=1: raise ValueError('Ambiguous historical checkpoint')
        directory=candidates[0]
        checkpoints.append({'step':step,'adapter':str(directory/'adapter'),'state':str(directory/'state.pt')})
        paths += [directory/'state.pt']+[directory/'adapter'/n for n in ('adapter_config.json','adapter_model.safetensors')]
    expected={str(p+1):[] for p in POINTS}
    with paths[6].open() as handle:
        for line in handle:
            q=json.loads(line)
            if str(q['update']) in expected:
                expected[str(q['update'])].append({k:q[k] for k in ('question_id','microbatch','chosen_ce','dpo_loss')})
    if any(len(v)!=16 for v in expected.values()): raise ValueError('Need original16 QA per point')
    write_json(ROOT/'EXPECTED_LOSSES.json',expected); paths.append(ROOT/'EXPECTED_LOSSES.json')
    paths += [PROJECT_ROOT/'src'/n for n in ('vlm_native_scope_v1/model.py','vlm_fp32_adapter/model.py',
              'vlm_literal_preference_v1/common.py','vlm_alignment_signal/loss.py','vlm_object_train/train.py','research_runtime/queue.py')]
    paths += [PROJECT_ROOT/'tools/run_native_scope_v1.py']
    write_json(ROOT/'PLAN.json',{'status':'frozen','checkpoints':checkpoints,'files':{**parent['files'],**{str(p):sha_file(p) for p in paths}},
        'formal_optimizer_updates':0,'discarded_diagnostic_updates':3,'new_generations':0,'max_active_seconds':300,
        'automatic_training':False,'initial_lora_sha256':parent['initial_lora_sha256']})
    print({'status':'prepared','points':list(POINTS),'new_training':False},flush=True)


def verify():
    for p,digest in read(ROOT/'PLAN.json')['files'].items():
        if sha_file(p)!=digest: raise ValueError('Frozen geometry input changed: '+p)


def worker():
    import torch
    from vlm_native_scope_v1.model import load,PARAMETERS
    from vlm_fp32_adapter.model import restore_fp32_adapter
    from vlm_human_sft.evaluate import lora_hash
    from vlm_object_train.train import learning_rate
    from vlm_literal_preference_v1.common import make_batch
    from vlm_alignment_signal.loss import sequence_logps,dpo_loss
    from vlm_sft.common import to_device
    from run_native_scope_v1 import mode
    verify(); plan=read(ROOT/'PLAN.json'); config=read(PARENT/'CONFIG.json'); started=time.monotonic()
    data=read(PARENT/'data/MANIFEST.json'); cpu=read(PARENT/'data/CPU_INPUTS.json')
    updates=read(PARENT/'data/SCHEDULE.json')['updates']; refs=read(PARENT/'reference/CACHE.json')['batches']
    expected=read(ROOT/'EXPECTED_LOSSES.json'); rows={q['question_id']:q for q in data['rows']}
    model,processor,_=load(config['initial_adapter'],'visual',True)
    named={n:p for n,p in model.named_parameters() if p.requires_grad}; params=list(named.values())
    if sum(p.numel() for p in params)!=PARAMETERS or len(params)!=60: raise ValueError('Wrong visual scope')
    torch.cuda.reset_peak_memory_stats(); points=[]; discarded=0
    for checkpoint in plan['checkpoints']:
        step=checkpoint['step']; restore_fp32_adapter(model,checkpoint['adapter']); before_hash=lora_hash(model)
        state=torch.load(checkpoint['state'],map_location='cpu',weights_only=False)
        if state['completed_updates']!=step or list(state['trainable_parameters'])!=list(named):
            raise ValueError('Historical optimizer coordinate/cursor mismatch')
        if any(not torch.equal(p.detach().cpu(),state['trainable_parameters'][n]) for n,p in named.items()):
            raise ValueError('Historical visual master values mismatch')
        optimizer=torch.optim.AdamW(params,lr=config['lr'],weight_decay=0.)
        optimizer.load_state_dict(state['optimizer']); del state
        for group in optimizer.param_groups: group['lr']=learning_rate(step,32,config)
        optimizer.zero_grad(set_to_none=True); mode(model,'visual')
        batches=[]; source_ids=[]
        for micro,ids in enumerate(updates[step]['micros']):
            chunk=[rows[i] for i in ids]; batch,key,chosen,rejected=make_batch(processor,data,chunk,cpu)
            source=classify(chunk); reference=refs.get(f'{step+1}:{micro}') if rejected else None
            if rejected and key!=reference['binding']: raise ValueError('Reference batch differs')
            batches.append((batch,source,reference));source_ids.append({'source':source,'question_ids':ids})
        if sorted(s for _,s,_ in batches)!=['hard','hard','native_replay','ocr_replay']:
            raise ValueError('Unexpected source allocation')
        vectors={s:torch.zeros(PARAMETERS,device=params[0].device,dtype=torch.float32)
                 for s in ('hard_ce','hard_dpo','native_replay','ocr_replay')}
        def objectives(batch,source,reference):
            gpu=to_device(batch,params[0].device); logp,counts=sequence_logps(model,gpu)
            ce=-logp[:4]/counts[:4]; dpo=None
            if source=='hard':
                if counts.cpu().tolist()!=reference['counts']: raise ValueError('Answer counts differ')
                a=torch.tensor(reference['chosen_logps'],device=logp.device,dtype=torch.float32)
                b=torch.tensor(reference['rejected_logps'],device=logp.device,dtype=torch.float32)
                dpo=dpo_loss(logp[:4],logp[4:],a,b,.1)
            return ce,dpo
        def flatten(gradient): return torch.cat([g.detach().reshape(-1) for g in gradient])
        totals={s:0. for s in vectors}; max_error=0.; by_micro={q['microbatch']:[] for q in expected[str(step+1)]}
        for q in expected[str(step+1)]:by_micro[q['microbatch']].append(q)
        for micro,(batch,source,reference) in enumerate(batches):
            ce,dpo=objectives(batch,source,reference)
            if not bool(torch.isfinite(ce).all()) or dpo is not None and not bool(torch.isfinite(dpo).all()):
                raise FloatingPointError('Nonfinite source objective')
            original=by_micro[micro]
            for index,q in enumerate(original):
                if q['question_id']!=source_ids[micro]['question_ids'][index]: raise ValueError('Trace order differs')
                max_error=max(max_error,abs(float(ce[index])-q['chosen_ce']),abs((float(dpo[index]) if dpo is not None else 0.)-q['dpo_loss']))
            ce_key='hard_ce' if source=='hard' else source
            vectors[ce_key]+=flatten(torch.autograd.grad(ce.sum()/16,params,retain_graph=dpo is not None))
            totals[ce_key]+=float(ce.detach().sum())
            if dpo is not None:
                vectors['hard_dpo']+=flatten(torch.autograd.grad(dpo.sum()/16,params));totals['hard_dpo']+=float(dpo.detach().sum())
            del ce,dpo
        if max_error>1e-5: raise ValueError('Historical next-step losses differ: '+str(max_error))
        total=sum(vectors.values()); offset=0
        for parameter in params:
            count=parameter.numel();parameter.grad=total[offset:offset+count].reshape_as(parameter).clone();offset+=count
        norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
        before=torch.cat([p.detach().reshape(-1) for p in params]).clone()
        optimizer.step(); discarded+=1
        delta=torch.cat([p.detach().reshape(-1) for p in params])-before
        after={s:0. for s in vectors}
        with torch.no_grad():
            for batch,source,reference in batches:
                ce,dpo=objectives(batch,source,reference);key='hard_ce' if source=='hard' else source
                after[key]+=float(ce.sum())
                if dpo is not None:after['hard_dpo']+=float(dpo.sum())
        for values in (totals,after):
            for key in values:values[key]/=8 if key.startswith('hard') else 4
        hard=vectors['hard_ce']+vectors['hard_dpo'];replay=vectors['native_replay']+vectors['ocr_replay']
        result={'checkpoint_step':step,'next_update':step+1,'sources':source_ids,'before':totals,'after':after,
            'cos_hard_actual_delta':cosine(hard,delta),'cos_replay_actual_delta':cosine(replay,delta),
            'cos_hard_replay_gradients':cosine(hard,replay),
            'per_source_delta_cosines':{k:cosine(g,delta) for k,g in vectors.items()},
            'per_source_gradient_norms':{k:float(g.norm()) for k,g in vectors.items()},
            'cos_hard_ce_dpo':cosine(vectors['hard_ce'],vectors['hard_dpo']),
            'finite_hard_objective_change':after['hard_ce']+after['hard_dpo']-totals['hard_ce']-totals['hard_dpo'],
            'finite_replay_ce_change':(after['native_replay']+after['ocr_replay']-totals['native_replay']-totals['ocr_replay'])/2,
            'actual_delta_norm':float(delta.norm()),'raw_total_gradient_norm':norm,'trace_loss_max_error':max_error,
            'extra_backward_calls':2,'extra_backward_batch_sequences':16,'extra_backward_chosen_qa':8}
        result['mechanism_gate_passed']=mechanism(result)
        restore_fp32_adapter(model,checkpoint['adapter'])
        if lora_hash(model)!=before_hash: raise ValueError('Diagnostic checkpoint not restored')
        points.append(result);write_json(ROOT/'runs/visual/PROGRESS.json',{'status':'source_gradient_and_actual_adam',
            'completed_points':len(points),'expected_points':3,'discarded_optimizer_updates':discarded,
            'formal_optimizer_updates':0,'new_generations':0,'active_seconds':time.monotonic()-started,
            'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'points':points})
        del optimizer,before,delta,total,hard,replay,vectors,batches
        if time.monotonic()-started>300 or torch.cuda.max_memory_allocated()/2**30>65:
            raise RuntimeError('Geometry active-time/memory budget exceeded')
    restore_fp32_adapter(model,config['initial_adapter'])
    if lora_hash(model)!=plan['initial_lora_sha256']: raise ValueError('Initial adapter not restored')
    verify(); supported=sum(p['mechanism_gate_passed'] for p in points)>=2
    result={'status':'completed','phase':'visual_replay_conflict_supported' if supported else 'visual_replay_conflict_not_supported',
        'points':points,'mechanism_signal':supported,'passed_points':sum(p['mechanism_gate_passed'] for p in points),
        'required_points':2,'formal_optimizer_updates':0,'discarded_diagnostic_optimizer_updates':discarded,
        'new_generations':0,'new_adapters':0,'initial_restored':True,'automatic_training':False,'goal_complete':False,
        'active_seconds':time.monotonic()-started,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30}
    write_json(ROOT/'COMPLETION.json',result)
    write_json(ROOT/'DECISION.json',{'status':'completed','new_training_admitted':False,
        'decision':'preregister_new_matched_visual_coordination_study' if supported else 'close_visual_replay_conflict_no_extra_points_or_training'})
    write_json(ROOT/'runs/visual/COMPLETE.json',{'status':'completed','points':3,'formal_optimizer_updates':0})
    print(result,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    if args.prepare:prepare();return
    if args.worker:worker();return
    from research_runtime.queue import run_queue
    verify();run_queue(ROOT/'queue',[{'key':'visual','command':[sys.executable,'-u',str(Path(__file__).resolve()),'--worker'],
        'completion_marker':str(ROOT/'runs/visual/COMPLETE.json')}],{'plan_sha256':sha_file(ROOT/'PLAN.json')})


if __name__=='__main__':main()
