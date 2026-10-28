import math
import os
import statistics
import time
from pathlib import Path
import torch
from vlm_deep.common import sha_file,write_json
from vlm_sft.common import to_device
from vlm_human_sft.evaluate import lora_hash
from vlm_fp32_adapter.model import restore_fp32_adapter
from vlm_object_train.train import lock_json
from .common import ROOT,read,actor,chosen_batch,cache_path
from .loss import forward_logits,action_logps,loss_values,frame_hash


def reference_packet(step,micro,cpu_batch,ids):
    packet=torch.load(cache_path(step,micro),map_location='cpu',weights_only=False)
    if packet['plan_sha256']!=sha_file(ROOT/'PLAN.json') or packet['question_ids']!=ids or packet['frame_hash']!=frame_hash(cpu_batch):
        raise ValueError('Cached teacher/actor padded frame differs')
    return packet['logps']


def build_cache():
    if (ROOT/'CACHE_COMPLETE.json').exists():
        receipt=read(ROOT/'CACHE_COMPLETE.json')
        if receipt['plan_sha256']!=sha_file(ROOT/'PLAN.json'):raise ValueError('Cache plan changed')
        for path,digest in receipt['files'].items():
            if sha_file(path)!=digest:raise ValueError('Teacher cache bytes changed')
        return receipt
    plan=read(ROOT/'PLAN.json');data=read(ROOT/'DATA.json');cpu=read(ROOT/'CPU_INPUTS.json');schedule=read(ROOT/'SCHEDULE.json')
    lookup={q['question_id']:q for q in data['training_rows']};model,processor,_=actor();model.eval();model.config.use_cache=False
    if lora_hash(model)!=plan['initial_lora_sha256']:raise ValueError('Wrong frozen teacher')
    lock_json(ROOT/'CACHE_INPUT_LOCK.json',{'plan_sha256':sha_file(ROOT/'PLAN.json'),'producer_uuid':os.environ['CUDA_VISIBLE_DEVICES'],
              'initial_lora_sha256':plan['initial_lora_sha256']})
    started=time.monotonic();files={};entropies=[];masses=[];actions=0
    with torch.inference_mode():
        for step,block in enumerate(schedule):
            for micro in (0,2):
                path=cache_path(step,micro);ids=block['micros'][micro]
                if path.exists():
                    packet=torch.load(path,map_location='cpu',weights_only=False)
                    if packet['plan_sha256']!=sha_file(ROOT/'PLAN.json') or packet['question_ids']!=ids:raise ValueError('Partial cache binding changed')
                else:
                    rows=[lookup[qid] for qid in ids];batch=chosen_batch(processor,rows,data['images'],cpu);signature=frame_hash(batch)
                    gpu=to_device(batch,next(model.parameters()).device);logits=forward_logits(model,gpu);logps,targets=action_logps(logits,gpu['labels'])
                    if not all(bool(torch.isfinite(x).all()) for x in logps):raise FloatingPointError('Nonfinite teacher distribution')
                    packet={'plan_sha256':sha_file(ROOT/'PLAN.json'),'question_ids':ids,'frame_hash':signature,
                            'logps':[x.detach().cpu().clone() for x in logps],
                            'entropy_qa':[float(-(x.exp()*x).sum(-1).mean()) for x in logps],
                            'non_gt_mass_qa':[float((1-x.gather(1,y[:,None]).squeeze(1).exp()).mean()) for x,y in zip(logps,targets)]}
                    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.tmp');torch.save(packet,temporary);temporary.replace(path)
                    del gpu,logits,logps,targets,batch
                files[str(path)]=sha_file(path);entropies+=packet['entropy_qa'];masses+=packet['non_gt_mass_qa'];actions+=sum(x.shape[0] for x in packet['logps'])
                write_json(ROOT/'CACHE_PROGRESS.json',{'status':'building_teacher','batches':len(files),'expected_batches':128,
                           'reference_qa':4*len(files),'active_seconds_this_attempt':time.monotonic()-started})
                if time.monotonic()-started>600:raise TimeoutError('600sec teacher cache budget')
                if torch.cuda.max_memory_allocated()/2**30>65:raise RuntimeError('65GiB cache allocation budget')
    if lora_hash(model)!=plan['initial_lora_sha256']:raise ValueError('Teacher weights changed')
    result={'status':'completed','plan_sha256':sha_file(ROOT/'PLAN.json'),'files':files,'qa':512,'assistant_actions':actions,
            'reference_entropy_qa_mean':statistics.mean(entropies),'reference_non_gt_mass_qa_mean':statistics.mean(masses),
            'bytes':sum(Path(p).stat().st_size for p in files),'active_seconds_this_attempt':time.monotonic()-started,'optimizer_updates':0}
    write_json(ROOT/'CACHE_COMPLETE.json',result);del model;torch.cuda.empty_cache();return result


def norm(vector):return float(torch.linalg.vector_norm(vector.double()))


def calibrate():
    plan=read(ROOT/'PLAN.json');model,processor,_=actor(training=True)
    if (ROOT/'CALIBRATION.json').exists():
        receipt=read(ROOT/'CALIBRATION.json')
        if receipt['plan_sha256']!=sha_file(ROOT/'PLAN.json') or receipt['cache_complete_sha256']!=sha_file(ROOT/'CACHE_COMPLETE.json'):
            raise ValueError('Calibration inputs changed')
        return model,processor,receipt
    data=read(ROOT/'DATA.json');cpu=read(ROOT/'CPU_INPUTS.json');schedule=read(ROOT/'SCHEDULE.json');lookup={q['question_id']:q for q in data['training_rows']}
    parameters=[p for p in model.parameters() if p.requires_grad];points=[];started=time.monotonic()
    for checkpoint in plan['calibration_checkpoints']:
        restore_fp32_adapter(model,checkpoint['adapter'])
        if lora_hash(model)!=checkpoint['trainable_sha256']:raise ValueError('Calibration checkpoint differs')
        model.train();model.get_base_model().visual.eval();model.config.use_cache=False
        vectors={key:[torch.zeros_like(p) for p in parameters] for key in ('ce','replay_ce','kl')}
        step=checkpoint['step']
        for micro,ids in enumerate(schedule[step]['micros']):
            rows=[lookup[qid] for qid in ids];batch=chosen_batch(processor,rows,data['images'],cpu)
            refs=reference_packet(step,micro,batch,ids) if micro in (0,2) else None
            gpu=to_device(batch,next(model.parameters()).device);logits=forward_logits(model,gpu);ce,kl=loss_values(logits,gpu['labels'],refs)
            grads=torch.autograd.grad(ce.sum()/16,parameters,retain_graph=refs is not None)
            for target,g in zip(vectors['ce'],grads):target.add_(g.detach())
            if refs is not None:
                for target,g in zip(vectors['replay_ce'],grads):target.add_(2*g.detach())
                grads=torch.autograd.grad(kl.sum()/8,parameters)
                for target,g in zip(vectors['kl'],grads):target.add_(g.detach())
            del batch,gpu,logits,ce,kl,grads,refs
        flat={key:torch.cat([x.flatten() for x in value]) for key,value in vectors.items()}
        n={key:norm(value) for key,value in flat.items()}
        if not all(math.isfinite(x) for x in n.values()) or n['kl']<=1e-8 or n['ce']<=0:raise ValueError('Uninformative calibration gradient')
        cosine=float(torch.dot(flat['kl'].double(),flat['replay_ce'].double())/max(n['kl']*n['replay_ce'],1e-30))
        residual=norm(flat['kl']-flat['replay_ce'])/max(n['replay_ce'],1e-30)
        points.append({'checkpoint_step':step,'gradient_norms':n,'ce_over_kl':n['ce']/n['kl'],
                       'cos_kl_vs_replay_ce':cosine,'relative_gradient_difference':residual})
        del flat,vectors
        if time.monotonic()-started>180:raise TimeoutError('180sec calibration budget')
    raw=statistics.median(q['ce_over_kl'] for q in points);beta=max(.1,min(100.,raw))
    redundant=all(q['cos_kl_vs_replay_ce']>=.999 and q['relative_gradient_difference']<=.01 for q in points)
    restore_fp32_adapter(model,plan['initial_adapter'])
    if lora_hash(model)!=plan['initial_lora_sha256']:raise ValueError('Initial policy restoration failed after calibration')
    receipt={'status':'completed','training_allowed':not redundant,'plan_sha256':sha_file(ROOT/'PLAN.json'),
             'cache_complete_sha256':sha_file(ROOT/'CACHE_COMPLETE.json'),'points':points,'raw_median_ratio':raw,'beta':beta,
             'beta_rule':'clip(median(||g_base_CE||/||g_replay_KL||),0.1,100)','validation_used':False,
             'redundant_with_hard_replay':redundant,'restored_initial_lora_sha256':lora_hash(model),
             'active_seconds':time.monotonic()-started,'new_optimizer_updates':0}
    temporary=ROOT/'CALIBRATION.tmp';write_json(temporary,receipt);temporary.replace(ROOT/'CALIBRATION.json')
    return model,processor,receipt
