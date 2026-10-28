import math
import statistics
import time
import torch
from vlm_deep.common import sha_file,write_json
from vlm_sft.common import to_device
from vlm_human_sft.evaluate import lora_hash
from vlm_fp32_adapter.model import restore_fp32_adapter
from vlm_reference_replay_v1.reference import reference_packet,norm
from .common import ROOT,PARENT,JOINT,read,actor,chosen_batch,frozen_qv_hash
from .loss import forward_logits,loss_values


def calibrate():
    plan=read(ROOT/'PLAN.json');data=read(ROOT/'DATA.json');cpu=read(ROOT/'CPU_INPUTS.json');schedule=read(ROOT/'SCHEDULE.json')
    lookup={q['question_id']:q for q in data['training_rows']};model,processor,_=actor(training=True)
    if lora_hash(model)!=plan['initial_lora_sha256']:raise ValueError('Wrong original initializer')
    parameters=[p for p in model.parameters() if p.requires_grad];started=time.monotonic();points=[];initial_kl=[]
    model.train();model.get_base_model().visual.eval();model.config.use_cache=False
    write_json(ROOT/'CALIBRATION_PROGRESS.json',{'stage':'initial_cached_teacher_parity','new_optimizer_updates':0})
    for micro in (0,2):
        ids=schedule[0]['micros'][micro];rows=[lookup[qid] for qid in ids];batch=chosen_batch(processor,rows,data['images'],cpu)
        refs=reference_packet(0,micro,batch,ids);gpu=to_device(batch,next(model.parameters()).device)
        logits=forward_logits(model,gpu);ce,kl=loss_values(logits,gpu['labels'],refs);initial_kl+=kl.detach().cpu().tolist()
        del batch,refs,gpu,logits,ce,kl
    if len(initial_kl)!=8 or not all(math.isfinite(x) and abs(x)<=1e-5 for x in initial_kl):raise ValueError('Reused original teacher not numerically compatible with zero-B MLP')
    for point in plan['calibration_checkpoints']:
        restore_fp32_adapter(model,point['adapter'])
        if frozen_qv_hash(model)!=model._frozen_qv_initial_sha256:raise ValueError('Historical frozen q/v changed')
        vectors={k:[torch.zeros_like(p) for p in parameters] for k in ('ce','replay_ce','kl')};step=point['step']
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
            del batch,refs,gpu,logits,ce,kl,grads
        flat={k:torch.cat([v.flatten() for v in values]) for k,values in vectors.items()};n={k:norm(v) for k,v in flat.items()}
        if not all(math.isfinite(x) for x in n.values()) or n['kl']<=1e-8 or n['ce']<=0 or n['replay_ce']<=0:raise ValueError('Uninformative MLP gradient calibration')
        cosine=float(torch.dot(flat['kl'].double(),flat['replay_ce'].double())/(n['kl']*n['replay_ce']))
        points.append({'checkpoint_step':step,'gradient_norms':n,'ce_over_kl':n['ce']/n['kl'],'cos_kl_vs_replay_ce':cosine,
                       'relative_gradient_difference':norm(flat['kl']-flat['replay_ce'])/n['replay_ce'],'actual_lora_sha256':lora_hash(model)})
        write_json(ROOT/'CALIBRATION_PROGRESS.json',{'stage':'train_only_mlp_gradient_calibration','completed_points':len(points),'points':points,'new_optimizer_updates':0})
        del flat,vectors
        if time.monotonic()-started>180:raise TimeoutError('180sec calibration budget')
        if torch.cuda.max_memory_allocated()/2**30>65:raise RuntimeError('65GiB calibration budget')
    beta=max(.1,min(100.,statistics.median(p['ce_over_kl'] for p in points)))
    redundant=all(p['cos_kl_vs_replay_ce']>=.999 and p['relative_gradient_difference']<=.01 for p in points)
    restore_fp32_adapter(model,JOINT/'initial_adapter')
    if lora_hash(model)!=plan['initial_lora_sha256'] or frozen_qv_hash(model)!=model._frozen_qv_initial_sha256:raise ValueError('Original initialization restore failed')
    result={'status':'completed','training_allowed':not redundant,'plan_sha256':sha_file(ROOT/'PLAN.json'),'points':points,'beta':beta,
            'beta_rule':'clip(median(||g_baseCE||/||g_KL_replay||),.1,100)','validation_used':False,'initial_cached_teacher_kl':initial_kl,
            'redundant_with_hard_replay':redundant,'restored_initial_lora_sha256':lora_hash(model),'frozen_qv_sha256':model._frozen_qv_initial_sha256,
            'active_seconds':time.monotonic()-started,'new_optimizer_updates':0,'new_teacher_generations':0,
            'scope':'Raw MLP gradients only; ratio does not guarantee balanced Adam steps or optimal beta.'}
    write_json(ROOT/'CALIBRATION.json',result);print('CALIBRATED_MLP_REFERENCE',result,flush=True);return result
