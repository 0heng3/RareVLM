"""Full fixed-Final training-gradient geometry; zero optimizer updates."""
import argparse
from collections import OrderedDict
from pathlib import Path
import sys
import time

from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import content_sha,sha_file,write_json
from vlm_grpo_iteration_v1.common import read
import run_native_stable_kl_v1 as parent

ROOT=parent.BASE/'vlm_stable_kl_gradient_v1'
PROTOCOL=PROJECT_ROOT/'docs/VLM_STABLE_KL_GRADIENT_V1_PROTOCOL.md'
KINDS=('hard_ce','native_ce','ocr_ce','native_kl','ocr_kl')


def prepare():
    if (ROOT/'PLAN.json').exists():raise FileExistsError('Already frozen')
    parent.verify();files=dict(read(parent.ROOT/'STUDY_INPUT_LOCK.json')['files'])
    paths=[Path(__file__),PROTOCOL,parent.ROOT/'STUDY_INPUT_LOCK.json',parent.ROOT/'COMPLETION.json',parent.ROOT/'SCREEN_ANALYSIS.json',parent.ROOT/'reference/CACHE.json',parent.ROOT/'reference/INPUT_LOCK.json']
    for role in parent.ROLES:
        directory=parent.ROOT/'runs'/role/'train';paths += [directory/'TRAIN_COMPLETE.json',directory/'adapter/adapter_model.safetensors',directory/'adapter/adapter_config.json']
    files.update({str(p):sha_file(p) for p in paths})
    write_json(ROOT/'PLAN.json',{'files':files,'roles':list(parent.ROLES),'optimizer_updates':0,'new_generations':0,'training_unique_qa_per_role':384,'weighted_exposures_per_role':512,'cap_seconds_per_role':900,'max_peak_allocated_gib':65})


def verify():
    plan=read(ROOT/'PLAN.json');parent.verify()
    for path,digest in plan['files'].items():
        if sha_file(path)!=digest:raise ValueError('Frozen gradient input changed: '+path)
    return plan


def worker(role):
    import torch
    from safetensors.torch import load_file,save_file
    from vlm_batch_stable_v1.loss import assistant_logps,replay_kl
    from vlm_sft.common import to_device
    from vlm_human_sft.evaluate import lora_hash
    plan=verify();data=read(parent.ROOT/'data/MANIFEST.json');cpu=read(parent.ROOT/'data/CPU_INPUTS.json');refs=read(parent.ROOT/'reference/CACHE.json')['batches'];groups=OrderedDict()
    for step,update in enumerate(read(parent.ROOT/'data/SCHEDULE.json')['updates']):
        for micro,ids in enumerate(update['micros']):
            key=tuple(ids)
            if key not in groups:groups[key]={'ids':ids,'multiplicity':0,'reference_key':f'{step+1}:{micro}' if micro in (0,2) else None}
            groups[key]['multiplicity']+=1
    if len(groups)!=96 or sum(q['multiplicity']*4 for q in groups.values())!=512:raise ValueError('Unexpected complete training schedule')
    final=read(parent.ROOT/'runs'/role/'train/TRAIN_COMPLETE.json');model,processor,_=parent.load(final['adapter'],training=True);model.train();model.get_base_model().visual.eval();model.config.use_cache=False
    before=lora_hash(model)
    if before!=final['final_lora_sha256']:raise ValueError('Wrong Final adapter')
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad];params=[p for n,p in named];size=sum(p.numel() for p in params)
    if size!=1843200 or any(p.dtype!=torch.float32 for p in params):raise ValueError('Wrong trainable scope')
    vectors={name:torch.zeros(size,dtype=torch.float64) for name in KINDS};losses={name:0. for name in KINDS};directory=ROOT/role;started=time.monotonic();torch.cuda.reset_peak_memory_stats();exposures=0
    def accumulate(kind,loss,retain_graph=False):
        gradients=torch.autograd.grad(loss,params,retain_graph=retain_graph,allow_unused=True)
        vector=torch.cat([(g.detach().reshape(-1) if g is not None else torch.zeros_like(p).reshape(-1)) for g,p in zip(gradients,params)]).cpu().double()
        if not bool(torch.isfinite(vector).all()):raise FloatingPointError('Nonfinite diagnostic gradient')
        vectors[kind].add_(vector);losses[kind]+=float(loss.detach())
    for index,group in enumerate(groups.values()):
        batch,binding,chunk,_=parent.batch_rows(processor,data,group['ids'],cpu);gpu=to_device(batch,next(model.parameters()).device);logp,owners,counts,ce=assistant_logps(model,gpu);weight=group['multiplicity']/512
        if group['reference_key'] is None:
            accumulate('hard_ce',ce.sum()*weight)
        else:
            ref=refs[group['reference_key']]
            if binding!=ref['binding'] or counts.cpu().tolist()!=ref['counts'] or sha_file(ref['path'])!=ref['sha256']:raise ValueError('Reference batch frame changed')
            domain=chunk[0]['training_domain']
            if domain not in ('native','ocr') or any(q['training_domain']!=domain for q in chunk):raise ValueError('Mixed replay domain')
            teacher=load_file(ref['path'])['log_probs'].to(logp.device);kl=replay_kl(logp,teacher,owners,counts);flags=torch.tensor(ref['admitted'],device=logp.device,dtype=torch.float32)
            accumulate(domain+'_ce',ce.sum()*weight,retain_graph=True);accumulate(domain+'_kl',(kl*flags).sum()*weight)
            del teacher,kl,flags
        del logp,owners,counts,ce,gpu,batch
        exposures+=4*group['multiplicity'];elapsed=time.monotonic()-started
        write_json(directory/'PROGRESS.json',{'status':'gradient_diagnostic','role':role,'completed_batches':index+1,'expected_batches':96,'weighted_exposures':exposures,'actual_forward_qa':4*(index+1),'optimizer_updates':0,'active_seconds':elapsed,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30})
        if elapsed>900 or torch.cuda.max_memory_allocated()/2**30>65:raise RuntimeError('Gradient diagnostic budget')
    if lora_hash(model)!=before or any(p.grad is not None for p in params):raise ValueError('Diagnostic changed weights or accumulated parameter gradients')
    ce=vectors['hard_ce']+vectors['native_ce']+vectors['ocr_ce'];kl=vectors['native_kl']+vectors['ocr_kl'];joint=ce+kl
    norm=lambda x:float(x.norm())
    def cosine(x,y):
        denominator=norm(x)*norm(y)
        return float(x.dot(y))/denominator if denominator>0 else None
    result={'status':'completed','role':role,'optimizer_updates':0,'new_generations':0,'unique_forward_batches':96,'actual_forward_qa':384,'weighted_exposures':exposures,'losses':losses,
            'gradient_norms':{**{k:norm(v) for k,v in vectors.items()},'ce_total':norm(ce),'kl_total':norm(kl),'joint_lambda1':norm(joint)},
            'kl_to_ce_norm_ratio':norm(kl)/norm(ce) if norm(ce)>0 else None,
            'cosines':{'ce_vs_kl':cosine(ce,kl),'hard_vs_native_ce':cosine(vectors['hard_ce'],vectors['native_ce']),'hard_vs_ocr_ce':cosine(vectors['hard_ce'],vectors['ocr_ce']),'hard_vs_native_kl':cosine(vectors['hard_ce'],vectors['native_kl']),'hard_vs_ocr_kl':cosine(vectors['hard_ce'],vectors['ocr_kl']),'ce_vs_joint':cosine(ce,joint)},
            'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'active_seconds':time.monotonic()-started,'model_hash_unchanged':True,'scope':'Static full-training objective gradients at fixed Final; not historical Adam updates, held-out gradients or capability gains.'}
    save_file(vectors,str(directory/'GRADIENTS.safetensors'));write_json(directory/'PARAMETERS.json',[{'name':n,'shape':list(p.shape),'numel':p.numel()} for n,p in named]);write_json(directory/'COMPLETE.json',result);print('FINAL_GRADIENT_COMPLETE',result,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--worker');args=parser.parse_args()
    if args.prepare:prepare();return
    if args.worker:worker(args.worker);return
    from research_runtime.queue import run_queue
    verify();run_queue(ROOT/'queue',[{'key':role,'command':[sys.executable,'-u',str(Path(__file__).resolve()),'--worker',role],'completion_marker':str(ROOT/role/'COMPLETE.json')} for role in parent.ROLES],{'plan_sha256':sha_file(ROOT/'PLAN.json')})
    write_json(ROOT/'COMPLETION.json',{'status':'completed','optimizer_updates':0,'new_generations':0,'roles':{r:read(ROOT/r/'COMPLETE.json') for r in parent.ROLES},'goal_complete':False})


if __name__=='__main__':main()
