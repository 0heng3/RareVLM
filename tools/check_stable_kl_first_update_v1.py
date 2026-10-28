"""Zero-optimizer GPU check of the exact new study's first 16-QA gradient."""
from pathlib import Path
import gc
import sys
import time
import torch

from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import sha_file,write_json
from vlm_grpo_iteration_v1.common import read
from vlm_batch_stable_v1.loss import assistant_logps,replay_kl
from vlm_human_sft.evaluate import lora_hash
from vlm_sft.common import to_device
import run_native_stable_kl_v1 as study

ROOT=study.ROOT/'gpu_first_update_check_v1'


def worker():
    for path,digest in read(ROOT/'PLAN.json')['files'].items():
        if sha_file(path)!=digest:raise ValueError('Smoke source changed')
    lock=study.verify();data=read(study.ROOT/'data/MANIFEST.json');cpu=read(study.ROOT/'data/CPU_INPUTS.json');micros=read(study.ROOT/'data/SCHEDULE.json')['updates'][0]['micros'];started=time.monotonic();teacher={}
    model,processor,_=study.load(training=True);model.train();model.get_base_model().visual.eval()
    for micro in (0,2):
        batch,_,_,_=study.batch_rows(processor,data,micros[micro],cpu);flat,owners,counts,ce=assistant_logps(model,to_device(batch,next(model.parameters()).device));teacher[micro]=flat.detach().cpu();del flat,owners,counts,ce,batch
    del model,processor;gc.collect();torch.cuda.empty_cache()
    model,processor,binding=study.load(training=True);model.train();model.get_base_model().visual.eval();before=lora_hash(model);params=[p for p in model.parameters() if p.requires_grad];torch.cuda.reset_peak_memory_stats();checks=[]
    if before!=lock['initial_lora_sha256']:raise ValueError('Wrong GPU check model')
    for micro,ids in enumerate(micros):
        batch,_,_,_=study.batch_rows(processor,data,ids,cpu);gpu=to_device(batch,next(model.parameters()).device);flat,owners,counts,ce=assistant_logps(model,gpu);kl=torch.zeros_like(ce)
        if micro in teacher:
            reference=teacher[micro].to(flat.device);kl=replay_kl(flat,reference,owners,counts);difference=float((flat.detach()-reference).abs().max());maximum=float(kl.detach().abs().max())
            if difference>1e-6 or maximum>1e-6:raise ValueError('GPU check KL identity failed')
            checks.append({'micro':micro,'max_logp_difference':difference,'max_abs_kl':maximum})
        loss=(ce+kl).sum()/16
        if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite GPU loss')
        loss.backward();del flat,owners,counts,ce,kl,loss,gpu,batch
        if micro in teacher:del reference
        write_json(ROOT/'PROGRESS.json',{'status':'gradient_check','completed_micros':micro+1,'expected_micros':4,'optimizer_updates':0})
    norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True));peak=torch.cuda.max_memory_allocated()/2**30
    if norm<=0 or peak>65 or lora_hash(model)!=before or time.monotonic()-started>180:raise ValueError('GPU gradient/memory/identity check failed')
    result={'status':'passed','optimizer_updates':0,'teacher_qa_forwards':8,'actor_qa_forwards':16,'gradient_norm':norm,'peak_allocated_gib':peak,'identity_checks':checks,'active_seconds':time.monotonic()-started,'model_hash_unchanged':True,'binding':binding,'scope':'Extra independent first-batch forward/backward, no optimizer step or capability result.'}
    write_json(ROOT/'COMPLETION.json',result);print('FIRST_UPDATE_GPU_CHECK',result,flush=True)


def main():
    if '--worker' in sys.argv:worker();return
    from research_runtime.queue import run_queue
    run_queue(ROOT/'queue',[{'key':'gpu_check','command':[sys.executable,'-u',str(Path(__file__).resolve()),'--worker'],'completion_marker':str(ROOT/'COMPLETION.json')}],{'plan_sha256':sha_file(ROOT/'PLAN.json')})


if __name__=='__main__':main()
