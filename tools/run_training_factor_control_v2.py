"""Finite real-data learning positive control; not a model-performance experiment."""
import argparse
from contextlib import nullcontext
import gc
import json
import os
from pathlib import Path
import random
import sys
import time

from rarevlm_paths import PROJECT_ROOT

OUT=Path('/opt/data/private/longtail_vlm/rarevlm_pilot/outputs/vlm_failure_factor_review_v2')
CONFIG=PROJECT_ROOT/'configs/vlm_sft_resolution_v1.json'


def read(p):return json.loads(Path(p).read_text())


def ce(model,batch):
    import torch.nn.functional as F
    labels=batch['labels'];counts=(labels!=-100).sum(1);keep=int(counts.max())+1
    hook=model.get_output_embeddings().register_forward_pre_hook(lambda module,args:(args[0][:,-keep:,:],))
    try:output=model(**{k:v for k,v in batch.items() if k!='labels'},use_cache=False)
    finally:hook.remove()
    targets=labels[:,-output.logits.shape[1]:][:,1:]
    logits=output.logits[:,:-1,:].float()
    losses=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),targets.reshape(-1),ignore_index=-100,reduction='none').reshape(targets.shape)
    assert bool((targets.ne(-100).sum(1)==counts).all())
    return losses.sum(1)/counts


def worker(arm):
    import numpy as np
    import torch
    from vlm_deep.common import collate,content_sha,encode,to_device,write_json
    from vlm_fp32_adapter.model import load
    from vlm_object_train.train import lock_json,trainable_hash,save_checkpoint,resume_checkpoint
    from vlm_probe.evaluate import evaluate_batch,processor_for
    from vlm_deep.common import consensus
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    plan=read(OUT/'CONTROL_PLAN.json');c=read(CONFIG);spec=plan['arms'][arm]
    directory=OUT/'controls'/arm;directory.mkdir(parents=True,exist_ok=True)
    if (directory/'COMPLETE.json').exists():return
    manifest=read(c['manifest']);byid={str(r['question_id']):r for r in manifest['splits']['train_all']}
    rows={name:[byid[str(i)] for i in plan[name+'_question_ids']] for name in ['fit','probe']}
    assert not ({r['image_id'] for r in rows['fit']}&{r['image_id'] for r in rows['probe']})
    device=torch.device('cuda:0');ref=read(c['reference_config'])
    model,binding=load(c['model_root'],c['initial_adapter'],device,training=True,dropout=0.,gradient_checkpointing=True)
    params=[p for p in model.parameters() if p.requires_grad]
    assert sum(p.numel() for p in params)==1843200
    initial_hash=trainable_hash(model)
    lineage={'plan':plan,'arm':arm,'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES'],'initial_tensor_sha256':initial_hash}
    lock_json(directory/'INPUT_LOCK.json',lineage);lineage_sha=content_sha(lineage)
    initial={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
    processor=processor_for(Path(c['model_root']),ref,1024);root=Path(c['manifest']).parent
    encoded={}
    for name,qs in rows.items():
        encoded[name]=[encode(processor,r,root/manifest['images'][r['image_id']]['path'],True) for r in qs]
        assert all(x['input_ids'].shape[1]<=2048 and int(x['image_grid_thw'].prod())//4<=1024 for x in encoded[name])
    token_id=processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
    def nll(name):
        values=[]
        with torch.inference_mode():
            for start in range(0,len(rows[name]),4):
                batch=to_device(collate(encoded[name][start:start+4],processor.tokenizer.pad_token_id),device)
                values.extend(ce(model,batch).float().cpu().tolist())
        return sum(values)/len(values)
    def evaluate(tag):
        file=directory/(tag+'.json')
        if file.exists():return read(file)
        model.eval();result={'status':'completed','same_eval_profile':'BF16 base/FP32 LoRA/no autocast'}
        for name,qs in rows.items():
            predictions=[]
            with torch.inference_mode():
                for start in range(0,len(qs),4):
                    chunk=qs[start:start+4]
                    answers,_=evaluate_batch(model,processor,chunk,manifest['images'],root,device,1024,token_id,32)
                    predictions.extend({'question_id':r['question_id'],'image_id':r['image_id'],
                        'prediction':a['prediction'],'score':consensus(a['prediction'],r['answers'])} for r,a in zip(chunk,answers))
            result[name]={'qa':len(qs),'mean_ce':nll(name),'score_percent':100*sum(p['score'] for p in predictions)/len(qs),'predictions':predictions}
        write_json(file,result);return result
    before=evaluate('INITIAL')
    blocks=[]
    for epoch in range(16):
        order=torch.randperm(32,generator=torch.Generator().manual_seed(42+epoch)).tolist()
        blocks.extend([order[:16],order[16:]])
    schedule_sha=content_sha(blocks)
    optimizer=torch.optim.AdamW(params,lr=spec['lr'],weight_decay=0.)
    completed=0;stats={'qa_exposures':0,'active_seconds':0.,'training_loss_sum':0.}
    if (directory/'LATEST.json').exists():completed,stats=resume_checkpoint(directory,model,optimizer,lineage_sha,schedule_sha,32)
    started=time.monotonic();old_seconds=stats['active_seconds'];curve=[]
    for step in range(completed,32):
        model.train();model.get_base_model().visual.eval();optimizer.zero_grad(set_to_none=True)
        indices=blocks[step];total=0.
        for start in range(0,16,4):
            batch=to_device(collate([encoded['fit'][i] for i in indices[start:start+4]],processor.tokenizer.pad_token_id),device)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=spec['global_bf16_autocast']):
                losses=ce(model,batch);loss=losses.sum()/16
            if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite control CE')
            loss.backward();total+=float(losses.detach().sum());del batch,losses,loss
        norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
        assert float(norm)>0
        optimizer.step();stats['qa_exposures']+=16;stats['training_loss_sum']+=total
        stats['active_seconds']=old_seconds+time.monotonic()-started
        progress={'status':'training','arm':arm,'completed_updates':step+1,'planned_updates':32,
            'gradient_norm':float(norm),'mean_step_ce':total/16,**stats}
        if (step+1)%4==0:
            write_json(directory/'PROGRESS.json',progress);print(json.dumps(progress),flush=True)
        if (step+1)%8==0:
            model.eval();value=nll('fit');curve.append({'updates':step+1,'fit_ce_same_eval_profile':value})
            write_json(directory/'CURVE.json',curve)
            save_checkpoint(directory,model,optimizer,step+1,stats,lineage_sha,schedule_sha)
    model.save_pretrained(directory/'adapter',safe_serialization=True)
    after=evaluate('FINAL')
    delta=sum(float((p.detach().cpu().double()-initial[n].double()).square().sum())
              for n,p in model.named_parameters() if p.requires_grad)**.5
    value={'status':'completed','arm':arm,'settings':spec,'updates':32,**stats,'binding':binding,
        'fit_ce_before':before['fit']['mean_ce'],'fit_ce_after':after['fit']['mean_ce'],
        'fit_score_before':before['fit']['score_percent'],'fit_score_after':after['fit']['score_percent'],
        'probe_ce_before':before['probe']['mean_ce'],'probe_ce_after':after['probe']['mean_ce'],
        'probe_score_before':before['probe']['score_percent'],'probe_score_after':after['probe']['score_percent'],
        'parameter_update_l2':delta,'initial_tensor_sha256':initial_hash,'final_tensor_sha256':trainable_hash(model),
        'positive_control_pass':after['fit']['mean_ce']<=.5*before['fit']['mean_ce'] and after['fit']['score_percent']>=before['fit']['score_percent'],
        'scope':'train-only label fitting diagnostic, not benchmark generalization or model capability gain','research_goal_complete':False}
    write_json(directory/'COMPLETE.json',value);write_json(directory/'PROGRESS.json',value);print(json.dumps(value),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',choices=['amp_lr5e6','fp32_lora_lr5e6','fp32_lora_lr1e4']);a=p.parse_args()
    if a.worker:worker(a.worker);return
    from research_runtime.queue import run_queue
    from vlm_deep.common import sha_file,write_json
    plan=read(OUT/'CONTROL_PLAN.json')
    jobs=[{'key':name,'command':[sys.executable,'-u',str(Path(__file__).resolve()),'--worker',name],
        'completion_marker':str(OUT/'controls'/name/'COMPLETE.json')} for name in plan['arms']]
    identity={'source_sha256':sha_file(Path(__file__)),'plan_sha256':sha_file(OUT/'CONTROL_PLAN.json'),
        'protocol_sha256':sha_file(PROJECT_ROOT/'docs/VLM_FAILURE_FACTORS_V2.md')}
    run_queue(OUT/'queue',jobs,identity)
    values={name:read(OUT/'controls'/name/'COMPLETE.json') for name in plan['arms']}
    write_json(OUT/'CONTROL_RESULTS.json',{'status':'completed','arms':values,'scope':'finite optimization diagnostics only','research_goal_complete':False})


if __name__=='__main__':main()
