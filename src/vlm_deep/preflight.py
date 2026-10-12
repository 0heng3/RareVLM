"""Real-image loss/gradient equivalence and component/memory contracts."""
from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path
from rarevlm_paths import PROJECT_ROOT
from .common import answer_loss,collate,content_sha,encode,frozen_parameter_hash,to_device,write_json
from .model import load,parameter_groups


def main():
    import torch
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--asset-root',type=Path,required=True);p.add_argument('--pilot-root',type=Path,required=True)
    p.add_argument('--config',type=Path,default=PROJECT_ROOT/'configs/vlm_deep_v2.json')
    p.add_argument('--method',choices=['qv_r8','qkvo_r4','qv_r4_merger'],required=True)
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args();config=json.loads(args.config.read_text())
    torch.cuda.set_device(args.device)
    torch.set_num_threads(2);random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    old=json.loads((args.pilot_root/'prepared/manifest.json').read_text())
    rows=[dict(r,target=r['answer']) for r in old['splits']['train'][:config['training_microbatch']]]
    model,processor=load(args.pilot_root/'model',config,args.device,args.method,training=True)
    before=frozen_parameter_hash(model)
    samples=[encode(processor,r,args.pilot_root/'prepared'/r['image'],True) for r in rows]
    batch=to_device(collate(samples,processor.tokenizer.pad_token_id),args.device)
    rng_cpu=torch.get_rng_state();rng_gpu=torch.cuda.get_rng_state()
    full,_,_=answer_loss(model,batch,tail=False);full.mean().backward()
    full_grad={name:p.grad.detach().cpu().clone() for name,p in model.named_parameters() if p.requires_grad}
    model.zero_grad(set_to_none=True)
    torch.set_rng_state(rng_cpu);torch.cuda.set_rng_state(rng_gpu)
    tail,_,_=answer_loss(model,batch,tail=True);tail.mean().backward()
    if not torch.allclose(full.detach(),tail.detach(),atol=2e-5,rtol=2e-5):raise ValueError('tail head loss mismatch')
    errors={};group_grad={'language':0.,'merger':0.}
    for name,p in model.named_parameters():
        if not p.requires_grad:
            if p.grad is not None:raise ValueError('frozen gradient leak')
            continue
        if p.grad is None or not torch.isfinite(p.grad).all():raise ValueError('missing/nonfinite LoRA gradient')
        actual=p.grad.detach().cpu();reference=full_grad[name]
        if not torch.allclose(actual,reference,atol=3e-6,rtol=1e-3):raise ValueError(f'tail gradient mismatch: {name}')
        errors[name]=float((actual-reference).abs().max())
        group_grad['merger' if '.visual.merger.' in name else 'language']+=float(actual.abs().sum())
    groups=parameter_groups(model)
    if group_grad['language']<=0 or (groups['merger'] and group_grad['merger']<=0):raise ValueError('component lacks a gradient')
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config['lr'])
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
    optimizer.step();model.zero_grad(set_to_none=True)
    if before!=frozen_parameter_hash(model):raise ValueError('frozen weights changed')
    # Resource stress uses training images and deliberately extended prompts only.
    stress=[]
    for row,sample in zip(rows,samples):
        item=dict(row);extra=max(config['max_sequence_length']-sample['input_ids'].shape[1]-12,0)
        item['question']+=' read'*extra
        x=encode(processor,item,args.pilot_root/'prepared'/row['image'],True)
        if x['input_ids'].shape[1]>config['max_sequence_length']:raise ValueError('stress exceeds declared limit')
        stress.append(x)
    stress_batch=to_device(collate(stress,processor.tokenizer.pad_token_id),args.device)
    torch.cuda.reset_peak_memory_stats()
    losses,_,_=answer_loss(model,stress_batch,tail=True);losses.mean().backward()
    if not torch.isfinite(losses).all():raise FloatingPointError('stress loss nonfinite')
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
    result={'passed':True,'method':args.method,'config_sha256':content_sha(config),
        'parameter_groups':groups,'trainable_parameters':sum(groups.values()),
        'tail_head_loss_max_absolute_error':float((full-tail).detach().abs().max()),
        'tail_head_gradient_max_absolute_error':max(errors.values()),'component_gradient_absolute_sums':group_grad,
        'frozen_parameters_unchanged':True,'microbatch':len(rows),
        'stress_sequence_tokens':stress_batch['input_ids'].shape[1],
        'stress_peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
        'gpu':torch.cuda.get_device_name(),'scope':'training-image correctness/resource preflight only; no benchmark score'}
    write_json(args.asset_root/'preflight'/f'{args.method}.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
