"""Registered finite-step follow-up; do not modify the frozen projection study."""
import argparse
import os
from pathlib import Path
import sys
import time
import torch
from vlm_deep.common import sha_file,write_json
from vlm_fp32_adapter.model import restore_fp32_adapter
from vlm_human_sft.evaluate import lora_hash
from vlm_object_train.train import learning_rate,lock_json
from vlm_sft.common import to_device
from vlm_reference_replay_v1.loss import forward_logits,loss_values
from vlm_replay_projection_v1.common import ROOT,PARENT,read,actor,chosen_batch,gradients,flat

DEST=ROOT/'finite_step_v1'
PROTOCOL=Path(__file__).resolve().parents[1]/'docs/VLM_REPLAY_FINITE_STEP_V1_PROTOCOL.md'


def worker():
    identity=read(DEST/'INPUT_LOCK.json')
    for path,digest in identity['files'].items():
        if sha_file(path)!=digest:raise ValueError('Finite-step input/source changed: '+path)
    plan=read(PARENT/'PLAN.json');data=read(PARENT/'DATA.json');cpu=read(PARENT/'CPU_INPUTS.json');schedule=read(PARENT/'SCHEDULE.json')
    lookup={q['question_id']:q for q in data['training_rows']}
    model,processor,_=actor(training=True);parameters=[p for p in model.parameters() if p.requires_grad]
    begin=time.monotonic();points=[]
    for checkpoint in plan['calibration_checkpoints']:
        restore_fp32_adapter(model,checkpoint['adapter'])
        if lora_hash(model)!=checkpoint['trainable_sha256']:raise ValueError('Checkpoint differs')
        state=torch.load(checkpoint['state_path'],map_location='cpu',weights_only=False)
        if list(state['trainable_parameters'])!=[n for n,p in model.named_parameters() if p.requires_grad]:raise ValueError('Parameter coordinates differ')
        optimizer=torch.optim.AdamW(parameters,lr=2e-4,weight_decay=0.);optimizer.load_state_dict(state['optimizer']);optimizer.zero_grad(set_to_none=True)
        step=checkpoint['step'];lr=learning_rate(step,64,{'lr':2e-4,'warmup_ratio':.125})
        for group in optimizer.param_groups:group['lr']=lr
        _,vectors,before,_=gradients(model,processor,schedule[step],data,cpu)
        original=flat(parameters).clone();torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);optimizer.step();delta=flat(parameters)-original
        predicted={s:2*float(torch.dot(g.double(),delta.double())) for s,g in vectors.items()}
        after={'hard':0.,'replay':0.}
        # Preserve train()/visual.eval(), four padded frames, head/tail readout and
        # use_cache=False. Only disable gradient recording for this second readout.
        with torch.no_grad():
            for micro,ids in enumerate(schedule[step]['micros']):
                source='replay' if micro in (0,2) else 'hard';rows=[lookup[qid] for qid in ids]
                batch=chosen_batch(processor,rows,data['images'],cpu);gpu=to_device(batch,next(model.parameters()).device)
                logits=forward_logits(model,gpu);ce,_=loss_values(logits,gpu['labels'])
                if not bool(torch.isfinite(ce).all()):raise FloatingPointError('Nonfinite finite-step loss')
                after[source]+=float(ce.sum());del batch,gpu,logits,ce
        points.append({'checkpoint_step':step,'next_update':step+1,'lr':lr,'source_question_counts':{'hard':8,'replay':8},
                       'before_ce':{s:v/8 for s,v in before.items()},'after_ce':{s:v/8 for s,v in after.items()},
                       'actual_ce_delta':{s:(after[s]-before[s])/8 for s in before},'linearized_ce_delta':predicted,
                       'same_question_ids':schedule[step]['micros']})
        write_json(DEST/'PROGRESS.json',{'status':'diagnosing','points':points,'active_seconds':time.monotonic()-begin})
        del optimizer,state,vectors,original,delta
        if time.monotonic()-begin>180:raise TimeoutError('180sec finite-step active budget')
        if torch.cuda.max_memory_allocated()/2**30>65:raise RuntimeError('65GiB finite-step budget')
    restore_fp32_adapter(model,plan['initial_adapter'])
    if lora_hash(model)!=plan['initial_lora_sha256']:raise ValueError('Original adapter not restored')
    write_json(DEST/'COMPLETION.json',{'status':'completed','points':points,'formal_optimizer_updates':0,'discarded_optimizer_updates':3,
                'new_generations':0,'active_seconds':time.monotonic()-begin,'restored_initial_lora_sha256':lora_hash(model),
                'scope':'Training minibatch finite-step CE only; not validation, independent VQA, or cause of global transfer failure.'})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    if args.worker:worker();return
    from research_runtime.queue import run_queue
    files=[Path(__file__).resolve(),PROTOCOL,ROOT/'PLAN.json',ROOT/'DIAGNOSTIC.json',PARENT/'PLAN.json',PARENT/'DATA.json',PARENT/'CPU_INPUTS.json',PARENT/'SCHEDULE.json']
    files += list((Path(__file__).resolve().parents[1]/'src/vlm_replay_projection_v1').glob('*.py'))
    for c in read(PARENT/'PLAN.json')['calibration_checkpoints']:
        files += [Path(c['state_path']),Path(c['adapter'])/'adapter_model.safetensors',Path(c['adapter'])/'adapter_config.json']
    lock_json(DEST/'INPUT_LOCK.json',{'files':{str(p):sha_file(p) for p in files},'purpose':'finite-step CE after zero-conflict directional diagnostic','automatic_training':False})
    run_queue(DEST/'queue',[{'key':'finite_step','command':[sys.executable,'-u',str(Path(__file__).resolve()),'--worker'],'completion_marker':str(DEST/'COMPLETION.json')}],{'input_lock_sha256':sha_file(DEST/'INPUT_LOCK.json')})
    print('FINITE_STEP_COMPLETED',flush=True)


if __name__=='__main__':main()
