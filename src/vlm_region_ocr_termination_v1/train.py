"""Derived from frozen region SFT loop; only auxiliary termination loss is ablated."""

import json
import os
from pathlib import Path
import random
import time


def train(config, role, study_lock):
    import numpy as np
    import torch
    from safetensors.torch import load_file
    from vlm_deep.common import collate,content_sha,encode,read_manifest,sha_file,to_device,write_json
    from vlm_fp32_readout_v1.model import load
    from .loss import answer_loss
    from vlm_human_sft.evaluate import lora_hash
    from vlm_object_train.train import learning_rate,lock_json,resume_checkpoint,save_checkpoint
    from vlm_probe.evaluate import processor_for

    root=Path(config['output_root']);directory=root/'runs'/role/'train';directory.mkdir(parents=True,exist_ok=True)
    reference=json.loads(Path(config['baseline_config']).read_text())
    data_root=Path(config['data_root'])
    manifest_path=data_root/f'data/{role.upper()}_TRAIN.json';manifest=read_manifest(manifest_path)
    rows={q['question_id']:q for q in manifest['splits']['train_all']}
    schedule_path=data_root/'data/SCHEDULE.json';updates=json.loads(schedule_path.read_text())['roles'][role]
    if len(updates)!=200 or len(rows)!=3200:raise ValueError('Full registered training view required')
    lineage={'study_lock_sha256':sha_file(study_lock),'role':role,'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES'],
        'schedule_sha256':sha_file(schedule_path),'manifest_sha256':sha_file(manifest_path),
        'config_sha256':content_sha(config),'source_sha256':sha_file(Path(__file__)),
        'numerics':'BF16 base / FP32 LoRA, frozen readout and CE / no global AMP / TF32 off'}
    lock_json(directory/'INPUT_LOCK.json',lineage);lineage_sha=content_sha(lineage)
    torch.set_num_threads(2);random.seed(config['seed']);np.random.seed(config['seed'])
    torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda:0')
    complete=directory/'TRAIN_COMPLETE.json'
    adapter=json.loads(complete.read_text())['adapter'] if complete.exists() else config['initial_adapter']
    model,binding=load(reference['model_root'],adapter,device,training=not complete.exists(),dropout=0.)
    processor=processor_for(Path(reference['model_root']),json.loads(Path(reference['reference_config']).read_text()),config['train_budget'])
    if complete.exists():return model,processor,json.loads(complete.read_text())
    initial_hash=lora_hash(model)
    if initial_hash!=json.loads(Path(study_lock).read_text())['initial_lora_sha256']:
        raise ValueError('Training initialization differs from the new baseline')
    parameters=[p for p in model.parameters() if p.requires_grad]
    if sum(p.numel() for p in parameters)!=config['expected_trainable'] or any(p.dtype!=torch.float32 for p in parameters):
        raise ValueError('Registered language-only FP32 trainable budget differs')
    optimizer=torch.optim.AdamW(parameters,lr=config['lr'],weight_decay=config['weight_decay'])
    cpu_records={v['question_id']:v for v in map(json.loads,(data_root/'data/CPU_INPUTS.jsonl').read_text().splitlines())}
    completed=0;stats={'questions':0,'loss_sum':0.,'active_seconds':0.,'trace_bytes':0,
        'source_counts':{},'source_loss_sums':{},'source_plain_loss_sums':{},'visual_tokens':0,'assistant_tokens':0,
        'peak_allocated_gib':0.,'peak_reserved_gib':0.}
    if (directory/'LATEST.json').exists():
        completed,stats=resume_checkpoint(directory,model,optimizer,lineage_sha,lineage['schedule_sha256'],len(updates))
    trace_path=directory/'INPUT_TRACES.jsonl';trace_path.touch(exist_ok=True)
    if trace_path.stat().st_size<stats['trace_bytes']:raise ValueError('Trace shorter than saved checkpoint boundary')
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    started=time.monotonic();prior=stats['active_seconds'];durations=[]
    with trace_path.open('r+b') as trace:
        trace.truncate(stats['trace_bytes']);trace.seek(0,os.SEEK_END)
        def checkpoint(step):
            stats['active_seconds']=prior+time.monotonic()-started;stats['trace_bytes']=trace.tell()
            trace.flush();os.fsync(trace.fileno())
            save_checkpoint(directory,model,optimizer,step,stats,lineage_sha,lineage['schedule_sha256'])
        for step in range(completed,len(updates)):
            if prior+time.monotonic()-started>=config['training_cap_seconds']:
                checkpoint(step);raise TimeoutError('Registered cumulative training cap reached')
            begin=time.monotonic();model.train();model.get_base_model().visual.eval();model.config.use_cache=False
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:group['lr']=learning_rate(step,len(updates),config)
            total=0.
            for micro_index,ids in enumerate(updates[step]['micros']):
                samples=[];evidence=[]
                if len(ids)!=4:raise ValueError('Expected a full four-QA microbatch')
                for qid in ids:
                    row=rows[qid];image=manifest['images'][row['image_id']]
                    sample=encode(processor,row,Path(image['path']),True);record=cpu_records[qid]
                    if (content_sha(sample['input_ids'].tolist())!=record['input_ids_sha256']
                            or content_sha(sample['labels'].tolist())!=record['labels_sha256']):
                        raise ValueError('Actual training QA differs from frozen CPU encoding')
                    samples.append(sample)
                    evidence.append({'update':step+1,'microbatch':micro_index,'question_id':qid,
                        'image_id':row['image_id'],'image_sha256':image['sha256'],'source':row['study_source'],
                        'target':row['target'],'input_tokens':record['input_tokens'],'visual_tokens':record['visual_tokens'],
                        'assistant_target_ids':record['assistant_target_ids'],
                        'input_ids':sample['input_ids'][0].tolist(),'labels':sample['labels'][0].tolist()})
                batch=to_device(collate(samples,processor.tokenizer.pad_token_id),device)
                with torch.autocast('cuda',enabled=False):
                    sources={rows[qid]['study_source'] for qid in ids}
                    if len(sources)!=1:raise ValueError('Expected source-homogeneous matched microbatch')
                    ce,plain,kept=answer_loss(model,batch,sources=={'region_aux'},processor.tokenizer.convert_tokens_to_ids('<|im_end|>'))
                    if ce.dtype!=torch.float32 or not bool(torch.isfinite(ce).all()):raise FloatingPointError('Nonfinite or non-FP32 CE')
                    loss=ce.sum()/config['effective_batch']
                loss.backward();values=ce.detach().cpu().tolist();plain_values=plain.detach().cpu().tolist();kept_values=kept.detach().cpu().tolist();total+=sum(values)
                for entry,value,plain_value,kept_count in zip(evidence,values,plain_values,kept_values):
                    source=entry['source'];entry.update(qa_ce=value,qa_plain_ce=plain_value,loss_active_tokens=kept_count)
                    trace.write((json.dumps(entry,ensure_ascii=False,allow_nan=False)+'\n').encode())
                    stats['source_counts'][source]=stats['source_counts'].get(source,0)+1
                    stats['source_loss_sums'][source]=stats['source_loss_sums'].get(source,0.)+value
                    stats['source_plain_loss_sums'][source]=stats['source_plain_loss_sums'].get(source,0.)+plain_value
                    stats['visual_tokens']+=entry['visual_tokens'];stats['assistant_tokens']+=len(entry['assistant_target_ids'])
                del batch,ce,plain,kept,loss,samples
            norm=float(torch.nn.utils.clip_grad_norm_(parameters,config['gradient_clip_norm'],error_if_nonfinite=True))
            if norm<=0:raise ValueError('Zero formal training gradient')
            optimizer.step();torch.cuda.synchronize()
            stats['questions']+=16;stats['loss_sum']+=total
            stats['active_seconds']=prior+time.monotonic()-started
            stats['peak_allocated_gib']=max(stats['peak_allocated_gib'],torch.cuda.max_memory_allocated()/2**30)
            stats['peak_reserved_gib']=max(stats['peak_reserved_gib'],torch.cuda.max_memory_reserved()/2**30)
            if step==0 or (step+1)%25==0:checkpoint(step+1)
            if step==0:
                saved=json.loads((directory/'LATEST.json').read_text())
                weights=load_file(str(directory/saved['directory']/'adapter/adapter_model.safetensors'))
                expected={n.replace('.default.','.'):v for n,v in model.named_parameters() if 'lora_' in n}
                if set(weights)!=set(expected) or any(not torch.equal(weights[n],v.detach().cpu()) for n,v in expected.items()):
                    raise ValueError('New FP32-head path changed saved LoRA tensors')
            durations.append(time.monotonic()-begin);durations=durations[-30:]
            if step==0 or (step+1)%10==0 or step+1==len(updates):
                speed=sum(durations)/len(durations)
                progress={'status':'training','role':role,'completed_updates':step+1,'planned_updates':len(updates),
                    'recent_seconds_per_update':speed,'remaining_training_seconds_estimate':speed*(len(updates)-step-1),
                    'last_update_mean_qa_ce':total/16,'gradient_norm':norm,'lr':optimizer.param_groups[0]['lr'],**stats}
                write_json(directory/'PROGRESS.json',progress);print(json.dumps(progress),flush=True)
                if step==0:write_json(directory/'FIRST_UPDATE.json',{**progress,'initial_lora_sha256':initial_hash,
                    'binding':binding,'discarded_preflight_updates':0,'saved_adapter_master_values_verified':True})
            if stats['active_seconds']>=config['training_cap_seconds']:
                checkpoint(step+1)
                if step+1<len(updates):raise TimeoutError('Registered training cap reached at saved optimizer boundary')
    if stats['questions']!=3200 or stats['source_counts'].get('common')!=1600:
        raise ValueError('Completed exposure counts differ from protocol')
    destination=directory/'adapter';model.save_pretrained(destination,safe_serialization=True)
    result={'status':'completed','role':role,'updates':200,'adapter':str(destination),
        'initial_lora_sha256':initial_hash,'final_lora_sha256':lora_hash(model),'lineage_sha256':lineage_sha,
        'binding':binding,'input_trace_sha256':sha_file(trace_path),'discarded_preflight_updates':0,**stats}
    write_json(complete,result);write_json(directory/'PROGRESS.json',result)
    return model,processor,result
