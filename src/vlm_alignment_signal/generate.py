"""Same image/question greedy+four samples; score references after generation."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import torch
from vlm_deep.common import encode,collate,to_device,consensus,NORMALIZER
from vlm_probe.evaluate import content_sha,inference_row,lock_file,now,package_versions,processor_for,describe_input,sha_file
from vlm_joint.evaluate import visible_gpu_binding
from vlm_precision.evaluate import load_path
from .common import write_json,group_advantages,preference,native
from .loss import sequence_logps,dpo_loss

def cached_records(path,rows,lock_sha):
    if not path.exists():return []
    raw=path.read_bytes().splitlines(keepends=True)
    if raw and not raw[-1].endswith(b'\n'):
        raw.pop();path.write_bytes(b''.join(raw))
    items=[json.loads(x) for x in raw]
    if len(items)>len(rows):raise ValueError('Too many cached questions')
    for item,row in zip(items,rows):
        if item['question_id']!=row['question_id'] or item['input_lock_sha256']!=lock_sha:raise ValueError('Resume question/lock changed')
        if len(item['samples'])!=4:raise ValueError('Incomplete sampling group')
        expected=[consensus(x['prediction'],row['answers']) for x in item['samples']]
        if expected!=item['rewards'] or item['greedy_score']!=consensus(item['greedy']['prediction'],row['answers']):raise ValueError('Resume reward changed')
    return items

def gpu_dpo_contract(model,processor,pairs,images,image_root,cfg):
    contract=[];params=[]
    for name,p in model.named_parameters():
        p.requires_grad_('lora_' in name)
        if p.requires_grad:params.append(p)
    if sum(p.numel() for p in params)!=1843200:raise ValueError('Unexpected LoRA budget')
    for row,pair in pairs[:cfg['dpo_gpu_contract_pairs_per_role']]:
        model.zero_grad(set_to_none=True)
        samples=[encode(processor,{**row,'target':pair[k]},image_root/images[row['image_id']]['path'],True) for k in ['chosen','rejected']]
        batch=to_device(collate(samples,processor.tokenizer.pad_token_id),torch.device('cuda',0))
        torch.cuda.reset_peak_memory_stats()
        logps,counts=sequence_logps(model,batch)
        reference=logps.detach().clone()
        loss=dpo_loss(logps[:1],logps[1:],reference[:1],reference[1:],cfg['dpo_beta']).mean()
        if abs(float(loss.detach())-math.log(2))>1e-5:raise ValueError('DPO identity loss differs')
        loss.backward()
        gradients=[p.grad for p in params if p.grad is not None]
        if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):raise ValueError('Nonfinite/no DPO gradient')
        norm=math.sqrt(sum(float(g.float().square().sum()) for g in gradients))
        if not math.isfinite(norm) or norm<=0:raise ValueError('No finite nonzero gradient')
        if any(p.grad is not None for name,p in model.named_parameters() if 'lora_' not in name):raise ValueError('Non-LoRA gradient')
        contract.append({'question_id':row['question_id'],'chosen':pair['chosen'],'rejected':pair['rejected'],
            'chosen_logp':float(logps[0].detach()),'rejected_logp':float(logps[1].detach()),
            'assistant_token_counts':counts.tolist(),'beta':cfg['dpo_beta'],'initial_dpo_loss':float(loss.detach()),
            'reference_source':'same fixed initial forward detached;no parameter update',
            'lora_gradient_l2':norm,'finite_gradients':True,'only_language_lora_gradients':True,
            'trainable_gradient_elements':1843200,'optimizer_updates':0,
            'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30})
        del logps,loss,batch,samples
    model.zero_grad(set_to_none=True);model.requires_grad_(False)
    return contract

def run(config_path,role):
    cfg=json.loads(config_path.read_text());root=Path(cfg['output_root']);roster=json.loads((root/'DATA_ROSTER.json').read_text())
    rows=roster['questions'][role::len(cfg['roles'])];images=roster['images'];image_root=Path(roster['image_root'])
    directory=root/'roles'/f'role{role}';directory.mkdir(parents=True,exist_ok=True)
    lock={'config_sha256':sha_file(config_path),'study_lock_sha256':sha_file(root/'STUDY_INPUT_LOCK.json'),
        'data_binding_sha256':sha_file(root/'DATA_BINDINGS.json'),'hardware':visible_gpu_binding(),'packages':package_versions(),
        'role':role,'ordered_question_ids':[r['question_id'] for r in rows],'answers_used_in_generation':False}
    lock_file(directory/'INPUT_LOCK.json',lock);lock_sha=content_sha(lock)
    cache=directory/'predictions.jsonl';items=cached_records(cache,rows,lock_sha)
    if (directory/'COMPLETE.json').exists():
        if json.loads((directory/'COMPLETE.json').read_text())['predictions_sha256']!=sha_file(cache):raise ValueError('Completed output changed')
        return
    torch.cuda.set_device(0);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda',0)
    reference=json.loads(Path(cfg['reference_config']).read_text())
    start=time.monotonic();model,dtypes=load_path(Path(cfg['model_root']),reference,device,cfg['numerical_path'],Path(cfg['adapter_dir']))
    load_seconds=time.monotonic()-start
    processor=processor_for(Path(cfg['model_root']),reference,cfg['visual_token_budget'])
    image_token_id=model.config.image_token_id
    begun=time.monotonic();previous=sum(x['resource']['total_group_seconds'] for x in items)
    with cache.open('a') as stream:
        for index in range(len(items),len(rows)):
            row=rows[index];group_started=time.monotonic()
            # inference_row excludes target and all ten human reference answers.
            sample=encode(processor,inference_row(row),image_root/images[row['image_id']]['path'],False)
            detail=describe_input(sample,row,images,processor,image_token_id,cfg['visual_token_budget'])
            if detail['input_tokens']>cfg['max_input_tokens']:raise ValueError('Input too long;no truncation allowed')
            seed=int(content_sha({'seed':cfg['sample_seed'],'qid':row['question_id']})[:8],16)
            detail.update(input_ids_sha256=content_sha(sample['input_ids'].tolist()),
                pixel_values_sha256=hashlib.sha256(sample['pixel_values'].numpy().tobytes()).hexdigest(),
                source_image_sha256=images[row['image_id']]['sha256'],input_references_used=False)
            def generate(copies,sampling):
                batch=to_device(collate([sample]*copies,processor.tokenizer.pad_token_id),device)
                torch.cuda.manual_seed_all(seed);torch.manual_seed(seed)
                torch.cuda.synchronize();t=time.monotonic()
                options={'temperature':cfg['sampling_temperature'],'top_p':cfg['sampling_top_p'],'top_k':cfg['sampling_top_k']} if sampling else {}
                with torch.inference_mode():generated=model.generate(**batch,max_new_tokens=cfg['max_new_tokens'],do_sample=sampling,use_cache=True,**options)
                torch.cuda.synchronize();elapsed=time.monotonic()-t;length=batch['input_ids'].shape[1]
                output=[]
                for ids in generated[:,length:]:
                    output.append({'prediction':processor.tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False).strip(),
                        'generated_nonpad_tokens':int((ids!=processor.tokenizer.pad_token_id).sum()),
                        'tail_slots':int(ids.numel())})
                del batch,generated;return output,elapsed
            torch.cuda.reset_peak_memory_stats();greedy,greedy_seconds=generate(1,False);responses,sample_seconds=generate(4,True)
            scores=[consensus(x['prediction'],row['answers']) for x in responses]
            pair=preference([x['prediction'] for x in responses],scores,NORMALIZER,cfg['preference_min_chosen_score'],cfg['preference_max_rejected_score'])
            item={'question_id':row['question_id'],'image_id':row['image_id'],'question':row['question'],
                'input_lock_sha256':lock_sha,'sampling_seed':seed,'input':detail,'greedy':greedy[0],
                'greedy_score':consensus(greedy[0]['prediction'],row['answers']),'samples':responses,'rewards':scores,
                'advantages':group_advantages(scores).tolist(),'preference':pair,
                'unique_normalized_samples':len({NORMALIZER(x['prediction']) for x in responses}),
                'train_target_frequency':row['train_target_frequency'],
                'resource':{'greedy_generation_seconds':greedy_seconds,'sample_generation_seconds':sample_seconds,
                    'total_group_seconds':time.monotonic()-group_started,
                    'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30}}
            stream.write(json.dumps(native(item),ensure_ascii=False,allow_nan=False)+'\n');stream.flush();items.append(item)
            if (index+1)%8==0 or index+1==len(rows):
                elapsed=previous+time.monotonic()-begun
                write_json(directory/'PROGRESS.json',{'status':'sampling','completed_questions':index+1,'planned_questions':len(rows),
                    'generated_responses':5*(index+1),'active_seconds':elapsed,'eta_seconds':(len(rows)-index-1)*elapsed/(index+1),
                    'eligible_preference_pairs':sum(x['preference'] is not None for x in items),'updated_at':now()})
                print('SIGNAL_SAMPLING',role,index+1,'/',len(rows),flush=True)
    by_qid={r['question_id']:r for r in rows};pairs=[]
    for item in items:
        if item['preference'] is not None:
            p=item['preference'];pairs.append((by_qid[item['question_id']],
                {'chosen':item['samples'][p['chosen_index']]['prediction'],'rejected':item['samples'][p['rejected_index']]['prediction']}))
    write_json(directory/'PROGRESS.json',{'status':'dpo_gradient_contract','completed_questions':len(rows),'updated_at':now()})
    contracts=gpu_dpo_contract(model,processor,pairs,images,image_root,cfg)
    result={'status':'completed','completed_at':now(),'role':role,'questions':len(rows),'generated_responses':5*len(rows),
        'model_load_seconds':load_seconds,'sampling_active_seconds':sum(x['resource']['total_group_seconds'] for x in items),
        'predictions_sha256':sha_file(cache),'input_lock_sha256':lock_sha,'dtypes':dtypes,
        'dpo_gradient_contracts':contracts,'new_optimizer_updates':0,'new_trained_adapters':0}
    write_json(directory/'COMPLETE.json',result);write_json(directory/'PROGRESS.json',{'status':'completed','completed_questions':len(rows),'updated_at':now()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--role',type=int,required=True);a=p.parse_args()
    run(a.config,a.role)
