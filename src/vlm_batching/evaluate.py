"""Resumable same-card full-dev and repeated offline batching measurements."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
import sys
sys.dont_write_bytecode = True
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION','python')
from rarevlm_paths import PROJECT_ROOT
from vlm_probe.evaluate import (content_sha, now, sha_file, write_json, lock_file,
    processor_for, package_versions, inference_row)
from vlm_deep.common import encode, collate, to_device
from vlm_deep.evaluate import score_prediction
from vlm_joint.evaluate import visible_gpu_binding, model_metadata_binding
from vlm_precision.evaluate import load_path
from vlm_ocr.evaluate import adapter_binding
from vlm_weighted.evaluate import dev_data
from .prepare import DEFAULT_CONFIG, read_plan


def build_batch(processor, chunk, images, image_root, expected):
    samples=[]; details=[]
    image_token=processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
    for row in chunk:
        qid=row['question_id']; item=images[row['image_id']]; path=image_root/item['path']
        if sha_file(path)!=item['sha256']:
            raise ValueError('input image changed')
        sample=encode(processor,inference_row(row),path,False)
        if 'labels' in sample:
            raise ValueError('labels entered generation')
        ids=sample['input_ids'][0].tolist()
        grids=sample['image_grid_thw'].tolist()
        pixels=sample['pixel_values'].detach().cpu().contiguous()
        pixel_sha=hashlib.sha256(pixels.numpy().tobytes()).hexdigest()
        detail={'question_id':qid,'image_id':row['image_id'],
            'image_grid_thw':grids,'input_tokens':len(ids),
            'visual_tokens':math.prod(grids[0])//4,
            'counted_image_tokens':int((sample['input_ids']==image_token).sum()),
            'input_ids_sha256':content_sha(ids),'pixel_values_sha256':pixel_sha,
            'pixel_values_shape':list(pixels.shape),'image_sha256':item['sha256'],
            'input_truncated':False,'answers_or_target_used':False,
            'question_sha256':content_sha(row['question'])}
        validate_detail(detail,expected[qid])
        samples.append(sample); details.append(detail)
    batch=collate(samples,processor.tokenizer.pad_token_id)
    if batch['attention_mask'].sum().item()!=sum(d['input_tokens'] for d in details):
        raise ValueError('left-padding mask changed actual inputs')
    padded=int(batch['input_ids'].shape[1])
    if padded!=max(d['input_tokens'] for d in details):
        raise ValueError('collation length differs')
    for d in details: d['padded_input_tokens']=padded
    return batch,details


def validate_detail(record,expected):
    keys=('question_id','image_id','image_grid_thw','input_tokens','visual_tokens',
          'counted_image_tokens','input_ids_sha256','pixel_values_sha256',
          'pixel_values_shape','image_sha256','question_sha256')
    for key in keys:
        if record[key]!=expected[key]:
            raise ValueError(f'actual per-QA encoding differs from frozen baseline:{key}')
    if record.get('input_truncated') is not False or record.get('answers_or_target_used') is not False:
        raise ValueError('generation contract failed')


def generate_batch(model,processor,chunk,images,image_root,expected,device,check_logits=False):
    import torch
    start=time.monotonic()
    cpu,details=build_batch(processor,chunk,images,image_root,expected)
    preprocessing=time.monotonic()-start
    torch.cuda.synchronize(device);start=time.monotonic()
    batch=to_device(cpu,device);torch.cuda.synchronize(device)
    transfer=time.monotonic()-start
    torch.cuda.reset_peak_memory_stats(device)
    finite=None
    if check_logits:
        result=model(**batch,use_cache=False)
        finite=bool(torch.isfinite(result.logits).all());del result
        if not finite: raise ValueError('non-finite logits')
    torch.cuda.synchronize(device);start=time.monotonic()
    generated=model.generate(**batch,max_new_tokens=32,do_sample=False,use_cache=True)
    torch.cuda.synchronize(device);generation=time.monotonic()-start
    width=int(batch['input_ids'].shape[1]);answers=[]
    for i,detail in enumerate(details):
        tail=generated[i,width:]
        answers.append({**detail,'prediction':processor.tokenizer.decode(tail,
            skip_special_tokens=True,clean_up_tokenization_spaces=False).strip(),
            'generated_nonpad_tokens':int((tail!=processor.tokenizer.pad_token_id).sum()),
            'generated_tail_slots':int(tail.numel())})
    resource={'batch_size':len(chunk),'padded_input_tokens':width,
        'useful_input_tokens':sum(d['input_tokens'] for d in details),
        'padding_slots':len(chunk)*width-sum(d['input_tokens'] for d in details),
        'generation_seconds':generation,'preprocessing_seconds':preprocessing,'h2d_seconds':transfer,
        'peak_allocated_gib':torch.cuda.max_memory_allocated(device)/2**30,
        'peak_reserved_gib':torch.cuda.max_memory_reserved(device)/2**30,'finite_logits':finite}
    if resource['peak_allocated_gib']>65:
        raise ValueError('predeclared memory limit exceeded; no automatic smaller batch fallback')
    return answers,resource


def cached_batches(path,schedule,lock,by_id,expected,scored):
    if not path.exists():return [],None
    raw=path.read_bytes(); repair=None
    if raw and not raw.endswith(b'\n'):
        keep=raw.rfind(b'\n')+1
        repair={'original_bytes':len(raw),'kept_bytes':keep,'reason':'interrupted final JSON batch'}
        path.write_bytes(raw[:keep]);raw=raw[:keep]
    batches=[json.loads(line) for line in raw.splitlines()]
    if len(batches)>len(schedule):raise ValueError('extra cached batches')
    for i,b in enumerate(batches):
        if b['batch_index']!=i or b['input_lock_sha256']!=content_sha(lock):
            raise ValueError('cached identity changed')
        if [r['question_id'] for r in b['records']]!=schedule[i]:
            raise ValueError('cached batch schedule differs')
        for r in b['records']:
            validate_detail(r,expected[r['question_id']])
            if scored and not math.isclose(r['score'],score_prediction(r['prediction'],by_id[r['question_id']],'textvqa'),abs_tol=1e-12):
                raise ValueError('cached scoring changed')
        for key in ('generation_seconds','preprocessing_seconds','h2d_seconds','peak_allocated_gib','peak_reserved_gib'):
            if not math.isfinite(b['resource'][key]) or b['resource'][key]<0:
                raise ValueError('invalid cached timing')
    return batches,repair


def summarize(batches,lock,session,scored):
    records=[r for b in batches for r in b['records']]
    steady=[b for b in batches if not b['timing_warmup']]
    if not steady:raise ValueError('no steady batches')
    n=sum(len(b['records']) for b in steady)
    times={k:sum(b['resource'][k] for b in steady) for k in ('generation_seconds','preprocessing_seconds','h2d_seconds')}
    slots=sum(b['resource']['batch_size']*b['resource']['padded_input_tokens'] for b in batches)
    useful=sum(r['input_tokens'] for r in records)
    result={'status':'completed','n':len(records),'images':len({r['image_id'] for r in records}),
        'input_lock_sha256':content_sha(lock),'session':session,
        'resource':{'steady_questions':n,'first_batch_per_session_excluded':False,'separate_extra_batch_warmup':True,
            'pure_generation_qa_per_second':n/times['generation_seconds'],
            'preprocess_h2d_generation_qa_per_second':n/sum(times.values()),
            'steady_seconds':times,'peak_allocated_gib':max(b['resource']['peak_allocated_gib'] for b in batches),
            'peak_reserved_gib':max(b['resource']['peak_reserved_gib'] for b in batches),
            'padded_input_slots':slots,'useful_input_tokens':useful,'padding_fraction':(slots-useful)/slots,
            'scope':'offline CPU decoding/hash validation/processor,H2D,greedy generation;no queue,network or service SLA'}}
    if scored:result['score_percent']=100*statistics.mean(r['score'] for r in records)
    return result


def run_cell(directory,schedule,policy,common,model,processor,by_id,images,image_root,expected,device,scored,update):
    import torch
    directory.mkdir(parents=True,exist_ok=True)
    lock={**common,'policy':policy,'schedule_sha256':content_sha(schedule),'scored':scored,
          'question_count':sum(map(len,schedule)),'directory_role':str(directory.name)}
    lock_file(directory/'INPUT_LOCK.json',lock)
    path=directory/'batches.jsonl'
    batches,repair=cached_batches(path,schedule,lock,by_id,expected,scored)
    done=directory/'EVALUATION.json'
    if done.exists():
        result=json.loads(done.read_text())
        if len(batches)!=len(schedule) or result['batches_sha256']!=sha_file(path):
            raise ValueError('completed cache changed')
        update(result['n'],'completed');return result
    first=len(batches)
    torch.cuda.empty_cache()
    if first<len(schedule):
        with torch.inference_mode():
            warm_answers,warm_resource=generate_batch(model,processor,[by_id[q] for q in schedule[first]],
                images,image_root,expected,device)
        write_json(directory/'WARMUP.json',{'completed_at':now(),'question_ids':schedule[first],
            'resource':warm_resource,'scope':'extra unscored generation before measured QA;not research predictions'})
        del warm_answers
    started=time.monotonic()
    session_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    session={'started_at':now(),'id':session_id,'resumed_batches':first,'repair':repair}
    with path.open('a') as stream,torch.inference_mode():
        for i in range(first,len(schedule)):
            chunk=[by_id[q] for q in schedule[i]]
            answers,resource=generate_batch(model,processor,chunk,images,image_root,expected,device)
            if scored:
                for r,row in zip(answers,chunk):r['score']=score_prediction(r['prediction'],row,'textvqa')
            record={'batch_index':i,'records':answers,'resource':resource,'timing_warmup':False,
                    'input_lock_sha256':content_sha(lock),'session_id':session_id}
            stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
            stream.flush();os.fsync(stream.fileno());batches.append(record)
            count=sum(len(b['records']) for b in batches)
            elapsed=time.monotonic()-started; new=count-sum(map(len,schedule[:first]))
            write_json(directory/'PROGRESS.json',{'status':'running','updated_at':now(),'completed_questions':count,
                'total_questions':sum(map(len,schedule)),'completed_batches':len(batches),
                'elapsed_seconds':elapsed,'eta_seconds':(sum(map(len,schedule))-count)*elapsed/new,'last_batch':resource})
            update(count,'running')
            if i==first or (i+1)%25==0:print('GENERATED',directory,count,flush=True)
    session.update(completed_at=now(),wall_seconds=time.monotonic()-started)
    result=summarize(batches,lock,session,scored);result['batches_sha256']=sha_file(path)
    records=[r for b in batches for r in b['records']]
    pred=directory/'predictions.jsonl'
    tmp=pred.with_suffix('.jsonl.tmp')
    tmp.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in records));tmp.replace(pred)
    result['predictions_sha256']=sha_file(pred)
    write_json(done,result);write_json(directory/'PROGRESS.json',{'status':'completed','completed_questions':len(records)})
    update(len(records),'completed');print('CELL COMPLETE',directory,result.get('score_percent'),result['resource'],flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group',choices=('batch16','batch32'),required=True)
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    args=parser.parse_args();cfg,root,plan=read_plan(args.config)
    worker=root/'workers'/args.group;worker.mkdir(parents=True,exist_ok=True)
    status={'status':'starting','started_at':now(),'group':args.group,'cells':{}}
    def update(key,count,state):
        status['cells'][key]={'completed_questions':count,'state':state}
        status['updated_at']=now();write_json(worker/'STATUS.json',status)
    model=hook=None
    try:
        rows,images,image_root,metadata=dev_data(Path(cfg['asset_root'])); by_id={r['question_id']:r for r in rows}
        input_contract=json.loads(Path(cfg['upstream_input_contract']).read_text())
        expected={r['question_id']:r for r in input_contract['records'] if r['condition']=='baseline'}
        reference=json.loads((PROJECT_ROOT/cfg['reference_config']).read_text())
        adapter=Path(cfg['fixed_adapter_dir']); model_root=Path(cfg['model_root'])
        files,training=adapter_binding(cfg,adapter,model_root)
        import torch
        if not torch.cuda.is_available() or int(os.environ.get('WORLD_SIZE','1'))!=1:
            raise ValueError('single admitted CUDA worker required')
        torch.cuda.set_device(0);torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest');device=torch.device('cuda',0)
        props=torch.cuda.get_device_properties(device)
        hardware={**visible_gpu_binding(),'name':props.name,'total_memory_bytes':props.total_memory,
                  'compute_capability':[props.major,props.minor],'torch_cuda_version':torch.version.cuda}
        common={'schema_version':'rarevlm.batching.worker.v1','config_sha256':sha_file(args.config),
            'plan_sha256':sha_file(root/'PLAN.json'),'source_hashes':plan['source_hashes'],
            'hardware':hardware,'packages':package_versions(),'data':metadata,
            'adapter_files_sha256':files,'training_binding':training,
            'model_binding':model_metadata_binding(model_root,reference),'group':args.group,
            'numerical_path':'original_bf16','visual_token_budget':1024,'max_new_tokens':32,
            'do_sample':False,'input_truncation':False,'attention_implementation':'sdpa',
            'tf32':False,'head_last_position_only':True,'input_contract_sha256':sha_file(Path(cfg['upstream_input_contract']))}
        lock_file(worker/'INPUT_LOCK.json',common);status['hardware']=hardware
        status['status']='loading_model';write_json(worker/'STATUS.json',status)
        begin=time.monotonic();model,dtypes=load_path(model_root,reference,device,'original_bf16',adapter)
        status['model_load_seconds']=time.monotonic()-begin;status['actual_dtypes']=dtypes
        hook=model.get_output_embeddings().register_forward_pre_hook(lambda module,items:(items[0][:,-1:,:],))
        processor=processor_for(model_root,cfg,1024)
        for policy in cfg['groups'][args.group]:
            pre=worker/'preflight'/policy;pre.mkdir(parents=True,exist_ok=True)
            if not (pre/'PREFLIGHT.json').exists():
                size=cfg['policies'][policy]['batch_size']
                longest=sorted(by_id,key=lambda q:(-expected[q]['input_tokens'],int(q)))[:size]
                stress=[plan['schedules'][policy][0],longest]
                batches=[]
                status['status']='preflight';update(policy,0,'preflight')
                with torch.inference_mode():
                    for ids in stress:
                        answers,resource=generate_batch(model,processor,[by_id[q] for q in ids],images,image_root,expected,device,True)
                        batches.append({'records':answers,'resource':resource})
                write_json(pre/'PREFLIGHT.json',{'status':'passed','completed_at':now(),
                    'common_lock_sha256':content_sha(common),'batches':batches,'scope':'input-only finite logits/memory/generation;no scores'})
            else:
                passed=json.loads((pre/'PREFLIGHT.json').read_text())
                if passed['status']!='passed' or passed['common_lock_sha256']!=content_sha(common):
                    raise ValueError('preflight cache mismatch')
            print('PREFLIGHT PASSED',args.group,policy,flush=True)
        status['status']='full_dev'
        for policy in cfg['groups'][args.group]:
            run_cell(root/'runs'/args.group/policy/'dev',plan['schedules'][policy],policy,common,
                model,processor,by_id,images,image_root,expected,device,True,
                lambda n,s,p=policy:update('dev/'+p,n,s))
        status['status']='timing_repeats'
        for repeat,order in enumerate(cfg['timing_orders'][args.group]):
            for policy in order:
                run_cell(root/'timing'/args.group/f'repeat{repeat}'/policy,plan['timing_schedules'][policy],policy,common,
                    model,processor,by_id,images,image_root,expected,device,False,
                    lambda n,s,p=policy,r=repeat:update(f'timing/{r}/{p}',n,s))
        read_plan(args.config)
        status.update(status='completed',completed_at=now());write_json(worker/'STATUS.json',status)
    except Exception as error:
        status.update(status='failed',error=repr(error),updated_at=now());write_json(worker/'STATUS.json',status)
        raise
    finally:
        if hook is not None:hook.remove()
        if model is not None:del model


if __name__=='__main__':main()
