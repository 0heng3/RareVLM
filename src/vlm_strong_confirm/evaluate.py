"""Fresh dense4 versus segmented16, paired on the same GPU per formal task."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
import torch
from rarevlm_paths import PROJECT_ROOT,SOURCE_ROOT
from vlm_probe.evaluate import now,write_json,sha_file,content_sha,lock_file,processor_for,inference_row,package_versions
from vlm_deep.common import encode,collate,to_device
from vlm_deep.evaluate import score_prediction
from vlm_formal.evaluate import task_data,METRICS,COUNTS
from vlm_joint.evaluate import visible_gpu_binding,model_metadata_binding
from vlm_ocr.evaluate import adapter_binding
from vlm_precision.evaluate import load_path
from vlm_segmented.attention import set_segmented

DEFAULT_CONFIG=PROJECT_ROOT/'configs/vlm_strong_confirm_v1.json'
CONDITIONS=('dense4','segmented16')
DETAIL_FIELDS=('question_id','image_id','image_grid_thw','input_tokens','visual_tokens',
    'counted_image_tokens','input_ids_sha256','pixel_values_sha256','pixel_values_shape','image_sha256','question_sha256')


def source_hashes():
    names=['vlm_strong_confirm/evaluate.py','vlm_strong_confirm/analysis.py','vlm_strong_confirm/pipeline.py',
        'vlm_strong_confirm/__init__.py','vlm_segmented/attention.py','vlm_precision/evaluate.py',
        'vlm_formal/evaluate.py','vlm_deep/common.py','vlm_deep/evaluate.py','vlm_sft/common.py',
        'vlm_deep/third_party/vqa_eval_metric.py','rarevlm_paths.py']
    return {n:sha_file(SOURCE_ROOT/n) for n in names}


def build(processor,rows,images,image_root):
    samples=[];records=[];image_token=processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
    for row in rows:
        item=images[row['image_id']];path=image_root/item['path']
        if sha_file(path)!=item['sha256']:raise ValueError('official image changed')
        sample=encode(processor,inference_row(row),path,False)
        if 'labels' in sample:raise ValueError('references entered generation')
        ids=sample['input_ids'][0].tolist();grid=sample['image_grid_thw'].tolist();pixels=sample['pixel_values'].cpu().contiguous()
        visual=math.prod(grid[0])//4;counted=int((sample['input_ids']==image_token).sum())
        if len(grid)!=1 or grid[0][0]!=1 or not 64<=visual<=1024 or visual!=counted:raise ValueError('native visual contract differs')
        record={'question_id':row['question_id'],'image_id':row['image_id'],'image_grid_thw':grid,
            'input_tokens':len(ids),'visual_tokens':visual,'counted_image_tokens':counted,
            'input_ids_sha256':content_sha(ids),'pixel_values_sha256':hashlib.sha256(pixels.numpy().tobytes()).hexdigest(),
            'pixel_values_shape':list(pixels.shape),'image_sha256':item['sha256'],'question_sha256':content_sha(row['question']),
            'input_truncated':False,'references_used_for_generation':False}
        records.append(record);samples.append(sample)
    batch=collate(samples,processor.tokenizer.pad_token_id)
    for r in records:r['padded_input_tokens']=int(batch['input_ids'].shape[1])
    return batch,records


def compare_input(record,other):
    for key in DETAIL_FIELDS:
        if record[key]!=other[key]:raise ValueError(f'per-QA dense/segmented inputs differ:{key}')
    if record['input_truncated'] or record['references_used_for_generation']:raise ValueError('generation integrity failed')


def generate(model,processor,rows,images,image_root,device,expected=None,finite=False):
    start=time.monotonic();cpu,records=build(processor,rows,images,image_root);preprocessing=time.monotonic()-start
    if expected is not None:
        for r in records:
            if r['question_id'] in expected:compare_input(r,expected[r['question_id']])
    torch.cuda.synchronize();start=time.monotonic();batch=to_device(cpu,device);torch.cuda.synchronize();h2d=time.monotonic()-start
    torch.cuda.reset_peak_memory_stats()
    if finite:
        output=model(**batch,use_cache=False)
        if not bool(torch.isfinite(output.logits).all()):raise ValueError('nonfinite logits')
        del output
    torch.cuda.synchronize();start=time.monotonic();out=model.generate(**batch,max_new_tokens=32,do_sample=False,use_cache=True)
    torch.cuda.synchronize();elapsed=time.monotonic()-start;width=int(batch['input_ids'].shape[1])
    for i,r in enumerate(records):
        tail=out[i,width:];r.update(prediction=processor.tokenizer.decode(tail,skip_special_tokens=True,clean_up_tokenization_spaces=False).strip(),
            generated_nonpad_tokens=int((tail!=processor.tokenizer.pad_token_id).sum()))
    resource={'batch_size':len(rows),'generation_seconds':elapsed,'preprocessing_seconds':preprocessing,'h2d_seconds':h2d,
        'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30}
    if resource['peak_allocated_gib']>65:raise ValueError('65GiB resource cap exceeded,no automatic fallback')
    return records,resource


def metric_percent(records,rows,task):
    if task!='mmstar':return 100*statistics.mean(r['score'] for r in records)
    by_id={r['question_id']:r for r in rows};groups={}
    for r in records:groups.setdefault(by_id[r['question_id']]['subcategory'],[]).append(r['score'])
    if len(groups)!=18:raise ValueError('complete18-subcategory macro required')
    return 100*statistics.mean(statistics.mean(v) for v in groups.values())


def run_cell(directory,rows,condition,task,common,model,processor,images,image_root,device,scored,expected,update):
    size=4 if condition=='dense4' else 16
    directory.mkdir(parents=True,exist_ok=True);path=directory/'batches.jsonl'
    lock={**common,'condition':condition,'batch_size':size,'scored':scored,'question_order_sha256':content_sha([r['question_id'] for r in rows]),
          'cell_path':str(directory),'n':len(rows)}
    lock_file(directory/'INPUT_LOCK.json',lock);batches=[]
    if path.exists():
        raw=path.read_bytes()
        if raw and not raw.endswith(b'\n'):path.write_bytes(raw[:raw.rfind(b'\n')+1])
        batches=[json.loads(line) for line in path.read_text().splitlines()]
        for i,b in enumerate(batches):
            if b['input_lock_sha256']!=content_sha(lock) or [r['question_id'] for r in b['records']]!=[r['question_id'] for r in rows[i*size:(i+1)*size]]:raise ValueError('cached complete batch identity differs')
            for r in b['records']:
                if expected and r['question_id'] in expected:compare_input(r,expected[r['question_id']])
                if scored:
                    row=next(row for row in rows[i*size:(i+1)*size] if row['question_id']==r['question_id'])
                    if not math.isclose(r['score'],score_prediction(r['prediction'],row,task),abs_tol=1e-12):raise ValueError('cached score differs')
    result_path=directory/'EVALUATION.json'
    if result_path.exists():
        result=json.loads(result_path.read_text())
        if result['batches_sha256']!=sha_file(path) or sum(len(b['records']) for b in batches)!=len(rows):raise ValueError('completed cache differs')
        return result,{r['question_id']:r for b in batches for r in b['records']}
    start=len(batches)*size;started=time.monotonic();torch.cuda.empty_cache()
    set_segmented(model,condition=='segmented16')
    if start<len(rows):
        with torch.inference_mode():generate(model,processor,rows[start:start+size],images,image_root,device,expected)
    with path.open('a') as stream,torch.inference_mode():
        for begin in range(start,len(rows),size):
            chunk=rows[begin:begin+size];records,resource=generate(model,processor,chunk,images,image_root,device,expected)
            if scored:
                for r,row in zip(records,chunk):r['score']=score_prediction(r['prediction'],row,task)
            b={'batch_start':begin,'records':records,'resource':resource,'input_lock_sha256':content_sha(lock)}
            stream.write(json.dumps(b,ensure_ascii=False,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno());batches.append(b)
            done=min(begin+size,len(rows));elapsed=time.monotonic()-started
            write_json(directory/'PROGRESS.json',{'status':'running','updated_at':now(),'completed_questions':done,'total_questions':len(rows),
                'eta_seconds':(len(rows)-done)*elapsed/max(done-start,1),'last_batch':resource})
            update(done,'running')
            if begin==start or len(batches)%25==0:print('GENERATED',task,condition,done,'/',len(rows),flush=True)
    records=[r for b in batches for r in b['records']];times={k:sum(b['resource'][k] for b in batches) for k in ['generation_seconds','preprocessing_seconds','h2d_seconds']}
    result={'status':'completed','task':task,'condition':condition,'n':len(rows),'completed_at':now(),
        'batches_sha256':sha_file(path),'input_lock_sha256':content_sha(lock),
        'phase_qa_per_second':len(rows)/sum(times.values()),'generation_qa_per_second':len(rows)/times['generation_seconds'],
        'peak_allocated_gib':max(b['resource']['peak_allocated_gib'] for b in batches),
        'peak_reserved_gib':max(b['resource']['peak_reserved_gib'] for b in batches),
        'session_wall_seconds':time.monotonic()-started,'seconds':times,
        'scope':'offline fixed public benchmark;extra warmup excluded,allplannedQA included;not serviceSLA'}
    if scored:result['score_percent']=metric_percent(records,rows,task)
    (directory/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    result['predictions_sha256']=sha_file(directory/'predictions.jsonl');write_json(result_path,result)
    write_json(directory/'PROGRESS.json',{'status':'completed','completed_questions':len(rows),'total_questions':len(rows)})
    update(len(rows),'completed');print('COMPLETE',task,condition,result,flush=True)
    return result,{r['question_id']:r for r in records}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--task',choices=['textvqa','docvqa','mmstar'],required=True)
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG);args=parser.parse_args();cfg=json.loads(args.config.read_text());root=Path(cfg['output_root'])
    task=args.task;worker=root/'workers'/task;worker.mkdir(parents=True,exist_ok=True)
    state={'status':'starting','task':task,'started_at':now(),'cells':{}}
    def update(key,n,status):state['cells'][key]={'completed_questions':n,'state':status};state['updated_at']=now();write_json(worker/'STATUS.json',state)
    hook=None
    try:
        rows,images,image_root,metadata=task_data(Path(cfg['asset_root']),task,cfg)
        rows=[{**r,'question_id':str(r['question_id'])} for r in rows]
        torch.cuda.set_device(0);torch.set_num_threads(2);device=torch.device('cuda',0)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
        hardware=visible_gpu_binding();state['hardware']=hardware
        reference=json.loads((PROJECT_ROOT/cfg['reference_config']).read_text());model_root=Path(cfg['model_root']);adapter=Path(cfg['fixed_adapter_dir'])
        files,training=adapter_binding(cfg,adapter,model_root)
        common={'config_sha256':sha_file(args.config),'source_hashes':source_hashes(),'hardware':hardware,
            'packages':package_versions(),'task':task,'data':metadata,'adapter_files_sha256':files,'training_binding':training,
            'model_binding':model_metadata_binding(model_root,reference),'attention_library_sha256':sha_file(Path(cfg['attention_library_path'])),
            'numerical_path':'original_bf16','visual_token_budget':1024,'max_new_tokens':32,'do_sample':False,
            'comparison':'dense4_vs_segmented16_joint_operating_configuration,notisolatedattentioneffect'}
        lock_file(worker/'INPUT_LOCK.json',common)
        processor=processor_for(model_root,cfg,1024)
        _,cpu=build(processor,rows[:32],images,image_root);expected_cpu={r['question_id']:r for r in cpu}
        write_json(worker/'CPU_PRECHECK.json',{'status':'passed','records':cpu,'scope':'first32 input-only;fullcandidateinputs matchedagainstfreshbaseline'})
        state['status']='loading_model';write_json(worker/'STATUS.json',state)
        model,dtypes=load_path(model_root,reference,device,'original_bf16',adapter);state['dtypes']=dtypes
        hook=model.get_output_embeddings().register_forward_pre_hook(lambda module,items:(items[0][:,-1:,:],))
        for condition in CONDITIONS:
            set_segmented(model,condition=='segmented16');size=4 if condition=='dense4' else 16
            with torch.inference_mode():answers,resource=generate(model,processor,rows[:size],images,image_root,device,expected_cpu,True)
            write_json(worker/'preflight'/condition/'PREFLIGHT.json',{'status':'passed','records':answers,'resource':resource,'scope':'engineering,no scores'})
        state['status']='full_formal';baseline={}
        for condition in CONDITIONS:
            result,records=run_cell(root/'runs'/task/condition,rows,condition,task,common,model,processor,images,image_root,device,True,
                expected_cpu if condition=='dense4' else baseline,lambda n,s,c=condition:update('full/'+c,n,s))
            if condition=='dense4':baseline=records
        state['status']='timing_repeats';selected=rows[:200]
        for repeat,order in enumerate(cfg['timing_orders']):
            for condition in order:
                run_cell(root/'timing'/task/f'repeat{repeat}'/condition,selected,condition,task,common,model,processor,images,image_root,device,False,
                    baseline,lambda n,s,c=condition,r=repeat:update(f'timing/{r}/{c}',n,s))
        state.update(status='completed',completed_at=now());write_json(worker/'STATUS.json',state)
    except Exception as error:
        state.update(status='failed',error=repr(error),updated_at=now());write_json(worker/'STATUS.json',state);raise
    finally:
        if hook is not None:hook.remove()


if __name__=='__main__':main()
