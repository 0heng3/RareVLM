"""Fixed-checkpoint, four-condition multi-image internal-dev evaluation.

Only the predeclared view allocation changes. References are scored after
generation; neither the view builder nor encoder receives reference answers.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gc
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

sys.dont_write_bytecode = True
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
from vlm_deep.evaluate import score_prediction
from vlm_formal.evaluate import question_prompt_sha
from vlm_joint.evaluate import model_metadata_binding, visible_gpu_binding
from vlm_precision.evaluate import PATH_SPECS, load_path
from vlm_probe.evaluate import (content_sha, distribution, lock_file, now,
    package_versions, private_path, read_json as strict_read_json, repo_path, sha_file, write_json)
from vlm_weighted.evaluate import dev_data

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / 'configs/vlm_region_token_v1.json'
CONDITIONS = ('whole', 'double_whole', 'text_region', 'random_matched')
METRIC = 'TextVQA leave-one-out consensus percent'
EXPECTED_TOKENS = {'whole': [1024], 'double_whole': [512,512],
    'text_region': [512,512], 'random_matched': [512,512]}


def read_json(path):
    """Optional newly-created status artifacts; required inputs are checked first."""
    return strict_read_json(path) if Path(path).is_file() else None


def source_hashes():
    # Do not couple prediction identity to controller or analysis source.
    names = ('vlm_region/evaluate.py','vlm_region/data.py','vlm_region/views.py',
        'vlm_probe/evaluate.py','vlm_precision/evaluate.py','vlm_joint/evaluate.py',
        'vlm_formal/evaluate.py','vlm_weighted/evaluate.py','vlm_deep/evaluate.py',
        'vlm_deep/common.py','vlm_deep/third_party/vqa_eval_metric.py')
    return {name:sha_file(REPO/name) for name in names}


def config_value(config, key, default=None):
    return config.get(key, config.get('evaluation', {}).get(key, default))


def validate_config(config, batch_size=4):
    if not isinstance(config,dict): raise ValueError('region scientific configuration missing')
    expected = {'numerical_path':'original_bf16','precision':'bf16','quantization':'original',
        'vision_precision':'bf16','visual_token_budget':1024,'batch_size':4,
        'min_pixels':50176,'max_pixels':802816,'max_new_tokens':32,'do_sample':False,
        'attention_implementation':'sdpa','allow_tf32':False,'input_truncation':False}
    expected.update(split='dev',expected_questions=1617,expected_images=1024,expected_predictions=6468,
        model_revision='66285546d2b821cf421d4f5eb2576359d3770cd3',bootstrap_replicates=2000,bootstrap_seed=20261012)
    for key,value in expected.items():
        if config_value(config,key,value)!=value: raise ValueError(f'region contract differs:{key}')
    values=config.get('conditions',list(CONDITIONS))
    keys=[v['key'] if isinstance(v,dict) else v for v in values]
    if keys!=list(CONDITIONS) or batch_size!=4: raise ValueError('expected frozen four conditions, batch4')
    if config.get('full_canvas_wh',[896,896])!=[896,896] or config.get('half_canvas_wh',[896,448])!=[896,448]:
        raise ValueError('exact visual-token canvases differ')
    for key,value in {'overall_min_delta_vs_random_pp':.5,'overall_min_delta_vs_double_pp':.5,'frequent_min_delta_vs_double_pp':-.5}.items():
        if config.get('dev_gate',{}).get(key)!=value: raise ValueError(f'predeclared region gate differs:{key}')
    return config


def artifact_paths(config, root):
    def get(key, fallback): return private_path(repo_path(config.get(key,str(root/fallback))))
    return {'view_lock':get('view_lock_path','VIEW_LOCK.json'),
        'cpu_precheck':get('input_precheck_path','INPUT_PRECHECK.json'),
        'input_contract':get('input_contract_path','INPUT_CONTRACT.json'),
        'detector_manifest':get('detector_manifest_path','detector/DETECTOR_READY.json')}


def prepared_binding(config, root):
    paths=artifact_paths(config,root)
    for key,path in paths.items():
        if not path.is_file(): raise ValueError(f'missing frozen {key}:{path}')
    views=read_json(paths['view_lock'])
    if len(views.get('images',{}))!=1024: raise ValueError('expected1024 frozen image view records')
    cpu=read_json(paths['cpu_precheck'])
    if cpu.get('status') not in ('passed','complete','completed') and cpu.get('passed') is not True:
        raise ValueError('CPU multi-image/token precheck not passed')
    if cpu.get('processor_checks')!=128 or cpu.get('questions')!=32 or cpu.get('view_lock_sha256')!=sha_file(paths['view_lock']):
        raise ValueError('CPU32 precheck counts/view binding differ')
    full=read_json(paths['input_contract'])
    if (full.get('status')!='passed' or full.get('processor_checks')!=6468 or full.get('questions')!=1617
        or full.get('view_lock_sha256')!=sha_file(paths['view_lock']) or len(full.get('records',[]))!=6468):
        raise ValueError('full6468 input-only processor contract not passed')
    for key,digest in views['source_hashes'].items():
        if sha_file(REPO/key)!=digest: raise ValueError('prepared view source changed')
    return {key:{'path':str(path),'sha256':sha_file(path)} for key,path in paths.items()},views


@lru_cache(maxsize=4)
def input_contract_lookup(filename,digest):
    path=private_path(filename)
    if sha_file(path)!=digest: raise ValueError('CPU encoded-input file changed')
    records=read_json(path)['records']
    mapping={(r['condition'],str(r['question_id'])):r for r in records}
    if len(mapping)!=6468: raise ValueError('CPU contract duplicate/missing QA')
    return mapping


def validate_encoded_contract(record,condition,lookup,include_padding=True):
    expected=lookup[(condition,str(record['question_id']))]
    for key in ('image_grid_thw','per_image_visual_tokens','visual_tokens','counted_image_tokens','image_slots',
        'input_tokens','padded_input_tokens','input_truncated','input_ids_sha256','text_prompt_sha256',
        'chat_template_sha256','rgb_frame_sha256','encoded_input_sha256','pixel_values_shape'):
        if key=='padded_input_tokens' and not include_padding: continue
        if record.get(key)!=expected.get(key): raise ValueError(f'actual CPU/GPU processor encoding differs:{key}')


def adapter_binding(config, adapter, model_root):
    fixed=private_path(repo_path(config['fixed_adapter_dir']))
    if adapter!=fixed: raise ValueError('expected the pre-existing uniform seed42 final adapter')
    training_path=adapter.parent/'TRAIN_RESULT.json'
    training=read_json(training_path)
    expected={'condition':'uniform','seed':42,'method':'qv_r8','split':'train_all',
        'planned_updates':2062,'completed_updates':2062,'actual_questions_seen':32985,
        'frozen_parameters_unchanged':True,'trainable_parameters':1843200}
    for key,value in expected.items():
        if training.get(key)!=value: raise ValueError(f'fixed adapter training identity differs:{key}')
    weighted_path=private_path(repo_path(config.get('weighted_config','configs/vlm_weighted_sft_v1.json')))
    if training['config_sha256']!=content_sha(read_json(weighted_path)):
        raise ValueError('adapter weighted-SFT scientific lineage differs')
    if training['model_provenance_sha256']!=sha_file(model_root.parent/'model_provenance.json'):
        raise ValueError('adapter model provenance differs')
    files={p.name:sha_file(p) for p in sorted(adapter.iterdir()) if p.is_file()}
    if training['adapter_files_sha256']!=files: raise ValueError('fixed adapter bytes differ')
    return files,{'path':str(training_path),'sha256':sha_file(training_path),**expected,
        'weighted_config_path':str(weighted_path),'weighted_config_content_sha256':training['config_sha256'],
        'hardware':training.get('hardware'),'runtime_layout':training.get('runtime_layout'),
        'training_numerical_path':'nf4_language_only_fp32','training_visual_token_budget':256,
        'checkpoint_selection':'pre-existing uniform final checkpoint; no region-result selection'}


def validate_input(record, condition):
    grids=record.get('image_grid_thw')
    if not isinstance(grids,list) or not grids or not isinstance(grids[0],list):
        raise ValueError('multi-image grid list missing')
    tokens=[]
    for grid in grids:
        if len(grid)!=3 or grid[0]!=1 or any(int(v)!=v or v<=0 for v in grid) or math.prod(grid)%4:
            raise ValueError('invalid still-image grid')
        tokens.append(math.prod(grid)//4)
    if tokens!=EXPECTED_TOKENS[condition] or record.get('visual_tokens')!=1024 or record.get('counted_image_tokens')!=1024:
        raise ValueError('actual per-slot1024 visual-token contract differs')
    if record.get('input_truncated') is not False or record.get('input_tokens',0)<1024:
        raise ValueError('truncated/invalid multi-image input')
    if 'per_image_visual_tokens' in record and record['per_image_visual_tokens']!=tokens:
        raise ValueError('per-image token annotation differs')
    if 'image_slots' in record and record['image_slots']!=len(tokens): raise ValueError('image-slot count differs')


def view_record_sha(record):
    # Timings are observations, not image/selection identity. Prepared records are
    # nevertheless file-locked, so tampering with times also invalidates the run.
    def clean(value):
        if isinstance(value,dict): return {k:clean(v) for k,v in value.items() if not k.endswith('_seconds') and k not in ('timings','cache_hit')}
        if isinstance(value,list): return [clean(v) for v in value]
        return value
    return content_sha(clean(record))


def validate_prediction(record,row,images,lock,index,total):
    condition=lock['condition']
    expected={'question_id':row['question_id'],'image_id':row['image_id'],
        'question_sha256':content_sha(row['question']),'question_prompt_sha256':question_prompt_sha(row),
        'image_sha256':images[row['image_id']]['sha256'],'condition':condition,'seed':42,
        'task':'textvqa','model_key':'uniform_seed42','input_lock_sha256':content_sha(lock),
        'source_binding_sha256':content_sha(lock['source_sha256']),'numerical_path':'original_bf16',
        'precision':'bf16','quantization':'original','visual_token_budget':1024,'metric':METRIC,
        'adapter_files_sha256':lock['adapter_files_sha256'],'model_revision':lock['model_revision'],
        'batch_start':(index//4)*4,'batch_size':min(4,total-(index//4)*4)}
    for key,value in expected.items():
        if record.get(key)!=value: raise ValueError(f'cached region binding differs:{key}')
    value=score_prediction(record['prediction'],row,'textvqa')
    if not math.isfinite(float(record['score'])) or not math.isclose(value,record['score'],abs_tol=1e-12,rel_tol=0):
        raise ValueError('cached consensus differs')
    validate_input(record,condition)
    contract=lock['prepared_binding']['input_contract']
    validate_encoded_contract(record,condition,input_contract_lookup(contract['path'],contract['sha256']))
    if record.get('view_record_sha256')!=view_record_sha(record['view_record']): raise ValueError('cached view metadata changed')
    if record['view_record'].get('view_sha256')!=input_contract_lookup(contract['path'],contract['sha256'])[(condition,str(row['question_id']))]['view_sha256']:
        raise ValueError('cached actual-view identity differs from CPU contract')
    for key in ('batch_generation_seconds','batch_preprocessing_seconds','batch_h2d_seconds',
        'batch_view_fetch_seconds','batch_peak_allocated_gib','batch_peak_reserved_gib'):
        if not math.isfinite(float(record[key])) or float(record[key])<0: raise ValueError('invalid cached resource timing')


def complete_cache(path,rows,images,lock):
    if not path.exists(): return [],None
    lines=path.read_bytes().splitlines(keepends=True)
    repairs=[]
    if lines and not lines[-1].endswith(b'\n'):
        lines.pop();repairs.append('discarded interrupted final JSONL line')
    records=[json.loads(line) for line in lines]
    if len(records)>len(rows): raise ValueError('cache exceeds frozen dev')
    if records:
        begin=((len(records)-1)//4)*4
        if len(records)-begin!=min(4,len(rows)-begin):
            records,lines=records[:begin],lines[:begin];repairs.append('discarded incomplete final batch')
    for index,(record,row) in enumerate(zip(records,rows)): validate_prediction(record,row,images,lock,index,len(rows))
    if repairs:
        with path.open('wb') as stream:
            stream.write(b''.join(lines));stream.flush();os.fsync(stream.fileno())
    return records,repairs or None


def evaluate_batch(model,processor,chunk,images,image_root,device,condition,config,views,check_logits=False):
    import torch
    from vlm_deep.common import to_device
    from .views import get_views
    from .data import build_multiimage_batch
    begun=time.monotonic()
    prepared=[get_views({'question_id':r['question_id'],'image_id':r['image_id'],'question':r['question']},
        images[r['image_id']],image_root,condition,config,view_lock=views) for r in chunk]
    view_seconds=time.monotonic()-begun
    begun=time.monotonic()
    # The builder receives label-free rows even if a future implementation changes.
    inputs=[{'question_id':r['question_id'],'image_id':r['image_id'],'question':r['question'],
        **({'prompt_suffix':r['prompt_suffix']} if 'prompt_suffix' in r else {})} for r in chunk]
    cpu_batch,details,messages=build_multiimage_batch(processor,inputs,[p[0] for p in prepared])
    preprocessing=time.monotonic()-begun
    binding=views.get('_input_contract_binding')
    if binding is None:
        contract=artifact_paths(config,private_path(config['output_root']))['input_contract']
        binding={'path':str(contract),'sha256':sha_file(contract)}
    lookup=input_contract_lookup(binding['path'],binding['sha256'])
    for detail in details:
        validate_input(detail,condition)
        validate_encoded_contract(detail,condition,lookup,include_padding=not check_logits)
    torch.cuda.synchronize(device)
    begun=time.monotonic();batch=to_device(cpu_batch,device);torch.cuda.synchronize(device)
    h2d=time.monotonic()-begun
    torch.cuda.reset_peak_memory_stats(device)
    finite=None
    if check_logits:
        output=model(**batch,use_cache=False)
        finite=bool(torch.isfinite(output.logits).all());del output
        if not finite: raise ValueError('nonfinite last-position precheck logits')
    torch.cuda.synchronize(device);begun=time.monotonic()
    generated=model.generate(**batch,max_new_tokens=32,do_sample=False,use_cache=True)
    torch.cuda.synchronize(device);generation=time.monotonic()-begun
    padded=int(batch['input_ids'].shape[1]);answers=[]
    for index in range(len(chunk)):
        ids=generated[index,padded:]
        answers.append({'prediction':processor.tokenizer.decode(ids,skip_special_tokens=True,
            clean_up_tokenization_spaces=False).strip(),'generated_nonpad_tokens':int((ids!=processor.tokenizer.pad_token_id).sum()),
            'generated_tail_slots':int(ids.numel()),**details[index],
            'view_record':prepared[index][1],'view_record_sha256':view_record_sha(prepared[index][1])})
    resource={'batch_size':len(chunk),'padded_input_tokens':padded,'batch_generation_seconds':generation,
        'batch_preprocessing_seconds':preprocessing,'batch_view_fetch_seconds':view_seconds,
        'batch_h2d_seconds':h2d,'batch_peak_allocated_gib':torch.cuda.max_memory_allocated(device)/2**30,
        'batch_peak_reserved_gib':torch.cuda.max_memory_reserved(device)/2**30,'finite_logits':finite}
    return answers,resource


def run_preflight(model,processor,args,lock,rows,images,image_root,device,views):
    selected=sorted(rows,key=lambda r:int(r['question_id']))[:32][:8]
    directory=args.output_root/'preflight'/lock['condition']
    prelock={**lock,'scope':'engineering input-only finite-logit/generation check; no scores',
        'selection':'fixed first8 of numeric-question-ID first32; no answer or prediction feedback',
        'question_ids':[r['question_id'] for r in selected]}
    lock_file(directory/'INPUT_LOCK.json',prelock)
    previous=read_json(directory/'PREFLIGHT.json')
    if previous:
        if previous.get('status')!='passed' or previous.get('input_lock_sha256')!=content_sha(prelock):
            raise ValueError('preflight lock differs')
        return previous
    import torch
    records,batches=[],[]
    with torch.inference_mode():
        for begin in range(0,len(selected),4):
            chunk=selected[begin:begin+4]
            answers,resources=evaluate_batch(model,processor,chunk,images,image_root,device,
                lock['condition'],args.configuration,views,check_logits=True)
            batches.append(resources)
            records.extend({'question_id':row['question_id'],'image_id':row['image_id'],**answer} for row,answer in zip(chunk,answers))
    result={'schema_version':'rarevlm.region_token.preflight.v1','status':'passed','completed_at':now(),
        'input_lock_sha256':content_sha(prelock),'condition':lock['condition'],'records':records,'batches':batches,
        'peak_allocated_gib':max(b['batch_peak_allocated_gib'] for b in batches),
        'scope':'32 CPU inputs/condition;8 GPU input-only generation checks/condition;no scored cache'}
    write_json(directory/'PREFLIGHT.json',result)
    print('PREFLIGHT PASSED',lock['condition'],result['peak_allocated_gib'],flush=True)
    return result


def summarize(records,lock,path,session):
    batches={}
    for row in records: batches[(row['session_id'],row['batch_start'])]=row
    steady=[r for r in batches.values() if not r['timing_warmup']]
    if not steady: steady=list(batches.values())
    n=sum(r['batch_size'] for r in steady)
    times={key:sum(r[key] for r in steady) for key in
        ('batch_generation_seconds','batch_preprocessing_seconds','batch_h2d_seconds','batch_view_fetch_seconds')}
    phase=sum(times.values())
    return {'schema_version':'rarevlm.region_token.dev_cell.v1','status':'completed','condition':lock['condition'],
        'seed':42,'n':len(records),'images':len({r['image_id'] for r in records}),
        'score_percent':100*statistics.mean(r['score'] for r in records),'metric':METRIC,
        'predictions_sha256':sha_file(path),'input_lock_sha256':content_sha(lock),'session':session,
        'resource':{'steady_questions':n,'timing_warmup_batches_excluded':sum(r['timing_warmup'] for r in batches.values()),
            'generation_questions_per_second':n/times['batch_generation_seconds'],
            'view_fetch_preprocess_h2d_generation_questions_per_second':n/phase,
            'steady_seconds':times,'visual_tokens':distribution([r['visual_tokens'] for r in records]),
            'input_tokens':distribution([r['input_tokens'] for r in records]),
            'generated_nonpad_tokens':distribution([r['generated_nonpad_tokens'] for r in records]),
            'peak_allocated_gib':max(r['batch_peak_allocated_gib'] for r in records),
            'peak_reserved_gib':max(r['batch_peak_reserved_gib'] for r in records),
            'scope':'observed cached-view VLM phases; detector/crop/letterbox preparation reported separately;not service SLA'},
        'scope':'single existing final adapter; new letterbox multi-image protocol; full internal dev only'}


def run_cell(model,processor,args,lock,rows,images,image_root,device,views,update):
    condition=lock['condition'];directory=args.output_root/'runs'/condition/'dev'
    lock_file(directory/'INPUT_LOCK.json',lock)
    path=directory/'predictions.jsonl'
    records,repaired=complete_cache(path,rows,images,lock)
    previous=read_json(directory/'EVALUATION.json')
    if previous:
        if len(records)!=len(rows) or previous.get('predictions_sha256')!=sha_file(path) or previous.get('input_lock_sha256')!=content_sha(lock):
            raise ValueError('completed region result differs')
        if not math.isclose(previous['score_percent'],100*statistics.mean(r['score'] for r in records),abs_tol=1e-10,rel_tol=0):
            raise ValueError('completed score differs')
        update(len(records),'completed');return previous
    import torch
    start=len(records);begun=time.monotonic()
    session_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    session={'session_id':session_id,'resumed_questions':start,'cache_repair':repaired,'started_at':now()}
    with path.open('a') as stream,torch.inference_mode():
        for begin in range(start,len(rows),4):
            chunk=rows[begin:begin+4]
            answers,resource=evaluate_batch(model,processor,chunk,images,image_root,device,condition,args.configuration,views)
            batch=[]
            for row,answer in zip(chunk,answers):
                record={'question_id':row['question_id'],'image_id':row['image_id'],
                    'question_sha256':content_sha(row['question']),'question_prompt_sha256':question_prompt_sha(row),
                    'image_sha256':images[row['image_id']]['sha256'],**answer,
                    'score':score_prediction(answer['prediction'],row,'textvqa'),'task':'textvqa',
                    'condition':condition,'seed':42,'model_key':'uniform_seed42','metric':METRIC,
                    'model_revision':lock['model_revision'],'adapter_files_sha256':lock['adapter_files_sha256'],
                    'source_binding_sha256':content_sha(lock['source_sha256']),
                    **{k:v for k,v in resource.items() if k!='finite_logits'},
                    'batch_start':begin,'session_id':session_id,'timing_warmup':begin==start,
                    'visual_token_budget':1024,'numerical_path':'original_bf16','precision':'bf16',
                    'quantization':'original','input_lock_sha256':content_sha(lock),
                    'reference_score_source':'frozen dev references only after label-free generation'}
                validate_prediction(record,row,images,lock,begin+len(batch),len(rows));batch.append(record)
            stream.write(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in batch))
            stream.flush();os.fsync(stream.fileno());records.extend(batch)
            elapsed=time.monotonic()-begun
            write_json(directory/'PROGRESS.json',{'status':'running','updated_at':now(),'condition':condition,
                'completed_questions':len(records),'total_questions':len(rows),'session_id':session_id,
                'session_elapsed_seconds':elapsed,'remaining_estimate_seconds':(len(rows)-len(records))*elapsed/(len(records)-start),
                'last_batch':resource})
            update(len(records),'running')
            if begin==start or ((begin-start)//4+1)%25==0: print('GENERATED',condition,len(records),'/',len(rows),flush=True)
    session.update(completed_at=now(),new_questions=len(records)-start,wall_seconds=time.monotonic()-begun)
    result=summarize(records,lock,path,session)
    write_json(directory/'EVALUATION.json',result)
    write_json(directory/'PROGRESS.json',{'status':'completed','updated_at':now(),'condition':condition,
        'completed_questions':len(records),'total_questions':len(rows)})
    update(len(records),'completed');print('CELL COMPLETE',condition,result['score_percent'],flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--conditions',nargs='+',choices=CONDITIONS,required=True)
    parser.add_argument('--asset-root',type=Path,required=True)
    parser.add_argument('--model-root',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--adapter-dir',type=Path)
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args()
    for name in ('asset_root','model_root','output_root','config'): setattr(args,name,private_path(getattr(args,name)))
    config=read_json(args.config);args.configuration=validate_config(config,args.batch_size)
    if len(set(args.conditions))!=len(args.conditions): raise ValueError('duplicated worker condition')
    args.adapter_dir=private_path(args.adapter_dir or repo_path(config['fixed_adapter_dir']))
    reference_path=private_path(repo_path(config.get('reference_config','configs/vlm_deep_v2.json')))
    reference=read_json(reference_path)
    files,training=adapter_binding(config,args.adapter_dir,args.model_root)
    rows,images,image_root,metadata=dev_data(args.asset_root)
    prepared,views=prepared_binding(config,args.output_root)
    views['_input_contract_binding']=prepared['input_contract']
    if views.get('source_manifest_sha256') not in (metadata['manifest_file_sha256'],metadata['manifest_sha256']):
        raise ValueError('prepared views source manifest differs')
    import torch
    if int(os.environ.get('WORLD_SIZE','1'))!=1 or not torch.cuda.is_available(): raise ValueError('expected one admitted CUDA worker')
    torch.cuda.set_device(0);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest');device=torch.device('cuda',0)
    props=torch.cuda.get_device_properties(device)
    if 'A100' not in props.name or props.total_memory<70*2**30: raise ValueError('requires admitted A10080GB')
    hardware={**visible_gpu_binding(),'name':props.name,'total_memory_bytes':props.total_memory,
        'compute_capability':[props.major,props.minor],'torch_cuda_version':torch.version.cuda}
    common={'schema_version':'rarevlm.region_token.dev_input.v1','model_key':'uniform_seed42','seed':42,
        'region_config_sha256':sha_file(args.config),'reference_config_sha256':sha_file(reference_path),
        **model_metadata_binding(args.model_root,reference),**metadata,'adapter_files_sha256':files,
        'training_binding':training,'prepared_binding':prepared,'source_sha256':source_hashes(),
        'packages':package_versions(),'hardware':hardware,'batch_size':4,'numerical_path':'original_bf16',
        **PATH_SPECS['original_bf16'],'visual_token_budget':1024,'actual_visual_tokens':1024,
        'max_pixels':802816,'min_pixels':50176,'max_new_tokens':32,'do_sample':False,
        'attention_implementation':'sdpa','tf32_matmul':False,'tf32_cudnn':False,
        'head_last_position_only':True,'autocast':False,'adapter_autocast_dtype':False,
        'explicit_lora_bf16_conversion':True,'input_truncation':False,
        'input':'frozen image views+question+fixed suffix;no references,target or recognized OCR text',
        'scope':'full internal-dev fixed-checkpoint four-condition exploratory view allocation;not training-seed replication'}
    worker_key='__'.join(args.conditions);status_path=args.output_root/'workers'/worker_key/'STATUS.json'
    status={'conditions':args.conditions,'hardware':hardware,'started_at':now(),'phase':'starting','cells':{}}
    def update(condition,count,state):
        status['cells'][condition]={'completed_questions':count,'total_questions':len(rows),'state':state}
        status['updated_at']=now();write_json(status_path,status)
    model,hook=None,None
    try:
        begun=time.monotonic();status['phase']='loading_model';write_json(status_path,status)
        model,dtypes=load_path(args.model_root,reference,device,'original_bf16',args.adapter_dir)
        status['model_load_seconds']=time.monotonic()-begun;common['actual_parameter_dtypes']=dtypes
        if int(model.config.image_token_id)!=151655: raise ValueError('model image token ID differs')
        hook=model.get_output_embeddings().register_forward_pre_hook(lambda module,items:(items[0][:,-1:,:],))
        from .data import processor_for
        processor=processor_for(args.model_root,config)
        for condition in args.conditions:
            lock={**common,'condition':condition,'expected_per_image_visual_tokens':EXPECTED_TOKENS[condition]}
            lock_file(args.output_root/'runs'/condition/'dev'/'INPUT_LOCK.json',lock)
            status.update(phase='preflight',current_condition=condition);update(condition,0,'preflight')
            run_preflight(model,processor,args,lock,rows,images,image_root,device,views)
            if not args.preflight_only:
                status['phase']='evaluating'
                run_cell(model,processor,args,lock,rows,images,image_root,device,views,
                    lambda count,state,c=condition:update(c,count,state))
        status.update(phase='preflight_completed' if args.preflight_only else 'completed',completed_at=now())
        write_json(status_path,status)
    except Exception as error:
        status.update(phase='failed',error_type=type(error).__name__,error=str(error),updated_at=now())
        write_json(status_path,status);raise
    finally:
        if hook is not None: hook.remove()
        del model;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__': main()
