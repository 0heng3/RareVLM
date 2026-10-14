"""Same-batch dense/grouped visual attention on full real dev, with feature checks."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time
import sys
sys.dont_write_bytecode=True
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION','python')
from rarevlm_paths import PROJECT_ROOT
from vlm_probe.evaluate import now,write_json,sha_file,content_sha,lock_file,processor_for,package_versions
from vlm_precision.evaluate import load_path
from vlm_joint.evaluate import model_metadata_binding,visible_gpu_binding
from vlm_ocr.evaluate import adapter_binding
from vlm_weighted.evaluate import dev_data
from vlm_batching.evaluate import build_batch,generate_batch,run_cell
from vlm_deep.common import to_device
from .prepare import DEFAULT_CONFIG,read_context
from .attention import attention_modules,segmented_forward,set_segmented


def feature_check(model,processor,rows,images,image_root,expected,device):
    import torch
    set_segmented(model,False)
    modules=list(attention_modules(model).items());chosen=[modules[0],modules[7]]
    records=[];handles=[]
    def hook(name):
        def compare(module,args,kwargs,original):
            candidate=segmented_forward(module,*args,**kwargs)
            difference=candidate.float()-original.float()
            relative=float(difference.square().mean().sqrt()/original.float().square().mean().sqrt().clamp_min(1e-8))
            bounds=kwargs.get('cu_seqlens',args[1] if len(args)>1 else None)
            finite=bool(torch.isfinite(candidate).all())
            record={'module':name,'hidden_shape':list(args[0].shape),'cu_seqlens':bounds.detach().cpu().tolist(),
                'relative_rms_error':relative,'max_abs_error':float(difference.abs().max()),
                'dense_output_dtype':str(original.dtype),'finite':finite,
                'scope':'local layer outputs from exactly same original hidden states/RoPE,not whole-model equivalence'}
            records.append(record)
            if not finite or relative>.01:raise ValueError('real feature contract failed;no full study')
        return compare
    try:
        for name,module in chosen:handles.append(module.register_forward_hook(hook(name),with_kwargs=True))
        cpu,_=build_batch(processor,rows[:4],images,image_root,expected)
        with torch.inference_mode():
            result=model(**to_device(cpu,device),use_cache=False)
            if not bool(torch.isfinite(result.logits).all()):raise ValueError('feature-precheck logits nonfinite')
            del result
    finally:
        for handle in handles:handle.remove()
        set_segmented(model,False)
    if len(records)!=2:raise ValueError('expected two local-layer real-image comparisons')
    return {'status':'passed','records':records,'question_ids':[r['question_id'] for r in rows[:4]],
            'relative_rms_limit':.01,'scope':'engineering real feature check;no references or scores'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--group',choices=['batch4','batch16'],required=True)
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG);args=parser.parse_args()
    cfg,root,plan,executor=read_context(args.config)
    worker=root/'workers'/args.group;worker.mkdir(parents=True,exist_ok=True)
    state={'status':'starting','started_at':now(),'group':args.group,'cells':{}}
    def update(key,n,phase):
        state['cells'][key]={'completed_questions':n,'state':phase};state['updated_at']=now();write_json(worker/'STATUS.json',state)
    model=handle=None
    try:
        rows,images,image_root,metadata=dev_data(Path(cfg['asset_root']));by_id={r['question_id']:r for r in rows}
        expected={r['question_id']:r for r in json.loads(Path(cfg['upstream_input_contract']).read_text())['records'] if r['condition']=='baseline'}
        reference=json.loads((PROJECT_ROOT/cfg['reference_config']).read_text());model_root=Path(cfg['model_root']);adapter=Path(cfg['fixed_adapter_dir'])
        files,training=adapter_binding(cfg,adapter,model_root)
        import torch
        if not torch.cuda.is_available():raise ValueError('admitted CUDA worker required')
        torch.cuda.set_device(0);torch.set_num_threads(2);device=torch.device('cuda',0)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest');props=torch.cuda.get_device_properties(device)
        hardware={**visible_gpu_binding(),'name':props.name,'total_memory_bytes':props.total_memory,
            'compute_capability':[props.major,props.minor],'torch_cuda_version':torch.version.cuda}
        common={'schema_version':'rarevlm.segmented.worker.v1','config_sha256':sha_file(args.config),
            'plan_sha256':sha_file(root/'PLAN.json'),'source_hashes':plan['source_hashes'],
            'segmented_executor_sha256':sha_file(root/'SEGMENTED_EXECUTOR_LOCK.json'),'segmented_sources':executor['source_hashes'],
            'hardware':hardware,'packages':package_versions(),'data':metadata,'condition_definitions':cfg['policies'],
            'adapter_files_sha256':files,'training_binding':training,'model_binding':model_metadata_binding(model_root,reference),
            'group':args.group,'numerical_path':'original_bf16','visual_token_budget':1024,
            'max_new_tokens':32,'do_sample':False,'input_truncation':False,'head_last_position_only':True}
        lock_file(worker/'INPUT_LOCK.json',common);state['hardware']=hardware;state['status']='loading_model';write_json(worker/'STATUS.json',state)
        model,dtypes=load_path(model_root,reference,device,'original_bf16',adapter)
        state['actual_dtypes']=dtypes
        handle=model.get_output_embeddings().register_forward_pre_hook(lambda module,items:(items[0][:,-1:,:],))
        processor=processor_for(model_root,cfg,1024)
        feature=worker/'FEATURE_CHECK.json'
        if not feature.exists():write_json(feature,{**feature_check(model,processor,rows,images,image_root,expected,device),'input_lock_sha256':content_sha(common)})
        else:
            previous=json.loads(feature.read_text())
            if previous['status']!='passed' or previous['input_lock_sha256']!=content_sha(common):raise ValueError('feature cache binding differs')
        print('REAL FEATURE CHECK PASSED',args.group,flush=True)
        for policy in cfg['groups'][args.group]:
            set_segmented(model,cfg['policies'][policy]['attention_mode']=='grouped_segments')
            path=worker/'preflight'/policy/'PREFLIGHT.json'
            if not path.exists():
                size=cfg['policies'][policy]['batch_size'];longest=sorted(by_id,key=lambda q:(-expected[q]['input_tokens'],int(q)))[:size]
                batches=[];state['status']='preflight';update(policy,0,'preflight')
                with torch.inference_mode():
                    for ids in [plan['schedules'][policy][0],longest]:
                        answers,resource=generate_batch(model,processor,[by_id[q] for q in ids],images,image_root,expected,device,True)
                        batches.append({'records':answers,'resource':resource})
                write_json(path,{'status':'passed','batches':batches,'input_lock_sha256':content_sha(common),'scope':'input-only,not scored'})
            else:
                previous=json.loads(path.read_text())
                if previous['status']!='passed' or previous['input_lock_sha256']!=content_sha(common):raise ValueError('preflight cache differs')
            print('PREFLIGHT PASSED',args.group,policy,flush=True)
        state['status']='full_dev'
        for policy in cfg['groups'][args.group]:
            set_segmented(model,cfg['policies'][policy]['attention_mode']=='grouped_segments')
            run_cell(root/'runs'/args.group/policy/'dev',plan['schedules'][policy],policy,common,model,processor,by_id,images,image_root,expected,device,True,
                lambda n,s,p=policy:update('dev/'+p,n,s))
        state['status']='timing_repeats'
        for repeat,order in enumerate(cfg['timing_orders'][args.group]):
            for policy in order:
                set_segmented(model,cfg['policies'][policy]['attention_mode']=='grouped_segments')
                run_cell(root/'timing'/args.group/f'repeat{repeat}'/policy,plan['timing_schedules'][policy],policy,common,model,processor,by_id,images,image_root,expected,device,False,
                    lambda n,s,p=policy,r=repeat:update(f'timing/{r}/{p}',n,s))
        read_context(args.config);state.update(status='completed',completed_at=now());write_json(worker/'STATUS.json',state)
    except Exception as error:
        state.update(status='failed',error=repr(error),updated_at=now());write_json(worker/'STATUS.json',state);raise
    finally:
        if handle is not None:handle.remove()


if __name__=='__main__':main()
