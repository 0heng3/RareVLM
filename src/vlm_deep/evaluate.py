"""Resumable, label-free generation and task-specific full-benchmark scoring."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path
from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT
from .common import collate,consensus,content_sha,encode,read_manifest,sha_file,to_device,write_json
from .model import load


def aggregate(rows,task):
    if task=='mmstar':
        groups=defaultdict(list)
        for r in rows:groups[r['subcategory']].append(r['score'])
        by_group={k:100*statistics.mean(v) for k,v in groups.items()}
        return 100*statistics.mean(v/100 for v in by_group.values()),by_group
    return 100*statistics.mean(r['score'] for r in rows),{}


def score_prediction(prediction,row,task):
    if task=='textvqa':return consensus(prediction,row['answers'])
    if task=='docvqa':
        from .scoring import anls
        return anls(prediction,row['answers'])
    if task=='mmstar':
        from .third_party.mcq_extract import extract_mcq_answer
        return float(extract_mcq_answer(prediction,choices=['A','B','C','D'])==row['target'].strip().upper())
    raise ValueError('unknown task')


def main():
    import torch
    import torch.distributed as dist
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--asset-root',type=Path,required=True);p.add_argument('--model-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--adapter-dir',type=Path)
    p.add_argument('--quantization',choices=['original','nf4_language_only'],default='nf4_language_only')
    p.add_argument('--task',choices=['textvqa','docvqa','mmstar'],default='textvqa')
    p.add_argument('--split',choices=['dev','final'],default='final')
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--config',type=Path,default=PROJECT_ROOT/'configs/vlm_deep_v2.json')
    args=p.parse_args();c=json.loads(args.config.read_text())
    rank=int(os.environ.get('RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'));local=int(os.environ.get('LOCAL_RANK','0'))
    torch.cuda.set_device(local);device=torch.device('cuda',local);torch.set_num_threads(2)
    if world>1:dist.init_process_group('nccl')
    if args.task=='textvqa':
        manifest=args.asset_root/'prepared/manifest.json';m=read_manifest(manifest)
        rows=m['splits'][args.split];images=m['images'];image_root=manifest.parent
        data_sha=m['manifest_sha256']
    else:
        manifest=args.asset_root/'transfer'/args.task/'manifest.json';m=json.loads(manifest.read_text())
        if content_sha({k:v for k,v in m.items() if k!='manifest_sha256'})!=m['manifest_sha256']:raise ValueError('transfer manifest changed')
        rows=m['records'];images=m['images'];image_root=manifest.parent;data_sha=m['manifest_sha256']
    assigned=rows[rank::world]
    adapter_files={f.name:sha_file(f) for f in args.adapter_dir.iterdir() if f.is_file()} if args.adapter_dir else None
    lock={'schema_version':'rarevlm.vlm_eval.v2','task':args.task,'split':args.split,
        'manifest_sha256':data_sha,'config_sha256':content_sha(c),
        'model_provenance_sha256':sha_file(args.model_root.parent/'model_provenance.json'),
        'quantization':args.quantization,'precision':'fp32','adapter_files_sha256':adapter_files,
        'world_size':world,'batch_size':args.batch_size,'n':len(rows),
        'generation':{'max_new_tokens':c['max_new_tokens'],'do_sample':False,'min_pixels':c['min_pixels'],'max_pixels':c['max_pixels'],
            'head_last_position_only':True},
        'input':'image/question only; references scored after generation',
        'source_sha256':{**{str(f.relative_to(Path(__file__).parent)):sha_file(f) for f in Path(__file__).parent.rglob('*.py')},
            'rarevlm_paths.py':sha_file(SOURCE_ROOT/'rarevlm_paths.py')}}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if rank==0:
        path=args.output_dir/'EVAL_INPUT_LOCK.json'
        if path.exists() and json.loads(path.read_text())!=lock:raise ValueError('evaluation input/code lock changed')
        write_json(path,lock)
    if world>1:dist.barrier(device_ids=[local])
    partial=args.output_dir/f'predictions_rank{rank}.jsonl'
    predictions=[]
    if partial.exists():
        text=partial.read_text();lines=text.splitlines(keepends=True)
        # A killed writer may leave only its final line incomplete.
        if lines and not lines[-1].endswith('\n'):
            partial.write_text(''.join(lines[:-1]));lines=lines[:-1]
        predictions=[json.loads(x) for x in lines]
        expected=[r['question_id'] for r in assigned[:len(predictions)]]
        if [r['question_id'] for r in predictions]!=expected:raise ValueError('partial prediction order changed')
    for active_rank in range(world):
        if rank==active_rank:
            model,processor=load(args.model_root,c,device,adapter_dir=args.adapter_dir,quantization=args.quantization)
            model.config.use_cache=True
            model.generation_config.temperature=1.0;model.generation_config.top_p=1.0;model.generation_config.top_k=50
            print('model loaded',args.task,args.quantization,'rank',rank,flush=True)
        if world>1:dist.barrier(device_ids=[local])
    head_hook=model.get_output_embeddings().register_forward_pre_hook(lambda module,items:(items[0][:,-1:,:],))
    torch.cuda.reset_peak_memory_stats();begun=time.monotonic();start=len(predictions)
    with partial.open('a') as stream,torch.inference_mode():
        for begin in range(start,len(assigned),args.batch_size):
            chunk=assigned[begin:begin+args.batch_size]
            samples=[encode(processor,r,image_root/images[r['image_id']]['path'],False) for r in chunk]
            batch=to_device(collate(samples,processor.tokenizer.pad_token_id),device)
            torch.cuda.synchronize();t=time.monotonic()
            generated=model.generate(**batch,max_new_tokens=c['max_new_tokens'],do_sample=False,use_cache=True)
            torch.cuda.synchronize();elapsed=time.monotonic()-t
            for i,row in enumerate(chunk):
                ids=generated[i,batch['input_ids'].shape[1]:]
                answer=processor.tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False).strip()
                count=int((ids!=processor.tokenizer.pad_token_id).sum())
                item={'question_id':row['question_id'],'image_id':row['image_id'],'prediction':answer,
                    'score':score_prediction(answer,row,args.task),
                    'question_sha256':content_sha(row['question']),'image_sha256':images[row['image_id']]['sha256'],
                    'batch_generation_seconds':elapsed,'batch_size':len(chunk),'generated_nonpad_tokens':count,
                    'timing_warmup':begin==0,'subcategory':row.get('subcategory','')}
                predictions.append(item);stream.write(json.dumps(item,ensure_ascii=False)+'\n')
            stream.flush()
            if rank==0 and (begin==start or (begin+len(chunk))%100==0):
                print('generated',args.task,args.split,len(predictions),'/',len(assigned),'rank-local questions',flush=True)
    head_hook.remove()
    local_stats={'rank':rank,'new_generation_seconds':time.monotonic()-begun,'resumed_questions':start,
        'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30}
    write_json(args.output_dir/f'RANK{rank}_RESOURCE.json',local_stats)
    if world>1:dist.barrier(device_ids=[local])
    if rank==0:
        all_rows=[]
        for r in range(world):
            all_rows+=[json.loads(x) for x in (args.output_dir/f'predictions_rank{r}.jsonl').read_text().splitlines()]
        if len(all_rows)!=len(rows) or len({r['question_id'] for r in all_rows})!=len(rows):raise ValueError('incomplete/duplicate benchmark predictions')
        lookup={r['question_id']:r for r in all_rows};ordered=[lookup[r['question_id']] for r in rows]
        value,group_scores=aggregate(ordered,args.task)
        write_json(args.output_dir/'predictions.json',ordered)
        result={**lock,'score_percent':value,'question_mean_score_percent':100*statistics.mean(r['score'] for r in ordered),
            'category_scores_percent':group_scores,'unique_images':len({r['image_id'] for r in rows}),
            'predictions_sha256':sha_file(args.output_dir/'predictions.json'),
            'resources':[json.loads((args.output_dir/f'RANK{r}_RESOURCE.json').read_text()) for r in range(world)],
            'scope':'public validation benchmark; pretraining overlap unknown; no final-based checkpoint selection'}
        write_json(args.output_dir/'EVALUATION.json',result)
        print('EVALUATION COMPLETE',json.dumps({k:result[k] for k in ['task','split','n','unique_images','score_percent']}),flush=True)
    if world>1:dist.barrier(device_ids=[local]);dist.destroy_process_group()


if __name__=='__main__':main()
