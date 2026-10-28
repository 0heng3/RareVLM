"""Teacher-forced preference diagnostic; no claims of generated accuracy."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from vlm_alignment_signal.loss import sequence_logps
from vlm_alignment_signal.common import write_json
from vlm_deep.common import collate,encode,to_device
from vlm_joint.evaluate import visible_gpu_binding
from vlm_precision.evaluate import load_path
from vlm_probe.evaluate import lock_file,sha_file,content_sha,package_versions,processor_for,now
from vlm_posttrain_eval.evaluate import adapter_binding,config_path

def run(args):
    cfg=json.loads(args.config.read_text());root=args.output_root
    manifest=Path(cfg['heldout_manifest']);doc=json.loads(manifest.read_text());rows=doc['rows']
    if len(rows)!=439 or len({r['pixel_sha256'] for r in rows})!=439:raise ValueError('Heldout population changed')
    train=json.loads(Path(cfg['train_manifest']).read_text())['rows']
    if {r['pixel_sha256'] for r in rows}&{r['pixel_sha256'] for r in train}:raise ValueError('Train/heldout image overlap')
    reference=json.loads(config_path(cfg,'reference_config').read_text());model_root=Path(cfg['model_root'])
    files,binding=adapter_binding(cfg,args.condition,args.seed,args.adapter_dir,model_root)
    directory=root/'runs'/args.condition/f'seed{args.seed}'/'heldout';directory.mkdir(parents=True,exist_ok=True)
    lock={'config_sha256':sha_file(args.config),'heldout_sha256':sha_file(manifest),
          'train_sha256':sha_file(Path(cfg['train_manifest'])),'adapter_files_sha256':files,'adapter_binding':binding,
          'hardware':visible_gpu_binding(),'packages':package_versions(),'source_sha256':sha_file(Path(__file__)),
          'loss_source_sha256':sha_file(Path(sequence_logps.__code__.co_filename)),
          'question_ids':[r['question_id'] for r in rows],
          'scope':'assistant sum logp teacher forcing;not generated answers or independent human reannotation',
          'prompt_suffix':'','pair_batch_size':2,'numerical_path':'original_bf16','visual_token_budget':1024}
    lock_file(directory/'INPUT_LOCK.json',lock);identity=content_sha(lock);cache=directory/'predictions.jsonl';items=[]
    if cache.exists():
        lines=cache.read_bytes().splitlines(keepends=True)
        if lines and not lines[-1].endswith(b'\n'):lines.pop();cache.write_bytes(b''.join(lines))
        items=[json.loads(x) for x in lines]
        if len(items)>len(rows):raise ValueError('Extra heldout cache rows')
        for item,row in zip(items,rows):
            if item['question_id']!=row['question_id'] or item['input_lock_sha256']!=identity:raise ValueError('Heldout resume mismatch')
    if (directory/'EVALUATION.json').exists():
        saved=json.loads((directory/'EVALUATION.json').read_text())
        if len(items)!=439 or saved['predictions_sha256']!=sha_file(cache):raise ValueError('Heldout completion changed')
        return
    torch.cuda.set_device(0);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda',0);processor=processor_for(model_root,reference,1024)
    model,dtypes=load_path(model_root,reference,device,'original_bf16',args.adapter_dir)
    torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    with cache.open('a') as stream,torch.inference_mode():
        for row in rows[len(items):]:
            samples=[encode(processor,{'question':row['question'],'target':row[key],'prompt_suffix':''},row['image_path'],True) for key in ['chosen','rejected']]
            if any(x['input_ids'].shape[1]>2048 for x in samples):raise ValueError('Heldout truncation prohibited')
            rgb_sha=row['pixel_sha256'];input_sha=content_sha([x['input_ids'].tolist() for x in samples])
            batch=to_device(collate(samples,processor.tokenizer.pad_token_id),device)
            logps,counts=sequence_logps(model,batch)
            if not bool(torch.isfinite(logps).all()):raise ValueError('Nonfinite heldout logp')
            lp=logps.tolist();n=counts.tolist();margin=lp[0]-lp[1]
            item={'question_id':row['question_id'],'image_id':row['image_id'],'pixel_sha256':rgb_sha,
                  'input_ids_sha256':input_sha,'input_lock_sha256':identity,
                  'chosen_logp':lp[0],'rejected_logp':lp[1],'chosen_tokens':n[0],'rejected_tokens':n[1],
                  'raw_sequence_margin':margin,'token_mean_margin':lp[0]/n[0]-lp[1]/n[1],
                  'raw_pair_hit':1. if margin>0 else .5 if margin==0 else 0.,
                  'input_tokens':int(batch['attention_mask'].sum())}
            stream.write(json.dumps(item,ensure_ascii=False,allow_nan=False)+'\n');stream.flush();items.append(item)
            if len(items)%25==0 or len(items)==439:
                write_json(directory/'PROGRESS.json',{'status':'heldout_logps','completed':len(items),'planned':439,'updated_at':now()})
                print('HELDOUT_LOGPS',args.condition,len(items),'/439',flush=True)
    result={'status':'completed','completed_at':now(),'condition':args.condition,'seed':args.seed,'n':439,
            'predictions_sha256':sha_file(cache),'input_lock_sha256':identity,'raw_pair_accuracy_percent':100*sum(x['raw_pair_hit'] for x in items)/439,
            'mean_raw_sequence_margin':sum(x['raw_sequence_margin'] for x in items)/439,
            'mean_token_mean_margin':sum(x['token_mean_margin'] for x in items)/439,
            'elapsed_seconds':time.monotonic()-started,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
            'dtypes':dtypes,'scope':lock['scope'],'length_bias':'summed logp depends on completion length;token-mean shown separately;neither is generated accuracy'}
    write_json(directory/'EVALUATION.json',result);write_json(directory/'PROGRESS.json',{'status':'completed','completed':439,'updated_at':now()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--condition',choices=['initial','sft','dpo'],required=True)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--adapter-dir',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);run(p.parse_args())
