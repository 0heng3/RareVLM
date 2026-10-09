"""Paired generated-answer evaluation; annotations enter scoring after generation."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

from .common import consensus_score,content_sha,encode_record,normalize_answer,read_manifest,sha_file,to_device,write_json
from .model import load_stack


def main():
    import torch
    import torch.distributed as dist
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--asset-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--adapter-dir',type=Path)
    p.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'configs/vlm_sft_textvqa.json')
    args=p.parse_args();config=json.loads(args.config.read_text())
    rank=int(os.environ.get('RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'));local=int(os.environ.get('LOCAL_RANK','0'))
    torch.cuda.set_device(local);device=torch.device('cuda',local);torch.set_num_threads(2)
    if world>1:dist.init_process_group('nccl')
    manifest_path=args.asset_root/'prepared/manifest.json';m=read_manifest(manifest_path)
    if args.adapter_dir:
        trained=json.loads((args.adapter_dir.parent/'TRAIN_RESULT.json').read_text())
        if trained['manifest_sha256']!=m['manifest_sha256'] or trained['config_sha256']!=content_sha(config):raise ValueError('adapter evaluated under changed data/config')
        for name,digest in trained['adapter_files_sha256'].items():
            if sha_file(args.adapter_dir/name)!=digest:raise ValueError('adapter checkpoint changed')
    if rank==0:
        write_json(args.output_dir/'EVAL_INPUT_LOCK.json',{'manifest_sha256':m['manifest_sha256'],
            'config_sha256':content_sha(config),'model_provenance_sha256':sha_file(args.asset_root/'model_provenance.json'),
            'evaluation_source_sha256':sha_file(Path(__file__)),
            'adapter_files_sha256':trained['adapter_files_sha256'] if args.adapter_dir else None,
            'selection':'fixed configuration; no final-answer parameter selection','n':len(m['splits']['final'])})
    for initialization_rank in range(world):
        if rank==initialization_rank:
            model,processor=load_stack(args.asset_root/'model',config,device,adapter_dir=args.adapter_dir)
            print('evaluation model initialized on rank',rank,flush=True)
        if world>1:dist.barrier(device_ids=[local])
    model.config.use_cache=True
    torch.cuda.reset_peak_memory_stats(device);started=time.monotonic();predictions=[]
    with torch.inference_mode():
        for i,row in enumerate(m['splits']['final'][rank::world]):
            batch=to_device(encode_record(processor,row,manifest_path.parent,False),device)
            torch.cuda.synchronize(device);begin=time.monotonic()
            generated=model.generate(**batch,max_new_tokens=config['max_new_tokens'],do_sample=config['do_sample'],use_cache=True)
            torch.cuda.synchronize(device);elapsed=time.monotonic()-begin
            ids=generated[0,batch['input_ids'].shape[1]:]
            answer=processor.tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False).strip()
            predictions.append({'image_id':row['image_id'],'question_id':row['question_id'],'prediction':answer,
                'consensus_score':consensus_score(answer,row['answers']),
                'normalized_any_answer_match':normalize_answer(answer) in {normalize_answer(a) for a in row['answers']},
                'generation_seconds':elapsed,'generated_tokens':len(ids),'timing_warmup':i==0})
            if rank==0 and (i==0 or (i+1)%16==0):print('generated',i+1,'rank-local samples',flush=True)
    local_result={'rank':rank,'duration_seconds':time.monotonic()-started,
                  'peak_gpu_allocated_gib':torch.cuda.max_memory_allocated(device)/2**30,
                  'peak_gpu_reserved_gib':torch.cuda.max_memory_reserved(device)/2**30,'predictions':predictions}
    gathered=[None]*world if rank==0 else None
    if world>1:dist.gather_object(local_result,gathered,dst=0)
    else:gathered=[local_result]
    if rank==0:
        flat=sorted([r for group in gathered for r in group['predictions']],key=lambda r:r['image_id'])
        if len(flat)!=len(m['splits']['final']) or len({r['image_id'] for r in flat})!=len(flat):raise ValueError('missing/duplicated final samples')
        timing=[r for r in flat if not r['timing_warmup']]
        durations=sorted(r['generation_seconds'] for r in timing)
        result={'manifest_sha256':m['manifest_sha256'],'config_sha256':content_sha(config),
            'model_provenance_sha256':sha_file(args.asset_root/'model_provenance.json'),
            'method':'language_lora_sft' if args.adapter_dir else 'pretrained_instruct',
            'n':len(flat),'vqa_consensus_percent':100*statistics.mean(r['consensus_score'] for r in flat),
            'normalized_any_answer_match_percent':100*statistics.mean(r['normalized_any_answer_match'] for r in flat),
            'metric_scope':config['metric'],'generation':{'do_sample':False,'max_new_tokens':config['max_new_tokens'],
                'min_pixels':config['min_pixels'],'max_pixels':config['max_pixels'],'precision':config['precision'],
                'quantization':config.get('quantization'),'use_cache':True},
            'timing':{'warmup_samples_excluded':world,'n':len(timing),'generation_p50_seconds':statistics.median(durations),
                'generation_p95_seconds':durations[int(.95*(len(durations)-1))],
                'tokens_per_generation_second':sum(r['generated_tokens'] for r in timing)/sum(durations),
                'parallel_end_to_end_images_per_second':len(flat)/max(g['duration_seconds'] for g in gathered),
                'peak_gpu_allocated_gib':max(g['peak_gpu_allocated_gib'] for g in gathered),
                'peak_gpu_reserved_gib':max(g['peak_gpu_reserved_gib'] for g in gathered),
                'world_size':world,'gpu_name':torch.cuda.get_device_name(device)},
            'trainable_adapter_files_sha256':trained['adapter_files_sha256'] if args.adapter_dir else None}
        args.output_dir.mkdir(parents=True,exist_ok=True)
        prediction_path=args.output_dir/'predictions.json';write_json(prediction_path,flat)
        result['predictions_sha256']=sha_file(prediction_path);write_json(args.output_dir/'EVALUATION.json',result)
        print('evaluation complete',json.dumps({k:result[k] for k in ('method','n','vqa_consensus_percent','normalized_any_answer_match_percent')}),flush=True)
    if world>1:dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
