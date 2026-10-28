"""Pinned BLIP: same-device visual cache, native preflight and resumable pairs."""
import argparse
import json
import os
from pathlib import Path
import time
import statistics
import numpy as np
import torch
from vlm_joint.evaluate import visible_gpu_binding
from vlm_probe.evaluate import sha_file,lock_file,now,package_versions
from .common import setup_cuda,load_matcher,cached_logits,write_json


def run(config_path,split,role,mode):
    worker_started=time.monotonic()
    cfg=json.loads(config_path.read_text());root=Path(cfg['output_root'])
    if split=='test' and not json.loads((root/'analysis/dev/ANALYSIS.json').read_text())['gate']['passed']:
        raise ValueError('No test matching before dev gate')
    device=setup_cuda();directory=root/'matcher'/split/f'role{role}';directory.mkdir(parents=True,exist_ok=True)
    ready=json.loads((root/'MODEL_READY.json').read_text())
    if sha_file(Path(cfg['model_root'])/'pytorch_model.bin')!=cfg['model_weight_sha256']:
        raise ValueError('Pinned matching weights changed')
    inp=root/'prepared'/split;meta=json.loads((inp/'INPUTS.json').read_text())
    candidate=root/'candidates'/split;record=json.loads((candidate/'CANDIDATES.json').read_text())
    identity={'config_sha256':sha_file(config_path),'model_ready_sha256':sha_file(root/'MODEL_READY.json'),
        'pixels_sha256':meta['pixels_sha256'],'tokens_sha256':meta['tokens_sha256'],
        'pair_keys_sha256':record['files']['pair_keys.npy'],'role':role,'hardware':visible_gpu_binding(),
        'source_sha256':sha_file(Path(__file__)),'packages':package_versions(),'dtype':'bfloat16','tf32':False}
    lock_file(directory/'INPUT_LOCK.json',identity)
    pixels=torch.load(inp/'pixels.pt',map_location='cpu',weights_only=True)
    token=torch.load(inp/'tokens.pt',map_location='cpu',weights_only=True)
    token={k:v.to(device) for k,v in token.items()}
    keys=np.load(candidate/'pair_keys.npy');model=load_matcher(cfg,device)
    torch.cuda.reset_peak_memory_stats();encoded_seconds=0.;cache=directory/'vision.pt'
    with torch.inference_mode():
        if cache.exists():
            saved=json.loads((directory/'VISION.json').read_text())
            if sha_file(cache)!=saved['sha256']:raise ValueError('Visual cache changed')
            vision=torch.load(cache,map_location=device,weights_only=True);encoded_seconds=saved['encode_seconds']
        else:
            blocks=[];torch.cuda.synchronize();start=time.monotonic()
            for begin in range(0,1000,cfg['vision_batch_size']):
                values=pixels[begin:begin+cfg['vision_batch_size']].to(device=device,dtype=torch.bfloat16)
                blocks.append(model.vision_model(pixel_values=values,return_dict=True).last_hidden_state)
            vision=torch.cat(blocks);torch.cuda.synchronize();encoded_seconds=time.monotonic()-start
            if tuple(vision.shape)!=(1000,577,768):raise ValueError('Unexpected vision cache shape')
            temp=cache.with_suffix('.pt.tmp');torch.save(vision.cpu(),temp);temp.replace(cache)
            write_json(directory/'VISION.json',{'status':'completed','sha256':sha_file(cache),
                'shape':list(vision.shape),'dtype':str(vision.dtype),'encode_seconds':encoded_seconds})
        def score_pair_rows(selected,full=False):
            values=[]
            for begin in range(0,len(selected),cfg['pair_batch_size']):
                part=selected[begin:begin+cfg['pair_batch_size']]
                images=torch.as_tensor(part//5000,device=device);texts=torch.as_tensor(part%5000,device=device)
                ids=token['input_ids'][texts];mask=token['attention_mask'][texts]
                if full:
                    raw=pixels[torch.as_tensor(part//5000)].to(device=device,dtype=torch.bfloat16)
                    # HF4.51.1 makes image_atts with a factory lacking explicit device.
                    # Scoped default device keeps its unmodified native GPU forward usable.
                    with torch.device(device):
                        logits=model(pixel_values=raw,input_ids=ids,attention_mask=mask,return_dict=True).itm_score.float()
                else:logits=cached_logits(model,vision[images],ids,mask)
                values.append(logits.cpu().numpy())
            return np.concatenate(values)
        if not (directory/'PREFLIGHT.json').exists():
            select=np.linspace(0,len(keys)-1,cfg['preflight_pairs'],dtype=np.int64);probe=keys[select]
            baseline=score_pair_rows(probe,True);cached=score_pair_rows(probe)
            difference=cached-baseline
            error=float(np.abs(difference).max());rms=float(np.linalg.norm(difference)/max(np.linalg.norm(baseline),1e-12))
            if error>cfg['native_cached_max_logit_abs'] or rms>cfg['native_cached_relative_rms']:
                raise ValueError(f'Cached/native preflight failed: maxabs={error},relativeRMS={rms}')
            full_times=[];cache_times=[]
            for repeat in range(cfg['preflight_repeats']):
                for full,target in [(True,full_times),(False,cache_times)]:
                    torch.cuda.synchronize();start=time.monotonic();score_pair_rows(probe,full)
                    torch.cuda.synchronize();target.append(time.monotonic()-start)
            estimate=len(keys)/cfg['pair_scoring_roles']*statistics.median(cache_times)/len(probe)
            allocated=torch.cuda.max_memory_allocated()/2**30
            if allocated>cfg['peak_allocated_limit_gib'] or estimate>cfg['max_estimated_pair_scoring_seconds_per_role']:
                raise ValueError(f'Resource preflight failed: allocated={allocated},estimatedseconds={estimate}')
            write_json(directory/'PREFLIGHT.json',{'status':'passed','created_at':now(),'role':role,
                'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES'],'hardware':visible_gpu_binding(),
                'probe_pair_keys':probe,'native_logits':baseline,'cached_logits':cached,
                'max_logit_abs_error':error,'relative_rms':rms,'probe_pairs':len(probe),
                'native_phase_seconds':full_times,'cached_phase_seconds':cache_times,
                'median_cached_pairs_per_second':len(probe)/statistics.median(cache_times),
                'estimated_matching_seconds_per_role':estimate,'vision_encode_seconds':encoded_seconds,
                'peak_allocated_gib':allocated,'scope':'64 diverse actual pairs,three timing repeats;not all-input/bitwise/online equivalence'})
            print('PREFLIGHT_PASSED',split,role,error,rms,estimate,flush=True)
        if mode=='preflight':return
        bounds=np.linspace(0,len(keys),cfg['pair_scoring_roles']+1,dtype=np.int64)
        start_index,end_index=int(bounds[role]),int(bounds[role+1])
        chunks=directory/'chunks';chunks.mkdir(exist_ok=True);started=time.monotonic();new_pairs=0
        for begin in range(start_index,end_index,cfg['pair_chunk_size']):
            end=min(begin+cfg['pair_chunk_size'],end_index);path=chunks/f'{begin:07d}.npz';meta_path=path.with_suffix('.json')
            if path.exists() and meta_path.exists():
                meta_chunk=json.loads(meta_path.read_text())
                if sha_file(path)!=meta_chunk['sha256'] or meta_chunk['end']!=end:raise ValueError('Pair checkpoint changed')
                continue
            torch.cuda.synchronize();stamp=time.monotonic();logits=score_pair_rows(keys[begin:end]);torch.cuda.synchronize()
            seconds=time.monotonic()-stamp;new_pairs+=end-begin
            margin=logits[:,cfg['positive_itm_class']]-logits[:,1-cfg['positive_itm_class']]
            if not np.isfinite(logits).all():raise ValueError('Nonfinite ITM output')
            temp=path.with_suffix('.npz.tmp')
            with temp.open('wb') as stream:np.savez(stream,keys=keys[begin:end],logits=logits,margin=margin)
            temp.replace(path)
            write_json(meta_path,{'status':'completed','begin':begin,'end':end,'sha256':sha_file(path),
                'matching_phase_seconds':seconds,'pairs':end-begin})
            elapsed=time.monotonic()-started
            write_json(directory/'PROGRESS.json',{'status':'running','updated_at':now(),'completed_pairs':end-start_index,
                'planned_pairs':end_index-start_index,'new_session_pairs':new_pairs,'elapsed_session_seconds':elapsed,
                'eta_seconds':(end_index-end)*elapsed/max(new_pairs,1)})
            if (begin-start_index)//cfg['pair_chunk_size']%8==0:print('MATCH',split,role,end-start_index,'/',end_index-start_index,flush=True)
        write_json(directory/'MATCH_COMPLETE.json',{'status':'completed','completed_at':now(),'role':role,
            'begin':start_index,'end':end_index,'pairs':end_index-start_index,
            'chunk_files':{p.name:sha_file(p) for p in sorted(chunks.glob('*.npz'))},
            'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'vision_encode_seconds':encoded_seconds,
            'worker_wall_seconds':time.monotonic()-worker_started})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--split',choices=['dev','test'],required=True);p.add_argument('--role',type=int,required=True)
    p.add_argument('--mode',choices=['preflight','score'],required=True)
    a=p.parse_args();run(a.config,a.split,a.role,a.mode)
