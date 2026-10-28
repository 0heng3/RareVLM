"""Fresh full/selected BLIP forwards, same-role timing and resumable quality."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from clip_retrieval_rerank.common import setup_cuda,load_matcher,cached_logits,write_json
from vlm_joint.evaluate import visible_gpu_binding
from vlm_probe.evaluate import lock_file,sha_file,now,package_versions


def run(config_path,split,role):
    worker_start=time.monotonic();cfg=json.loads(config_path.read_text());root=Path(cfg['output_root'])
    if role not in range(cfg['pair_scoring_roles']):raise ValueError('Invalid GPU role')
    if split=='test' and not json.loads((root/'analysis/dev/ANALYSIS.json').read_text())['gate']['passed']:raise ValueError('No test before joint dev gate')
    device=setup_cuda();d=root/'matcher'/split/f'role{role}';d.mkdir(parents=True,exist_ok=True)
    plan=json.loads((root/'routes'/split/'PLAN.json').read_text());inputs=root/'prepared'/split
    meta=json.loads((inputs/'INPUTS.json').read_text())
    for filename,key in [('pixels.pt','pixels_sha256'),('tokens.pt','tokens_sha256')]:
        if sha_file(inputs/filename)!=meta[key]:raise ValueError('Fixed BLIP input changed')
    if sha_file(Path(cfg['model_root'])/'pytorch_model.bin')!=cfg['model_weight_sha256']:raise ValueError('Pinned weights changed')
    identity={'config_sha256':sha_file(config_path),'policy_sha256':sha_file(root/'POLICY.json'),
        'route_plan_sha256':sha_file(root/'routes'/split/'PLAN.json'),'pixels_sha256':meta['pixels_sha256'],
        'tokens_sha256':meta['tokens_sha256'],'model_weight_sha256':cfg['model_weight_sha256'],
        'hardware':visible_gpu_binding(),'source_sha256':sha_file(Path(__file__)),'packages':package_versions(),'role':role,'dtype':'bfloat16','tf32':False}
    lock_file(d/'INPUT_LOCK.json',identity)
    pixels=torch.load(inputs/'pixels.pt',map_location='cpu',weights_only=True)
    tokens=torch.load(inputs/'tokens.pt',map_location='cpu',weights_only=True);tokens={k:v.to(device) for k,v in tokens.items()}
    all_keys={variant:np.load(root/'routes'/split/(variant+'_keys.npy')) for variant in ['full','adaptive']}
    for variant,keys in all_keys.items():
        if sha_file(root/'routes'/split/(variant+'_keys.npy'))!=plan['files'][variant+'_keys.npy']:raise ValueError('Pair plan changed')
        if len(keys) and (not np.all(keys[1:]>keys[:-1]) or keys.min()<0 or keys.max()>=5000000):raise ValueError('Invalid/duplicated planned pair')
    selected={};bounds={}
    for variant,keys in all_keys.items():
        cuts=np.linspace(0,len(keys),cfg['pair_scoring_roles']+1,dtype=np.int64)
        a,b=int(cuts[role]),int(cuts[role+1]);selected[variant]=keys[a:b];bounds[variant]=(a,b)
    model=load_matcher(cfg,device);torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        cache=d/'vision.pt'
        if cache.exists():
            record=json.loads((d/'VISION.json').read_text())
            if sha_file(cache)!=record['sha256']:raise ValueError('Own vision cache changed')
            vision=torch.load(cache,map_location=device,weights_only=True);encode_seconds=record['encode_seconds'];vision_reused=True
        else:
            blocks=[];torch.cuda.synchronize();stamp=time.monotonic()
            for start in range(0,1000,cfg['vision_batch_size']):
                batch=pixels[start:start+cfg['vision_batch_size']].to(device=device,dtype=torch.bfloat16)
                blocks.append(model.vision_model(pixel_values=batch,return_dict=True).last_hidden_state)
            vision=torch.cat(blocks);torch.cuda.synchronize();encode_seconds=time.monotonic()-stamp
            if tuple(vision.shape)!=(1000,577,768):raise ValueError('Unexpected vision shape')
            tmp=cache.with_suffix('.pt.tmp');torch.save(vision.cpu(),tmp);tmp.replace(cache);vision_reused=False
            write_json(d/'VISION.json',{'status':'completed','sha256':sha_file(cache),'shape':list(vision.shape),'dtype':str(vision.dtype),
                'encode_seconds':encode_seconds,'scope':'freshly encoded this study;common full/adaptive preprocessing'})

        def score_rows(keys,native=False):
            if not len(keys):return np.empty((0,2),dtype=np.float32)
            values=[]
            for start in range(0,len(keys),cfg['pair_batch_size']):
                part=keys[start:start+cfg['pair_batch_size']]
                ii=torch.as_tensor(part//5000,device=device);tt=torch.as_tensor(part%5000,device=device)
                ids=tokens['input_ids'][tt];mask=tokens['attention_mask'][tt]
                if native:
                    batch=pixels[torch.as_tensor(part//5000)].to(device=device,dtype=torch.bfloat16)
                    with torch.device(device):logits=model(pixel_values=batch,input_ids=ids,attention_mask=mask,return_dict=True).itm_score.float()
                else:logits=cached_logits(model,vision[ii],ids,mask)
                values.append(logits.cpu().numpy())
            out=np.concatenate(values)
            if out.shape!=(len(keys),2) or not np.isfinite(out).all():raise ValueError('Invalid actual matching outputs')
            return out

        if not (d/'PREFLIGHT.json').exists():
            full=all_keys['full'];probe=full[np.linspace(0,len(full)-1,cfg['preflight_pairs'],dtype=np.int64)]
            native=score_rows(probe,True);cached=score_rows(probe)
            diff=cached-native;error=float(np.abs(diff).max());rms=float(np.linalg.norm(diff)/max(np.linalg.norm(native),1e-12))
            if error>cfg['native_cached_max_logit_abs'] or rms>cfg['native_cached_relative_rms']:raise ValueError('Native/cached preflight failed')
            write_json(d/'PREFLIGHT.json',{'status':'passed','role':role,'probe_pairs':len(probe),'probe_pair_keys':probe,
                'native_logits':native,'cached_logits':cached,'max_logit_abs_error':error,'relative_rms':rms,
                'scope':'actual64pair finite input check;not allinput bitwise equivalence'})

        def quality_pass(variant):
            target=d/variant;chunks=target/'chunks';chunks.mkdir(parents=True,exist_ok=True)
            keys=selected[variant];base,_=bounds[variant];had_existing=bool(list(chunks.glob('*.json')))
            seconds=0.;new_pairs=0
            for start in range(0,len(keys),cfg['pair_chunk_size']):
                end=min(start+cfg['pair_chunk_size'],len(keys));p=chunks/f'{base+start:07d}.npz';marker=p.with_suffix('.json')
                if p.exists() and marker.exists():
                    m=json.loads(marker.read_text())
                    if sha_file(p)!=m['sha256'] or m['end']!=base+end:raise ValueError('Resumed quality chunk changed')
                else:
                    torch.cuda.synchronize();stamp=time.monotonic();logits=score_rows(keys[start:end]);torch.cuda.synchronize()
                    elapsed=time.monotonic()-stamp;new_pairs+=end-start
                    margin=logits[:,cfg['positive_itm_class']]-logits[:,1-cfg['positive_itm_class']]
                    tmp=p.with_suffix('.npz.tmp')
                    with tmp.open('wb') as stream:np.savez(stream,keys=keys[start:end],logits=logits,margin=margin)
                    tmp.replace(p)
                    m={'status':'completed','begin':base+start,'end':base+end,'pairs':end-start,
                       'matching_phase_seconds':elapsed,'sha256':sha_file(p)};write_json(marker,m)
                seconds+=m['matching_phase_seconds']
                write_json(d/'PROGRESS.json',{'status':'quality_'+variant,'updated_at':now(),'completed_pairs':end,
                    'planned_pairs':len(keys),'new_session_forward_pairs':new_pairs,'role':role})
                if start//cfg['pair_chunk_size']%12==0:print('MATCH',split,role,variant,end,'/',len(keys),flush=True)
            record={'status':'completed','condition':variant,'begin':bounds[variant][0],'end':bounds[variant][1],
                'pairs':len(keys),'fresh_complete_timing_valid':not had_existing and new_pairs==len(keys),
                'matching_phase_seconds':seconds,'new_session_forward_pairs':new_pairs,
                'chunk_files':{p.name:sha_file(p) for p in sorted(chunks.glob('*.npz'))}}
            write_json(target/'QUALITY_COMPLETE.json',record);return record

        quality={variant:quality_pass(variant) for variant in ['full','adaptive']}
        if (d/'TIMING.json').exists():
            timing=json.loads((d/'TIMING.json').read_text())
        else:
            samples=[];extra_forwards={'full':0,'adaptive':0}
            if all(v['fresh_complete_timing_valid'] for v in quality.values()):
                samples.append({'repeat':0,'order':['full','adaptive'],
                    'full_seconds':quality['full']['matching_phase_seconds'],'adaptive_seconds':quality['adaptive']['matching_phase_seconds'],
                    'scope':'fresh complete quality forward;sum synchronized chunk phases,writeIO excluded'})
            while len(samples)<cfg['timing_repeats']:
                repeat=len(samples);order=['full','adaptive'] if repeat%2==0 else ['adaptive','full'];times={}
                for variant in order:
                    total=0.
                    for start in range(0,len(selected[variant]),cfg['pair_chunk_size']):
                        keys=selected[variant][start:start+cfg['pair_chunk_size']]
                        torch.cuda.synchronize();stamp=time.monotonic();score_rows(keys);torch.cuda.synchronize()
                        total+=time.monotonic()-stamp;extra_forwards[variant]+=len(keys)
                    times[variant]=total
                    write_json(d/'PROGRESS.json',{'status':'timing','updated_at':now(),'repeat':repeat,'last_condition':variant,
                        'full_pairs':len(selected['full']),'adaptive_pairs':len(selected['adaptive'])})
                    print('TIMING',split,role,repeat,variant,total,flush=True)
                samples.append({'repeat':repeat,'order':order,'full_seconds':times['full'],'adaptive_seconds':times['adaptive'],
                    'scope':'fresh complete matching forward;no quality chunk/old score reuse'})
            timing={'status':'completed','samples':samples,'extra_timing_forward_pairs':extra_forwards,'pair_batch_size':cfg['pair_batch_size'],
                'scope':'same GPU/same batch/common vision,matcher gather+GPU+CPU output;not wholepipeline or online latency'}
            write_json(d/'TIMING.json',timing)
        peak=torch.cuda.max_memory_allocated()/2**30
        if peak>cfg['peak_allocated_limit_gib']:raise ValueError('Resource cap exceeded')
        result={'status':'completed','completed_at':now(),'role':role,'conditions':quality,
            'timing_sha256':sha_file(d/'TIMING.json'),'vision_encode_seconds':encode_seconds,'vision_cache_reused_on_resume':vision_reused,
            'peak_allocated_gib':peak,'worker_wall_seconds':time.monotonic()-worker_start,
            'hardware':visible_gpu_binding(),'native_preflight_sha256':sha_file(d/'PREFLIGHT.json'),
            'fresh_quality_forward_pairs':{variant:q['new_session_forward_pairs'] for variant,q in quality.items()},
            'extra_timing_forward_pairs':timing['extra_timing_forward_pairs'],
            'scope':'selected keys were physically forwarded independently of full scores;3timing repeats not new QA samples'}
        write_json(d/'MATCH_COMPLETE.json',result);write_json(d/'PROGRESS.json',{'status':'completed','role':role})
        return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--split',choices=['dev','test'],required=True);p.add_argument('--role',type=int,required=True)
    a=p.parse_args();run(a.config,a.split,a.role)
