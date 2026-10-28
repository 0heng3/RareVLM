"""Scan only calibration labels, freeze two thresholds, then plan actual keys."""
import json
from pathlib import Path
import time
import numpy as np
from vlm_probe.evaluate import lock_file,sha_file,now
from clip_retrieval_rerank.common import pair_indices,ranking_metrics,write_json
from .route import route_mask,selected_keys


def choose_direction(margins,clip,full,qmask,cfg,direction):
    if not qmask.any():raise ValueError('Empty calibration fold')
    candidates=np.unique(np.r_[-1.,np.quantile(np.concatenate([m[qmask] for m in margins]),cfg['calibration_quantiles']),2.1])
    scan=[];chosen=None
    for threshold in candidates:
        masks=[route_mask(m,float(threshold)) for m in margins];deltas={}
        for metric in ['R@1','nDCG@10']:
            name=direction+'_'+metric
            values=[np.where(mask,f[name],c[name])[qmask]-f[name][qmask] for mask,f,c in zip(masks,full,clip)]
            deltas[metric]=100*float(np.mean(values))
        feasible=deltas['R@1']>=-cfg['calibration_max_r1_loss_pp'] and deltas['nDCG@10']>=-cfg['calibration_max_ndcg_loss_pp']
        item={'threshold':float(threshold),'cal_r1_delta_pp':deltas['R@1'],'cal_ndcg_delta_pp':deltas['nDCG@10'],
            'cal_rerank_query_fraction':float(np.mean([m[qmask].mean() for m in masks])),'feasible':bool(feasible)}
        scan.append(item)
        if chosen is None and feasible:chosen=item
    if chosen is None:raise ValueError('Always-rerank fallback unexpectedly infeasible')
    return chosen,scan


def calibrate(config_path):
    cfg=json.loads(config_path.read_text());root=Path(cfg['output_root']);old=Path(cfg['rerank_root'])
    partition=json.loads((root/'PARTITION.json').read_text());cal=np.zeros(1000,dtype=bool);cal[partition['cal_image_indices']]=True
    selection={};scan={}
    for direction in ['T2I','I2T']:
        qmask=np.repeat(cal,5) if direction=='T2I' else cal;margins=[];clip=[];full=[]
        for seed in cfg['training_seeds']:
            with np.load(root/'confidence/dev'/f'seed{seed}.npz') as z:margins.append(z[direction+'_margin'].copy())
            with np.load(old/'candidates/dev'/f'seed{seed}.npz') as z:clip.append(ranking_metrics(z[direction.lower()],direction))
            with np.load(old/'evaluation'/f'rerank_seed{seed}'/'dev/per_query.npz') as z:
                full.append({k:z[k].copy() for k in [direction+'_R@1',direction+'_nDCG@10']})
        # Confirmation labels never enter the scan objective or candidate thresholds.
        selection[direction],scan[direction]=choose_direction(margins,clip,full,qmask,cfg,direction)
    policy={'status':'frozen','selected_at_rule':'smallest feasible threshold;timestamps excluded for deterministic resume',
        'config_sha256':sha_file(config_path),'partition_sha256':sha_file(root/'PARTITION.json'),
        'dev_confidence_sha256':sha_file(root/'confidence/dev/CONFIDENCE.json'),'assets_sha256':sha_file(root/'REUSED_ASSETS.json'),
        'selection':selection,'scope':'labels from500calibrationimages only;not confirmation/test selection'}
    lock_file(root/'POLICY.json',policy)
    write_json(root/'CALIBRATION.json',{'status':'completed','completed_at':now(),'policy_sha256':sha_file(root/'POLICY.json'),
        'scan':scan,'selected':selection,'quality_scope':'calibration only,not a new benchmark result'})
    return policy


def plan(config_path,split):
    cfg=json.loads(config_path.read_text());root=Path(cfg['output_root']);old=Path(cfg['rerank_root'])
    if split=='test' and not json.loads((root/'analysis/dev/ANALYSIS.json').read_text())['gate']['passed']:raise ValueError('No test planning before dev gate')
    policy=json.loads((root/'POLICY.json').read_text());d=root/'routes'/split;d.mkdir(parents=True,exist_ok=True)
    lock_file(d/'INPUT_LOCK.json',{'config_sha256':sha_file(config_path),'policy_sha256':sha_file(root/'POLICY.json'),
        'confidence_sha256':sha_file(root/'confidence'/split/'CONFIDENCE.json'),'source_sha256':sha_file(Path(__file__))})
    if (d/'PLAN.json').exists():
        o=json.loads((d/'PLAN.json').read_text())
        if any(sha_file(d/k)!=v for k,v in o['files'].items()):raise ValueError('Route artifacts changed')
        return o
    began=time.monotonic();all_full=[];all_selected=[];records=[]
    for seed in cfg['training_seeds']:
        with np.load(old/'candidates'/split/f'seed{seed}.npz') as z:pools={direction:z[direction.lower()].copy() for direction in ['T2I','I2T']}
        with np.load(root/'confidence'/split/f'seed{seed}.npz') as z:margins={direction:z[direction+'_margin'].copy() for direction in ['T2I','I2T']}
        masks={direction:route_mask(margins[direction],policy['selection'][direction]['threshold']) for direction in pools}
        full_keys=np.unique(np.concatenate([pair_indices(pool,direction) for direction,pool in pools.items()]))
        chosen=selected_keys(pools,masks);all_full.append(full_keys);all_selected.append(chosen)
        np.savez(d/f'seed{seed}.npz',**{direction+'_rerank':mask for direction,mask in masks.items()})
        records.append({'seed':seed,'full_unique_pairs':len(full_keys),'adaptive_unique_pairs':len(chosen),
            'pair_reduction':1-len(chosen)/len(full_keys),'reranked_queries':{direction:int(mask.sum()) for direction,mask in masks.items()},
            'query_counts':{'T2I':5000,'I2T':1000}})
    full_union=np.unique(np.concatenate(all_full));chosen_union=np.unique(np.concatenate(all_selected))
    expected=np.load(old/'candidates'/split/'pair_keys.npy')
    if not np.array_equal(full_union,expected):raise ValueError('Full pair union differs')
    np.save(d/'full_keys.npy',full_union);np.save(d/'adaptive_keys.npy',chosen_union)
    names=['full_keys.npy','adaptive_keys.npy',*[f'seed{s}.npz' for s in cfg['training_seeds']]]
    result={'status':'completed','completed_at':now(),'split':split,'full_unique_pairs':len(full_union),'adaptive_unique_pairs':len(chosen_union),
        'unique_pair_reduction':1-len(chosen_union)/len(full_union),'full_logical_pairs':900000,
        'adaptive_logical_pairs':sum(50*sum(r['reranked_queries'].values()) for r in records),
        'per_seed':records,'routing_and_union_cpu_seconds':time.monotonic()-began,
        'files':{name:sha_file(d/name) for name in names},'policy_sha256':sha_file(root/'POLICY.json'),
        'scope':'planned actual subset,no reference labels;not yet measured forward savings or latency'}
    write_json(d/'PLAN.json',result);return result
