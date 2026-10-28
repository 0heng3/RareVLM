"""Only calibration labels choose20/50 cutoffs; runtime budgets use margins."""
import json
from pathlib import Path
import time
import numpy as np
from clip_retrieval_rerank.common import ranking_metrics,reorder,pair_indices,write_json
from vlm_probe.evaluate import lock_file,sha_file,now
from .route import budgets_from_margin,budget_keys


def choose_cutoff(margins,metrics,mask,cfg,direction,fast):
    required=[m[mask&(m<=fast)] for m in margins];values=np.concatenate(required)
    grid=np.unique(np.r_[-1.,np.quantile(values,cfg['calibration_quantiles']) if len(values) else [],2.1])
    scan=[];chosen=None
    for cutoff in grid:
        budgets=[budgets_from_margin(m,fast,float(cutoff)) for m in margins];deltas={}
        for suffix in ['R@1','nDCG@10']:
            name=direction+'_'+suffix
            difference=[]
            for b,condition in zip(budgets,metrics):
                predicted=np.select([b==0,b==20],[condition[0][name],condition[20][name]],default=condition[50][name])
                difference.append((predicted-condition[50][name])[mask])
            deltas[suffix]=100*float(np.mean(difference))
        feasible=deltas['R@1']>=-cfg['calibration_max_r1_loss_pp'] and deltas['nDCG@10']>=-cfg['calibration_max_ndcg_loss_pp']
        row={'cutoff':float(cutoff),'cal_r1_delta_pp':deltas['R@1'],'cal_ndcg_delta_pp':deltas['nDCG@10'],
            'cal_budget_fractions':{str(k):float(np.mean([np.mean(b[mask]==k) for b in budgets])) for k in [0,20,50]},'feasible':bool(feasible)}
        scan.append(row)
        if chosen is None and feasible:chosen=row
    if chosen is None:raise ValueError('Frozen fast+50 fallback no longer feasible')
    return chosen,scan


def calibrate(config_path):
    cfg=json.loads(config_path.read_text());root=Path(cfg['output_root']);old=Path(cfg['rerank_root']);adaptive=Path(cfg['adaptive_root'])
    fast_policy=json.loads((adaptive/'POLICY.json').read_text());record=json.loads((root/'REUSED_FAST_POLICY.json').read_text())
    if sha_file(adaptive/'POLICY.json')!=record['old_policy_sha256']:raise ValueError('Fast policy changed')
    cal=np.zeros(1000,dtype=bool);cal[json.loads((root/'PARTITION.json').read_text())['cal_image_indices']]=True
    selection={};scans={};retention={}
    for direction in ['T2I','I2T']:
        qmask=np.repeat(cal,5) if direction=='T2I' else cal;margins=[];metrics=[];retention[direction]=[]
        for seed in cfg['training_seeds']:
            with np.load(root/'confidence/dev'/f'seed{seed}.npz') as z:margins.append(z[direction+'_margin'].copy())
            with np.load(old/'candidates/dev'/f'seed{seed}.npz') as z:pool=z[direction.lower()].copy()
            with np.load(old/'evaluation'/f'rerank_seed{seed}'/'dev/per_query.npz') as z:
                score=z[direction.lower()+'_scores_original_pool'].copy()
                full={direction+'_'+s:z[direction+'_'+s].copy() for s in ['R@1','nDCG@10']}
            if pool.shape!=score.shape or pool.shape[1]!=50:raise ValueError('Cached full score/pool mismatch')
            prefix=pool.copy();prefix[:,:20]=reorder(pool[:,:20],score[:,:20])
            metrics.append({0:ranking_metrics(pool,direction),20:ranking_metrics(prefix,direction),50:full})
            relevant=pool[:,:20]//5==np.arange(len(pool))[:,None] if direction=='I2T' else pool[:,:20]==(np.arange(len(pool))//5)[:,None]
            full_correct=qmask&(full[direction+'_R@1']>0);hits=relevant.any(1)
            retention[direction].append({'seed':seed,'cal_full_correct_queries':int(full_correct.sum()),
                'cal_full_correct_queries_with_positive_in_top20':int((full_correct&hits).sum()),
                'retention_fraction':float(hits[full_correct].mean()),'scope':'afterprediction reference diagnostic,never runtime routing'})
        fast=fast_policy['selection'][direction]['threshold']
        selection[direction],scans[direction]=choose_cutoff(margins,metrics,qmask,cfg,direction,fast)
    policy={'status':'frozen','config_sha256':sha_file(config_path),'partition_sha256':sha_file(root/'PARTITION.json'),
        'assets_sha256':sha_file(root/'REUSED_ASSETS.json'),'reused_fast_policy_sha256':sha_file(root/'REUSED_FAST_POLICY.json'),
        'selection':{d:{'threshold':fast_policy['selection'][d]['threshold']} for d in selection},'budget_selection':selection,
        'scope':'fast thresholds unchanged;only500calimage labels choose20/50 cutoff;not confirmation/test selection'}
    lock_file(root/'POLICY.json',policy)
    write_json(root/'CALIBRATION.json',{'status':'completed','completed_at':now(),'policy_sha256':sha_file(root/'POLICY.json'),
        'selected':selection,'scan':scans,'cal_top20_reference_retention':retention})
    return policy


def plan(config_path,split):
    cfg=json.loads(config_path.read_text());root=Path(cfg['output_root']);old=Path(cfg['rerank_root'])
    if split=='test' and not json.loads((root/'analysis/dev/ANALYSIS.json').read_text())['gate']['passed']:raise ValueError('No test before dev gate')
    policy=json.loads((root/'POLICY.json').read_text());d=root/'routes'/split;d.mkdir(parents=True,exist_ok=True)
    lock_file(d/'INPUT_LOCK.json',{'config_sha256':sha_file(config_path),'policy_sha256':sha_file(root/'POLICY.json'),
        'confidence_sha256':sha_file(root/'confidence'/split/'CONFIDENCE.json'),'source_sha256':sha_file(Path(__file__))})
    if (d/'PLAN.json').exists():
        result=json.loads((d/'PLAN.json').read_text())
        if any(sha_file(d/k)!=digest for k,digest in result['files'].items()):raise ValueError('Budget plan changed')
        return result
    started=time.monotonic();all_full=[];all_budget=[];records=[]
    for seed in cfg['training_seeds']:
        with np.load(old/'candidates'/split/f'seed{seed}.npz') as z:pools={direction:z[direction.lower()].copy() for direction in ['T2I','I2T']}
        with np.load(root/'confidence'/split/f'seed{seed}.npz') as z:margins={direction:z[direction+'_margin'].copy() for direction in ['T2I','I2T']}
        budgets={direction:budgets_from_margin(margins[direction],policy['selection'][direction]['threshold'],policy['budget_selection'][direction]['cutoff']) for direction in pools}
        full=np.unique(np.concatenate([pair_indices(p,direction) for direction,p in pools.items()]));chosen=budget_keys(pools,budgets)
        all_full.append(full);all_budget.append(chosen)
        np.savez(d/f'seed{seed}.npz',**{direction+'_budget':budget for direction,budget in budgets.items()})
        records.append({'seed':seed,'full_unique_pairs':len(full),'budget_unique_pairs':len(chosen),'pair_reduction':1-len(chosen)/len(full),
            'budget_counts':{direction:{str(k):int((b==k).sum()) for k in [0,20,50]} for direction,b in budgets.items()},
            'budget_logical_pairs':int(sum(b.sum() for b in budgets.values()))})
    full=np.unique(np.concatenate(all_full));chosen=np.unique(np.concatenate(all_budget))
    if not np.array_equal(full,np.load(old/'candidates'/split/'pair_keys.npy')):raise ValueError('Original full union changed')
    np.save(d/'full_keys.npy',full);np.save(d/'adaptive_keys.npy',chosen)
    names=['full_keys.npy','adaptive_keys.npy',*[f'seed{s}.npz' for s in cfg['training_seeds']]]
    result={'status':'completed','completed_at':now(),'split':split,'full_unique_pairs':len(full),'adaptive_unique_pairs':len(chosen),
        'unique_pair_reduction':1-len(chosen)/len(full),'full_logical_pairs':900000,
        'adaptive_logical_pairs':sum(r['budget_logical_pairs'] for r in records),'per_seed':records,
        'routing_and_union_cpu_seconds':time.monotonic()-started,'policy_sha256':sha_file(root/'POLICY.json'),
        'files':{name:sha_file(d/name) for name in names},
        'scope':'adaptive storage name means0/20/50budgeted;planned selected keys,no measured speedup yet'}
    write_json(d/'PLAN.json',result);return result
