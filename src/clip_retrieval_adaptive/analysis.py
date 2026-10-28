"""Fresh prediction quality on confirmation clusters and measured pair cost."""
import json
from pathlib import Path
import statistics
import time
import numpy as np
from clip_retrieval.analysis import interval_image
from clip_retrieval_rerank.common import ranking_metrics,write_json,retrieve_pair_scores
from vlm_probe.evaluate import sha_file,now
from .route import routed_ranking,selected_keys
from .prepare import verify_guard


def load_scores(root,split,variant,roles):
    expected=np.load(root/'routes'/split/(variant+'_keys.npy'));pieces=[]
    for role in range(roles):
        d=root/'matcher'/split/f'role{role}'/variant
        complete=json.loads((d/'QUALITY_COMPLETE.json').read_text())
        for name,digest in sorted(complete['chunk_files'].items()):
            p=d/'chunks'/name
            if sha_file(p)!=digest:raise ValueError('Actual score chunk changed')
            with np.load(p) as z:
                if z['logits'].shape!=(len(z['keys']),2) or not np.array_equal(z['margin'],z['logits'][:,1]-z['logits'][:,0]):raise ValueError('ITM margin/logit contract failed')
                pieces.append((z['keys'].copy(),z['margin'].copy()))
    pieces.sort(key=lambda p:int(p[0][0]))
    keys=np.concatenate([p[0] for p in pieces]) if pieces else np.empty(0,dtype=np.int64)
    scores=np.concatenate([p[1] for p in pieces]) if pieces else np.empty(0,dtype=np.float32)
    if not np.array_equal(keys,expected) or not np.isfinite(scores).all():raise ValueError('Actual computation differs from planned pair union')
    return keys,scores


def analyze(config_path,split):
    cfg=json.loads(config_path.read_text());root=Path(cfg['output_root']);old=Path(cfg['rerank_root'])
    plan=json.loads((root/'routes'/split/'PLAN.json').read_text());pair_scores={v:load_scores(root,split,v,cfg['pair_scoring_roles']) for v in ['full','adaptive']}
    partition=json.loads((root/'PARTITION.json').read_text());confirm=np.ones(1000,dtype=bool)
    if split=='dev':confirm[:]=False;confirm[partition['confirm_image_indices']]=True
    conditions={'full':[],'adaptive':[]};checks=[];sort_times={'full':0.,'adaptive':0.}
    for seed in cfg['training_seeds']:
        with np.load(old/'candidates'/split/f'seed{seed}.npz') as z:pools={d:z[d.lower()].copy() for d in ['T2I','I2T']}
        with np.load(root/'routes'/split/f'seed{seed}.npz') as z:masks={d:z[d+'_rerank'].copy() for d in ['T2I','I2T']}
        predicted=selected_keys(pools,masks)
        if len(predicted):retrieve_pair_scores(predicted,*pair_scores['adaptive'])
        old_file=old/'evaluation'/f'rerank_seed{seed}'/split/'per_query.npz'
        with np.load(old_file) as z:original={k:z[k].copy() for k in z.files if 'scores_original_pool' not in k}
        result_ranks={};old_exact=True
        for variant in ['full','adaptive']:
            outputs={};rankings={};start=time.monotonic()
            for direction,pool in pools.items():
                mask=np.ones(len(pool),dtype=bool) if variant=='full' else masks[direction]
                ranked=routed_ranking(pool,direction,mask,*pair_scores[variant])
                if not np.array_equal(np.sort(ranked,1),np.sort(pool,1)):raise ValueError('Candidate membership changed')
                if variant=='adaptive' and not np.array_equal(ranked[~mask],pool[~mask]):raise ValueError('Fast path changed original CLIP ranks')
                metric=ranking_metrics(ranked,direction)
                if not np.array_equal(metric[direction+'_R@50'],ranking_metrics(pool,direction)[direction+'_R@50']):raise ValueError('Candidate recall upper bound changed')
                outputs.update(metric);rankings[direction.lower()+'_top50']=ranked
                if variant=='full':old_exact=old_exact and np.array_equal(ranked,original[direction.lower()+'_top50'])
            sort_times[variant]+=time.monotonic()-start;conditions[variant].append(outputs);result_ranks[variant]=rankings
            d=root/'evaluation'/f'{variant}_seed{seed}'/split;d.mkdir(parents=True,exist_ok=True)
            np.savez(d/'per_query.npz',**outputs,**rankings)
            write_json(d/'EVALUATION.json',{'status':'completed','completed_at':now(),'condition':variant,'seed':seed,'split':split,
                'metrics_full_gallery_percent':{k:100*float(v.mean()) for k,v in outputs.items()},'per_query_sha256':sha_file(d/'per_query.npz'),
                'scope':'fresh matching scores;adaptive skipped queries retain CLIP order;not independent BLIP seeds'})
        checks.append({'seed':seed,'old_full_top50_exact':bool(old_exact),'candidate_membership_and_R50_unchanged':True,
            'fast_path_CLIP_exact':True,'selected_pair_coverage_exact':True})
    def summarize(image_mask):
        out={}
        for variant,vectors in conditions.items():
            out[variant]={}
            for metric in vectors[0]:
                mask=np.repeat(image_mask,5) if metric.startswith('T2I') else image_mask
                values=[100*float(v[metric][mask].mean()) for v in vectors]
                out[variant][metric]={'mean':statistics.mean(values),'sample_sd':statistics.stdev(values),
                    'per_seed':dict(zip(map(str,cfg['training_seeds']),values))}
        return out
    summaries=summarize(confirm);all_gallery=summarize(np.ones(1000,dtype=bool))
    def per_image(variant,metric):
        return np.mean([v[metric].reshape(1000,5).mean(1) if metric.startswith('T2I') else v[metric] for v in conditions[variant]],axis=0)[confirm]
    comparisons={d:interval_image(per_image('adaptive',d+'_R@1')-per_image('full',d+'_R@1'),
        cfg['bootstrap_seed'],cfg['bootstrap_replicates'],cfg['gate']['each_primary_one_sided_alpha']) for d in ['T2I','I2T']}
    roles=[]
    for role in range(cfg['pair_scoring_roles']):
        d=root/'matcher'/split/f'role{role}';complete=json.loads((d/'MATCH_COMPLETE.json').read_text())
        if complete['timing_sha256']!=sha_file(d/'TIMING.json'):raise ValueError('Timing records changed')
        timing=json.loads((d/'TIMING.json').read_text())
        if len(timing['samples'])!=cfg['timing_repeats']:raise ValueError('Incomplete complete-forward timing repeats')
        roles.append({'role':role,'samples':timing['samples'],'peak_allocated_gib':complete['peak_allocated_gib'],
            'vision_encode_seconds':complete['vision_encode_seconds'],'worker_wall_seconds':complete['worker_wall_seconds'],
            'fresh_quality_forward_pairs':complete['fresh_quality_forward_pairs'],'extra_timing_forward_pairs':complete['extra_timing_forward_pairs'],
            'hardware':complete['hardware']})
    timing=[]
    for repeat in range(cfg['timing_repeats']):
        full=max(role['samples'][repeat]['full_seconds'] for role in roles)
        adaptive=max(role['samples'][repeat]['adaptive_seconds'] for role in roles)+plan['routing_and_union_cpu_seconds']
        timing.append({'repeat':repeat,'parallel_full_match_seconds':full,'parallel_adaptive_match_plus_route_seconds':adaptive,
            'speedup':full/max(adaptive,1e-12)})
    queue=json.loads((root/(split+'_match_queue')/'STATUS.json').read_text())
    resource={'query_failures':queue['resource_query_failures'],'contention_observations':sum(v['contention_observations'] for v in queue['jobs'].values())}
    ndcg={d:summaries['adaptive'][d+'_nDCG@10']['mean']-summaries['full'][d+'_nDCG@10']['mean'] for d in ['T2I','I2T']}
    g=cfg['gate'];gate_checks={d.lower()+'_conditional_noninferiority':comparisons[d]['one_sided_lower_pp']>-g['each_direction_noninferiority_margin_pp'] for d in ['T2I','I2T']}
    gate_checks.update(unique_pair_reduction=plan['unique_pair_reduction']>=g['unique_pair_reduction_min'],
        measured_median_speedup=statistics.median(t['speedup'] for t in timing)>=g['parallel_match_plus_route_speedup_median_min'],
        measured_each_repeat_speedup=all(t['speedup']>=g['each_repeat_speedup_min'] for t in timing),
        ndcg_point_protection=all(v>=g['each_direction_ndcg_point_delta_min_pp'] for v in ndcg.values()),
        peak_resource_cap=all(r['peak_allocated_gib']<=cfg['peak_allocated_limit_gib'] for r in roles),
        resource_uncontended=resource['query_failures']==0 and resource['contention_observations']==0,
        historical_full_reproduction=all(c['old_full_top50_exact'] for c in checks))
    result={'status':'completed','completed_at':now(),'split':split,'analysis_set':'500devconfirmationimages' if split=='dev' else '1000historicaltestimages',
        'quality_image_clusters':int(confirm.sum()),'summaries':summaries,'comparisons':comparisons,
        'full_gallery_exploratory_summaries':all_gallery,'calibration_summaries':summarize(~confirm) if split=='dev' else None,
        'gate':{'passed':all(gate_checks.values()),'checks':gate_checks},'ndcg_point_delta_pp':ndcg,'baseline_and_pool_checks':checks,
        'cost':{'full_unique_pairs':plan['full_unique_pairs'],'adaptive_unique_pairs':plan['adaptive_unique_pairs'],
            'unique_pair_reduction':plan['unique_pair_reduction'],'full_logical_pairs':plan['full_logical_pairs'],
            'adaptive_logical_pairs':plan['adaptive_logical_pairs'],'per_seed':plan['per_seed'],
            'routing_and_union_cpu_seconds':plan['routing_and_union_cpu_seconds'],'timing_repeats':timing,
            'median_parallel_match_plus_route_speedup':statistics.median(t['speedup'] for t in timing),
            'quality_sort_seconds':sort_times,'matching_roles':roles,'resource_observations':resource,
            'scope':'fresh actual selected pair forwards,cached-gallery shared3checkpoint union;not cold-preparation/online/singlemodel latency'},
        'new_training_runs':0,'new_prediction_cells':6,'policy_sha256':sha_file(root/'POLICY.json'),
        'frozen_guard':verify_guard(root),'limits':cfg['limits'],
        'interval_scope':'mean3fixedCLIP/checkpoint outputs then paired image clusters;SDseparate;two NI endpoints alpha=.05/2;historically exposed dev/test'}
    d=root/'analysis'/split;d.mkdir(parents=True,exist_ok=True);write_json(d/'ANALYSIS.json',result)
    return result
