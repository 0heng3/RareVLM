"""Complete fixed-candidate reranking with image-cluster quality gates."""
import json
from pathlib import Path
import statistics
import time
import numpy as np
from clip_retrieval.analysis import interval_image
from vlm_probe.evaluate import sha_file,now
from .common import pair_indices,retrieve_pair_scores,reorder,ranking_metrics,write_json
from .prepare import verify_guard


def analyze(config_path,split):
    cfg=json.loads(config_path.read_text());root=Path(cfg['output_root']);directory=root/'candidates'/split
    expected=np.load(directory/'pair_keys.npy');pieces=[];roles=[]
    for role in range(cfg['pair_scoring_roles']):
        path=root/'matcher'/split/f'role{role}';done=json.loads((path/'MATCH_COMPLETE.json').read_text())
        phase_seconds=0.
        for name,digest in sorted(done['chunk_files'].items()):
            p=path/'chunks'/name
            if sha_file(p)!=digest:raise ValueError('Pair score checkpoint changed')
            marker=json.loads(p.with_suffix('.json').read_text());phase_seconds+=marker['matching_phase_seconds']
            with np.load(p) as z:pieces.append((z['keys'].copy(),z['margin'].copy()))
        roles.append({'role':role,'pairs':done['pairs'],'matching_phase_seconds':phase_seconds,
                      'vision_encode_seconds':done['vision_encode_seconds'],'peak_allocated_gib':done['peak_allocated_gib'],
                      'worker_wall_seconds':done['worker_wall_seconds']})
    pieces.sort(key=lambda v:int(v[0][0]));keys=np.concatenate([p[0] for p in pieces]);scores=np.concatenate([p[1] for p in pieces])
    if not np.array_equal(keys,expected) or not np.isfinite(scores).all():raise ValueError('Incomplete/duplicated candidate union')
    conditions={'baseline':[],'rerank':[]};baseline_checks=[];sort_seconds=0.
    for seed in cfg['training_seeds']:
        with np.load(directory/f'seed{seed}.npz') as z:pools={d:z[d.lower()].copy() for d in ['T2I','I2T']}
        old=Path(cfg['baseline_root'])/f'evaluation/distill_seed{seed}/{split}'
        with np.load(old/'per_query.npz') as z:base={k:z[k].astype(np.float64) for k in z.files if 'top10' not in k}
        outputs={};ranks={};matched_scores={};begin=time.monotonic()
        for direction,pool in pools.items():
            score=retrieve_pair_scores(pair_indices(pool,direction).reshape(pool.shape),keys,scores)
            ranked=reorder(pool,score);m=ranking_metrics(ranked,direction);before=ranking_metrics(pool,direction)
            for metric,array in before.items():
                if metric.endswith('R@50'):base[metric]=array
                elif 'nDCG' in metric:
                    if np.max(np.abs(array-base[metric]))>8*np.finfo(np.float32).eps:raise ValueError('Baseline nDCG differs')
                elif not np.array_equal(array,base[metric]):raise ValueError('Baseline Recall differs')
            if not np.array_equal(before[direction+'_R@50'],m[direction+'_R@50']):raise ValueError('Rerank changed candidate recall upper bound')
            if not np.array_equal(np.sort(ranked,axis=1),np.sort(pool,axis=1)):raise ValueError('Rerank changed candidate membership')
            outputs.update(m);ranks[direction.lower()+'_top50']=ranked;matched_scores[direction.lower()+'_scores_original_pool']=score
        sort_seconds+=time.monotonic()-begin
        path=root/'evaluation'/f'rerank_seed{seed}'/split;path.mkdir(parents=True,exist_ok=True)
        np.savez(path/'per_query.npz',**outputs,**ranks,**matched_scores)
        write_json(path/'EVALUATION.json',{'status':'completed','completed_at':now(),'seed':seed,'split':split,
            'metrics_percent':{k:100*float(v.mean()) for k,v in outputs.items()},'per_query_sha256':sha_file(path/'per_query.npz'),
            'candidate_file_sha256':sha_file(directory/f'seed{seed}.npz'),'no_candidate_membership_change':True,'candidate_recall50_unchanged':True})
        conditions['baseline'].append(base);conditions['rerank'].append(outputs)
        baseline_checks.append({'seed':seed,'baseline_recall_exact':True,'baseline_ndcg_dtype_bound':float(8*np.finfo(np.float32).eps),'pool_unchanged':True})
    summary={c:{m:{'mean':statistics.mean(100*float(v[m].mean()) for v in values),
                  'sample_sd':statistics.stdev(100*float(v[m].mean()) for v in values),
                  'per_seed':{str(s):100*float(v[m].mean()) for s,v in zip(cfg['training_seeds'],values)}}
                  for m in values[0]} for c,values in conditions.items()}
    def per_image(c,metric):return np.mean([v[metric].reshape(-1,5).mean(1) if metric.startswith('T2I') else v[metric] for v in conditions[c]],axis=0)
    comparisons={d:interval_image(per_image('rerank',d+'_R@1')-per_image('baseline',d+'_R@1'),
        cfg['bootstrap_seed'],cfg['bootstrap_replicates'],cfg['gate']['each_primary_one_sided_alpha']) for d in ['T2I','I2T']}
    wins=sum(a['T2I_R@1'].mean()>b['T2I_R@1'].mean() for a,b in zip(conditions['rerank'],conditions['baseline']))
    ndcg={d:summary['rerank'][d+'_nDCG@10']['mean']-summary['baseline'][d+'_nDCG@10']['mean'] for d in ['T2I','I2T']}
    g=cfg['gate'];checks={'t2i_minimum_gain':comparisons['T2I']['delta_mean_pp']>=g['t2i_gain_min_pp'],
        't2i_corrected_lower_positive':comparisons['T2I']['one_sided_lower_pp']>0,
        'i2t_conditional_noninferiority':comparisons['I2T']['one_sided_lower_pp']>-g['i2t_noninferiority_margin_pp'],
        't2i_seed_directions':wins>=g['t2i_positive_seed_directions_min'],
        'ndcg_point_protection':min(ndcg.values())>=g['ndcg_point_min_delta_pp'],
        'role_resource_cap':all(r['peak_allocated_gib']<=cfg['peak_allocated_limit_gib'] for r in roles)}
    result={'status':'completed','completed_at':now(),'split':split,'summaries':summary,'comparisons':comparisons,
        'gate':{'passed':all(checks.values()),'checks':checks},'t2i_positive_seed_directions':wins,
        'ndcg_delta_pp':ndcg,'new_training_runs':0,'new_reranked_cells':3,'reused_baseline_cells':3,
        'logical_pairs':900000,'unique_pairs':len(keys),'matching_roles':roles,'rerank_sort_seconds':sort_seconds,
        'baseline_checks':baseline_checks,'frozen_guard':verify_guard(root),'limits':cfg['limits'],
        'interval_scope':'conditional paired-image intervals on3 fixed CLIP checkpoints+1 fixed sharedBLIP; separate seedSD;two endpoints alpha=.05/2'}
    folder=root/'analysis'/split;folder.mkdir(parents=True,exist_ok=True);write_json(folder/'ANALYSIS.json',result)
    lines=['# 固定Top50图文匹配重排','',f'划分：{split}；0新训练，三已有CLIP checkpoint+一固定BLIP。','',
        '| 条件 | 文搜图R@1均值±SD | 图搜文R@1均值±SD |','| --- | ---: | ---: |']
    for c,s in summary.items():
        lines.append(f'| {c} | {s["T2I_R@1"]["mean"]:.3f} ± {s["T2I_R@1"]["sample_sd"]:.3f} | {s["I2T_R@1"]["mean"]:.3f} ± {s["I2T_R@1"]["sample_sd"]:.3f} |')
    lines+=['',f'预设质量/资源门槛：{result["gate"]["passed"]}；逻辑匹配900000，实际唯一pair{len(keys)}。',
        '新增BLIP容量、384图像和BF16计算，不称等计算、纯注意力算法或线上服务；历史公开集曝光/pretrain未知。',
        '完整Recall@1/5/10/50、nDCG、图像簇区间、成本、源锁和限制见ANALYSIS.json。']
    (folder/'结果.md').write_text('\n'.join(lines)+'\n');return result
