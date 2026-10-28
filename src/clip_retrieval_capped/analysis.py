"""Two fixed C/A primary endpoints; B comparisons explicitly exploratory."""
import json
from pathlib import Path
import statistics
import numpy as np
from clip_retrieval.analysis import interval_image
from vlm_probe.evaluate import now,sha_file,write_json


def independent_score(arrays):
    for direction,key,count in [('T2I','t2i_top10',5000),('I2T','i2t_top10',1000)]:
        order=arrays[key]
        if order.shape!=(count,10):raise ValueError('Incomplete saved rankings')
        positive=order==np.arange(count)[:,None]//5 if direction=='T2I' else order//5==np.arange(count)[:,None]
        for k in [1,5,10]:
            if not np.array_equal(positive[:,:k].any(1),arrays[f'{direction}_R@{k}']):raise ValueError('Independent Recall mismatch')
        discount=1/np.log2(np.arange(10)+2);ideal=discount[:1 if direction=='T2I' else 5].sum()
        score=(positive*discount).sum(1)/ideal
        if not np.allclose(score,arrays[f'{direction}_nDCG@10'],atol=8*np.finfo(np.float32).eps,rtol=0):raise ValueError('Independent nDCG mismatch')


def analyze(config_path,split):
    cfg=json.loads(config_path.read_text());root=Path(cfg['output_root']);summary={};vectors={};verified=[]
    for condition in cfg['conditions']:
        cells=[];vectors[condition]=[]
        for seed in cfg['training_seeds']:
            directory=root/'evaluation'/f'{condition}_seed{seed}'/split
            cell=json.loads((directory/'EVALUATION.json').read_text())
            if cell['per_query_sha256']!=sha_file(directory/'per_query.npz'):raise ValueError('Saved rankings changed')
            with np.load(directory/'per_query.npz') as a:arrays={name:a[name].copy() for name in a.files}
            independent_score(arrays)
            for name,value in arrays.items():
                if 'top10' not in name and abs(100*value.astype(np.float64).mean()-cell['metrics_percent'][name])>1e-4:raise ValueError('Reported score differs')
            vectors[condition].append({k:v.astype(np.float64) for k,v in arrays.items() if 'top10' not in k});cells.append(cell)
            verified.append({'condition':condition,'seed':seed,'per_query_sha256':cell['per_query_sha256'],'top10_rescore':'passed'})
        summary[condition]={name:{'mean':statistics.mean(c['metrics_percent'][name] for c in cells),
            'sample_sd':statistics.stdev(c['metrics_percent'][name] for c in cells),
            'per_seed':{str(seed):cell['metrics_percent'][name] for seed,cell in zip(cfg['training_seeds'],cells)}}
            for name in cells[0]['metrics_percent']}
    bindings=json.loads((root/'REUSED_BASELINES.json').read_text());contracts=[]
    for seed in cfg['training_seeds']:
        a=json.loads((root/'runs/distill'/f'seed{seed}'/'TRAIN_RESULT.json').read_text());record={'seed':seed,'conditions':{}}
        for method in cfg['new_training_conditions']:
            directory=root/'runs'/method/f'seed{seed}';b=json.loads((directory/'TRAIN_RESULT.json').read_text())
            for name in ['initial_state_sha256','completed_updates','positive_caption_pair_exposures','trainable_parameters','negative_image_exposures','distill_weight']:
                if a[name]!=b[name]:raise ValueError('Matched training differs:'+name)
            if b['anchor_stream_sha256']!=bindings['checkpoints'][str(seed)]['expected_anchor_stream_sha256']:raise ValueError('Anchor sequence differs')
            for filename,name in [('epoch_hard_counts.npy','epoch_hard_counts_sha256'),('epoch_random_counts.npy','epoch_random_counts_sha256'),('negative_occurrences.npy','negative_occurrences_sha256')]:
                if sha_file(directory/filename)!=b[name]:raise ValueError('Saved exposure changed')
            h=np.load(directory/'epoch_hard_counts.npy');r=np.load(directory/'epoch_random_counts.npy');total=np.load(directory/'negative_occurrences.npy')
            if not np.array_equal((h+r).sum(0),total) or int(total.sum())!=8750*64:raise ValueError('Incomplete exposure accounting')
            if method=='capped_distill' and h.max()>cfg['hard_exposure_cap_per_image_per_epoch']:raise ValueError('Hard quota violated')
            record['conditions'][method]={k:b[k] for k in ['anchor_stream_sha256','sampling_stream_sha256',
                'hard_image_exposures','random_image_exposures','hard_shortage_updates','max_hard_image_exposure_in_one_epoch',
                'max_negative_image_exposure','top1pct_total_negative_exposure_share','active_seconds']}
        contracts.append(record)
    def per_image(method,metric):
        return np.mean([v[metric].reshape(-1,5).mean(1) if metric.startswith('T2I') else v[metric] for v in vectors[method]],axis=0)
    comparisons={}
    for left,right,label in [('capped_distill','distill','capped_minus_distill'),('mixed_distill','distill','mixed_minus_distill'),('capped_distill','mixed_distill','capped_minus_mixed')]:
        comparisons[label]={direction:interval_image(per_image(left,direction+'_R@1')-per_image(right,direction+'_R@1'),
            cfg['bootstrap_seed'],cfg['bootstrap_replicates'],cfg['gate']['each_primary_one_sided_alpha'] if label=='capped_minus_distill' else .05)
            for direction in ['T2I','I2T']}
        comparisons[label]['scope']='two predeclared corrected primary endpoints' if label=='capped_minus_distill' else 'exploratory pointwise mechanism contrast,not selectable candidate'
    primary=comparisons['capped_minus_distill'];g=cfg['gate']
    wins=int(sum(c['T2I_R@1'].mean()>a['T2I_R@1'].mean() for c,a in zip(vectors['capped_distill'],vectors['distill'])))
    ndcg={d:summary['capped_distill'][d+'_nDCG@10']['mean']-summary['distill'][d+'_nDCG@10']['mean'] for d in ['T2I','I2T']}
    checks={'t2i_minimum_gain':primary['T2I']['delta_mean_pp']>=g['t2i_capped_minus_distill_min_pp'],
        't2i_positive_corrected_lower':primary['T2I']['one_sided_lower_pp']>0,
        'i2t_conditional_noninferiority':primary['I2T']['one_sided_lower_pp']>-g['i2t_noninferiority_margin_pp'],
        'positive_seed_directions':wins>=g['positive_t2i_seed_directions_min'],
        'both_ndcg_point_protection':all(v>=g['each_direction_ndcg_point_min_pp'] for v in ndcg.values())}
    result={'schema_version':'rarevlm.retrieval_capped.analysis.v1','status':'completed','completed_at':now(),'split':split,
        'config_sha256':sha_file(config_path),'source_sha256':sha_file(Path(__file__)),'summaries':summary,'comparisons':comparisons,
        'gate':{'passed':all(checks.values()),'checks':checks},'t2i_positive_seed_directions':wins,'ndcg_delta_pp':ndcg,
        'training_contracts':contracts,'independent_top10_checks':verified,'new_training_runs':6,
        'new_scored_cells':6,'reused_cells':3,'image_queries_per_cell':1000,'caption_queries_per_cell':5000,
        'interval_scope':'fixed3checkpoint average,1000paired image clusters;SD separate;C/A primary alpha=.05/2;historically exposed data',
        'limits':cfg['limits']}
    folder=root/'analysis'/split;folder.mkdir(parents=True,exist_ok=True);write_json(folder/'ANALYSIS.json',result)
    lines=['# 曝光受控困难负例结果','',f'划分：{split}；主候选为预先固定的capped_distill。','',
        '| 方法 | 文搜图R1均值±样本SD | 图搜文R1均值±样本SD |','| --- | ---: | ---: |']
    for method,s in summary.items():lines.append(f'| {method} | {s["T2I_R@1"]["mean"]:.3f} ± {s["T2I_R@1"]["sample_sd"]:.3f} | {s["I2T_R@1"]["mean"]:.3f} ± {s["I2T_R@1"]["sample_sd"]:.3f} |')
    lines+=['',f'预设联合门控：{result["gate"]}。','曝光上限只作用hard，不是所有负例。曝光改变不证明语义误负或唯一因果；dev失败不新增test。']
    (folder/'结果.md').write_text('\n'.join(lines)+'\n')
    return result
