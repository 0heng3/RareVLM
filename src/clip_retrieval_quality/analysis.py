"""Two paired comparisons; four primary endpoints at alpha=.05/4."""
import json
from pathlib import Path
import statistics
import numpy as np
from clip_retrieval.analysis import interval_image
from vlm_probe.evaluate import now,sha_file,write_json
from .prepare import verify_guard


def analyze(config_path,split,branches=None):
    cfg=json.loads(config_path.read_text());root=Path(cfg['output_root']);branches=branches or ['clean','noisy']
    arrays={};summaries={};comparisons={};contracts=[]
    for condition in [b+'_'+m+'4' for b in branches for m in ['random','score']]:
        cells=[];vectors=[]
        for seed in cfg['training_seeds']:
            d=root/'evaluation'/f'{condition}_seed{seed}'/split
            cell=json.loads((d/'EVALUATION.json').read_text())
            if cell['per_query_sha256']!=sha_file(d/'per_query.npz'):raise ValueError('Ranking changed')
            cells.append(cell)
            with np.load(d/'per_query.npz') as x:vectors.append({k:x[k].astype(np.float64) for k in x.files if 'top10' not in k})
        arrays[condition]=vectors
        summaries[condition]={metric:{'mean':statistics.mean(c['metrics_percent'][metric] for c in cells),
            'sample_sd':statistics.stdev(c['metrics_percent'][metric] for c in cells),
            'per_seed':{str(s):c['metrics_percent'][metric] for s,c in zip(cfg['training_seeds'],cells)}}
            for metric in cells[0]['metrics_percent']}
    baseline=Path(cfg['baseline_root']);reference={}
    for metric in ['T2I_R@1','I2T_R@1']:
        values=[json.loads((baseline/'evaluation'/f'distill_seed{s}'/split/'EVALUATION.json').read_text())['metrics_percent'][metric]
            for s in cfg['training_seeds']]
        reference[metric]={'mean':statistics.mean(values),'sample_sd':statistics.stdev(values),
            'scope':'historical clean-full5;more annotations and different candidate width,not matched4 control'}
    for branch in branches:
        a=branch+'_score4';b=branch+'_random4';intervals={}
        for metric in ['T2I_R@1','I2T_R@1']:
            delta=np.mean([x[metric]-y[metric] for x,y in zip(arrays[a],arrays[b])],axis=0)
            if metric.startswith('T2I'):delta=delta.reshape(1000,5).mean(1)
            intervals[metric]=interval_image(delta,cfg['bootstrap_seed'],cfg['bootstrap_replicates'],cfg['gate']['each_primary_one_sided_alpha'])
        wins=sum(x['T2I_R@1'].mean()>y['T2I_R@1'].mean() for x,y in zip(arrays[a],arrays[b]))
        nd={m:summaries[a][m]['mean']-summaries[b][m]['mean'] for m in ['T2I_nDCG@10','I2T_nDCG@10']}
        checks={'t2i_min_gain':intervals['T2I_R@1']['delta_mean_pp']>=cfg['gate'][branch+'_t2i_min_gain_pp'],
            't2i_lower_positive':intervals['T2I_R@1']['one_sided_lower_pp']>0,
            'i2t_noninferiority':intervals['I2T_R@1']['one_sided_lower_pp']>-cfg['gate']['i2t_noninferiority_margin_pp'],
            'positive_seeds':wins>=cfg['gate']['positive_seed_directions_min'],
            'ndcg_point_protection':all(v>=cfg['gate']['ndcg_point_min_delta_pp'] for v in nd.values())}
        comparisons[branch]={'intervals':intervals,'positive_seed_directions':int(wins),'ndcg_delta_pp':nd,
            'gate':{'passed':all(checks.values()),'checks':checks},
            'score_minus_clean_full5_point_pp':{m:summaries[a][m]['mean']-reference[m]['mean'] for m in reference}}
        for seed in cfg['training_seeds']:
            x=json.loads((root/'runs'/a/f'seed{seed}'/'TRAIN_RESULT.json').read_text())
            y=json.loads((root/'runs'/b/f'seed{seed}'/'TRAIN_RESULT.json').read_text())
            keys=['initial_state_sha256','completed_updates','positive_caption_pair_exposures','trainable_parameters',
                'negative_occurrences_sha256','sampling_stream_sha256']
            if any(x[k]!=y[k] for k in keys):raise ValueError('Matched branch training differs:'+branch)
            contracts.append({'branch':branch,'seed':seed,**{k:x[k] for k in keys}})
    result={'status':'completed','completed_at':now(),'split':split,'summaries':summaries,
        'reference_clean_full5':reference,'comparisons':comparisons,'training_contracts':contracts,
        'gate':{'passed':any(c['gate']['passed'] for c in comparisons.values()),
            'passed_branches':[b for b,c in comparisons.items() if c['gate']['passed']]},
        'protected_files_unchanged':verify_guard(root),'limits':cfg['limits'],
        'interval_scope':'fixed-three-checkpoint conditional image-cluster;6000 draws;four primary endpoints alpha .05/4;SD separate'}
    write_json(root/'analysis'/split/'ANALYSIS.json',result)
    return result
