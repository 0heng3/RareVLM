"""Predeclared three primary image-cluster tests and separate seed SD."""
from __future__ import annotations
import json
from pathlib import Path
import statistics
import numpy as np

from clip_retrieval.analysis import interval_image
from vlm_probe.evaluate import now, sha_file, write_json


def analyze(config_path,split):
    cfg=json.loads(config_path.read_text());root=Path(cfg['output_root']);summary={};arrays={}
    for condition in ['frozen',*cfg['conditions']]:
        seeds=[0] if condition=='frozen' else cfg['training_seeds'];cells=[];vectors=[]
        for seed in seeds:
            key='frozen' if condition=='frozen' else f'{condition}_seed{seed}'
            path=root/'evaluation'/key/split
            cell=json.loads((path/'EVALUATION.json').read_text())
            if cell['per_query_sha256']!=sha_file(path/'per_query.npz'):raise ValueError('Ranking hash changed')
            cells.append(cell)
            with np.load(path/'per_query.npz') as x:vectors.append({k:x[k].astype(np.float64) for k in x.files if 'top10' not in k})
        summary[condition]={name:{'mean':statistics.mean(c['metrics_percent'][name] for c in cells),
            'sample_sd':statistics.stdev(c['metrics_percent'][name] for c in cells) if len(cells)>1 else None,
            'per_seed':{str(seed):cell['metrics_percent'][name] for seed,cell in zip(seeds,cells)}}
            for name in cells[0]['metrics_percent']}
        arrays[condition]=vectors
    contracts=[]
    for seed in cfg['training_seeds']:
        a=json.loads((root/'runs/uniform_mean'/f'seed{seed}'/'TRAIN_RESULT.json').read_text())
        b=json.loads((root/'runs/distill'/f'seed{seed}'/'TRAIN_RESULT.json').read_text())
        for name in ['initial_state_sha256','completed_updates','positive_caption_pair_exposures',
                     'trainable_parameters','negative_occurrences_sha256','sampling_stream_sha256']:
            if a[name]!=b[name]:raise ValueError(f'Matched training differs:{name}')
        contracts.append({'seed':seed,'updates':a['completed_updates'],'positive_caption_pair_exposures':a['positive_caption_pair_exposures'],
            'initial_state_sha256':a['initial_state_sha256'],'negative_occurrences_sha256':a['negative_occurrences_sha256'],
            'sampling_stream_sha256':a['sampling_stream_sha256']})
    def per_image(condition,metric):
        values=[v[metric].reshape(-1,5).mean(1) if metric.startswith('T2I') else v[metric] for v in arrays[condition]]
        return np.mean(values,axis=0)
    def interval(a,b,metric):
        return interval_image(per_image(a,metric)-per_image(b,metric),cfg['bootstrap_seed'],
            cfg['bootstrap_replicates'],cfg['gate']['each_primary_one_sided_alpha'])
    comparisons={'i2t_distill_minus_mean':interval('distill','uniform_mean','I2T_R@1'),
        't2i_distill_minus_mean':interval('distill','uniform_mean','T2I_R@1'),
        'i2t_distill_minus_frozen':interval('distill','frozen','I2T_R@1')}
    wins=int(sum(a['I2T_R@1'].mean()>b['I2T_R@1'].mean() for a,b in zip(arrays['distill'],arrays['uniform_mean'])))
    ndcg=summary['distill']['I2T_nDCG@10']['mean']-summary['uniform_mean']['I2T_nDCG@10']['mean']
    frozen_gain=summary['distill']['T2I_R@1']['mean']-summary['frozen']['T2I_R@1']['mean']
    g=cfg['gate'];checks={
        'i2t_gain_minimum':comparisons['i2t_distill_minus_mean']['delta_mean_pp']>=g['i2t_gain_vs_mean_min_pp'],
        'i2t_gain_lower_positive':comparisons['i2t_distill_minus_mean']['one_sided_lower_pp']>0,
        't2i_conditional_noninferiority':comparisons['t2i_distill_minus_mean']['one_sided_lower_pp']>-g['t2i_noninferiority_margin_pp'],
        'i2t_frozen_conditional_noninferiority':comparisons['i2t_distill_minus_frozen']['one_sided_lower_pp']>-g['i2t_vs_frozen_noninferiority_margin_pp'],
        'i2t_ndcg_point_protection':ndcg>=g['i2t_ndcg_vs_mean_point_min_pp'],
        't2i_frozen_point_gain':frozen_gain>=g['t2i_gain_vs_frozen_point_min_pp'],
        'positive_seed_directions':wins>=g['i2t_positive_seed_directions_min']}
    result={'schema_version':'rarevlm.retrieval_distill.analysis.v1','status':'completed','completed_at':now(),
        'split':split,'config_sha256':sha_file(config_path),'source_sha256':sha_file(Path(__file__)),
        'summaries':summary,'comparisons':comparisons,'i2t_ndcg_delta_pp':ndcg,'t2i_frozen_delta_pp':frozen_gain,
        'i2t_positive_seed_directions':wins,'gate':{'passed':all(checks.values()),'checks':checks},
        'training_contracts':contracts,'new_training_runs':3,'full_gallery_cells':7,'reused_dev_cells':4 if split=='dev' else 0,'new_scored_cells':3 if split=='dev' else 7,
        'image_queries_per_cell':1000,'caption_queries_per_cell':5000,'limits':cfg['limits'],
        'interval_scope':'conditional fixed three-checkpoint average; separate sample SD; alpha=.05/3 across three primary endpoints'}
    directory=root/'analysis'/split;directory.mkdir(parents=True,exist_ok=True);write_json(directory/'ANALYSIS.json',result)
    lines=['# 冻结CLIP教师关系蒸馏结果','',f'划分：{split}；三seed Final@10，完整图库。','',
        '| 方法 | 文搜图R1均值±样本SD | 图搜文R1均值±样本SD | 图搜文nDCG10 |','| --- | ---: | ---: | ---: |']
    for condition,s in summary.items():
        def fmt(metric):
            v=s[metric];return f'{v["mean"]:.3f}' + (f' ± {v["sample_sd"]:.3f}' if v['sample_sd'] is not None else '')
        lines.append(f'| {condition} | {fmt("T2I_R@1")} | {fmt("I2T_R@1")} | {s["I2T_nDCG@10"]["mean"]:.3f} |')
    lines+=['',f'预设门槛通过：{result["gate"]["passed"]}；I2T正方向种子：{wins}/3。',
        '区间按图像整簇计算，不把5条caption或seed×image当独立样本。未通过dev门槛则不追加本轮test评测。',
        '完整指标/保护条件与三项校正后的单侧下界见ANALYSIS.json。历史test已曝光，预训练重叠未知，不称盲测。']
    (directory/'结果.md').write_text('\n'.join(lines)+'\n');return result
