"""Three-seed summaries and paired-image retrieval evidence,never seed×image IID."""
from __future__ import annotations
import json
from pathlib import Path
import statistics
import numpy as np
from vlm_probe.evaluate import write_json,now,sha_file
from .data import DEFAULT_CONFIG


def interval_image(delta,seed,replicates,alpha):
    rng=np.random.default_rng(seed);values=[];n=len(delta)
    for start in range(0,replicates,256):
        picks=rng.integers(0,n,size=(min(256,replicates-start),n));values.extend((100*delta[picks].mean(axis=1)).tolist())
    return {'delta_mean_pp':100*float(delta.mean()),'one_sided_lower_pp':float(np.quantile(values,alpha)),
        'exploratory_95_ci_pp':np.quantile(values,[.025,.975]).tolist(),'one_sided_confidence':1-alpha,
        'image_clusters':n,'replicates':replicates,'seed':seed,
        'scope':'pairedimage clusters after averaging the3 fixed checkpoint outputs;SDseparate;not seed-imageIID'}


def analyze(config_path=DEFAULT_CONFIG):
    cfg=json.loads(config_path.read_text());root=Path(cfg['output_root']);summaries={};arrays={}
    for condition in ['frozen',*cfg['conditions']]:
        seeds=[0] if condition=='frozen' else cfg['training_seeds'];summaries[condition]={};arrays[condition]={}
        for split in ['dev','test']:
            cells=[];values=[]
            for seed in seeds:
                key='frozen' if condition=='frozen' else f'{condition}_seed{seed}';directory=root/'evaluation'/key/split
                cell=json.loads((directory/'EVALUATION.json').read_text())
                if cell['per_query_sha256']!=sha_file(directory/'per_query.npz'):raise ValueError('ranking artifacts changed')
                cells.append(cell)
                with np.load(directory/'per_query.npz') as x:values.append({name:x[name].copy() for name in x.files if 'top10' not in name})
            metric_names=list(cells[0]['metrics_percent']);stats={}
            for name in metric_names:
                v=[c['metrics_percent'][name] for c in cells]
                stats[name]={'mean':statistics.mean(v),'sample_sd':statistics.stdev(v) if len(v)>1 else None,'per_seed':dict(zip(map(str,seeds),v))}
            summaries[condition][split]=stats;arrays[condition][split]=values
    initial=[]
    for seed in cfg['training_seeds']:
        a=json.loads((root/'runs/uniform_negatives'/f'seed{seed}'/'TRAIN_RESULT.json').read_text())
        b=json.loads((root/'runs/mixed_hard_negatives'/f'seed{seed}'/'TRAIN_RESULT.json').read_text())
        for key in ['initial_state_sha256','completed_updates','positive_caption_pair_exposures','trainable_parameters']:
            if a[key]!=b[key]:raise ValueError(f'matchedtrainingcontractdiffers:{key}')
        initial.append({'seed':seed,'initial_sha256':a['initial_state_sha256'],'updates':a['completed_updates'],
            'positive_pair_exposures':a['positive_caption_pair_exposures'],'negative_exposures_uniform':a['negative_image_exposures'],
            'negative_exposures_mixed':b['negative_image_exposures']})
    comparisons={};gates={}
    for split in ['dev','test']:
        mixed=np.mean([v['T2I_R@1'].reshape(-1,5).mean(axis=1) for v in arrays['mixed_hard_negatives'][split]],axis=0)
        uniform=np.mean([v['T2I_R@1'].reshape(-1,5).mean(axis=1) for v in arrays['uniform_negatives'][split]],axis=0)
        frozen=arrays['frozen'][split][0]['T2I_R@1'].reshape(-1,5).mean(axis=1)
        against_uniform=interval_image(mixed-uniform,cfg['bootstrap_seed'],cfg['bootstrap_replicates'],cfg['gate']['each_primary_one_sided_alpha'])
        against_frozen=interval_image(mixed-frozen,cfg['bootstrap_seed'],cfg['bootstrap_replicates'],cfg['gate']['each_primary_one_sided_alpha'])
        i2t=summaries['mixed_hard_negatives'][split]['I2T_R@1']['mean']-summaries['uniform_negatives'][split]['I2T_R@1']['mean']
        wins=sum(a['T2I_R@1'].mean()>b['T2I_R@1'].mean() for a,b in zip(arrays['mixed_hard_negatives'][split],arrays['uniform_negatives'][split]))
        checks={'t2i_gain_vs_uniform':against_uniform['delta_mean_pp']>=cfg['gate']['mixed_minus_uniform_t2i_r1_mean_min_pp'],
            't2i_gain_vs_frozen':against_frozen['delta_mean_pp']>=cfg['gate']['mixed_minus_frozen_t2i_r1_mean_min_pp'],
            'paired_lower_vs_uniform':against_uniform['one_sided_lower_pp']>0,'paired_lower_vs_frozen':against_frozen['one_sided_lower_pp']>0,
            'i2t_point_protection':i2t>=-1,'training_seed_directions':wins>=2}
        comparisons[split]={'mixed_minus_uniform':against_uniform,'mixed_minus_frozen':against_frozen,
            'i2t_mean_delta_pp':i2t,'positive_seed_directions':wins}
        gates[split]={'passed':all(checks.values()),'checks':checks}
    result={'schema_version':'rarevlm.retrieval.analysis.v1','status':'completed','completed_at':now(),
        'config_sha256':sha_file(config_path),'source_sha256':sha_file(Path(__file__)),
        'summaries':summaries,'comparisons':comparisons,'gates':gates,'training_contracts':initial,
        'new_training_runs':6,'metrics_not_combined':True,'pretraining_overlap':'unknown','limits':cfg['limits']}
    (root/'analysis').mkdir(exist_ok=True);write_json(root/'analysis/ANALYSIS.json',result)
    lines=['# 图文检索训练结果','',f'完成：{result["completed_at"]}。冻结CLIP，两侧残差适配器132480参数；三seed均值与样本SD分开报告。','',
        '| 数据划分 | 方法 | T2I R@1均值±SD | I2T R@1均值±SD |','| --- | --- | --- | --- |']
    for c,by_split in summaries.items():
        for split,s in by_split.items():
            t,i=s['T2I_R@1'],s['I2T_R@1'];lines.append(f'| {split} | {c} | {t["mean"]:.3f} ± {t["sample_sd"] if t["sample_sd"] is not None else "NA"} | {i["mean"]:.3f} ± {i["sample_sd"] if i["sample_sd"] is not None else "NA"} |')
    lines+=['','完整Recall@1/5/10、nDCG、逐query、三seed与图像簇CI见JSON/NPZ；Final@10固定，全组完整test一次，不据test回调。该真实公开检索任务不等于工业级商品检索、VLMEmbedding或Reranker部署。']
    (root/'analysis/结果.md').write_text('\n'.join(lines)+'\n');return result
