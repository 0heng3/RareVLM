"""Matched original-vs-shuffled relations, two predeclared directional endpoints."""
import json
from pathlib import Path
import statistics
import numpy as np
from clip_retrieval.analysis import interval_image
from vlm_probe.evaluate import now, sha_file, write_json
from .prepare import verify_guard


def rescore_top10(path):
    with np.load(path/'per_query.npz') as z:
        expected = {}
        for prefix, count in [('I2T',1000),('T2I',5000)]:
            ranking = z[prefix.lower()+'_top10']
            positive = ranking//5 == np.arange(count)[:,None] if prefix=='I2T' else ranking == (np.arange(count)//5)[:,None]
            for k in [1,5,10]:
                expected[f'{prefix}_R@{k}'] = positive[:,:k].any(1).astype(float)
            weights = 1/np.log2(np.arange(10)+2)
            ideal = weights[:5].sum() if prefix=='I2T' else weights[0]
            expected[f'{prefix}_nDCG@10'] = (positive*weights).sum(1)/ideal
        for key,value in expected.items():
            if not np.allclose(value,z[key],atol=1e-7,rtol=0):
                raise ValueError('Independent Top10 rescore differs: '+str(path)+' '+key)
    return True


def analyze(config_path, split):
    cfg = json.loads(config_path.read_text()); root = Path(cfg['output_root'])
    summaries = {}; arrays = {}; rescored = []
    for condition in ['frozen',*cfg['conditions']]:
        seeds = [0] if condition=='frozen' else cfg['training_seeds']; cells=[]; vectors=[]
        for seed in seeds:
            key = 'frozen' if condition=='frozen' else f'{condition}_seed{seed}'
            path = root/'evaluation'/key/split
            cell = json.loads((path/'EVALUATION.json').read_text())
            if cell['per_query_sha256'] != sha_file(path/'per_query.npz'):
                raise ValueError('Rankings changed')
            rescore_top10(path); rescored.append(key)
            cells.append(cell)
            with np.load(path/'per_query.npz') as z:
                vectors.append({k:z[k].astype(np.float64) for k in z.files if 'top10' not in k})
        summaries[condition] = {name:{'mean':statistics.mean(c['metrics_percent'][name] for c in cells),
            'sample_sd':statistics.stdev(c['metrics_percent'][name] for c in cells) if len(cells)>1 else None,
            'per_seed':{str(s):c['metrics_percent'][name] for s,c in zip(seeds,cells)}} for name in cells[0]['metrics_percent']}
        arrays[condition] = vectors
    contracts = []
    for seed in cfg['training_seeds']:
        results = [json.loads((root/'runs'/c/f'seed{seed}'/'TRAIN_RESULT.json').read_text()) for c in cfg['conditions']]
        names = ['initial_state_sha256','completed_updates','positive_caption_pair_exposures',
                 'trainable_parameters','negative_occurrences_sha256','sampling_stream_sha256']
        for name in names:
            if len({r[name] for r in results})!=1:
                raise ValueError('Matched training differs: '+name)
        contracts.append({'seed':seed,**{name:results[0][name] for name in names}})
    def per_image(condition, metric):
        values = [v[metric].reshape(-1,5).mean(1) if metric.startswith('T2I') else v[metric] for v in arrays[condition]]
        return np.mean(values,axis=0)
    comparisons = {}; wins = {}; checks = {}
    for direction in ['I2T','T2I']:
        metric = direction+'_R@1'
        c = interval_image(per_image('distill',metric)-per_image('negative_shuffle',metric),
                           cfg['bootstrap_seed'],cfg['bootstrap_replicates'],cfg['gate']['each_primary_one_sided_alpha'])
        comparisons[direction.lower()+'_original_minus_shuffle'] = c
        wins[direction] = int(sum(a[metric].mean()>b[metric].mean() for a,b in zip(arrays['distill'],arrays['negative_shuffle'])))
        checks[direction+'_minimum_effect'] = c['delta_mean_pp']>=cfg['gate']['min_original_minus_shuffle_pp']
        checks[direction+'_corrected_lower_positive'] = c['one_sided_lower_pp']>0
        checks[direction+'_seed_directions'] = wins[direction]>=cfg['gate']['positive_seed_directions_min']
    exploratory = {direction:interval_image(per_image('negative_shuffle',direction+'_R@1')-per_image('uniform_mean',direction+'_R@1'),
        cfg['bootstrap_seed'],cfg['bootstrap_replicates'],.025) for direction in ['I2T','T2I']}
    guard = verify_guard(root)
    result = {'schema_version':'rarevlm.retrieval_shuffle.analysis.v1','status':'completed','completed_at':now(),
        'split':split,'config_sha256':sha_file(config_path),'source_sha256':sha_file(Path(__file__)),
        'summaries':summaries,'comparisons':comparisons,'exploratory_shuffle_minus_mean':exploratory,
        'positive_seed_directions':wins,'gate':{'passed':all(checks.values()),'checks':checks},
        'training_contracts':contracts,'new_training_runs':3,'new_scored_cells':3,'reused_cells':7,
        'image_queries_per_cell':1000,'caption_queries_per_cell':5000,
        'verification':{'independent_top10_rescore':True,'rescored_cells':rescored,'frozen_guard':guard},
        'interval_scope':'Two primary endpoints alpha=.05/2; conditional image-cluster intervals on fixed three-checkpoint average; seedSD separate',
        'limits':cfg['limits']}
    directory = root/'analysis'/split; directory.mkdir(parents=True,exist_ok=True)
    write_json(directory/'ANALYSIS.json',result)
    lines = ['# 教师负关系置乱机制结果','',f'划分：{split}；新增3训练，复用6训练checkpoint。','',
             '| 条件 | 文搜图R@1均值±SD | 图搜文R@1均值±SD | 图搜文nDCG@10 |','| --- | ---: | ---: | ---: |']
    for condition,s in summaries.items():
        def fmt(metric):
            x=s[metric];return f'{x["mean"]:.3f}'+(f' ± {x["sample_sd"]:.3f}' if x['sample_sd'] is not None else '')
        lines.append(f'| {condition} | {fmt("T2I_R@1")} | {fmt("I2T_R@1")} | {s["I2T_nDCG@10"]["mean"]:.3f} |')
    lines += ['',f'两方向机制门槛：{result["gate"]["passed"]}；正向seed数：{wins}。',
              '原蒸馏减置乱：每方向至少+.5pp、单侧97.5%下界>0且至少2seed同向。',
              '置乱保留正项教师概率、损失系数与目标熵；只能检验负项关系，不归因全部语义收益。',
              '历史公开集已曝光、pretrain未知，不称盲测；种子SD和条件图像簇CI分开。',
              '完整指标/区间/独立重评分与训练契约见ANALYSIS.json。']
    (directory/'结果.md').write_text('\n'.join(lines)+'\n')
    return result
