"""Export descriptive seed summaries and paired image-cluster intervals."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from .common import sha_file,write_json


def paired_interval(base,adapted,seed=20261009,n_boot=2000):
    import numpy as np
    lookup={r['question_id']:r for r in base};groups=defaultdict(lambda:[0.,0])
    if set(lookup)!={r['question_id'] for r in adapted}:raise ValueError('unpaired questions')
    for r in adapted:
        b=lookup[r['question_id']]
        if b['image_id']!=r['image_id']:raise ValueError('unpaired images')
        groups[r['image_id']][0]+=r['score']-b['score'];groups[r['image_id']][1]+=1
    sums=np.array([v[0] for v in groups.values()]);counts=np.array([v[1] for v in groups.values()])
    rng=np.random.default_rng(seed);samples=[]
    for _ in range(n_boot):
        indices=rng.integers(0,len(sums),len(sums));samples.append(100*sums[indices].sum()/counts[indices].sum())
    return {'paired_question_mean_delta_pp':float(100*sums.sum()/counts.sum()),
        'image_cluster_bootstrap_95_pp':np.quantile(samples,[.025,.975]).tolist(),
        'clusters':len(groups),'questions':int(counts.sum()),
        'scope':'conditional image-cluster uncertainty; training-seed variability reported separately'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--asset-root',type=Path,required=True)
    p.add_argument('--run-root',type=Path,required=True);p.add_argument('--public-output',type=Path)
    args=p.parse_args();root=args.run_root;records=[];numerical=[]
    for file in sorted((root/'numerical').glob('*/EVALUATION.json')):
        e=json.loads(file.read_text())
        numerical.append({'condition':file.parent.name,'score_percent':e['score_percent'],'n':e['n'],
                          'precision':e['precision'],'quantization':e['quantization'],'result_sha256':sha_file(file)})
    baselines={}
    for task in ['textvqa','docvqa','mmstar']:
        p=root/'baselines'/task/'EVALUATION.json'
        if p.exists():baselines[task]=json.loads(p.read_text())
    for stage in ['stage1','stage2']:
        for file in sorted((root/stage).glob('*/seed*/train/TRAIN_RESULT.json')):
            t=json.loads(file.read_text());directory=file.parent.parent
            for task in ['dev','textvqa','docvqa','mmstar']:
                evaluation=directory/f'eval_{task}'/'EVALUATION.json'
                if not evaluation.exists():continue
                e=json.loads(evaluation.read_text())
                record={'stage':stage,'method':t['method'],'seed':t['seed'],'task':task,'score_percent':e['score_percent'],
                    'question_mean_score_percent':e['question_mean_score_percent'],'n':e['n'],'images':e['unique_images'],
                    'training_questions':t['question_count'],'updates':t['completed_updates'],
                    'trainable_parameters':t['trainable_parameters'],'training_active_seconds':t['active_seconds'],
                    'training_peak_allocated_gib':t['peak_gpu_allocated_gib'],'frozen_weights_unchanged':t['frozen_parameters_unchanged'],
                    'train_result_sha256':sha_file(file),'evaluation_sha256':sha_file(evaluation)}
                if task=='textvqa' and task in baselines:
                    b=json.loads((root/'baselines'/task/'predictions.json').read_text())
                    a=json.loads((evaluation.parent/'predictions.json').read_text())
                    record.update(paired_interval(b,a))
                records.append(record)
    grouped=defaultdict(list)
    for row in records:grouped[(row['stage'],row['method'],row['task'])].append(row)
    summaries=[]
    for (stage,method,task),members in sorted(grouped.items()):
        scores=[x['score_percent'] for x in members]
        summaries.append({'stage':stage,'method':method,'task':task,'seeds':[x['seed'] for x in members],
            'n_runs':len(scores),'mean_score_percent':statistics.mean(scores),
            'seed_sd_pp':statistics.stdev(scores) if len(scores)>1 else None,
            'baseline_score_percent':baselines[task]['score_percent'] if task in baselines else None,
            'selection_scope':'dev only; all other tasks excluded from method selection'})
    selection=root/'STAGE2_SELECTION.json';status=json.loads((root/'STATUS.json').read_text()) if (root/'STATUS.json').exists() else {}
    result={'schema_version':'rarevlm.vlm_deep.summary.v2','status':status,
        'scope':'full public validation benchmarks; pretraining overlap unknown; fixed one-epoch, counted-parameter strategy comparisons',
        'numerical_path_matrix':numerical,'baseline_metrics':{k:{'score_percent':v['score_percent'],'n':v['n'],'images':v['unique_images']} for k,v in baselines.items()},
        'seed_records':records,'summaries':summaries,
        'stage2_selection':json.loads(selection.read_text()) if selection.exists() else None,
        'report_source_sha256':sha_file(Path(__file__)),'weights_and_dataset_images_bundled':False}
    write_json(root/'SUMMARY.json',result)
    if args.public_output:write_json(args.public_output,result)
    if records:
        keys=sorted({k for r in records for k in r})
        with (root/'seed_metrics.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(records)
    print('REPORT',len(numerical),'numerical paths;',len(records),'seed/task records;',len(summaries),'summaries',flush=True)


if __name__=='__main__':main()
