"""Task-specific paired-image noninferiority and repeated strong-baseline costs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics
import numpy as np
from vlm_probe.evaluate import now,write_json,sha_file,content_sha
from vlm_formal.evaluate import task_data
from vlm_deep.evaluate import score_prediction
from .evaluate import DEFAULT_CONFIG,source_hashes,metric_percent,compare_input


def paired_interval(rows,delta,task,replicates,seed,alpha):
    images=sorted({r['image_id'] for r in rows});index={iid:i for i,iid in enumerate(images)}
    categories=sorted({r['subcategory'] for r in rows}) if task=='mmstar' else ['all']
    lookup={c:i for i,c in enumerate(categories)};sums=np.zeros((len(images),len(categories)));counts=np.zeros_like(sums)
    for r,d in zip(rows,delta):
        i=index[r['image_id']];j=lookup[r['subcategory']] if task=='mmstar' else 0
        sums[i,j]+=d;counts[i,j]+=1
    rng=np.random.default_rng(seed);samples=[]
    for begin in range(0,replicates,64):
        picks=rng.integers(0,len(images),size=(min(64,replicates-begin),len(images)))
        total=sums[picks].sum(axis=1);denominator=counts[picks].sum(axis=1)
        if np.any(denominator==0):raise ValueError('bootstrap omitted required category')
        samples.extend((100*(total/denominator).mean(axis=1)).tolist())
    point=100*float((sums.sum(axis=0)/counts.sum(axis=0)).mean())
    return {'delta_pp':point,'exploratory_95_ci_pp':np.quantile(samples,[.025,.975]).tolist(),
        'one_sided_lower_pp':float(np.quantile(samples,alpha)),'one_sided_confidence':1-alpha,
        'image_clusters':len(images),'bootstrap_replicates':replicates,'seed':seed,
        'aggregation':'equal18subcategorymacro' if task=='mmstar' else 'QA mean',
        'scope':'fixed one checkpoint,known benchmark exposure and unknownpretraining overlap;not seed replication or blind test'}


def read_cell(directory,rows,task,scored):
    result=json.loads((directory/'EVALUATION.json').read_text());lock=json.loads((directory/'INPUT_LOCK.json').read_text())
    path=directory/'predictions.jsonl'
    if result['predictions_sha256']!=sha_file(path) or result['batches_sha256']!=sha_file(directory/'batches.jsonl') or result['input_lock_sha256']!=content_sha(lock):raise ValueError('completed artifacts changed')
    records=[json.loads(line) for line in path.read_text().splitlines()]
    if [r['question_id'] for r in records]!=[r['question_id'] for r in rows]:raise ValueError('fullquestion coverage/order differs')
    if scored:
        for record,row in zip(records,rows):
            if abs(record['score']-score_prediction(record['prediction'],row,task))>1e-12:raise ValueError('independent score verification failed')
        if abs(metric_percent(records,rows,task)-result['score_percent'])>1e-10:raise ValueError('aggregation differs')
    if lock['source_hashes']!=source_hashes():raise ValueError('scientific source changed')
    return result,records,lock


def analyze(config_path=DEFAULT_CONFIG):
    cfg=json.loads(config_path.read_text());root=Path(cfg['output_root']);queue=json.loads((root/'STATUS.json').read_text())
    output={'schema_version':'rarevlm.strong_confirmation.analysis.v1','status':'completed','completed_at':now(),
        'config_sha256':sha_file(config_path),'cells':{},'comparisons':{},'gates':{},'full_scored_predictions':23698,
        'timing_observations':3600,'unique_questions':11849,'new_training':0,'limits':cfg['limits']}
    for task in cfg['tasks']:
        rows,images,image_root,metadata=task_data(Path(cfg['asset_root']),task,cfg);rows=[{**r,'question_id':str(r['question_id'])} for r in rows]
        base,base_records,base_lock=read_cell(root/'runs'/task/'dense4',rows,task,True)
        candidate,candidate_records,candidate_lock=read_cell(root/'runs'/task/'segmented16',rows,task,True)
        if base_lock['hardware']!=candidate_lock['hardware']:raise ValueError('strong comparison not sameGPU')
        for a,b in zip(base_records,candidate_records):compare_input(a,b)
        delta=[b['score']-a['score'] for a,b in zip(base_records,candidate_records)]
        interval=paired_interval(rows,delta,task,cfg['bootstrap_replicates'],cfg['bootstrap_seed'],cfg['one_sided_alpha'])
        timing=[]
        for repeat in range(3):
            old,_,_=read_cell(root/'timing'/task/f'repeat{repeat}'/'dense4',rows[:200],task,False)
            new,_,_=read_cell(root/'timing'/task/f'repeat{repeat}'/'segmented16',rows[:200],task,False)
            timing.append(new['phase_qa_per_second']/old['phase_qa_per_second'])
        checks={'conditional_task_noninferiority':interval['one_sided_lower_pp']>-cfg['noninferiority_margin_pp'][task],
            'median_phase_speedup':statistics.median(timing)>=1.5,'repeat_consistency':min(timing)>=1.1,
            'resource_cap':candidate['peak_allocated_gib']<=65,
            'observed_resources_clean':queue['jobs'][task]['contention_observations']==0 and queue['resource_query_failures']==0}
        output['cells'][task]={'dense4':base,'segmented16':candidate}
        output['comparisons'][task]={'quality':interval,'phase_speedup_repeats':timing,'median_phase_speedup':statistics.median(timing),
            'full_phase_speedup':candidate['phase_qa_per_second']/base['phase_qa_per_second'],
            'allocated_delta_gib':candidate['peak_allocated_gib']-base['peak_allocated_gib'],
            'raw_answers_changed':sum(a['prediction']!=b['prediction'] for a,b in zip(base_records,candidate_records)),
            'scope':'jointbatch+attentionpackage;single-factor samebatch evidence archived separately'}
        output['gates'][task]={'passed':all(checks.values()),'checks':checks}
    output['all_tasks_passed']=all(g['passed'] for g in output['gates'].values())
    (root/'analysis').mkdir(exist_ok=True);write_json(root/'analysis/ANALYSIS.json',output)
    lines=['# 强基线正式确认结果','',f'完成：{output["completed_at"]}。单个固定adapter，6格23698条model×QA；不合成跨任务总分。','',
        '| 任务 | dense4 | segmented16 | 分差pp | 三轮阶段吞吐比中位数 | allocated前→后GiB | 门槛 |','| --- | ---: | ---: | ---: | ---: | --- | --- |']
    for task,cells in output['cells'].items():
        a,b=cells['dense4'],cells['segmented16'];comp=output['comparisons'][task]
        lines.append(f'| {task} | {a["score_percent"]:.3f} | {b["score_percent"]:.3f} | {comp["quality"]["delta_pp"]:.3f} | {comp["median_phase_speedup"]:.3f} | {a["peak_allocated_gib"]:.3f}→{b["peak_allocated_gib"]:.3f} | {output["gates"][task]["passed"]} |')
    lines+=['','基线为原始batch4，候选为分段SDPA batch16，速度是联合配置收益；显存可能增加。质量CI及单侧阈值见JSON，不把跨零区间称等价/显著提分，不称线上SLA或三训练种子验证。']
    (root/'analysis/结果.md').write_text('\n'.join(lines)+'\n')
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG);print(analyze(parser.parse_args().config)['gates'])
