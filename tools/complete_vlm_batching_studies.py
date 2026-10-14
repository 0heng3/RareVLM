"""Preserve failed batch32, finish original feasible cells, repeat them on idle GPU.

This is an explicit post-capacity engineering continuation, not a rewritten
five-cell success or new training-seed replication. Frozen v1 modules stay intact.
"""
from __future__ import annotations
import argparse
import csv
import fcntl
import html
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
sys.dont_write_bytecode=True
from rarevlm_paths import PROJECT_ROOT,SOURCE_ROOT
from vlm_joint.pipeline import gpu_snapshot,qualified
from vlm_precision.pipeline import GPU_LOCK_ROOT
from vlm_probe.evaluate import content_sha,now,write_json,sha_file,lock_file
from vlm_weighted.evaluate import dev_data
from vlm_ocr.analysis import prepare_slices
from vlm_batching.prepare import read_plan
from vlm_batching.analysis import read_cell,cluster_interval,build_reader


def analyze_feasible(config_path,resource_observations,excluded=None):
    cfg,root,plan=read_plan(config_path);group='batch16'
    policies=['fifo4','fifo16','bucket16']
    rows,images,image_root,metadata=dev_data(Path(cfg['asset_root']));by_id={r['question_id']:r for r in rows}
    contract=json.loads(Path(cfg['upstream_input_contract']).read_text())
    expected={r['question_id']:r for r in contract['records'] if r['condition']=='baseline'}
    slices=prepare_slices(Path(cfg['asset_root']),root);features={r['question_id']:r for r in slices['features']}
    results={group:{}};quality={group:{}};timing={group:{}};locks={};comparisons={};gates={}
    for policy in policies:
        result,records,lock=read_cell(root/'runs'/group/policy/'dev',plan['schedules'][policy],by_id,expected,True)
        if lock['source_hashes']!=plan['source_hashes'] or lock['config_sha256']!=sha_file(config_path):
            raise ValueError('scientific source/config mismatch')
        results[group][policy]=result;quality[group][policy]=records;locks[policy]=lock
        timing[group][policy]=[]
        for repeat in range(3):
            timed,_,tl=read_cell(root/'timing'/group/f'repeat{repeat}'/policy,plan['timing_schedules'][policy],by_id,expected,False)
            if tl['hardware']!=lock['hardware']:raise ValueError('full/repeated timing device differs')
            timing[group][policy].append(timed)
    base=quality[group]['fifo4'];threshold=cfg['gate']
    for policy in policies[1:]:
        if locks[policy]['hardware']!=locks['fifo4']['hardware']:raise ValueError('candidate/baseline different GPUs')
        candidate=quality[group][policy]
        delta=[candidate[r['question_id']]['score']-base[r['question_id']]['score'] for r in rows]
        interval=cluster_interval(rows,delta,cfg['bootstrap_replicates'],cfg['bootstrap_seed'],threshold['one_sided_alpha_per_candidate'])
        sliced={}
        for label in slices['slice_counts']:
            selected=[r for r in rows if features[r['question_id']]['sft_target_answer_frequency']==label]
            sliced[label]={'n':len(selected),
                'baseline':100*statistics.mean(base[r['question_id']]['score'] for r in selected),
                'candidate':100*statistics.mean(candidate[r['question_id']]['score'] for r in selected),
                'delta_pp':100*statistics.mean(candidate[r['question_id']]['score']-base[r['question_id']]['score'] for r in selected),
                'scope':'point only,not slice noninferiority'}
        ratios=[c['resource']['preprocess_h2d_generation_qa_per_second']/b['resource']['preprocess_h2d_generation_qa_per_second']
            for c,b in zip(timing[group][policy],timing[group]['fifo4'])]
        comparisons[policy]={'group':group,'quality':interval,'slices':sliced,
            'raw_answers_changed':sum(candidate[r['question_id']]['prediction']!=base[r['question_id']]['prediction'] for r in rows),
            'score_changed':sum(abs(d)>1e-12 for d in delta),'paired_phase_speedup_repeats':ratios,
            'median_paired_phase_speedup':statistics.median(ratios),
            'full_dev_phase_speedup':results[group][policy]['resource']['preprocess_h2d_generation_qa_per_second']/results[group]['fifo4']['resource']['preprocess_h2d_generation_qa_per_second']}
        checks={'conditional_overall_noninferiority':interval['one_sided_lower_bound_pp']>-threshold['overall_noninferiority_margin_pp'],
            'rare_point_protection':sliced['rare_1_5']['delta_pp']>=threshold['rare_point_delta_min_pp'],
            'frequent_point_protection':sliced['frequent_gt20']['delta_pp']>=threshold['frequent_point_delta_min_pp'],
            'median_phase_speedup':statistics.median(ratios)>=threshold['median_paired_phase_speedup_min'],
            'repeat_speedup_consistency':min(ratios)>=threshold['each_repeat_phase_speedup_min'],
            'allocated_memory_cap':results[group][policy]['resource']['peak_allocated_gib']<=threshold['peak_allocated_gib_max'],
            'resource_observation_clean':resource_observations['contention_observations']==0 and resource_observations['query_failures']==0 and not resource_observations.get('monitoring_gap_after_capacity_rejection',False)}
        gates[policy]={'passed':all(checks.values()),'checks':checks}
    # Planned matched-batch control: separate sorting from increasing batch.
    a=quality[group]['bucket16'];b=quality[group]['fifo16']
    delta=[a[r['question_id']]['score']-b[r['question_id']]['score'] for r in rows]
    secondary={'contrast':'bucket16_minus_fifo16','quality':cluster_interval(rows,delta,cfg['bootstrap_replicates'],cfg['bootstrap_seed'],.025),
        'timing_phase_ratio_repeats':[a['resource']['preprocess_h2d_generation_qa_per_second']/b['resource']['preprocess_h2d_generation_qa_per_second']
            for a,b in zip(timing[group]['bucket16'],timing[group]['fifo16'])],
        'scope':'predeclared same-batch secondary control;95%two-sided exploratory interval,not an additional advancement test'}
    if excluded:gates['bucket32']={'passed':False,'stage':'capacity_preflight','reason':excluded['error'],'quality_not_evaluated':True}
    result={'schema_version':'rarevlm.batching.feasible_completion.v1','status':'completed_feasible_subset' if excluded else 'completed_engineering_repeat',
        'completed_at':now(),'config_sha256':sha_file(config_path),'plan_sha256':sha_file(root/'PLAN.json'),
        'completion_source_sha256':sha_file(Path(__file__)),'cells':results,'timing':timing,
        'comparisons':comparisons,'same_batch_secondary':secondary,'gates':gates,
        'planned_full_cells':cfg['scope']['new_full_dev_cells'],'actual_full_cells':3,
        'planned_scored_predictions':cfg['scope']['new_full_scored_predictions'],'actual_scored_predictions':4851,
        'actual_unscored_timing_observations':2304,'unique_qa':1617,'unique_images':1024,
        'excluded':excluded,'resource_observations':resource_observations,'limits':cfg['limits'],
        'statistics':'original .05/3 one-sided screen retained after capacity exclusion;fixed checkpoint;no blind or seed-stability claim'}
    (root/'analysis').mkdir(exist_ok=True);write_json(root/'analysis/ANALYSIS.json',result)
    with (root/'analysis/cells.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['group','policy','score_percent','phase_qa_per_second','generation_qa_per_second','peak_allocated_gib','padding_fraction'])
        writer.writeheader()
        for policy,cell in results[group].items():
            r=cell['resource'];writer.writerow({'group':group,'policy':policy,'score_percent':cell['score_percent'],
                'phase_qa_per_second':r['preprocess_h2d_generation_qa_per_second'],
                'generation_qa_per_second':r['pure_generation_qa_per_second'],'peak_allocated_gib':r['peak_allocated_gib'],
                'padding_fraction':r['padding_fraction']})
    display_cfg={**cfg,'groups':{'batch16':policies}}
    build_reader(display_cfg,root,plan,result,rows,images,image_root,quality)
    page=root/'learning/学习页.html';text=page.read_text()
    description=('原计划5格，batch32在65GiB容量预检被拒绝；本归档完成可行3格，每格1617QA，在同一GPU重跑FIFO4基线。' if excluded else
        '独立硬件复核3格，每格1617QA，在同一GPU重跑FIFO4基线；与原研究共享QA和checkpoint，不能作为训练种子或独立数据重复。')
    text=text.replace('5格完整dev各1617QA；两组在各自GPU上重跑FIFO4基线。',description).replace('五格完整dev','三格完整dev')
    text=text.replace('<h2>工程实测</h2>','<h2>工程实测</h2><p>'+html.escape(description)+'</p>')
    page.write_text(text)
    provenance_path=root/'learning/FIGURE_PROVENANCE.json';provenance=json.loads(provenance_path.read_text())
    provenance.update(completion_source_sha256=sha_file(Path(__file__)),actual_full_cells=3,excluded_candidate=excluded)
    write_json(provenance_path,provenance)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-config',type=Path,required=True)
    parser.add_argument('--repeat-config',type=Path,required=True)
    args=parser.parse_args()
    original,original_root,original_plan=read_plan(args.original_config)
    repeat,repeat_root,repeat_plan=read_plan(args.repeat_config)
    owner=(repeat_root/'PIPELINE.lock').open('a+');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock_file(repeat_root/'COMPLETION_INPUT_LOCK.json',{'source_sha256':sha_file(Path(__file__)),
        'original_plan_sha256':sha_file(original_root/'PLAN.json'),'repeat_plan_sha256':sha_file(repeat_root/'PLAN.json'),
        'original_config_sha256':sha_file(args.original_config),'repeat_config_sha256':sha_file(args.repeat_config),
        'cause':'batch32 rejected at existing capacity gate; original batch16 worker continues; new same-policy hardware repeat'})
    excluded=json.loads((original_root/'workers/batch32/STATUS.json').read_text())
    if excluded['status']!='failed' or 'memory limit exceeded' not in excluded['error']:
        raise ValueError('expected preserved capacity rejection')
    state={'status':'waiting_resources','started_at':now(),'controller_pid':os.getpid(),
        'excluded_original_group':'batch32','existing_tasks_untouched':True,'repeat_pid':None,
        'original_worker_pid':json.loads((original_root/'STATUS.json').read_text())['groups']['batch16']['pid'],
        'resource':{'original':{'query_failures':0,'contention_observations':0,'monitoring_gap_after_capacity_rejection':True},'repeat':{'query_failures':0,'contention_observations':0,'monitoring_gap_after_capacity_rejection':False}}}
    idle={};process=None;stream=log=None;finished={}
    GPU_LOCK_ROOT.mkdir(mode=0o700,parents=True,exist_ok=True);(repeat_root/'logs').mkdir(exist_ok=True)
    repeat_ws=repeat_root/'workers/batch16/STATUS.json'
    while len(finished)<2:
        try:devices=gpu_snapshot()
        except Exception as error:
            for key in state['resource']:state['resource'][key]['query_failures']+=1
            idle.clear();state.update(status='resource_query_failed',updated_at=now(),last_resource_error=repr(error))
            write_json(repeat_root/'STATUS.json',state);write_json(original_root/'COMPLETION_STATUS.json',state)
            time.sleep(15);continue
        with (repeat_root/'RESOURCE_SNAPSHOTS.jsonl').open('a') as f:f.write(json.dumps({'time':now(),'devices':devices})+'\n')
        for key,root,cfg_path in [('original',original_root,args.original_config),('repeat',repeat_root,args.repeat_config)]:
            ws=root/'workers/batch16/STATUS.json'
            if not ws.exists():continue
            worker=json.loads(ws.read_text());uid=worker.get('hardware',{}).get('uuid')
            d=next((d for d in devices.values() if d['uuid']==uid),None)
            if uid and (d is None or len(d['compute_processes'])>1):state['resource'][key]['contention_observations']+=1
            if worker['status']=='failed':
                state.update(status='failed',error=worker.get('error'));write_json(repeat_root/'STATUS.json',state);return 1
            if worker['status']=='completed' and key not in finished:
                if key=='original':
                    prior=json.loads((original_root/'STATUS.json').read_text())
                    state['resource'][key]['contention_observations']+=prior['groups']['batch16']['contention_observations']
                    state['resource'][key]['query_failures']+=prior['resource_query_failures']
                finished[key]=analyze_feasible(cfg_path,state['resource'][key],excluded if key=='original' else None)
                print('ANALYSIS COMPLETE',key,finished[key]['gates'],flush=True)
        if process is None and 'repeat' not in finished:
            for d in devices.values():idle[d['uuid']]=idle.get(d['uuid'],0)+1 if qualified(d) else 0
            candidates=[d for d in devices.values() if idle.get(d['uuid'],0)>=3]
            if candidates:
                d=sorted(candidates,key=lambda d:d['index'])[0]
                stream=(GPU_LOCK_ROOT/(d['uuid']+'.lock')).open('a+')
                try:
                    fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    fresh=gpu_snapshot()[d['index']]
                    if not qualified(fresh):stream.close();stream=None
                except BlockingIOError:stream.close();stream=None
                if stream:
                    log=(repeat_root/'logs/batch16.log').open('a')
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=d['uuid'],PYTHONPATH=str(SOURCE_ROOT),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python')
                    command=[sys.executable,'-m','vlm_batching.evaluate','--group','batch16','--config',str(args.repeat_config)]
                    process=subprocess.Popen(command,cwd=PROJECT_ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,pass_fds=(stream.fileno(),))
                    state.update(repeat_pid=process.pid,repeat_gpu=d,repeat_admitted_at=now(),status='running')
                    print('ADMITTED REPEAT',d['index'],process.pid,flush=True)
        if process is not None and process.poll() is not None and 'repeat' not in finished and process.returncode!=0:
            state.update(status='failed',repeat_exit_code=process.returncode);write_json(repeat_root/'STATUS.json',state);return 1
        state['updated_at']=now();state['completed_roots']=list(finished)
        write_json(repeat_root/'STATUS.json',state);write_json(original_root/'COMPLETION_STATUS.json',state)
        if len(finished)<2:time.sleep(15)
    state.update(status='completed',completed_at=now(),completed_roots=list(finished))
    write_json(repeat_root/'STATUS.json',state);write_json(original_root/'COMPLETION_STATUS.json',state)
    if stream:stream.close()
    if log:log.close()
    return 0


if __name__=='__main__':sys.exit(main())
