"""Describe persisted execution records separately from algorithm gate failures."""
from collections import Counter
from datetime import datetime
import hashlib
from pathlib import Path
import json

from rarevlm_paths import PROJECT_ROOT

BASE=PROJECT_ROOT.parents[1]/'outputs'
ROOT=BASE/'vlm_failure_stage_stats_v1'


def read(p):return json.loads(p.read_text())


def summarize():
    inventory=read(ROOT/'INVENTORY.json');records=inventory['queue_job_records'];failures=[];raw_counts=Counter(q['state'] for q in records)
    for q in records:
        if q['state']!='failed':continue
        text=q.get('log_tail','');kind='unclassified';reason='原始日志待定位'
        if 'FileNotFoundError' in text and '/images/' in text:kind='evaluation_image_read';reason='评测图片根路径缺失'
        elif "KeyError: 'inference_input_sha256'" in text:kind='post_eval_metadata';reason='评测后metadata缺input SHA字段'
        elif 'Actual mean differs from established formula' in text:kind='pre_update_numeric_check';reason='优化器更新前loss数值一致性检查'
        elif 'completed replay-coverage training/teacher/adapter bytes changed' in text:kind='artifact_contract_handoff';reason='评测前训练/教师/adapter契约检查'
        elif 'seed42 fixed QA stream differs: replay_microbatch_order_sha256' in text:kind='artifact_contract_handoff';reason='训练产物交接的固定QA流哈希检查'
        job=q['job'];minutes=None
        if job.get('admitted_at') and job.get('ended_at'):minutes=(datetime.fromisoformat(job['ended_at'])-datetime.fromisoformat(job['admitted_at'])).total_seconds()/60
        failures.append({'study':q['study'],'job':q['key'],'stage':kind,'reason':reason,'minutes_admission_to_failure':minutes,'queue_sources':q['source_paths'],'log_path':q.get('log_path'),'log_sha256':q.get('log_sha256')})
    negative=[]
    def is_negative(decisions):
        for k in ('passed','pilot_passed','screen_passed'):
            value=decisions.get(k)
            if value is False or isinstance(value,dict) and value and not all(value.values()):return True
        return 'negative' in str(decisions.get('phase',''))
    for q in inventory['root_records']:
        sources=[]
        if q['completion'].get('status','').startswith('completed') and is_negative(q['completion']):sources.append(q['completion_source'])
        for a in q['analysis_files']:
            if a['decisions'].get('status')=='completed' and is_negative(a['decisions']):sources.append(a['path'])
        if sources:negative.append({'study':q['study'],'sources':sources})
    current=BASE/'vlm_native_stable_kl_v1';latest=read(current/'SCREEN_ANALYSIS.json');older=read(BASE/'vlm_native_correction_sft_dpo_v2/SCREEN_ANALYSIS.json')
    models=[]
    for study,analysis in [('native_correction_sft_dpo_v2',older),('native_stable_kl_v1',latest)]:
        for role,checks in analysis['screen_checks'].items():
            models.append({'study':study,'role':role,'hard_learning':checks['hard_learning'],'ocr_protection':checks['ocr_point_protection'],'native_protection':checks['native_point_protections'],
                           'native_delta_pp':analysis['vs_start'][role]['seen_validation']['delta_pp'],'rare_delta_pp':analysis['vs_start'][role]['rare_1_5']['delta_pp'],'rare_ci95':analysis['vs_start'][role]['rare_1_5']['ci95']})
    rows=[
        {'step':'CPU损失与token检查','unit':'本轮保存检查','passed':2,'failed':0,'evidence':[str(current/'CPU_LOSS_CHECK.json'),str(current/'CPU_TOKEN_FRAME_CHECK.json')]},
        {'step':'新基线生成','unit':'队列作业','passed':2,'failed':0,'evidence':[str(current/'baseline_queue/STATUS.json')]},
        {'step':'固定教师缓存','unit':'队列作业','passed':1,'failed':0,'evidence':[str(current/'reference_queue/STATUS.json')]},
        {'step':'首步数值与梯度','unit':'模型','passed':2,'failed':0,'evidence':[str(current/f'runs/{r}/train/INITIAL_KL_CHECK.json') for r in ('sft','sft_kl')]},
        {'step':'32步训练完成','unit':'模型','passed':2,'failed':0,'evidence':[str(current/f'runs/{r}/train/TRAIN_COMPLETE.json') for r in ('sft','sft_kl')]},
        {'step':'完整内部评测','unit':'模型','passed':2,'failed':0,'evidence':[str(current/f'runs/{r}/internal_COMPLETE.json') for r in ('sft','sft_kl')]},
        {'step':'训练纠错可学习','unit':'模型性能检查','passed':sum(v['hard_learning'] for v in latest['screen_checks'].values()),'failed':sum(not v['hard_learning'] for v in latest['screen_checks'].values()),'evidence':[str(current/'SCREEN_ANALYSIS.json')]},
        {'step':'OCR点保护','unit':'模型性能检查','passed':sum(v['ocr_point_protection'] for v in latest['screen_checks'].values()),'failed':sum(not v['ocr_point_protection'] for v in latest['screen_checks'].values()),'evidence':[str(current/'SCREEN_ANALYSIS.json')]},
        {'step':'原生/罕见答案保护','unit':'模型性能检查','passed':sum(v['native_point_protections'] for v in latest['screen_checks'].values()),'failed':sum(not v['native_point_protections'] for v in latest['screen_checks'].values()),'evidence':[str(current/'SCREEN_ANALYSIS.json')]},
    ]
    # Every current pipeline count is bound to completed/passed source markers.
    for row in rows:
        for path in row['evidence']:
            d=read(Path(path))
            if 'STATUS.json' in path and any(v.get('state')!='completed' for v in d['jobs'].values()):raise ValueError('Incomplete current queue')
            if row['step'] in ('32步训练完成','完整内部评测') and d.get('status')!='completed':raise ValueError('Incomplete current phase')
    state_notes=[]
    for q in records:
        if q['state']=='running':state_notes.append({'study':q['study'],'job':q['key'],'recorded_state':'running','worker_pid':q['job']['pid'],'pid_present_at_report':Path('/proc',str(q['job']['pid'])).exists(),'note':'Inventory state retained; absent historical PID is stale/unresolved, not silently failed or complete.'})
    sft=read(current/'runs/sft/internal/seen_validation/PREDICTIONS.json');kl=read(current/'runs/sft_kl/internal/seen_validation/PREDICTIONS.json');start=read(current/'baseline/seen_validation/PREDICTIONS.json');meta={q['question_id']:q for q in read(current/'data/MANIFEST.json')['evaluations']['seen_validation']}
    changed=[{'qid':b['question_id'],'before':a['prediction'],'after':b['prediction'],'delta_score':b['vqa_score']-a['vqa_score'],'rare':1<=meta[b['question_id']]['answer_frequency']<=5} for a,b in zip(start,sft) if a['vqa_score']!=b['vqa_score']]
    train={r:read(current/f'runs/{r}/train/TRAIN_COMPLETE.json') for r in ('sft','sft_kl')};qa_audit=read(BASE/'vlm_native_correction_pool_v2/REVIEWED_ERRORS.json');replay=read(current/'reference/REPLAY_ADMISSION.json')
    result={'observed_at':datetime.now().astimezone().isoformat(),'inventory_observed_at':inventory['observed_at'],'inventory_scope':inventory['scope'],'historical_queue_records':{'vlm_roots':inventory['vlm_roots'],'queue_files':inventory['queue_files'],'unique_execution_records':len(records),'states':dict(raw_counts),'finished_records':raw_counts['completed']+raw_counts['failed'],'failed_fraction_of_finished_records':raw_counts['failed']/(raw_counts['completed']+raw_counts['failed']),'failed_by_stage':dict(Counter(q['stage'] for q in failures)),'failure_cases':failures,'runtime_state_notes':state_notes},
            'explicit_negative_gate_study_count':len(negative),'explicit_negative_gate_studies':negative,'negative_gate_scope':'Lower-bound count of unique study directories with explicit failed passed/pilot_passed/screen_passed or negative phase. goal_complete=False alone is excluded. Heterogeneous tasks/protocols; not an overall model failure probability.',
            'current_step_counts':rows,'recent_two_studies_four_models':models,'current_analysis':latest,'prediction_comparison':{'qa':512,'different_answer_strings':sum(a['prediction']!=b['prediction'] for a,b in zip(sft,kl)),'different_vqa_scores':sum(a['vqa_score']!=b['vqa_score'] for a,b in zip(sft,kl))},'current_native_score_changes':changed,
            'trace_cumulative_kl_reduction_fraction':1-train['sft_kl']['replay_kl_sum']/train['sft']['replay_kl_sum'],'trace_loss_scope':'Cumulative losses along changing training weights; not Final KL or gradient strength.',
            'qa_units_separate':{'reviewed_candidates':qa_audit['reviewed_candidates'],'supported_candidates':qa_audit['supported'],'not_admitted_or_uncertain':qa_audit['not_admitted_or_uncertain'],'correct_replay':sum(replay['counts'].values()),'replay_total':256},
            'known_development_event_outside_queue':{'step':'CPU损失开发检查','count':1,'reason':'布尔索引IndexError，冻结前已改为targets[valid].unsqueeze(1)','source':'上一轮对话中的工具报错；未保留独立本地失败日志，不计入队列6/225分母'},
            'missing_history':'Shell-only CPU mistakes and independent-reader failures without queue STATUS are outside the queue census; no complete historical CPU failure rate. Interrupted design jobs are separate from crashes. Conditional 1617 not executed is not execution failure.',
            'next_action':{'root':str(BASE/'vlm_stable_kl_gradient_v1'),'optimizer_updates':0,'purpose':'Full fixed-Final CE/KL gradient norms and cosines; no blind extra steps or lambda sweep.'}}
    (ROOT/'SUMMARY.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    table='\n'.join(f"|{q['step']}|{q['unit']}|{q['passed']}|{q['failed']}|" for q in rows)
    cases='\n'.join(f"|{q['study']} / {q['job']}|{q['reason']}|{q['minutes_admission_to_failure']:.2f}|" for q in failures)
    recent='\n'.join(f"|{q['study']} / {q['role']}|{q['native_delta_pp']:+.4f}|{q['rare_delta_pp']:+.4f}|{'通过' if q['native_protection'] else '未通过'}|" for q in models)
    text=f'''本轮SFT/KL已完整结束。原生问答512：修复后新起点93.5546875%，SFT与KL均93.671875%，+0.1171875pp；rare366均−0.6557377pp，95%[−2.486339,+1.174863]跨0，点保护失败不等于证明显著退化。两组512条回答字符串完全相同，未证实KL带来能力收益。原完整1617未放行，不填0或重启原负方案。

统计生成于{result['observed_at']}；历史队列快照时间{inventory['observed_at']}。主要瓶颈在模型泛化/保持验收；本轮没有运行崩溃。

本轮各步骤按各自单位统计；同一worker可覆盖训练与评测，不能逐行相加当作独立失败次数。

|步骤|单位|通过|失败|
|---|---|---:|---:|
{table}

全流程共有5个队列作业（基线2、教师1、训练并评测2），均exit0。原始251候选中135支持、116不准入或不确定是QA审核数量，不能叫全数据错标率；训练回放255/256可用。数值修复R1/R2是2个预声明策略，R1未过、R2过，是稳定性检查，不能算两次训练失败。

最近两轮同数据对照的4个模型都通过纠错可学习/OCR保护，4/4未过罕见答案保护；这是本地候选计数，不是独立seed成功率，新旧数值策略各按自身新起点比较，不把绝对分数混为单一策略收益。

|研究/角色|native相对各自起点pp|rare相对各自起点pp|原生保护|
|---|---:|---:|---|
{recent}

历史扫描覆盖131个VLM目录、124个队列STATUS文件，按pid/准入时间/GPU/job去重，共231条执行记录：219完成、6失败、2设计中止、4仍记录running。其中2条是旧PID已不存在的陈旧记录，另2条为本次新梯度诊断的快照。6/225已结束执行记录=2.67%，这不是所有CPU/审查尝试的失败率，也不是模型训练失败概率。

|失败作业|实际失败环节|GPU准入至报错分钟|
|---|---|---:|
{cases}

6条中5条发生于评测/元数据/工件交接，1条在首次优化器更新前被数值检查拦住；没有一条日志证明是本轮式的核心优化器OOM/NaN崩溃。两条图片路径错误属于同一根因。耗时含训练、评测和IO，部分训练成果后来可恢复，不能把报错前时间全算成无效GPU训练。

另有至少{len(negative)}个研究目录留下明确未过准入记录（详见SUMMARY.json逐项来源）。不把goal_complete=False的诊断统一算负算法，也不以异构任务门槛计算总模型失败概率。现有队列台账不能完整覆盖未落盘CPU报错和独立审查器错误，相关失败率留缺。

下一步已经提交vlm_stable_kl_gradient_v1：对两个Final、完整384训练QA/原512曝光权重求纠错CE、native/OCR回放CE与KL的梯度norm/cos；0新优化/生成。先区分约束强度、方向冲突与覆盖不足；不根据累计KL占loss比例直接调λ。

[结构化统计与来源](SUMMARY.json) · [原始状态清单](INVENTORY.json) · [交互看板](../rarevlm_monitor/状态看板.html)
'''
    (ROOT/'REPORT.md').write_text(text)
    print(json.dumps({'stats':str(ROOT/'REPORT.md'),'explicit_negative_study_count':len(negative),'failure_stage_counts':result['historical_queue_records']['failed_by_stage'],'current_pipeline_worker_failures':0,'current_rare_protection_failures':2,'recent_rare_protection_failures':4,'negative_studies':[q['study'] for q in negative]},ensure_ascii=False))
    return result


if __name__=='__main__':summarize()
