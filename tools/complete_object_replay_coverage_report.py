#!/usr/bin/env python
"""Export terminal replay-coverage metadata, without model/tensor/scoring work."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
PRIVATE=Path('/opt/data/private')
TERMINAL={'completed_negative','engineering_rejected','confirmation_passed_pending_independent_review'}


class PendingResult(RuntimeError):
    pass


def digest(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def absolute(value):
    file=Path(value)
    if not file.is_absolute() or PRIVATE not in file.resolve().parents:
        raise ValueError(f'absolute persistent path required: {value}')
    return file.resolve()


class Metadata:
    """Read each JSON/JSONL once; never open weights, tensors, images, predictions."""
    def __init__(self):
        self.cache={};self.hashes={}

    def read(self,file,expected=None):
        file=absolute(file)
        if file.name=='predictions.jsonl' or file.suffix not in ('.json','.jsonl'):
            raise ValueError(f'metadata-only exporter refuses this file: {file}')
        if file not in self.cache:
            raw=file.read_bytes();self.hashes[str(file)]=hashlib.sha256(raw).hexdigest()
            self.cache[file]=([json.loads(line) for line in raw.splitlines()] if file.suffix=='.jsonl' else json.loads(raw))
        if expected is not None and self.hashes[str(file)]!=expected:
            raise ValueError(f'locked metadata SHA differs: {file}')
        return self.cache[file]

    def verify_unchanged(self):
        for name,expected in self.hashes.items():
            if digest(Path(name))!=expected:raise ValueError(f'metadata changed during export: {name}')


def pick(value,keys):
    return {key:value[key] for key in keys if key in value}


def all_files(root,pattern):
    return sorted(root.glob(pattern))


def overall(rows):
    return next(row for row in rows if row['slice']=='overall')


def source_guard(study,config_file,config):
    guarded=dict(study['source_sha256'])
    for file in (config_file,absolute(config['study_protocol']),absolute(config['dataset_protocol']),
                 absolute(config['cpu_contracts']),absolute(config['output_root'])/'STUDY_INPUT_LOCK.json',
                 absolute(config['output_root'])/'STUDY.json'):
        guarded[str(file)]=digest(file)
    for name,expected in guarded.items():
        # These are source/config/registration metadata only. Model provenance
        # remains the recorded frozen STUDY binding; weight files are not read.
        if Path(name).suffix not in ('.py','.json','.md') or digest(Path(name))!=expected:
            raise ValueError(f'frozen source/config metadata changed: {name}')
    return guarded


def training_metadata(reader,root):
    result=[]
    for file in all_files(root,'runs/*/seed*/train/TRAIN_RESULT.json'):
        train=reader.read(file)
        if (train.get('status')!='completed' or train.get('schema_version')!='rarevlm.object_replay_coverage.train.v1'
            or train.get('condition') not in ('wide32','repeat8x4')
            or (train['condition']=='repeat8x4' and train['seed']!=42)):
            raise ValueError('unexpected completed coverage training metadata')
        source=reader.read(file.parent/'TEACHER_SOURCE.json',train['teacher_source_file_sha256'])
        if source!=train['teacher_source']:raise ValueError('teacher source metadata differs from completed training')
        result.append({'file':str(file),'sha256':reader.hashes[str(file)],
            **pick(train,('condition','seed','completed_updates','planned_updates','skipped_updates','actual_object_exposures',
                'actual_replay_exposures','actual_questions_seen','replay_questions_per_update','replay_coverage',
                'training_microbatch_questions','training_accumulation','object_effective_batch','trainable_parameters',
                'lora_parameter_precision','visual_token_budget','kl_temperature','replay_lambda','mean_object_ce',
                'mean_replay_kl','mean_replay_ce','mean_replay_ce_scope','training_active_seconds','model_load_seconds',
                'teacher_reference_setup_wall_seconds','preflight_seconds_current_session','session_wall_seconds',
                'resumed_at_update','peak_gpu_allocated_gib','peak_gpu_reserved_gib','compute_accounting')),
            'teacher_source':source})
    return result


def costs_metadata(reader,root,training):
    histories=all_files(root,'engineering_runs/runs/*/seed*/train/PREFLIGHT_HISTORY.jsonl')
    histories+=all_files(root,'runs/*/seed*/train/PREFLIGHT_HISTORY.jsonl')
    engineering=[]
    for file in histories:
        for index,row in enumerate(reader.read(file)):
            engineering.append({'file':str(file),'attempt_index_in_file':index,
                **pick(row,('status','condition','reference_scope','discarded_engineering_optimizer_updates',
                    'formal_optimizer_updates','initial_reference_parity_questions','after_temporary_update_reference_probe_questions',
                    'engineering_actor_forward_backward_questions','actual_new_teacher_forward_questions',
                    'historical_teacher_bank_questions','peak_allocated_gib','peak_reserved_gib','memory_cap_gib',
                    'memory_cap_passed','completed_at','error'))})
    teacher_uses=[];fresh={};historical={}
    source_files=all_files(root,'engineering_runs/runs/*/seed*/train/TEACHER_SOURCE.json')
    source_files+=all_files(root,'runs/*/seed*/train/TEACHER_SOURCE.json')
    for file in source_files:
        source=reader.read(file);teacher_uses.append({'file':str(file),**source})
        key=(source['source_manifest_path'],source['source_manifest_sha256'])
        if source['actual_new_teacher_forward_questions']:
            value=pick(source,('source_manifest_path','source_manifest_sha256','scope','kind','producer_gpu_uuid',
                'actual_new_teacher_forward_questions','actual_new_teacher_forward_seconds'))
            if key in fresh and fresh[key]!=value:raise ValueError('same new teacher manifest has conflicting cost metadata')
            fresh[key]=value
        if source['historical_teacher_bank_questions']:
            value=pick(source,('source_manifest_path','source_manifest_sha256','producer_gpu_uuid',
                              'historical_teacher_bank_questions','historical_teacher_bank_forward_seconds'))
            if key in historical and historical[key]!=value:raise ValueError('same historical teacher bank has conflicting costs')
            historical[key]=value
    evaluations=[]
    for file in all_files(root,'runs/*/seed*/*/EVALUATION.json'):
        value=reader.read(file)
        if (value.get('schema_version')!='rarevlm.object_replay_coverage.eval_cell.v1' or value.get('status')!='completed'
            or value.get('condition') not in ('wide32','repeat8x4')):
            raise ValueError('only completed new-arm formal evaluations may contribute new generation counts')
        evaluations.append({'file':str(file),'sha256':reader.hashes[str(file)],**pick(value,('condition','seed','task',
            'questions','unique_images','score_percent','resources','session_id','resumed_questions',
            'current_session_wall_seconds','predictions_sha256','input_lock_sha256'))})
    eval_preflights=[]
    for file in all_files(root,'preflight/*/seed*/*/PREFLIGHT.json'):
        value=reader.read(file)
        eval_preflights.append({'file':str(file),'status':value['status'],'condition':file.parents[2].name,
            'seed':int(file.parents[1].name.removeprefix('seed')),'task':file.parent.name,
            'persisted_generation_questions':len(value['records']),'batches':value['batches'],
            **pick(value,('memory_cap_gib','memory_cap_passed','error'))})
    calls=[]
    for file in all_files(root,'workers/*/attempts/*.call.json'):
        value=reader.read(file)
        calls.append({'file':str(file),**pick(value,('phase','command','returncode','completed_at','log_file',
            'log_file_sha256','log_start_byte','log_end_byte','log_segment_sha256','fresh_failure_receipts'))})
    failures=[]
    for pattern in ('runs/*/seed*/train/teacher_cache/BUILD_FAILURE.json',
                    'engineering_runs/runs/*/seed*/train/teacher_cache/BUILD_FAILURE.json'):
        for file in all_files(root,pattern):failures.append({'file':str(file),'metadata':reader.read(file)})
    successful=[row for row in engineering if row['status']=='passed']
    return {'formal_optimizer_updates':sum(t['completed_updates'] for t in training),
        'formal_object_teacher_forced_forward_backward_questions':sum(t['actual_object_exposures'] for t in training),
        'formal_replay_teacher_forced_forward_backward_questions':sum(t['actual_replay_exposures'] for t in training),
        'formal_actor_teacher_forced_forward_backward_questions':sum(t['actual_questions_seen'] for t in training),
        'discarded_engineering_optimizer_updates_all_recorded_attempts':sum(row['discarded_engineering_optimizer_updates'] for row in engineering),
        'planned_pilot_discarded_engineering_optimizer_updates':4,
        'successful_engineering_actor_forward_backward_questions':sum(row['engineering_actor_forward_backward_questions'] for row in successful),
        'successful_engineering_initial_reference_forward_only_questions':sum(row['initial_reference_parity_questions'] for row in successful),
        'successful_engineering_post_temporary_update_reference_forward_only_questions':sum(row['after_temporary_update_reference_probe_questions'] for row in successful),
        'unsuccessful_engineering_attempts':sum(row['status']!='passed' for row in engineering),
        'new_teacher_forward_only_questions_unique_manifests':sum(row['actual_new_teacher_forward_questions'] for row in fresh.values()),
        'new_teacher_forward_seconds_unique_manifests':sum(row['actual_new_teacher_forward_seconds'] for row in fresh.values()),
        'historical_teacher_bank_questions_unique_manifests':sum(row['historical_teacher_bank_questions'] for row in historical.values()),
        'historical_teacher_bank_new_forward_questions':0,
        'formal_new_generation_model_x_QA_persisted_complete_cells':sum(row['questions'] for row in evaluations),
        'engineering_generation_model_x_QA_persisted_records':sum(row['persisted_generation_questions'] for row in eval_preflights),
        'engineering_finite_logit_forward_only_QA_passed_checks':sum(row['persisted_generation_questions'] for row in eval_preflights if row['status']=='passed'),
        'subprocess_attempts':len(calls),'subprocess_failures':sum(row['returncode']!=0 for row in calls),
        'preflight_attempts':engineering,'teacher_source_uses':teacher_uses,'new_teacher_manifests':list(fresh.values()),
        'historical_teacher_banks':list(historical.values()),'formal_evaluations':evaluations,
        'generation_preflights':eval_preflights,'subprocess_calls':calls,'retained_teacher_build_failures':failures,
        'accounting_limits':['QA counts are model/computation exposures, not independent examples or equal-FLOPs evidence',
            'historical bank is deduplicated by original manifest; its1448QA are not new teacher forward calls',
            'planned pilot4 discarded steps are separate from actual history, retries and extra seed preflights',
            'failed or interrupted engineering attempts may have partial compute not inferable from completed metadata',
            'formal generation counts cover persisted complete cells; abandoned tails are not invented',
            'forward-only reference checks, finite-logit checks, generation and backward passes remain separate']}


def collect(config_file):
    reader=Metadata();config_file=absolute(config_file);config=reader.read(config_file)
    root=absolute(config['output_root']);complete_file=root/'COMPLETION.json'
    if not complete_file.exists():
        raise PendingResult(f'尚无终态 COMPLETION.json：{root}；未写完成报告。')
    complete=reader.read(complete_file)
    if complete.get('status') not in TERMINAL:
        raise PendingResult(f"当前状态 {complete.get('status')!r} 不是受支持终态；未写完成报告。")
    if (complete.get('schema_version')!='rarevlm.object_replay_coverage.trial.v1'
        or complete.get('goal_complete') is not False or complete.get('candidate')!='wide32'
        or complete.get('control_can_be_selected') is not False):
        raise ValueError('terminal coverage/candidate/goal identity differs')
    study=reader.read(root/'STUDY_INPUT_LOCK.json',complete['study_input_lock_sha256'])
    if study['config_file_sha256']!=reader.hashes[str(config_file)]:raise ValueError('study configuration bytes changed')
    reader.read(root/'STUDY.json')
    guarded=source_guard(study,config_file,config)
    analyses=[]
    for name,expected in complete['analysis_files_sha256'].items():
        value=reader.read(Path(name),expected)
        if value.get('status')!='complete' or value.get('candidate')!='wide32':raise ValueError('official full analysis incomplete')
        analyses.append({'file':name,'sha256':expected,'analysis':value})
    def stage_order(item):
        value=item['analysis'];phase=value['schema_version'].split('.')[-2]
        return (0 if value['seeds']==[42] else 1) if phase=='screen' else 2 if phase=='external' else 3
    analyses.sort(key=stage_order)
    if complete['status']=='completed_negative' and (not analyses or analyses[-1]['analysis']['gate']['passed']):
        raise ValueError('negative status lacks the corresponding failed original gate')
    if complete['status']=='confirmation_passed_pending_independent_review' and (
        not complete.get('confirmation_passed') or not analyses or not analyses[-1]['analysis']['gate']['passed']):
        raise ValueError('pending-review confirmation lacks passing original evidence')
    training=training_metadata(reader,root);costs=costs_metadata(reader,root,training)
    historical=[]
    for task in ('object','text','external','pope'):
        file=root/'runs/initial/seed42'/task/'HISTORICAL_REFERENCE.json'
        reference=reader.read(file)
        if reference.get('new_generation_questions')!=0:raise ValueError('all initial baselines must be historical')
        source=reference['source_data'];evaluation=reader.read(Path(source['evaluation_file']),source['evaluation_sha256'])
        historical.append({'task':task,'file':str(file),'sha256':reader.hashes[str(file)],
            'original_questions':evaluation.get('questions',evaluation.get('n')),'new_generation_questions':0,
            'source_data':source,'original_hardware_and_source_retained':True})
    reviews=[]
    for file in all_files(root,'independent_review/*.json'):
        value=reader.read(file)
        reviews.append({'file':str(file),'sha256':reader.hashes[str(file)],
            **pick(value,('schema_version','status','scope','screen_passed','goal_complete','checks',
                          'protected_file_count','protected_bytes_unchanged_before_after'))})
    launch=reader.read(root/'LAUNCH.json') if (root/'LAUNCH.json').exists() else {}
    recovery=reader.read(root/'RECOVERY.json') if (root/'RECOVERY.json').exists() else None
    wall={'launch':launch,'completion_at':complete.get('completed_at'),
          'current_controller_session_seconds':complete.get('current_controller_session_seconds'),
          'scope':'queue/loading/training/evaluation/CPU/writer wall time; never pure GPU or serving latency'}
    if launch.get('launched_at') and complete.get('completed_at'):
        wall['launch_to_terminal_wall_seconds']=(datetime.fromisoformat(complete['completed_at'])-datetime.fromisoformat(launch['launched_at'])).total_seconds()
    if recovery:wall['cpu_recovery']=pick(recovery,('original_workflow_started_at','original_failure_at','original_workflow_wall_seconds','cpu_recovery_seconds'))
    metrics={'schema_version':'rarevlm.object_replay_coverage.report.v1','exported_at':datetime.now(timezone.utc).isoformat(),
        'status':complete['status'],'last_phase':complete['last_phase'],'candidate':'wide32','exploratory_control':'repeat8x4',
        'control_eligible':False,'goal_complete':False,'confirmation_passed':complete.get('confirmation_passed',False),
        'root_independent_prediction_review_required':True,'completion':complete,'analyses':analyses,
        'training':training,'costs':costs,'historical_initial_references':historical,'independent_reviews':reviews,
        'wall_time':wall,'equal_teacher_compute_claim':False,'exporter_model_loads':0,'exporter_model_forwards':0,
        'exporter_gpu_operations':0,'exporter_new_predictions':0,'exporter_optimizer_updates':0,
        'statistical_scope':'read original gates and CIs only; paired image clusters143/1024/internal or1791/external or500/sharedPOPE; fixed-seed means then clusters, sampleSD separate',
        'metadata_files_sha256':dict(reader.hashes),'frozen_source_config_metadata_sha256':guarded}
    reader.verify_unchanged()
    for name,expected in guarded.items():
        if digest(Path(name))!=expected:raise ValueError('frozen scientific text changed during export')
    return metrics,reader,guarded


def number(value,digits=3):
    return '—' if value is None else f'{value:.{digits}f}'


def markdown(metrics):
    status=metrics['status']
    meaning={'completed_negative':'本轮在原预注册科学门槛处停止，未达到真实提升验收。',
        'engineering_rejected':'本轮在工程门控处停止；工程失败不能记作准确率负结果。',
        'confirmation_passed_pending_independent_review':'量化确认门槛已通过，仍待父代理独立逐题预测与病例审查。'}[status]
    lines=['# 回放覆盖对照 v1：真实结果', '',f'状态：`{status}`。{meaning} `goal_complete=false`。','',
        '唯一候选为 wide32；repeat8x4 仅是 seed42 探索对照，不能替补候选或追加失败控制的种子。两组物体流、Final 更新规则与每步32条 replay 曝光相同；wide32 扩展 unique 覆盖，control 重复原8条四次。两组教师新建成本和信息覆盖不同，不作等教师计算或等 FLOPs 声明。','',
        '以下点值、区间和门槛直接读取原 ANALYSIS，不重评分或重算 bootstrap。四个主端点只属于 wide32，alpha=.05/4；按图像整簇、6000次抽样，单侧98.75%下界。顺序 seed 先均值后抽图，sample SD 另列；model×QA 不能视为独立样本。','']
    for item in metrics['analyses']:
        analysis=item['analysis'];lines.extend([f"官方阶段：`{analysis['stage']}`，seeds={analysis['seeds']}。",'',
            '| 端点 | initial % | wide32 % | Δ pp | 95%图簇CI | 单侧98.75%下界 |',
            '|---|---:|---:|---:|---|---:|'])
        for task,rows in analysis['contrasts'].items():
            selected=[overall(rows)]
            if task=='text':selected += [r for r in rows if r['slice'] in ('rare_1_5','frequent_gt20')]
            for row in selected:
                ci=row['image_cluster_bootstrap_95_pp']
                lines.append(f"| {task}/{row['slice']} | {number(row['before_score_percent'])} | {number(row['after_score_percent'])} | {number(row['delta_pp'])} | {('['+number(ci[0])+', '+number(ci[1])+']') if ci is not None else '—'} | {number(row['one_sided_98_75_lower_pp'])} |")
        lines.extend(['',f"官方 gate：`{analysis['gate']['passed']}`。",''])
        for name,passed in analysis['gate']['checks'].items():lines.append(f'- `{name}`：`{passed}`')
        if 'class_balanced' in analysis:
            row=overall(analysis['class_balanced']['contrasts'])
            lines.extend(['',f"逐类 yes/no 均衡固定子集：{row['questions']} QA，{row['image_clusters']} 图，Δ={number(row['delta_pp'])}pp。每类正负相等不表示各类题数相等；该子集不替代完整主端点，也不是宏平均。"])
        if analysis.get('exploratory_control_contrasts'):
            lines.extend(['','单 seed 探索对照（不用于选择候选）：','','| 比较 | Object Δ pp | Text Δ pp |','|---|---:|---:|'])
            for label,parts in analysis['exploratory_control_contrasts']['contrasts'].items():
                lines.append(f"| {label} | {number(overall(parts['object'])['delta_pp'])} | {number(overall(parts['text'])['delta_pp'])} |")
        lines.extend(['',f"原证据：[{Path(item['file']).name}]({item['file']})。",''])
    lines.extend(['rare/frequent 是原32985条 train_all 实际监督目标的答案频次代理；点值保护与总体 Text .5pp 非劣检验分别报告。切片/类别区间是条件于固定 checkpoint 的探索区间，不称切片非劣或预训练未见。','',
        '| 已完成训练 | seed | object QA | replay QA | replay unique | 正式更新 | teacher新前向QA | teacher历史bankQA |',
        '|---|---:|---:|---:|---:|---:|---:|---:|'])
    for t in metrics['training']:
        src=t['teacher_source'];lines.append(f"| {t['condition']} | {t['seed']} | {t['actual_object_exposures']} | {t['actual_replay_exposures']} | {t['replay_coverage']['unique_questions_consumed']} | {t['completed_updates']} | {src['actual_new_teacher_forward_questions']} | {src['historical_teacher_bank_questions']} |")
    costs=metrics['costs']
    lines.extend(['','| 实际成本项 | 数量 |','|---|---:|'])
    names={'formal_optimizer_updates':'正式优化器更新','discarded_engineering_optimizer_updates_all_recorded_attempts':'实际记录的临时工程更新（已丢弃）',
        'formal_object_teacher_forced_forward_backward_questions':'正式物体teacher-forced QA前后向',
        'formal_replay_teacher_forced_forward_backward_questions':'正式回放teacher-forced QA前后向',
        'new_teacher_forward_only_questions_unique_manifests':'新教师缓存QA前向（按manifest去重）',
        'historical_teacher_bank_questions_unique_manifests':'历史教师bank QA（按manifest去重）',
        'historical_teacher_bank_new_forward_questions':'历史bank在本研究新增教师前向',
        'successful_engineering_actor_forward_backward_questions':'通过工程预检的临时actor QA前后向',
        'successful_engineering_initial_reference_forward_only_questions':'工程初始reference QA前向',
        'successful_engineering_post_temporary_update_reference_forward_only_questions':'工程临时更新后reference QA前向',
        'formal_new_generation_model_x_QA_persisted_complete_cells':'正式新生成model×QA（完整保存cell）',
        'engineering_generation_model_x_QA_persisted_records':'预检新生成model×QA',
        'engineering_finite_logit_forward_only_QA_passed_checks':'通过的无标签finite-logit QA前向',
        'subprocess_attempts':'记录的子进程调用','subprocess_failures':'记录的子进程失败'}
    for key,label in names.items():lines.append(f'| {label} | {costs[key]} |')
    lines.extend(['','计划 pilot 的4个丢弃工程更新仅是计划数；上表来自全部实际 history，追加种子或重试会单列增加。历史1448题 bank 的原计算保留为历史，不能与本轮新教师 forward 相加。失败/中断尝试的未保存计算不凭计划推算，完整 calls 与失败 metadata 保留在便携 JSON。','',
        '| 训练臂/seed | active 秒 | teacher setup壁钟秒 | allocated GiB | reserved GiB |','|---|---:|---:|---:|---:|'])
    for t in metrics['training']:
        lines.append(f"| {t['condition']}/{t['seed']} | {number(t.get('training_active_seconds'))} | {number(t.get('teacher_reference_setup_wall_seconds'))} | {number(t.get('peak_gpu_allocated_gib'))} | {number(t.get('peak_gpu_reserved_gib'))} |")
    wall=metrics['wall_time']
    lines.extend(['',f"原启动至终态壁钟：{number(wall.get('launch_to_terminal_wall_seconds'))}秒；当前控制器 session：{number(wall.get('current_controller_session_seconds'))}秒。均包含相应队列、加载、训练、评测或CPU阶段，不作纯GPU时间／线上延迟解释。"])
    if wall.get('cpu_recovery'):lines.extend(['',f"独立CPU补报时间单列：`{json.dumps(wall['cpu_recovery'],ensure_ascii=False)}`。"])
    lines.extend(['','历史 initial 的 Object/Text/external/POPE 均只引用原完整预测，本轮新增 baseline 生成0题。独立审查只按已有收据链接，不重做旧审查。',''])
    for review in metrics['independent_reviews']:lines.append(f"- [{Path(review['file']).name}]({review['file']})，SHA `{review['sha256']}`。")
    lines.extend(['',f"完整数值与成本元数据：[便携 JSON]({REPO/'results/vlm_object_replay_coverage_v1_metrics.json'})。导出器仅标准库读取元数据，0模型加载、0forward、0GPU、0新增预测或更新。",''])
    return '\n'.join(lines)


def export(config_file):
    metrics,reader,guarded=collect(config_file)
    text=markdown(metrics)
    destination=REPO/'docs/VLM_OBJECT_REPLAY_COVERAGE_V1_RESULTS.md'
    portable=REPO/'results/vlm_object_replay_coverage_v1_metrics.json'
    destination.parent.mkdir(parents=True,exist_ok=True);portable.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(text,encoding='utf-8')
    portable.write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    reader.verify_unchanged()
    for name,expected in guarded.items():
        if digest(Path(name))!=expected:raise ValueError('frozen scientific source changed during report write')
    root=absolute(reader.read(absolute(config_file))['output_root'])
    receipt={'schema_version':'rarevlm.object_replay_coverage.report_export.v1','status':'completed',
        'completion_sha256':reader.hashes[str(root/'COMPLETION.json')],
        'source_file':str(Path(__file__).resolve()),'source_sha256':digest(Path(__file__).resolve()),
        'artifacts_sha256':{str(destination):digest(destination),str(portable):digest(portable)},
        'read_metadata_files':len(reader.hashes),'guarded_source_config_metadata_files':len(guarded),
        'metadata_and_frozen_scientific_text_unchanged':True,'model_loads':0,'model_forwards':0,
        'gpu_operations':0,'new_predictions':0,'optimizer_updates':0,'goal_complete':False}
    (root/'REPORT_EXPORT.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=REPO/'configs/vlm_object_replay_coverage_v1.json')
    args=parser.parse_args()
    try:receipt=export(args.config)
    except PendingResult as error:
        print(str(error),file=sys.stderr);return 2
    print(json.dumps(receipt,ensure_ascii=False),flush=True);return 0


if __name__=='__main__':raise SystemExit(main())
