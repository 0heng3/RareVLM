"""Read-only adapter for the three matched KD student roles."""
from pathlib import Path
import time


def collect(collector, root, labels, now, alive):
    read = collector.read; source = collector.source; base = root.parent
    config = read(root / 'CONFIG.json'); data = read(root / 'data/MANIFEST.json')
    completion = read(root / 'COMPLETION.json'); analysis = read(root / 'ANALYSIS.json')
    queues = {name: read(root / name / 'STATUS.json') for name in ('training_queue', 'evaluation_queue')}
    launch = read(root / 'LAUNCH.json'); errors = []; roles = []
    colors = {'hard': '#64748b', 'uniform': '#316f9e', 'kd': '#127467'}
    for role, label in labels.items():
        directory = root / 'runs' / role; progress = read(directory / 'train/PROGRESS.json')
        trained = read(directory / 'train/TRAIN_COMPLETE.json'); internal = read(directory / 'internal/PROGRESS.json')
        transfer = read(directory / 'transfer/PROGRESS.json')
        internal_done = (directory / 'internal_COMPLETE.json').exists(); transfer_done = (directory / 'transfer_COMPLETE.json').exists()
        step = trained.get('updates', progress.get('completed_updates', 0))
        job = queues['training_queue'].get('jobs', {}).get(role, {})
        evaluation = next((v for k, v in queues['evaluation_queue'].get('jobs', {}).items() if k.startswith(role + '_') and v.get('state') == 'running'), {})
        phase = '已完成' if transfer_done and internal_done else '完整评测' if evaluation else '训练已完成，待评测' if trained else '训练' if step else '等待GPU'
        roles.append({'id': role, 'label': label, 'color': colors[role], 'step': step, 'planned': 32, 'phase': phase,
            'train_seconds': trained.get('active_seconds', progress.get('active_seconds')),
            'seconds_per_update': progress.get('recent_seconds_per_update'),
            'peak_gib': trained.get('peak_allocated_gib', progress.get('peak_allocated_gib')),
            'trainable': 3686400, 'internal': internal, 'internal_done': internal_done,
            'transfer': transfer, 'transfer_done': transfer_done, 'queue': evaluation or job,
            'curve': {'points': read(directory / 'train/CURVE.json', []),
                      'source': source(directory / 'train/CURVE.json', label + '真实更新曲线')},
            'source': source(directory / 'train/PROGRESS.json', label + '训练状态')})
    for name, queue in queues.items():
        for key, job in queue.get('jobs', {}).items():
            if job.get('state') == 'failed':
                log = root / name / 'logs' / (key + '.log')
                errors.append({'job': key, 'queue': name, 'tail': log.read_text(errors='replace')[-3000:] if log.exists() else '日志待写入', 'source': source(log)})
    if not completion and alive(launch) is False and not errors:
        errors.append({'job': 'controller', 'queue': 'controller', 'tail': '控制器已退出但尚无完成记录；核对原日志，不自动重启。', 'source': source(root / 'controller.log')})
    running = any(q.get('status') == 'running' for q in queues.values())
    state_name = 'failed' if errors else ('completed' if analysis.get('student_signal') else 'negative') if completion else 'running' if running else 'waiting'
    internal_total = sum(960 if r['internal_done'] else collector.internal_count(r['internal']) for r in roles)
    full_total = sum(1617 if r['transfer_done'] else r['transfer'].get('completed', 0) for r in roles)
    def stage(identifier, label, value, total):
        return {'id': identifier, 'label': label, 'value': value, 'total': total, 'state': 'completed' if value == total else 'running' if value else 'waiting'}
    stages = [stage('baseline', '复用起点（0生成）', 2577, 2577), stage('data', '固定训练QA', 384, 384),
              stage('teacher', '复用7B软监督', 103, 103), stage('train', '三臂匹配训练', sum(r['step'] for r in roles), 96),
              stage('internal', '内部960×3', internal_total, 2880), stage('transfer', '完整1617×3', full_total, 4851)]
    baseline_metrics = {}
    for cohort in ('native_hard', 'literal_validation', 'seen_validation', 'retention', 'transfer_dev'):
        rows = read(root / 'baseline' / cohort / 'PREDICTIONS.json', [])
        if rows:
            baseline_metrics[cohort] = 100 * sum(float(q['iiit_alnum_exact']) if cohort == 'literal_validation' else q['vqa_score'] for q in rows) / len(rows)
    metrics = analysis.get('metrics', {}) or {'start': baseline_metrics}
    scores = {}
    old = read(base / 'vlm_visual_preference_v1/TRANSFER_ANALYSIS.json').get('metrics', {}).get('ce_dpo', {})
    for cohort in ('transfer_dev', 'seen_validation', 'literal_validation', 'official_test'):
        scores[cohort] = [{'id': 'start', 'label': '较强SFT共同起点', 'value': baseline_metrics.get(cohort)}]
        scores[cohort] += [{'id': role, 'label': label + '32', 'value': metrics.get(role, {}).get(cohort)} for role, label in labels.items()]
        scores[cohort].append({'id': 'old_dpo', 'label': '历史全视觉DPO', 'value': old.get(cohort)})
    failures = []
    if analysis:
        comparisons = [('对较强SFT起点', analysis['vs_start']['kd']['transfer_dev'])]
        comparisons += [('对' + labels[r], analysis['candidate_vs_controls'][r]['transfer_dev']) for r in ('hard', 'uniform')]
        comparisons.append(('对历史强DPO', analysis['candidate_vs_historical_strong_dpo']))
        for label, value in comparisons:
            if value['delta_pp'] < .5 or value['ci95'][0] <= 0:
                failures.append({'role': 'kd', 'cohort': 'transfer_dev', 'label': '完整1617 ' + label,
                    **value, 'threshold': .5, 'evidence': '固定Final32，完整配对门槛；不以训练拟合或较弱控制宣称提分。'})
        for reference, values in [('start', analysis['vs_start']['kd'])] + list(analysis['candidate_vs_controls'].items()):
            names = ('literal_validation','seen_validation','singleword','multiword','rare_1_5','retention','full_rare342') if reference=='start' else ('literal_validation','seen_validation','rare_1_5','retention','full_rare342')
            for name in names:
                value = values[name]
                if value['delta_pp'] < -.5:
                    failures.append({'role':'kd','cohort':name,'label':name+' 对'+reference,**value,
                        'threshold':-.5,'evidence':'预定点保护失败；不将跨零区间称为显著退化。'})
    teacher = read(base / 'vlm_full_teacher_signal_v1/COMPLETION.json')
    history = [{'name': '完整7B教师准入', 'status': '五门槛通过', 'detail': f"已见训练128题：教师{teacher.get('teacher_literal_correct')}/学生{teacher.get('student_literal_correct')}；新增纠正71、退化2，73题有结构信息；不是学生提分。",
        'source': source(base / 'vlm_full_teacher_signal_v1/COMPLETION.json')}]
    sources = [source(root / 'STUDY_INPUT_LOCK.json', '本轮科学输入锁'), source(root / 'CONFIG.json', '三臂同预算配置'),
        source(root.parents[1] / 'packages/RareVLM_minimal_20261001/docs/VLM_FULL_TEACHER_KD_V1_PROTOCOL.md', '蒸馏学生固定协议'),
        source(root / 'teacher/CACHE_INDEX.json', '103题既有教师缓存、0新教师前向'),
        source(root.parents[1] / 'packages/RareVLM_minimal_20261001/docs/VLM_FULL_TEACHER_KD_V1_STATUS.md','三组实际进度与范围说明'),
        source(base / 'vlm_full_teacher_signal_v1/教师监督_真实结果.png', '实际教师强弱、纠正和退化图像'),
        source(root / 'COMPLETION.json', '本轮最终能力验收')]
    review = read(base / 'vlm_native_correction_pool_v2/REVIEWED_ERRORS.json')
    cases = [q for q in collector.cases(data) if q['group'] == '评测变化'] if any(r['internal_done'] for r in roles) else []
    research = ('KD完整门槛通过，仍待三seed与外部确认' if analysis.get('student_signal') else 'KD完整门槛未过，停止本配置') if completion else (f'三臂训练完成；完整评测{internal_total+full_total}/7731' if all(r['step']==32 for r in roles) else '7B→3B：GT/均匀尾/教师KD三臂训练，能力待完整评测')
    state = {'schema': 1, 'observed_at': now(), 'mode': 'snapshot', 'project': 'RareVLM', 'study': root.name,
        'state': state_name, 'research': research, 'controller': {'pid': launch.get('pid'), 'alive': alive(launch)},
        'runtime_errors': errors, 'completion': completion, 'roles': roles, 'stages': stages, 'gpu': collector.gpu(),
        'screen': analysis, 'analysis': {**analysis, 'metrics': metrics}, 'failures': failures,
        'gates': [{'role': 'kd', 'key': k, 'passed': v} for k, v in analysis.get('gates', {}).items()],
        'scores': scores, 'history': history, 'sources': sources, 'batch_diagnostic': {}, 'auxiliary_runs': [],
        'independent_validation': {}, 'source_holdout_validation': {}, 'full_teacher_validation': {},
        'regularizer': 'kd', 'numeric_policy': config['policy'],
        'active_jobs': [{'label': name.replace('_queue', '') + ' / ' + key, **job} for name, q in queues.items() for key, job in q.get('jobs', {}).items() if job.get('state') == 'running'],
        'dataset': {'reviewed': review.get('reviewed_candidates'), 'supported': review.get('supported'),
                    'excluded': review.get('not_admitted_or_uncertain'), 'selected': 128, 'unique_targets': 127, 'overlap': 0},
        'cases': cases, 'warnings': collector.warnings,
        'scope': 'Read-only matched KD research; all accuracy values await complete Final evaluation; teacher training-pool result is separate.'}
    # Auxiliary diagnostic: original three-arm training stays complete and negative.
    restore_root = base / 'vlm_kd_component_restore_v1'
    restore_plan = read(restore_root / 'STUDY_INPUT_LOCK.json')
    if restore_plan:
        restore_queue = read(restore_root / 'queue/STATUS.json')
        restore_done = read(restore_root / 'COMPLETION.json')
        restore_counts = {}; restore_labels = {'language_only': '保留语言更新', 'visual_only': '保留视觉更新'}
        for role, label in restore_labels.items():
            directory = restore_root / 'runs' / role
            count = sum(len(read(directory / cohort / 'PREDICTIONS.json', [])) for cohort in restore_plan['cohorts'])
            restore_counts[role] = count
            job = restore_queue.get('jobs', {}).get(role, {})
            if job.get('state') == 'running':
                state['active_jobs'].append({'label': '零训练恢复 / ' + label, **job})
            if job.get('state') == 'failed':
                log = restore_root / 'queue/logs' / (role + '.log')
                state['runtime_errors'].append({'job': role, 'queue': 'component_restore',
                    'tail': log.read_text(errors='replace')[-3000:] if log.exists() else '日志待写入', 'source': source(log)})
        total = sum(restore_counts.values())
        detail = f'原KD阴性保持；新零训练恢复{total}/3618，0更新/0新adapter，不作新能力候选。'
        state['auxiliary_runs'].append({'name': '语言/视觉参数恢复', 'status': restore_queue.get('status', 'waiting'),
            'detail': detail, 'jobs': restore_queue.get('jobs', {}), 'source': source(restore_root / 'STUDY_INPUT_LOCK.json')})
        state['history'].append({'name': '已执行下一步', 'status': '零训练机制定位', 'detail': detail,
            'source': source(restore_root / 'STUDY_INPUT_LOCK.json')})
        state['component_restore_validation'] = {'completion': restore_done, 'counts': restore_counts, 'expected': 3618}
        state['research'] = 'KD未过；组件恢复已完成' if restore_done else f'KD未过；零训练恢复{total}/3618'
        state['stages'] = [stage('train', '原三臂训练', 96, 96), stage('internal', '原内部评测', 2880, 2880),
            stage('transfer', '原完整评测', 4851, 4851), stage('reused', '复用两端点', 3618, 3618),
            stage('language_restore', '保留语言更新', restore_counts['language_only'], 1809),
            stage('visual_restore', '保留视觉更新', restore_counts['visual_only'], 1809)]
        state['sources'] += [source(restore_root / 'STUDY_INPUT_LOCK.json', '新零训练消融登记'),
            source(restore_root / 'queue/STATUS.json', '组件恢复实际队列'),
            source(restore_root / 'ANALYSIS.json', '组件恢复完整结果'),
            source(restore_root / 'result_review_v1/组件恢复_结果与真实案例.png', '组件恢复实际图像与完整结果'),
            source(root / 'result_review_v1/真实结果与回答变化.png', '三臂完整结果与真实案例')]
        if state['runtime_errors']:
            state['state'] = 'failed'
        state['scope'] = 'Completed negative KD plus separately registered zero-update component diagnosis; no automatic training or model promotion.'
    teacher_root = base / 'vlm_full_teacher_transfer_v1'
    teacher_plan = read(teacher_root / 'PLAN.json')
    if teacher_plan:
        teacher_queue = read(teacher_root / 'queue/STATUS.json'); teacher_done = read(teacher_root / 'COMPLETION.json')
        teacher_counts = {}
        for shard, cohorts in teacher_plan['shards'].items():
            directory = teacher_root / 'shards' / shard
            done = read(directory / 'COMPLETE.json'); progress = read(directory / 'PROGRESS.json')
            teacher_counts[shard] = done.get('completed', progress.get('completed', 0))
            job = teacher_queue.get('jobs', {}).get('teacher_' + shard, {})
            if job.get('state') == 'running':
                state['active_jobs'].append({'label': '7B完整任务 / 分片' + shard, **job})
            if job.get('state') == 'failed':
                log = teacher_root / 'queue/logs' / ('teacher_' + shard + '.log')
                state['runtime_errors'].append({'job': 'teacher_' + shard, 'queue': 'full_teacher_transfer',
                    'tail': log.read_text(errors='replace')[-3000:] if log.exists() else '日志待写入', 'source': source(log)})
        total = sum(teacher_counts.values())
        state['research'] = 'KD阴性；7B完整验证已结束' if teacher_done else f'KD阴性；7B完整验证{total}/1681'
        state['stages'] = [stage('old_eval', '三臂评测完成', 7731, 7731), stage('restore', '组件恢复完成', 3618, 3618)]
        state['stages'] += [stage('teacher_' + shard, '7B分片' + shard, teacher_counts[shard], sum(cohorts.values())) for shard, cohorts in teacher_plan['shards'].items()]
        state['full_teacher_transfer_validation'] = {'completion': teacher_done, 'counts': teacher_counts, 'expected': 1681}
        detail = f'完整1617+保持64，实际{total}/1681生成；0训练/0GT前向，原7B资产复用，不能当作学生提分。'
        state['auxiliary_runs'].append({'name': '7B完整任务教师验证', 'status': teacher_queue.get('status', 'waiting'),
            'detail': detail, 'jobs': teacher_queue.get('jobs', {}), 'source': source(teacher_root / 'PLAN.json')})
        state['history'].append({'name': '已执行下一步', 'status': teacher_queue.get('status', 'waiting'), 'detail': detail, 'source': source(teacher_root / 'PLAN.json')})
        state['sources'] += [source(teacher_root / 'PLAN.json', '7B完整教师验证登记'),
            source(teacher_root / 'queue/STATUS.json', '7B四分片实际队列'), source(teacher_root / 'ANALYSIS.json', '7B完整任务最终结果')]
        if state['runtime_errors']:
            state['state'] = 'failed'
        state['scope'] = 'Negative student KD and completed component restoration; new zero-update 7B task evaluation is not a student improvement.'
    collector.cached_state = state; collector.cached_at = time.monotonic()
    return state
