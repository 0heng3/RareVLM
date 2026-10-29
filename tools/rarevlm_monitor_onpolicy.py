"""Read-only view for the new student-prefix KD role; no historical steps reused."""
import time


def collect(collector, root, labels, now, alive):
    read = collector.read; source = collector.source; base = root.parent; role = 'op_kd'
    data = read(root / 'data/MANIFEST.json'); config = read(root / 'CONFIG.json')
    directory = root / 'runs' / role
    trained = read(directory / 'train/TRAIN_COMPLETE.json'); progress = read(directory / 'train/PROGRESS.json')
    internal = read(directory / 'internal/PROGRESS.json'); transfer = read(directory / 'transfer/PROGRESS.json')
    internal_done = (directory / 'internal_COMPLETE.json').exists(); transfer_done = (directory / 'transfer_COMPLETE.json').exists()
    analysis = read(root / 'ANALYSIS.json'); completion = read(root / 'COMPLETION.json'); launch = read(root / 'LAUNCH.json')
    queues = {n: read(root / n / 'STATUS.json') for n in ('training_queue', 'evaluation_queue')}
    jobs = [{'label': name.replace('_queue', '') + '/' + key, **job} for name, queue in queues.items() for key, job in queue.get('jobs', {}).items() if job.get('state') == 'running']
    errors = []
    for name, queue in queues.items():
        for key, job in queue.get('jobs', {}).items():
            if job.get('state') == 'failed':
                log = root / name / 'logs' / (key + '.log')
                errors.append({'job': key, 'queue': name, 'tail': log.read_text(errors='replace')[-3000:] if log.exists() else '日志待写入', 'source': source(log)})
    if not completion and alive(launch) is False and not errors:
        errors.append({'job': 'controller', 'queue': 'controller', 'tail': '控制器已退出但没有完成记录；查看原日志，不自动重启。', 'source': source(root / 'controller.log')})
    step = trained.get('updates', progress.get('completed_updates', 0))
    phase = '已完成' if internal_done and transfer_done else '完整评测' if trained else '训练' if step else '加载学生与教师' if jobs else '等待空闲GPU'
    roles = [{'id': role, 'label': '学生前缀KD', 'color': '#127467', 'step': step, 'planned': 32, 'phase': phase,
        'train_seconds': trained.get('active_seconds', progress.get('active_seconds')),
        'seconds_per_update': progress.get('recent_seconds_per_update'), 'peak_gib': trained.get('peak_allocated_gib', progress.get('peak_allocated_gib')),
        'trainable': 3686400, 'internal': internal, 'internal_done': internal_done, 'transfer': transfer, 'transfer_done': transfer_done,
        'queue': jobs[0] if jobs else {}, 'curve': {'points': read(directory / 'train/CURVE.json', []), 'source': source(directory / 'train/CURVE.json')},
        'source': source(directory / 'train/PROGRESS.json')}]
    state_name = 'failed' if errors else ('completed' if analysis.get('student_signal') else 'negative') if completion else 'running' if jobs else 'waiting'
    metrics = analysis.get('metrics', {})
    old = read(base / 'vlm_full_teacher_kd_v1/ANALYSIS.json').get('metrics', {})
    teacher = read(base / 'vlm_full_teacher_transfer_v1/ANALYSIS.json')
    if not metrics:
        metrics = {'start': old.get('start', {})}
    old_dpo = read(base / 'vlm_visual_preference_v1/TRANSFER_ANALYSIS.json').get('metrics', {}).get('ce_dpo', {})
    scores = {}
    for cohort in ('transfer_dev', 'seen_validation', 'literal_validation', 'official_test'):
        scores[cohort] = [{'id': 'start', 'label': '强SFT起点', 'value': metrics.get('start', {}).get(cohort)},
            {'id': 'old_kd', 'label': '旧GT前缀KD', 'value': old.get('kd', {}).get(cohort)},
            {'id': role, 'label': '新学生前缀KD', 'value': metrics.get(role, {}).get(cohort)},
            {'id': 'old_dpo', 'label': '历史视觉DPO', 'value': old_dpo.get(cohort)},
            {'id': 'teacher', 'label': '7B教师参考', 'value': teacher.get('metrics', {}).get('teacher', {}).get(cohort)}]
    failures = []
    if analysis:
        for reference, cohorts in analysis['candidate_vs_controls'].items():
            value = cohorts['transfer_dev']
            if value['delta_pp'] < .5 or value['ci95'][0] <= 0:
                failures.append({'role': role, 'cohort': 'transfer_dev', 'label': '完整1617 对' + reference,
                    **value, 'threshold': .5, 'evidence': '固定Final32完整比较，教师结果不能代替学生提分。'})
            names = ('literal_validation','seen_validation','singleword','multiword','rare_1_5','retention','full_rare342') if reference == 'start' else ('literal_validation','seen_validation','rare_1_5','retention','full_rare342')
            for cohort in names:
                value = cohorts[cohort]
                if value['delta_pp'] < -.5:
                    failures.append({'role': role, 'cohort': cohort, 'label': cohort + ' 对' + reference,
                        **value, 'threshold': -.5, 'evidence': '原点保护门槛保持，跨零区间不称显著退化。'})
        value = analysis['candidate_vs_historical_strong_dpo']
        if value['delta_pp'] < .5 or value['ci95'][0] <= 0:
            failures.append({'role': role, 'cohort': 'transfer_dev', 'label': '完整1617 对历史强DPO', **value, 'threshold': .5, 'evidence': '原强参考保持。'})
    def stage(key, label, value, total):
        return {'id': key, 'label': label, 'value': value, 'total': total, 'state': 'completed' if value == total else 'running' if value else 'waiting'}
    stages = [stage('baseline', '复用强起点', 2577, 2577), stage('cpu', '新目标CPU检查', 4, 4),
        stage('mask', '原103题准入', 103, 103), stage('train', '新学生训练', step, 32),
        stage('internal', '内部960', 960 if internal_done else collector.internal_count(internal), 960),
        stage('transfer', '完整1617', 1617 if transfer_done else transfer.get('completed', 0), 1617)]
    review = read(base / 'vlm_native_correction_pool_v2/REVIEWED_ERRORS.json')
    research = ('新学生完整门槛通过，待三seed/外部' if analysis.get('student_signal') else '新学生门槛未过，停止本配置') if completion else '学生前缀KD：能力待固定Final完整验证'
    if state_name == 'waiting': research = '学生前缀KD排队；0/32新更新' if not step else '新训练等待资源；旧步数不冒充新更新'
    state = {'schema': 1, 'observed_at': now(), 'mode': 'snapshot', 'project': 'RareVLM', 'study': root.name,
        'state': state_name, 'research': research, 'controller': {'pid': launch.get('pid'), 'alive': alive(launch)},
        'runtime_errors': errors, 'completion': completion, 'roles': roles, 'stages': stages, 'gpu': collector.gpu(),
        'screen': analysis, 'analysis': {**analysis, 'metrics': metrics}, 'failures': failures,
        'gates': [{'role': role, 'key': key, 'passed': value} for key, value in analysis.get('gates', {}).items()],
        'scores': scores, 'history': [{'name': '7B教师完整任务', 'status': '三门槛通过',
            'detail': '93.537415% 对88.027211%，+5.510204pp/95%[4.232778,6.834321]；这是教师机会，不是学生提分。',
            'source': source(base / 'vlm_full_teacher_transfer_v1/ANALYSIS.json')}],
        'sources': [source(root / 'STUDY_INPUT_LOCK.json', '新学生前缀研究锁'), source(root / 'CPU_PREFIX_SIGNAL.json', '47题/127个不同前缀位置准入'),
            source(root / 'CONFIG.json', '固定原超参/新前缀机制'), source(directory / 'train/PROGRESS.json', '新32步实际进度'),
            source(directory / 'train/ROLLOUT_SUMMARY.json', '实际训练生成与教师前向'), source(root / 'COMPLETION.json', '新学生最终验收'),
            source(base / 'vlm_full_teacher_transfer_v1/result_review_v1/完整教师_结果与真实案例.png', '教师完整结果与真实案例')],
        'batch_diagnostic': {}, 'auxiliary_runs': [], 'independent_validation': {}, 'source_holdout_validation': {}, 'full_teacher_validation': {},
        'regularizer': 'kd', 'prefix_source': config['prefix_source'], 'numeric_policy': config['policy'], 'active_jobs': jobs,
        'dataset': {'reviewed': review.get('reviewed_candidates'), 'supported': review.get('supported'),
            'excluded': review.get('not_admitted_or_uncertain'), 'selected': 128, 'unique_targets': 127, 'overlap': 0},
        'cases': [q for q in collector.cases(data) if q['group'] == '评测变化'] if internal_done else [],
        'warnings': collector.warnings, 'scope': 'New student-prefix KD; old negative configurations stay closed, full teacher advantage is not student gain.'}
    collector.cached_state = state; collector.cached_at = time.monotonic(); return state
