"""Read-only RareVLM observability. Never imports a model or changes an experiment."""
import argparse
import base64
from collections import defaultdict
import csv
from datetime import datetime, timezone, timedelta
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import quote, unquote, urlparse

PROJECT = Path(__file__).resolve().parents[1]
WORK = PROJECT.parents[2]
BASE = PROJECT.parents[1] / 'outputs'
OUT = BASE / 'rarevlm_monitor'
CURRENT = BASE / 'vlm_native_correction_sft_dpo_v2'
TEMPLATE = PROJECT / 'docs/monitor/template.html'
TZ = timezone(timedelta(hours=8))
ROLE_LABELS = {'sft': 'SFT', 'dpo_sft': 'DPO＋SFT'}
COHORTS = {'native_hard': '训练纠错128', 'literal_validation': '内部OCR256', 'seen_validation': '原生问答512',
           'retention': '保持集64', 'transfer_dev': '整图问答1617', 'singleword': '单词答案', 'multiword': '多词答案', 'rare_1_5': '罕见答案366'}


def now():
    return datetime.now(TZ).isoformat(timespec='seconds')


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False, encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        temporary = stream.name
    os.replace(temporary, path)


def terminal_state(completion, queue):
    """A valid completion takes precedence over stale running/failed queue snapshots."""
    if completion.get('status') == 'completed':
        return 'negative' if 'negative' in completion.get('phase', '') or completion.get('passed') is False else 'completed'
    if queue.get('status') == 'failed':
        return 'failed'
    return 'running' if queue.get('status') == 'running' else 'waiting'


def process_alive(record):
    pid = record.get('pid', record.get('controller_pid'))
    expected = record.get('start_ticks', record.get('process_start_ticks'))
    if not pid or not expected: return None
    try:
        fields = Path('/proc', str(pid), 'stat').read_text().rsplit(') ', 1)[1].split()
        return fields[19] == str(expected) and fields[0] != 'Z'
    except (OSError, IndexError): return False


class Collector:
    def __init__(self):
        self.lock = threading.RLock()
        self.memo = {}
        self.artifacts = {}
        self.images = {}
        self.cached_state = None
        self.cached_at = 0.
        self.warnings = []

    def read(self, path, default=None):
        path = Path(path)
        if not path.exists():
            return {} if default is None else default
        key = str(path); stat = path.stat(); stamp = (stat.st_mtime_ns, stat.st_size)
        old = self.memo.get(key)
        if old and old[0] == stamp:
            return old[1]
        try:
            value = json.loads(path.read_text())
            self.memo[key] = (stamp, value)
            return value
        except (OSError, ValueError):
            self.warnings.append('文件正在写入或无法读取：' + str(path.relative_to(WORK)))
            return old[1] if old else ({} if default is None else default)

    def source(self, path, label=None):
        path = Path(path).resolve()
        path.relative_to(WORK)
        identifier = hashlib.sha256(str(path).encode()).hexdigest()[:18]
        self.artifacts[identifier] = path
        return {'id': identifier, 'label': label or path.name, 'path': str(path.relative_to(WORK)),
                'href': quote(os.path.relpath(path, OUT), safe='/._-'),
                'updated_at': datetime.fromtimestamp(path.stat().st_mtime, TZ).isoformat(timespec='seconds') if path.exists() else None,
                'exists': path.exists()}

    def thumb(self, path, digest):
        from PIL import Image, ImageOps
        path = Path(path).resolve(); path.relative_to(WORK)
        cache = OUT / 'cache/thumbs' / (digest[:24] + '.jpg')
        if not cache.exists():
            cache.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(path) as raw:
                ImageOps.contain(raw.convert('RGB'), (480, 320)).save(cache, quality=65, optimize=True)
        return 'data:image/jpeg;base64,' + base64.b64encode(cache.read_bytes()).decode()

    def trace_curve(self, role):
        path = CURRENT / 'runs' / role / 'train/INPUT_TRACES.jsonl'
        if not path.exists():
            return {'points': [], 'qa_read': 0}
        cache_path = OUT / 'cache' / ('curve_' + CURRENT.name + '_' + role + '.json')
        state = self.read(cache_path, {'offset': 0, 'steps': {}, 'tail_hash': None})
        state = json.loads(json.dumps(state)); size = path.stat().st_size
        with path.open('rb') as stream:
            offset = state.get('offset', 0)
            if offset > size:
                state = {'offset': 0, 'steps': {}, 'tail_hash': None}; offset = 0
            if offset:
                stream.seek(max(0, offset - 256))
                if hashlib.sha256(stream.read(min(256, offset))).hexdigest() != state.get('tail_hash'):
                    state = {'offset': 0, 'steps': {}, 'tail_hash': None}; offset = 0
            stream.seek(offset)
            while True:
                begin = stream.tell(); line = stream.readline()
                if not line or not line.endswith(b'\n'):
                    stream.seek(begin); break
                try:
                    row = json.loads(line)
                except ValueError:
                    stream.seek(begin); self.warnings.append('训练trace尾行写入中：' + role); break
                step = state['steps'].setdefault(str(row['update']), {'qa': 0, 'ce': 0., 'dpo': 0., 'preference': 0, 'kl': 0., 'replay': 0})
                step['qa'] += 1; step['ce'] += row['chosen_ce']
                if row['source'] == 'common' and 'kl_loss' in row:
                    step['replay'] += 1; step['kl'] += row['kl_loss'] * row.get('kl_admitted', True)
                if row['source'] == 'preference':
                    step['preference'] += 1; step['dpo'] += row['dpo_loss']
            state['offset'] = stream.tell()
            stream.seek(max(0, state['offset'] - 256))
            state['tail_hash'] = hashlib.sha256(stream.read(min(256, state['offset']))).hexdigest()
        atomic_json(cache_path, state)
        points = [{'step': int(k), 'ce': v['ce'] / v['qa'], 'dpo': v['dpo'] / v['preference'] if v['preference'] else None,
                   'kl': v.get('kl', 0.) / v['replay'] if v.get('replay') else None, 'qa': v['qa']}
                  for k, v in sorted(state['steps'].items(), key=lambda item: int(item[0])) if v['qa'] == 16]
        return {'points': points, 'qa_read': sum(v['qa'] for v in state['steps'].values()), 'source': self.source(path, role + '逐QA训练trace')}

    def gpu(self):
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,temperature.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=4, check=True)
            devices = []
            for row in csv.reader(io.StringIO(result.stdout)):
                index, uuid, name, total, used, utilization, temperature = [s.strip() for s in row]
                devices.append({'index': int(index), 'uuid': uuid, 'name': name, 'total_gib': float(total) / 1024,
                                'used_gib': float(used) / 1024, 'utilization': float(utilization), 'temperature': float(temperature), 'processes': []})
            apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_gpu_memory', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=4)
            if apps.returncode == 0:
                for row in csv.reader(io.StringIO(apps.stdout)):
                    if len(row) != 3 or not row[0].strip().isdigit(): continue
                    pid, uuid, memory = [s.strip() for s in row]
                    for d in devices:
                        if d['uuid'] == uuid:
                            d['processes'].append({'pid': int(pid), 'memory_mib': float(memory) if memory.replace('.', '', 1).isdigit() else None})
            return {'status': 'observed', 'devices': devices, 'process_query_status': 'observed' if apps.returncode == 0 else 'unavailable', 'observed_at': now()}
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            return {'status': 'unavailable', 'devices': [], 'error': str(error)[:250], 'observed_at': now()}

    def cases(self, data):
        notes_path = PROJECT / ('docs/monitor/native_scope_notes.json' if CURRENT.name == 'vlm_native_scope_v1' else 'docs/monitor/native_stable_kl_notes.json' if CURRENT.name == 'vlm_native_stable_kl_v1' else 'docs/monitor/case_notes.json')
        notes = self.read(notes_path)
        note_map = {(e['role'], e['qid'], e['before'], e['after']): e for e in notes.get('entries', [])} if notes.get('study') == CURRENT.name else {}
        audit_notes = {e['ordinal']: e for e in notes.get('audit_entries', [])} if notes.get('study') == CURRENT.name else {}
        review = self.read(BASE / 'vlm_native_correction_pool_v2/REVIEWED_ERRORS.json')
        pool = self.read(BASE / 'vlm_native_correction_pool_v2/ERROR_CANDIDATES.json')
        rows = {q['question_id']: q for q in pool.get('rows', [])}
        selected = {q['question_id'] for q in data.get('rows', []) if q.get('study_source') == 'preference'} if CURRENT.name == 'vlm_native_diversity_v1' else {q['question_id'] for q in data.get('evaluations', {}).get('native_hard', [])}
        result = []
        for e in review.get('entries', []):
            q = rows[e['question_id']]; im = pool['images'][q['image_id']]; identifier = 't-' + str(e['ordinal'])
            self.images[identifier] = Path(im['path'])
            result.append({'id': identifier, 'ordinal': e['ordinal'], 'qid': q['question_id'], 'group': '监督审核',
                           'status': e['status'], 'selected': q['question_id'] in selected, 'question': q['question'], 'target': q['target'],
                           'prediction': q['initial_prediction']['prediction'], 'reason': e['evidence'], 'category': e.get('category', ''),
                           'image': self.thumb(im['path'], im['sha256']), 'original_href': quote(os.path.relpath(im['path'], OUT), safe='/._-'),
                           'source': self.source(BASE / 'vlm_native_correction_pool_v2/REVIEWED_ERRORS.json', '逐例审核记录')})
            note = audit_notes.get(e['ordinal'])
            if note and (note['qid'], note['target'], note['prediction']) == (q['question_id'], q['target'], q['initial_prediction']['prediction']):
                result[-1]['reason'] = note['note']; result[-1]['note_source'] = self.source(notes_path, '仅展示的事后案例说明')
        # Display observed score changes; these are not new label adjudications or training data.
        baseline = self.read(CURRENT / 'baseline/seen_validation/PREDICTIONS.json', [])
        meta = {q['question_id']: q for q in data.get('evaluations', {}).get('seen_validation', [])}
        for role in ROLE_LABELS:
            pred = self.read(CURRENT / 'runs' / role / 'internal/seen_validation/PREDICTIONS.json', [])
            if len(pred) != len(baseline) or not baseline: continue
            for a, b in zip(baseline, pred):
                if a['question_id'] != b['question_id']: continue
                delta = b['vqa_score'] - a['vqa_score']
                if abs(delta) < 1e-9: continue
                q = meta[b['question_id']]; im = data['images'][q['image_id']]; identifier = 'v-' + role + '-' + q['question_id']
                self.images[identifier] = Path(im['path'])
                result.append({'id': identifier, 'qid': q['question_id'], 'group': '评测变化', 'role': role, 'selected': False,
                               'status': 'score_loss' if delta < 0 else 'score_gain', 'question': q['question'], 'target': q['target'],
                               'prediction': b['prediction'], 'before': a['prediction'], 'score_before': a['vqa_score'], 'score_after': b['vqa_score'],
                               'delta': delta, 'rare': 1 <= q.get('answer_frequency', 999) <= 5,
                               'reason': '同一评测题的保存回答与VQA分数变化。尚未逐图重标，具体因果未独立验证；此题不会加入本轮训练。',
                               'image': self.thumb(im['path'], im['sha256']), 'original_href': quote(os.path.relpath(im['path'], OUT), safe='/._-'),
                               'source': self.source(CURRENT / 'runs' / role / 'internal/seen_validation/PREDICTIONS.json', ROLE_LABELS[role] + '完整预测')})
                note = note_map.get((role, q['question_id'], a['prediction'], b['prediction']))
                if note:
                    result[-1]['reason'] = note['note']; result[-1]['category'] = note['category']
                    result[-1]['note_source'] = self.source(notes_path, '仅展示的事后案例说明')
        return result

    def collect(self, force=False):
        with self.lock:
            if not force and self.cached_state and time.monotonic() - self.cached_at < 8:
                return self.cached_state
            self.warnings = []
            if CURRENT.name == 'vlm_onpolicy_teacher_kd_v1':
                from rarevlm_monitor_onpolicy import collect
                return collect(self, CURRENT, ROLE_LABELS, now, process_alive)
            if CURRENT.name == 'vlm_full_teacher_kd_v1':
                from rarevlm_monitor_kd import collect
                return collect(self, CURRENT, ROLE_LABELS, now, process_alive)
            completion = self.read(CURRENT / 'COMPLETION.json'); screen = self.read(CURRENT / 'SCREEN_ANALYSIS.json'); transfer = self.read(CURRENT / 'TRANSFER_ANALYSIS.json')
            analysis = transfer or screen
            queue = self.read(CURRENT / 'training_queue/STATUS.json'); data = self.read(CURRENT / 'data/MANIFEST.json')
            kl_study = CURRENT.name == 'vlm_native_stable_kl_v1'
            diversity_study = CURRENT.name == 'vlm_native_diversity_v1'
            rank_study = CURRENT.name == 'vlm_late_rank_matched_dpo_v1'
            source_replay_study = CURRENT.name == 'vlm_stvqa_replay_v1'
            late_study = CURRENT.name == 'vlm_late_visual_dpo_v1' or rank_study
            preference_study = CURRENT.name == 'vlm_visual_preference_v1' or diversity_study or late_study or source_replay_study
            scope_study = CURRENT.name == 'vlm_native_scope_v1' or preference_study
            fresh_study = kl_study or scope_study
            protocol_name = 'docs/VLM_NATIVE_SCOPE_V1_PROTOCOL.md' if scope_study else 'docs/VLM_NATIVE_STABLE_KL_V1_PROTOCOL.md' if kl_study else 'docs/VLM_NATIVE_CORRECTION_SFT_DPO_V2_PROTOCOL.md'
            status_name = 'docs/VLM_NATIVE_SCOPE_V1_STATUS.md' if scope_study else 'docs/VLM_NATIVE_STABLE_KL_V1_STATUS.md' if kl_study else 'docs/VLM_NATIVE_CORRECTION_V2_STATUS.md'
            if preference_study:
                protocol_name = 'docs/VLM_VISUAL_PREFERENCE_V1_PROTOCOL.md'
                status_name = 'docs/VLM_VISUAL_PREFERENCE_V1_STATUS.md'
            if diversity_study:
                protocol_name = 'docs/VLM_NATIVE_DIVERSITY_V1_PROTOCOL.md'
                status_name = 'docs/VLM_NATIVE_DIVERSITY_V1_STATUS.md'
            if late_study:
                protocol_name = 'docs/VLM_LATE_VISUAL_DPO_V1_PROTOCOL.md'
                status_name = 'docs/VLM_LATE_VISUAL_DPO_V1_STATUS.md'
            if rank_study:
                protocol_name = 'docs/VLM_LATE_RANK_MATCHED_DPO_V1_PROTOCOL.md'
                status_name = 'docs/VLM_LATE_RANK_MATCHED_DPO_V1_STATUS.md'
            if source_replay_study:
                protocol_name = 'docs/VLM_STVQA_REPLAY_V1_PROTOCOL.md'
                status_name = 'docs/VLM_STVQA_REPLAY_V1_STATUS.md'
                analysis = self.read(CURRENT / 'ANALYSIS.json') or analysis
                if analysis: screen = analysis; transfer = analysis
            observed_queues = {name: self.read(CURRENT / name / 'STATUS.json') for name in ('preflight_queue', 'baseline_queue', 'reference_queue', 'training_queue', 'transfer_queue')}
            if source_replay_study: observed_queues['evaluation_queue'] = self.read(CURRENT / 'evaluation_queue/STATUS.json')
            active_queue = next((q for q in observed_queues.values() if q.get('status') == 'running'), queue)
            root_state = terminal_state(completion, active_queue)
            controller = self.read(CURRENT / 'LAUNCH.json')
            controller_running = process_alive(controller)
            runtime_errors = []
            if not completion:
                for queue_name in ('preflight_queue', 'baseline_queue', 'reference_queue', 'training_queue', 'transfer_queue', 'evaluation_queue', 'repeated_test_queue'):
                    qs = self.read(CURRENT / queue_name / 'STATUS.json')
                    for key, job in qs.get('jobs', {}).items():
                        if job.get('state') == 'failed':
                            log = CURRENT / queue_name / 'logs' / (key + '.log')
                            tail = log.read_text(errors='replace')[-3000:] if log.exists() else '日志尚未生成'
                            runtime_errors.append({'job': key, 'queue': queue_name, 'tail': tail, 'source': self.source(log, '失败worker日志')})
                if runtime_errors: root_state = 'failed'
                elif controller_running is False:
                    root_state = 'orphaned'
                    runtime_errors.append({'job': 'controller', 'queue': 'controller', 'tail': '控制器PID及启动时间未匹配，且没有有效完成标记。请核对控制器日志；不自动重启。', 'source': self.source(CURRENT / 'controller.log')})
            positive = analysis.get('native_positive_signal', {})
            if completion:
                research = '未达本轮门槛' if root_state == 'negative' or not any(positive.values()) else '局部信号待复现'
            else:
                research = '等待迁移评测' if screen and any(screen.get('screen_passed', {}).values()) else '尚在评测'
            roles = []
            for role, label in ROLE_LABELS.items():
                directory = CURRENT / 'runs' / role; final = self.read(directory / 'train/TRAIN_COMPLETE.json'); progress = self.read(directory / 'train/PROGRESS.json')
                step = final.get('updates', progress.get('completed_updates', 0)); internal = self.read(directory / 'internal/PROGRESS.json')
                internal_done = (directory / 'internal_COMPLETE.json').exists(); transfer_done = (directory / 'transfer_COMPLETE.json').exists()
                phase = '已完成' if completion else ('迁移评测' if internal_done and screen and any(screen.get('screen_passed', {}).values()) else '内部评测' if final else '训练' if step else '等待')
                transfer_job = observed_queues['transfer_queue'].get('jobs', {}).get('transfer' if late_study else role, {})
                if source_replay_study: transfer_job = observed_queues['evaluation_queue'].get('jobs', {}).get(role + '_transfer', {})
                if not completion and screen.get('expand_transfer'):
                    phase = '1617已完成' if transfer_done else '1617评测' if transfer_job.get('state') == 'running' else '等待GPU（1617）'
                if diversity_study and not completion and not (CURRENT / 'units' / role / 'reference/COMPLETE.json').exists():
                    phase = '建立reference' if queue.get('jobs', {}).get(role, {}).get('state') == 'running' else '等待GPU'
                if late_study and not completion and final:
                    phase = '完整1617已完成，等内部960' if transfer_done else '完整1617与内部960并行' if transfer_job.get('state') == 'running' else '等待完整1617评测'
                if source_replay_study and not completion and final:
                    phase = '完整1617与内部评测已完成' if transfer_done and internal_done else '完整1617评测' if transfer_job.get('state') == 'running' else 'Final32后内部与完整评测'
                roles.append({'id': role, 'label': label, 'step': step, 'planned': 32, 'phase': phase,
                              'train_seconds': final.get('active_seconds', progress.get('active_seconds')), 'seconds_per_update': progress.get('recent_seconds_per_update'),
                              'peak_gib': final.get('peak_allocated_gib', progress.get('peak_allocated_gib')), 'internal': internal, 'internal_done': internal_done,
                              'transfer': self.read(directory / 'transfer/PROGRESS.json'), 'transfer_done': transfer_done,
                              'queue': transfer_job or queue.get('jobs', {}).get(role, {}), 'curve': self.trace_curve(role), 'trainable': self.read(CURRENT / 'CONFIG.json').get('expected_trainable', 614400) if late_study else 1843200,
                              'source': self.source(directory / 'train/PROGRESS.json', label + '训练状态')})
            reference = self.read(CURRENT / 'reference/COMPLETE.json'); reference_progress = self.read(CURRENT / 'reference/PROGRESS.json')
            review = self.read(BASE / 'vlm_native_correction_pool_v2/REVIEWED_ERRORS.json'); receipt = self.read(CURRENT / 'DATA_RECEIPT.json')
            if source_replay_study: receipt = self.read(BASE / 'vlm_visual_preference_v1/DATA_RECEIPT.json')
            if diversity_study:
                references = [self.read(CURRENT / 'units' / role / 'reference/COMPLETE.json') for role in ROLE_LABELS]
                reference = {'status': 'completed'} if all(references) else {}
                reference_progress = {'completed_batches': sum(self.read(CURRENT / 'units' / role / 'reference/PROGRESS.json').get('completed_batches', 0) for role in ROLE_LABELS)}
                review = self.read(BASE / 'vlm_native_diversity_mining_v1/REVIEWED_ERRORS.json')
            mining = [self.read(BASE / name / 'COMPLETION.json') for name in ('vlm_native_correction_mining_v2', 'vlm_native_correction_supplement_v2')]
            stages = [
                {'id': 'data', 'label': '筛选原生问答', 'value': sum(m.get('mined_qa', 0) for m in mining), 'total': 3584, 'state': 'completed' if all(m.get('status') == 'completed' for m in mining) else 'waiting'},
                {'id': 'review', 'label': '看图审核', 'value': review.get('reviewed_candidates', 0), 'total': 251, 'state': 'completed' if review else 'waiting'},
                {'id': 'reference', 'label': '固定参考缓存', 'value': 64 if reference else reference_progress.get('completed_batches', 0), 'total': 64, 'state': 'completed' if reference else 'running'},
                {'id': 'train', 'label': '同数据训练', 'value': sum(r['step'] for r in roles), 'total': 32 * len(roles), 'state': 'completed' if all(r['step'] == 32 for r in roles) else 'running'},
                {'id': 'internal', 'label': '内部评测', 'value': sum(960 if r['internal_done'] else self.internal_count(r['internal']) for r in roles), 'total': 960 * len(roles), 'state': 'completed' if all(r['internal_done'] for r in roles) else 'running'},
                {'id': 'transfer', 'label': '完整1617', 'value': sum(1617 if r['transfer_done'] else r['transfer'].get('completed', 0) for r in roles), 'total': 3234,
                 'state': 'not_admitted' if screen and not any(screen.get('screen_passed', {}).values()) else 'completed' if all(r['transfer_done'] for r in roles) else 'running' if screen else 'conditional'},
            ]
            if fresh_study:
                baseline_count = 0
                baseline_total = 960 if scope_study else 1216
                for part, names in [('a', ('native_hard', 'literal_validation', 'retention')), ('b', ('seen_validation',) if scope_study else ('seen_validation', 'replay_ocr', 'replay_native'))]:
                    baseline_count += sum(len(self.read(CURRENT / ('baseline_' + part) / n / 'PREDICTIONS.json', [])) for n in names)
                stages[0] = {'id': 'baseline', 'label': '复用共同起点（0生成）' if preference_study else '新策略基线', 'value': baseline_count, 'total': baseline_total, 'state': 'completed' if baseline_count == baseline_total else 'running' if observed_queues['baseline_queue'].get('status') == 'running' else 'waiting'}
                if scope_study and not preference_study:
                    preflight = self.read(CURRENT / 'PREFLIGHT_ANALYSIS.json')
                    stages[2] = {'id': 'preflight', 'label': '适配梯度预检', 'value': sum((CURRENT / 'preflight' / r / 'COMPLETE.json').exists() for r in ROLE_LABELS), 'total': 2, 'state': 'completed' if preflight.get('passed') else 'not_admitted' if preflight else 'running'}
                stages[-1]['total'] = 4851
                stages[-1]['value'] += 1617 if (CURRENT / 'baseline_TRANSFER_COMPLETE.json').exists() else self.read(CURRENT / 'baseline/PROGRESS.json').get('completed', 0)
                if screen and not screen.get('expand_transfer'): stages[-1]['state'] = 'not_admitted'
                if baseline_count < baseline_total: research = '生成共同新基线，尚无训练结论'
                elif scope_study:
                    if not completion:
                        research = '内部初筛通过；三模型1617验证中' if screen.get('expand_transfer') else '内部初筛未过，等待收尾' if screen else '适配范围对照：训练中，Final指标待评测' if any(r['step'] < 32 for r in roles) else '适配范围对照：内部评测中'
                        if preference_study:
                            research = '视觉CE/DPO：内部保护通过，两Final1617验证中' if screen.get('expand_transfer') else '视觉CE/DPO：内部保护未过，等待收尾' if screen else '视觉CE/DPO：建立固定reference' if not reference else '视觉CE/DPO：32步匹配训练中' if any(r['step'] < 32 for r in roles) else '视觉CE/DPO：内部评测中'
                        if diversity_study:
                            research = '覆盖对照：两Final1617验证中' if screen.get('expand_transfer') else '覆盖对照：内部门槛未过，等待收尾' if screen else '覆盖对照：reference与训练并行' if not reference and any(r['step'] for r in roles) else '覆盖对照：建立reference' if not reference else '覆盖对照：96题与192题同曝光训练中' if any(r['step'] < 32 for r in roles) else '覆盖对照：内部评测中'
                elif not reference: research = '建立固定教师，尚无训练结论'
            if diversity_study:
                stages[1] = {'id': 'review', 'label': '准入审核（前152）', 'value': review.get('reviewed_candidates', 0), 'total': 152, 'state': 'completed' if review.get('status') == 'passed' else 'waiting'}
                stages[2] = {'id': 'reference', 'label': '两组固定参考', 'value': 128 if reference else reference_progress['completed_batches'], 'total': 128, 'state': 'completed' if reference else 'running'}
            if late_study:
                stages[2]['label'] = '复用固定参考（0前向）'
                stages[-1] = {'id': 'transfer', 'label': '固定Final完整1617', 'value': sum(1617 if r['transfer_done'] else r['transfer'].get('completed', 0) for r in roles), 'total': 1617, 'state': 'completed' if all(r['transfer_done'] for r in roles) else 'running' if observed_queues['transfer_queue'].get('status') == 'running' else 'waiting'}
                research = ('后10层内部信号通过，待独立确认' if any(positive.values()) else '后10层完整任务或保持未达标，停止该配置') if completion else '后10层：32步训练中，完整任务结果待测' if any(r['step'] < 32 for r in roles) else '后10层：内部960与完整1617评测中'
                if rank_study:
                    parity = self.read(CURRENT / 'INITIALIZATION_GPU_PARITY.json')
                    stages[1] = {'id': 'initialization', 'label': '新秩初始化输出对齐', 'value': 8 if parity else 0, 'total': 8, 'state': 'completed' if parity.get('status') == 'passed' else 'waiting'}
                    research = research.replace('后10层', '后10层等训练参数')
            gate_rows = []
            if source_replay_study:
                stages[0] = {'id':'baseline','label':'复用共同起点（0生成）','value':960,'total':960,'state':'completed'}
                stages[1] = {'id':'source','label':'新整图回放128','value':128,'total':128,'state':'completed'}
                stages[2]['label'] = '复用偏好参考（0前向）'
                stages[-1]['total'] = 3234
                stages[-1]['value'] = sum(1617 if r['transfer_done'] else r['transfer'].get('completed',0) for r in roles)
                stages[-1]['state'] = 'completed' if all(r['transfer_done'] for r in roles) else 'running' if any(r['step']==32 for r in roles) else 'waiting'
            for role in ROLE_LABELS:
                for key, passed in screen.get('screen_checks', {}).get(role, {}).items():
                    gate_rows.append({'role': role, 'key': key, 'passed': passed})
            failures = []
            if diversity_study:
                for role in ROLE_LABELS:
                    if screen and screen.get('screen_checks', {}).get(role, {}).get('hard_learning') is False:
                        result = screen['vs_start'][role]['native_hard']
                        predictions = self.read(CURRENT / 'runs' / role / 'internal/native_hard/PREDICTIONS.json', [])
                        full = sum(q['vqa_score'] >= 1 - 1e-9 for q in predictions)
                        failures.append({'role': role, 'cohort': 'native_hard', 'label': '原128诊断推进条件', **result,
                                         'threshold': 20., 'evidence': f'满分{full}/128（要求≥32）；增益{result["delta_pp"]:.6f}pp（要求≥20）。这是旧诊断推进条件，不是完整任务能力判定。'})
            if scope_study and transfer:
                for role in ROLE_LABELS:
                    result = transfer.get('vs_start', {}).get(role, {}).get('transfer_dev')
                    if result and (result['delta_pp'] < .5 or result['ci95'][0] <= 0):
                        failures.append({'role': role, 'cohort': 'transfer_dev', 'label': COHORTS['transfer_dev'], **result,
                                         'threshold': .5, 'evidence': '完整1617未达到至少+0.5pp且95%区间下界>0；不因优于退化控制而宣布能力提升。'})
                if late_study:
                    result = transfer.get('candidate_vs_full_dpo', {}).get('transfer_dev')
                    if result and (result['delta_pp'] < .5 or result['ci95'][0] <= 0):
                        failures.append({'role': 'ce_dpo', 'cohort': 'transfer_dev', 'label': '1617 对全层DPO强对照', **result, 'threshold': .5, 'evidence': '同数据/损失/32步，只限制更新范围；完整任务需优于强控制。'})
            for role in ROLE_LABELS:
                for name in ('literal_validation', 'seen_validation', 'retention', 'singleword', 'multiword', 'rare_1_5'):
                    result = screen.get('vs_start', {}).get(role, {}).get(name)
                    if result and result['delta_pp'] < -.5:
                        failures.append({'role': role, 'cohort': name, 'label': COHORTS[name], **result,
                                         'threshold': -.5, 'evidence': '点值保护未通过；区间跨0时不能称显著退化。'})
            old_native = self.read(BASE / 'vlm_iiit_native_transfer_v1/COMPLETION.json')
            old_scope = self.read(BASE / 'vlm_iiit_scope_v1/ANALYSIS.json')
            replay = self.read(BASE / 'vlm_iiit_replay_repair_v1/recovery_v1/PILOT_ANALYSIS.json')
            scores = {k: [] for k in ('transfer_dev', 'seen_validation', 'literal_validation', 'official_test')}
            for cohort in scores:
                for role, label in [('initial', '原SFT初始'), ('sparse_language', 'OCR增强起点')]:
                    score = old_native.get('metrics', {}).get(role) if cohort == 'transfer_dev' else old_scope.get('metrics', {}).get(role, {}).get(cohort, {}).get('score_percent')
                    scores[cohort].append({'id': role, 'label': label, 'value': score})
                scores[cohort].append({'id': 'repair42', 'label': '回放修复', 'value': replay.get('metrics', {}).get('repair42', {}).get(cohort, {}).get('score_percent')})
                for role in ROLE_LABELS:
                    scores[cohort].append({'id': role, 'label': ROLE_LABELS[role] + '32', 'value': analysis.get('metrics', {}).get(role, {}).get(cohort)})
                if fresh_study:
                    baseline = self.read(CURRENT / 'baseline' / cohort / 'PREDICTIONS.json', [])
                    expected = len(data.get('evaluations', {}).get(cohort, []))
                    value = None
                    if baseline and len(baseline) == expected:
                        value = 100 * sum(float(q['iiit_alnum_exact']) if cohort == 'literal_validation' else q['vqa_score'] for q in baseline) / len(baseline)
                    scores[cohort] = [{'id': 'start', 'label': '共同复合起点' if scope_study else '修复后新起点', 'value': value}] + [r for r in scores[cohort] if r['id'] in ROLE_LABELS]
                if late_study:
                    control = self.read(BASE / 'vlm_visual_preference_v1/TRANSFER_ANALYSIS.json').get('metrics', {}).get('ce_dpo', {}).get(cohort)
                    scores[cohort].insert(1, {'id': 'full_visual_dpo', 'label': '复用全层DPO强对照', 'value': control})
                    if rank_study:
                        low = self.read(BASE / 'vlm_late_visual_dpo_v1/TRANSFER_ANALYSIS.json').get('metrics', {}).get('ce_dpo', {}).get(cohort)
                        scores[cohort].insert(2, {'id': 'late12', 'label': '后10层r12低参数', 'value': low})
                if source_replay_study:
                    old_metrics = self.read(BASE / 'vlm_visual_preference_v1/TRANSFER_ANALYSIS.json').get('metrics', {})
                    for position, old_role, label in [(1,'ce','旧裁词CE'),(2,'ce_dpo','旧裁词CE+DPO强对照')]:
                        scores[cohort].insert(position, {'id':'old_'+old_role,'label':label,'value':old_metrics.get(old_role,{}).get(cohort)})
            recovered = self.read(BASE / 'vlm_iiit_replay_repair_v1/recovery_v1/COMPLETION.json')
            batch_root = BASE / 'vlm_native_batch_sensitivity_v1'
            batch_diagnostic = self.read(batch_root / 'COMPLETION.json')
            history = [
                {'name': 'IIIT5K词图适配', 'status': 'OCR局部提升', 'detail': '94.9667% → 96.9333%；不能外推整图问答', 'source': self.source(BASE / 'vlm_iiit_scope_v1/ANALYSIS.json')},
                {'name': '整图迁移补测', 'status': '已完成', 'detail': '6468次生成，0训练；原起点87.3284%，OCR增强87.2604%', 'source': self.source(BASE / 'vlm_iiit_native_transfer_v1/COMPLETION.json')},
                {'name': '回放分布修复', 'status': '已恢复汇总 · 未达标' if recovered else '需核对', 'detail': '收尾缺字段已独立恢复，0重训；1617为87.2480%', 'source': self.source(BASE / 'vlm_iiit_replay_repair_v1/recovery_v1/COMPLETION.json')},
            ]
            if batch_diagnostic:
                history.append({'name': '同起点分组敏感性诊断', 'status': '零更新检查已完成',
                                'detail': f"{batch_diagnostic['new_generations']}次生成；6个事后训练病例各组重复均稳定，跨组差异{batch_diagnostic['batch_disagreements']}/6；实际位置编码不一致{batch_diagnostic['unpad_position_mismatches']}。非全验证集不稳定率。",
                                'source': self.source(batch_root / 'COMPLETION.json')})
            auxiliary = []
            independent_validation = {}
            for dirname, label in [('vlm_batch_origin_trace_v1', '首次偏差定位'), ('vlm_batch_numeric_repair_v1', '数值修复验收'), ('vlm_native_stable_kl_v1/gpu_first_update_check_v1', '新损失GPU检查'), ('vlm_stable_kl_gradient_v1', 'Final完整训练梯度诊断'), ('vlm_native_scope_readout_v1', '视觉残错：答案读出诊断'), ('vlm_native_diversity_mining_v1', '数据覆盖：新增纠错筛选'), ('vlm_native_diversity_transfer_v1', '固定Final独立1617验证'), ('vlm_native_divergence_readout_v1', '整段偏好与首分歧决策'), ('vlm_review_resolution_repair_v1', '原图审核纠偏（固定9条）'), ('vlm_crop_teacher_signal_v1', '训练候选局部教师可用性'), ('vlm_crop_teacher_admission_v1', '足量局部教师数据准入'), ('vlm_visual_task_geometry_v1', '视觉CE/DPO与回放的实际更新'), ('vlm_stvqa_source_v1', '新ST-VQA训练来源准备'), ('vlm_stvqa_headroom_v1', '固定新训练槽位与来源保持')]:
                directory = BASE / dirname
                if not (directory / 'PLAN.json').exists() and not (dirname == 'vlm_stvqa_source_v1' and (directory / 'SOURCE_PLAN.json').exists()) and not (dirname == 'vlm_native_diversity_mining_v1' and (directory / 'SELECTION.json').exists()) and not (dirname in ('vlm_native_diversity_transfer_v1', 'vlm_native_divergence_readout_v1') and (directory / 'STUDY_INPUT_LOCK.json').exists()): continue
                done = self.read(directory / 'COMPLETION.json'); progress = self.read(directory / 'PROGRESS.json'); qstate = self.read(directory / 'queue/STATUS.json')
                phase = '已完成' if done else '运行异常' if qstate.get('status') == 'failed' else '等待GPU' if qstate.get('status') == 'waiting_resources' else '运行中'
                counts = (f"{progress.get('completed_forwards', 0)}/2前向" if dirname == 'vlm_batch_origin_trace_v1' else f"{progress.get('completed_micros', 0)}/4反向micro" if dirname.endswith('gpu_first_update_check_v1') else f"{progress.get('completed_sequences', 0)}条生成")
                if dirname == 'vlm_stable_kl_gradient_v1':
                    counts = str(sum(self.read(directory / role / 'PROGRESS.json').get('completed_batches', 0) for role in ('sft', 'sft_kl'))) + '/192批梯度'
                if dirname == 'vlm_native_scope_readout_v1':
                    counts = str(sum(self.read(directory / role / 'PROGRESS.json').get('completed_groups', 0) for role in ('start', 'visual'))) + '/64组读出；0新生成'
                if dirname == 'vlm_native_diversity_mining_v1':
                    if not (directory / 'PLAN.json').exists():
                        phase = 'CPU输入编码'
                        counts = str(self.read(directory / 'CPU_PROGRESS.json').get('completed', 0)) + '/3072条新输入'
                    else:
                        counts = str(sum(self.read(directory / ('shard_' + str(s)) / 'PROGRESS.json').get('completed', 0) for s in (0, 1))) + '/3072条筛选'
                    if done: phase = '原图准入通过' if self.read(directory / 'REVIEWED_ERRORS.json').get('status') == 'passed' else '筛选已完成，待原图审核'
                    if CURRENT.name == 'vlm_visual_preference_v1':
                        research = '上轮未达标；新数据' + ('输入编码 ' if phase == 'CPU输入编码' else '筛选 ') + counts if not done else '上轮未达标；补充纠错候选待原图审核'
                if dirname == 'vlm_native_diversity_transfer_v1':
                    completed = sum(self.read(directory / 'runs' / role / 'transfer/PROGRESS.json').get('completed', 0) for role in ('repeat96', 'diverse192'))
                    counts = f'{completed}/3234生成；0训练'
                    if diversity_study:
                        research = ('独立验证出现任务信号，待复现；原筛选未过' if done.get('task_signal') else '独立1617未达任务门槛；原筛选判定保留') if done else '原诊断门槛未过；固定Final独立验证 ' + counts
                        if done:
                            independent_validation = done
                            scores['transfer_dev'] = [{'id': r, 'label': ('前DPO128强参考' if r == 'previous_dpo128' else '独立验证 ' + ROLE_LABELS[r] if r in ROLE_LABELS else '共同复合起点'), 'value': done['metrics'][r]} for r in ('start', 'previous_dpo128', 'repeat96', 'diverse192')]
                            for reference_role in ('start', 'repeat96'):
                                comparison = done['candidate_comparisons'][reference_role]
                                if comparison['delta_pp'] < .5 or comparison['ci95'][0] <= 0:
                                    failures.append({'role': 'diverse192', 'cohort': 'transfer_dev', 'label': '独立1617 对' + ('起点' if reference_role == 'start' else '重复96'), **comparison, 'threshold': .5, 'evidence': '独立固定Final任务标准；原研究筛选判定不变。'})
                if dirname == 'vlm_native_divergence_readout_v1':
                    completed = sum((self.read(directory / 'runs' / r / 'PROGRESS.json').get('completed_questions', self.read(directory / 'runs' / r / 'COMPLETE.json').get('questions', 0))) for r in ('start', 'ce', 'ce_dpo'))
                    counts = f'{completed}/384题读出；0训练/0生成'
                    if done:
                        count = done['summaries']['ce_dpo']['sequence_win_local_loss_persistent']
                        phase = '机制线索成立，未训练' if done['mechanism_signal'] else '机制线索不足，停止该分支'
                        counts += f'；目标错型{count}/门槛16'
                        if not late_study:
                            research = '原完整任务未达标；首分歧诊断' + ('有线索，待独立训练协议' if done['mechanism_signal'] else '不足，不追加该训练')
                    else:
                        research = '原完整任务未达标；首分歧机制诊断 ' + counts
                        if qstate.get('status') in ('running', 'waiting_resources'):
                            root_state = 'running' if qstate['status'] == 'running' else 'waiting'
                if dirname == 'vlm_review_resolution_repair_v1':
                    counts = f"{done.get('reviewed', 0)}/9原图复核；支持{done.get('supported_after_full_resolution_review', '待定')}/不确定{done.get('still_uncertain', '待定')}；0训练/0生成"
                    if done:
                        phase = '复核完成，旧审核独立更正'
                        if rank_study and completion: research = '完整问答未达标，已关闭后层路线；原图审核纠偏完成'
                if dirname == 'vlm_crop_teacher_signal_v1':
                    completed = sum(self.read(directory / 'runs' / r / 'PROGRESS.json').get('completed_pairs', self.read(directory / 'runs' / r / 'COMPLETE.json').get('pairs', 0)) for r in ('start', 'full_dpo'))
                    counts = f'{completed}/14双输入检查；共28生成/28前缀前向，0训练'
                    if done:
                        a = done['summaries']['start']
                        counts += f'；主教师整图{a["full"]["strict_correct"]}/7、局部{a["crop"]["strict_correct"]}/7'
                        phase = '完整教师信号通过，仍未训练' if done['primary_complete_teacher_signal'] else '完整教师信号不足，未训练'
                    if rank_study and completion:
                        research = '原任务未达标；训练候选局部教师' + ('检查完成，未训练' if done else '可用性诊断中')
                    if not done and qstate.get('status') == 'running': root_state = 'running'
                if dirname == 'vlm_crop_teacher_admission_v1':
                    admission_plan = self.read(directory / 'PLAN.json')
                    checked = sum(self.read(directory / 'runs' / ('shard_' + str(s)) / 'PROGRESS.json').get('completed_pairs', 0) for s in (0, 1))
                    counts = f'{checked}/{admission_plan.get("pairs", "待定")}不同图的整图/局部检查；0训练'
                    if done:
                        phase = '足量数据通过，未训练' if all(done['gates'].values()) else '数据门槛未过，停止扩池'
                        counts += f'；可用{done["eligible_images"]}，选入{done["selected_images"]}图/{done["selected_unique_targets"]}答案/{done["selected_full_wrong_to_crop_correct"]}GT字面转换（非语义全审）'
                        admission_review = self.read(directory / 'RESULT_REVIEW.json')
                        if admission_review:
                            upper = admission_review['candidate_upper_bound_after_known_flag']
                            counts += f'；排已知疑点后最多{upper["images"]}图/{upper["targets"]}答案/{upper["gt_literal_conversions"]}转换'
                        research = '原能力目标未达标；局部教师' + ('数据足量，待匹配训练协议' if all(done['gates'].values()) else '数据不足，不启动训练')
                    else:
                        research = '原能力目标未达标；教师数据准入 ' + counts
                        if qstate.get('status') in ('running', 'waiting_resources'):
                            root_state = 'running' if qstate['status'] == 'running' else 'waiting'
                if dirname == 'vlm_visual_task_geometry_v1':
                    geometry_progress = self.read(directory / 'runs/visual/PROGRESS.json')
                    checked = geometry_progress.get('completed_points', 3 if done else 0)
                    counts = f'{checked}/3固定checkpoint；{geometry_progress.get("discarded_optimizer_updates", done.get("discarded_diagnostic_optimizer_updates", 0))}丢弃更新，0正式训练/0生成'
                    if done:
                        phase = '局部机制有信号，未训练' if done['mechanism_signal'] else '回放冲突证据不足，关闭假设'
                        counts += f'；通过{done["passed_points"]}/要求2'
                        research = '原能力目标未达标；视觉回放冲突' + ('有局部信号，待匹配训练协议' if done['mechanism_signal'] else '未达机制门槛，不追加训练')
                    else:
                        research = '原能力目标未达标；视觉参数梯度/实际Adam诊断 ' + counts
                        if qstate.get('status') in ('running', 'waiting_resources'):
                            root_state = 'running' if qstate['status'] == 'running' else 'waiting'
                if dirname == 'vlm_stvqa_source_v1':
                    source_inventory = self.read(directory / 'SOURCE_INVENTORY.json')
                    source_download = self.read(directory / 'DOWNLOAD_COMPLETE.json')
                    source_progress = self.read(directory / 'TRANSPORT_PROGRESS.json') or self.read(directory / 'DOWNLOAD_PROGRESS.json')
                    if source_inventory:
                        phase = '训练来源盘点完成，尚未训练'
                        counts = f'原训练图{source_inventory.get("source_images", "待定")}；可准备{source_inventory.get("eligible_images", "待定")}；0模型调用/0训练'
                    elif source_download:
                        phase = '固定源下载完成，核验原图与家族'
                        counts = '720MB训练镜像+上游标注SHA通过；0模型调用/0训练'
                    else:
                        phase = '固定训练源下载中'
                        counts = f'{source_progress.get("downloaded_bytes", 0)/2**20:.1f}/{source_progress.get("expected_bytes", 0)/2**20:.1f}MiB，完成文件{source_progress.get("completed_files", 0)}；0模型调用/0训练'
                    research = '原能力目标未达标；新ST-VQA来源：' + phase
                    root_state = 'preparing'
                if dirname == 'vlm_stvqa_headroom_v1':
                    source_checked = sum(self.read(directory / 'runs' / ('shard_' + str(s)) / 'PROGRESS.json').get('completed',0) for s in (0,1))
                    counts = f'{source_checked}/384源内生成与GT前向；128固定训练槽位+256来源保持；0训练'
                    if done:
                        phase = '新监督有信号，待匹配训练协议' if done['supervision_signal'] else '固定新监督不足，停止该子集'
                        a = done['summaries']['train_slots']
                        counts += f'；训练错{a["qa"]-a["literal_correct"]}/128，正文NLL{a["mean_body_nll"]:.3f}'
                        research = '原能力目标未达标；新ST-VQA监督' + ('有训练信号，尚未训练' if done['supervision_signal'] else '未达信号门槛')
                        root_state = 'negative'
                    else:
                        research = '原能力目标未达标；新ST-VQA固定槽位评估 ' + counts
                        root_state = 'running' if qstate.get('status')=='running' else 'waiting' if qstate.get('status')=='waiting_resources' else 'preparing'
                record = {'name': label, 'status': phase, 'detail': counts + ('；仅丢弃诊断步，不发布新权重。' if dirname == 'vlm_visual_task_geometry_v1' else '；零参数更新，具体结论以完成工件为准。'),
                          'source': self.source(directory / ('COMPLETION.json' if done else 'CPU_PROGRESS.json' if phase == 'CPU输入编码' else 'queue/STATUS.json')), 'jobs': qstate.get('jobs', {})}
                auxiliary.append(record); history.append(record)
            sources = [self.source(CURRENT / name, label) for name, label in [
                ('COMPLETION.json', '本轮终态'), ('SCREEN_ANALYSIS.json', '完整内部指标与门槛'), ('TRANSFER_ANALYSIS.json', '条件迁移指标'),
                ('STUDY_INPUT_LOCK.json', '冻结训练输入'), ('DATA_RECEIPT.json', '样本与隔离检查'), ('training_queue/STATUS.json', 'GPU队列状态')]]
            sources += [self.source(PROJECT / protocol_name, '实验协议'),
                        self.source(PROJECT / ('docs/monitor/native_scope_notes.json' if scope_study else 'docs/monitor/native_stable_kl_notes.json' if kl_study else 'docs/monitor/case_notes.json'), '事后案例说明（不改分数）'),
                        self.source(BASE / 'vlm_iiit_replay_repair_v1/pilot_queue/logs/repair42.log', '已恢复的历史错误日志'),
                        self.source(PROJECT / status_name, '人工状态说明')]
            if batch_diagnostic:
                sources += [self.source(batch_root / 'RESULTS.json', '6题分组与实际logits'),
                            self.source(BASE / 'vlm_native_bottleneck_check_v1/REPORT.json', '当前瓶颈与逐题贡献')]
            if kl_study:
                sources += [self.source(PROJECT / 'docs/VLM_NATIVE_STABLE_KL_V1_PROTOCOL.md', '当前SFT/KL冻结协议'),
                            self.source(CURRENT / 'CONFIG.json', '新数值与算法配置'),
                            self.source(BASE / 'vlm_batch_numeric_repair_v1/COMPLETION.json', '修复验收14训练病例')]
            if scope_study and not preference_study:
                sources += [self.source(CURRENT / 'INITIALIZATION.json', '复合初始化与等参数预算'), self.source(CURRENT / 'PREFLIGHT_ANALYSIS.json', '两组真实首批CE一致性'), self.source(CURRENT / 'preflight/visual/COMPLETE.json', '视觉真实梯度与显存'), self.source(CURRENT / 'CONFIG.json', '本轮适配范围配置')]
                review_root = BASE / 'vlm_native_scope_readout_v1'
                if (review_root / 'TRANSFER_REVIEW.json').exists():
                    sources += [self.source(review_root / 'TRANSFER_REVIEW.json', '1617完整升降与事后分层'), self.source(review_root / '结果与原图.html', '1617结果与真实图像学习页')]
            if preference_study:
                sources += [self.source(CURRENT / 'BASELINE_REUSE.json', '共同起点预测复用：0新生成'), self.source(CURRENT / 'reference/COMPLETE.json', '视觉DPO固定reference'), self.source(CURRENT / 'CONFIG.json', '视觉CE/DPO同预算配置'), self.source(BASE / 'vlm_native_scope_readout_v1/TRANSFER_REVIEW.json', '上轮完整1617升降'), self.source(BASE / 'vlm_native_scope_readout_v1/COMPLETION.json', '训练残错概率诊断')]
                if (CURRENT / 'RESULT_REVIEW.json').exists():
                    sources += [self.source(CURRENT / 'RESULT_REVIEW.json', '视觉DPO完整升降及1617频次分层'), self.source(CURRENT / '完整结果与原图.html', 'DPO公式、真实原图与结论边界')]
                if (BASE / 'vlm_native_diversity_mining_v1/SELECTION.json').exists():
                    sources += [self.source(PROJECT / 'docs/VLM_NATIVE_DIVERSITY_MINING_V1_PROTOCOL.md', '新增纠错数据筛选协议'), self.source(BASE / 'vlm_native_diversity_mining_v1/SELECTION.json', '3072条新训练池问答与隔离条件')]
            if diversity_study:
                sources += [self.source(CURRENT / 'UNIT_BINDINGS.json', '冻结训练循环的两unit绑定'), self.source(CURRENT / 'DATA_RECEIPT_CLARIFICATION.json', '历史继承字段说明与本轮真实覆盖'), self.source(BASE / 'vlm_native_diversity_mining_v1/REVIEWED_ERRORS.json', '前152原图审核：64新单词准入')]
                sources += [self.source(CURRENT / 'units' / role / 'reference/PROGRESS.json', ROLE_LABELS[role] + '实际reference') for role in ROLE_LABELS]
                if (BASE / 'vlm_native_diversity_transfer_v1/STUDY_INPUT_LOCK.json').exists():
                    sources += [self.source(PROJECT / 'docs/VLM_NATIVE_DIVERSITY_TRANSFER_V1_PROTOCOL.md', '独立固定Final验证协议：原门槛不改'), self.source(BASE / 'vlm_native_diversity_transfer_v1/queue/STATUS.json', '两份Final验证进度')]
                    if independent_validation:
                        sources += [self.source(BASE / 'vlm_native_diversity_transfer_v1/ANALYSIS.json', '独立1617完整指标与条件区间')]
            if (BASE / 'vlm_failure_stage_stats_v1/SUMMARY.json').exists():
                sources += [self.source(BASE / 'vlm_failure_stage_stats_v1/SUMMARY.json', '按步骤失败统计（独立计数单位）')]
            if (BASE / 'vlm_native_divergence_readout_v1/STUDY_INPUT_LOCK.json').exists():
                sources += [self.source(PROJECT / 'docs/VLM_NATIVE_DIVERGENCE_READOUT_V1_PROTOCOL.md', '首分歧机制诊断协议'), self.source(BASE / 'vlm_native_divergence_readout_v1/COMPLETION.json', '三个固定模型的机制结果'), self.source(BASE / 'vlm_native_divergence_readout_v1/整段偏好与首分歧.html', '真实原图、序列概率与词元决策')]
            if (BASE / 'vlm_review_resolution_repair_v1/COMPLETION.json').exists():
                sources += [self.source(BASE / 'vlm_review_resolution_repair_v1/REVIEW_CORRECTIONS.json', '9条原图复核：7支持2不确定，非总体错标率'), self.source(BASE / 'vlm_review_resolution_repair_v1/原图复核与分支关闭.html', 'EPSOM真实原图、显示裁剪和审核更正')]
            if (BASE / 'vlm_crop_teacher_signal_v1/PLAN.json').exists():
                sources += [self.source(PROJECT / 'docs/VLM_CROP_TEACHER_SIGNAL_V1_PROTOCOL.md', '仅训练候选的标注辅助教师诊断协议'), self.source(BASE / 'vlm_crop_teacher_signal_v1/局部教师可用性与真实输入.html', '真实教师输入、回答与正文/EOS概率')]
            if (BASE / 'vlm_crop_teacher_admission_v1/PLAN.json').exists():
                sources += [self.source(PROJECT / 'docs/VLM_CROP_TEACHER_ADMISSION_V1_PROTOCOL.md', '足量教师数据与同权重SFT对照规则'), self.source(BASE / 'vlm_crop_teacher_admission_v1/INVENTORY.json', '不同图像、答案数与隔离库存'), self.source(BASE / 'vlm_crop_teacher_admission_v1/COMPLETION.json', '教师数据准入结果'), self.source(BASE / 'vlm_crop_teacher_admission_v1/局部教师数据准入_原理与源码.html', '固定候选真实输入、CE/KD公式及源码')]
                if (BASE / 'vlm_crop_teacher_admission_v1/RESULT_REVIEW.json').exists():
                    sources += [self.source(BASE / 'vlm_crop_teacher_admission_v1/RESULT_REVIEW.json', '完整失败漏斗、教师信息与语义上限'), self.source(BASE / 'vlm_crop_teacher_admission_v1/准入失败漏斗与损失信息.png', '本轮失败原因的量化图')]
            if (BASE / 'vlm_visual_task_geometry_v1/PLAN.json').exists():
                sources += [self.source(PROJECT / 'docs/VLM_VISUAL_TASK_GEOMETRY_V1_PROTOCOL.md', '视觉实际Adam诊断事前门槛与旧语言研究区别'), self.source(BASE / 'vlm_visual_task_geometry_v1/runs/visual/PROGRESS.json', '固定3点真实梯度、位移及有限步损失'), self.source(BASE / 'vlm_visual_task_geometry_v1/COMPLETION.json', '视觉回放冲突诊断终态')]
                if (BASE / 'vlm_visual_task_geometry_v1/学算法_梯度方向与实际损失.html').exists():
                    sources += [self.source(BASE / 'vlm_visual_task_geometry_v1/学算法_梯度方向与实际损失.html', '实际参数梯度、Adam原理、科学代码及真实病例'), self.source(BASE / 'vlm_visual_task_geometry_v1/参数梯度与实际Adam_真实案例.png', '三个固定点的实测图解')]
            if (BASE / 'vlm_stvqa_source_v1/SOURCE_PLAN.json').exists():
                sources += [self.source(PROJECT / 'docs/VLM_STVQA_SOURCE_V1_PROTOCOL.md', '新场景数据的来源、隔离与未来假设'), self.source(BASE / 'vlm_stvqa_source_v1/SOURCE_PLAN.json', '固定训练版本与字节SHA'), self.source(BASE / 'vlm_stvqa_source_v1/DOWNLOAD_COMPLETE.json', '源数据下载结果'), self.source(BASE / 'vlm_stvqa_source_v1/SOURCE_INVENTORY.json', '源家族和可用数量盘点')]
            if (BASE / 'vlm_stvqa_headroom_v1/PLAN.json').exists():
                sources += [self.source(PROJECT / 'docs/VLM_STVQA_HEADROOM_V1_PROTOCOL.md', '固定128新训练槽位与256来源保持的事前门槛'), self.source(BASE / 'vlm_stvqa_headroom_v1/SOURCE_INFERENCE_ADAPTER_PLAN.json', '源内无标签生成与评分适配'), self.source(BASE / 'vlm_stvqa_headroom_v1/COMPLETION.json', '新监督信号评估终态')]
            if late_study:
                sources += [self.source(CURRENT / 'CONTROL_REUSE.json', '原强控制与参考的精确复用'), self.source(CURRENT / ('后10层等参数DPO.html' if rank_study else '后10层视觉适配.html'), 'LoRA原理、真实层范围和案例'), self.source(CURRENT / 'transfer_queue/STATUS.json', 'Final后两卡完整评测队列')]
                if rank_study:
                    sources += [self.source(CURRENT / 'INITIALIZATION.json', '实际PEFT秩、嵌套初始化与CPU往返'), self.source(CURRENT / 'INITIALIZATION_GPU_PARITY.json', '实际8条初始输出对齐，不是能力提升')]
            cases = self.cases(data)
            if source_replay_study:
                root_state = terminal_state(completion, active_queue)
                if runtime_errors: root_state = 'failed'
                research = ('新整图回放：内部任务有信号，仍待独立确认' if completion.get('primary_data_replacement_signal') else '主DPO未达标；SFT有信号但未过强对照') if completion else ('新整图回放CE/DPO训练中；尚无Final能力分数' if any(r['step']<32 for r in roles) else '新整图回放Final32：内部960与完整1617评测中')
                sources += [self.source(CURRENT/'DATA_REPLACEMENT.json','128裁词到128整图的固定映射'),self.source(CURRENT/'CONTROL_REUSE.json','原CE/DPO强控制的精确复用'),self.source(CURRENT/'evaluation_queue/STATUS.json','Final后完整评测队列')]
                sources += [self.source(CURRENT/'学算法_新整图监督与匹配SFT_DPO.html','新source真实原图、初始回答、loss及训练代码'),self.source(CURRENT/'新整图回放_真实样本与算法.png','本轮新训练数据与算法图解')]
                if (CURRENT/'report_recovery_v1/COMPLETION.json').exists():
                    sources += [self.source(CURRENT/'report_recovery_v1/COMPLETION.json','原汇总导入错误已恢复，0重训/0生成，42保护文件未变')]
                    history.append({'name':'本轮汇总恢复','status':'已修复','detail':'原控制器paired导入错误；CPU恢复全部预设统计，训练和四评测均exit0，非训练失败。','source':self.source(CURRENT/'report_recovery_v1/COMPLETION.json')})
                if (CURRENT/'result_review_v1/COMPLETION.json').exists():
                    sources += [self.source(CURRENT/'RESULT_REVIEW.md','完整量化结果、四组交互与停止决定'),self.source(CURRENT/'result_review_v1/学习页.html','本轮真实SFT改善、DPO退化及正反案例'),self.source(CURRENT/'result_review_v1/真实回答变化.png','本轮实际图像与保存回答对照')]
                matched_source_comparison = analysis.get('candidate_vs_ce',{}).get('transfer_dev')
                if matched_source_comparison and (matched_source_comparison['delta_pp']<.5 or matched_source_comparison['ci95'][0]<=0):
                    failures.append({'role':'ce_dpo','cohort':'transfer_dev','label':'1617 对同数据SFT',**matched_source_comparison,'threshold':.5,'evidence':'DPO必须超过相同数据、预算与初始化的SFT；本轮条件区间完全低于0。'})
                strong_source_comparison = analysis.get('vs_historical_controls_transfer',{}).get('ce_dpo',{}).get('ce_dpo')
                if strong_source_comparison and (strong_source_comparison['delta_pp']<.5 or strong_source_comparison['ci95'][0]<=0):
                    failures.append({'role':'ce_dpo','cohort':'transfer_dev','label':'1617 对旧裁词DPO强控制',**strong_source_comparison,'threshold':.5,'evidence':'新整图数据策略仍须超过旧强控制，不以训练loss或较弱控制替代。'})
                for old,comparison in analysis.get('full_rare342_comparisons',{}).items():
                    if comparison['delta_pp']<-.5:
                        failures.append({'role':'ce_dpo','cohort':'full_rare_1_5','label':'完整rare342对'+old,**comparison,'threshold':-.5,'evidence':'完整1617罕见答案点保护；区间与点门槛分列。'})
            source_holdout_validation = {}
            holdout_root = BASE/'vlm_stvqa_holdout_transfer_v1'
            if source_replay_study and (holdout_root/'PLAN.json').exists():
                holdout_queue = self.read(holdout_root/'queue/STATUS.json')
                holdout_done = self.read(holdout_root/'COMPLETION.json')
                holdout_counts = {role:sum(self.read(holdout_root/'roles'/role/'runs'/('shard_'+str(shard))/'PROGRESS.json').get('completed',0) for shard in (0,1)) for role in ('old_ce','old_dpo','new_ce','new_dpo')}
                total_checked = sum(holdout_counts.values())
                holdout_phase = '完整来源保持验证已完成' if holdout_done else '来源保持验证运行中' if holdout_queue.get('status')=='running' else '等待空闲GPU' if holdout_queue.get('status')=='waiting_resources' else '协议已冻结，待执行'
                holdout_record = {'name':'来源保持四Final','status':holdout_phase,'detail':f'{total_checked}/1024新生成与GT前向；256图×4冻结Final；0训练更新；非原native VQA。','jobs':holdout_queue.get('jobs',{}),'source':self.source(holdout_root/('COMPLETION.json' if holdout_done else 'queue/STATUS.json'))}
                auxiliary.append(holdout_record);history.append(holdout_record)
                source_holdout_validation = {'completed':total_checked,'expected':1024,'optimizer_updates':0,'roles':holdout_counts,'completion':holdout_done}
                sources += [self.source(PROJECT/'docs/VLM_STVQA_HOLDOUT_TRANSFER_V1_PROTOCOL.md','四固定Final的来源保持独立协议'),self.source(holdout_root/'PLAN.json','固定256图、四权重、0更新与原停止保持'),self.source(holdout_root/'来源保持_原理代码与实际回答.html','来源保持的公式、代码与预测前固定原图')]
                if not holdout_done:
                    root_state = 'running' if holdout_queue.get('status')=='running' else 'waiting' if holdout_queue.get('status')=='waiting_resources' else 'preparing'
                    research = f'主训练已结束；来源保持验证{total_checked}/1024（0更新）'
                else:
                    research = '主DPO未达标；四Final来源保持验证已完成'
                for job, record in holdout_queue.get('jobs',{}).items():
                    if record.get('state')=='failed':
                        log = holdout_root/'queue/logs'/(job+'.log')
                        runtime_errors.append({'job':job,'queue':'source_holdout','tail':log.read_text(errors='replace')[-3000:] if log.exists() else '等待失败日志','source':self.source(log)})
                        root_state = 'failed'
            full_teacher_validation = {}
            full_teacher_root = BASE/'vlm_full_teacher_signal_v1'
            if (full_teacher_root/'REGISTRATION.json').exists():
                teacher_assets = BASE/'vlm_full_teacher_assets_v1'
                acquisition = self.read(teacher_assets/'TRANSPORT_PROGRESS.json')
                acquisition_done = self.read(teacher_assets/'TRANSPORT_COMPLETE.json')
                teacher_queue = self.read(full_teacher_root/'queue/STATUS.json')
                teacher_done = self.read(full_teacher_root/'COMPLETION.json')
                teacher_cpu = self.read(full_teacher_root/'CPU_PROGRESS.json')
                progress_by_role = {r:self.read(full_teacher_root/'runs'/r/'PROGRESS.json') for r in ('student','teacher')}
                generated_teacher = progress_by_role['teacher'].get('new_generations',0)
                forced_teacher = sum(q.get('teacher_forced_sequences',0) for q in progress_by_role.values())
                if teacher_done:
                    teacher_phase = '整图教师有新增监督信号' if teacher_done.get('teacher_signal') else '固定整图教师未过信号门槛'
                    teacher_detail = f'128教师生成+256 GT前向完成；0更新；仅训练池教师准入，不是学生提分。'
                elif not acquisition_done:
                    teacher_phase = '获取并校验7B教师权重'
                    teacher_detail = f"{acquisition.get('bytes',0)/2**30:.2f}/{acquisition.get('expected_bytes',0)/2**30:.2f}GiB，{acquisition.get('verified_files',0)}/{acquisition.get('expected_files',15)}文件已验；0 GPU调用/0更新。"
                elif not (full_teacher_root/'PLAN.json').exists():
                    teacher_phase = '核验教师与学生实际token/图像输入'
                    teacher_detail = f"{teacher_cpu.get('completed',0)}/128 CPU输入；词表填充尺寸不同，实际token映射须一致。"
                else:
                    teacher_phase = '整图教师验证运行中' if teacher_queue.get('status')=='running' else '整图教师验证等待空卡'
                    teacher_detail = f'{generated_teacher}/128新教师生成、{forced_teacher}/256 GT前向；0训练更新。'
                teacher_record = {'name':'7B整图教师信号','status':teacher_phase,'detail':teacher_detail,
                    'jobs':teacher_queue.get('jobs',{}),'source':self.source(full_teacher_root/('COMPLETION.json' if teacher_done else 'queue/STATUS.json' if teacher_queue else 'STATUS.json'))}
                auxiliary.append(teacher_record);history.append(teacher_record)
                full_teacher_validation = {'phase':teacher_phase,'new_generations':generated_teacher,'teacher_forced_sequences':forced_teacher,
                    'optimizer_updates':0,'assets':acquisition,'completion':teacher_done}
                sources += [self.source(PROJECT/'docs/VLM_FULL_TEACHER_SIGNAL_V1_PROTOCOL.md','不同完整7B教师的事前监督信息门槛'),
                    self.source(full_teacher_root/'REGISTRATION.json','固定128训练池、较强SFT学生和新模型身份'),
                    self.source(BASE/'vlm_stvqa_holdout_transfer_v1/来源保持_结果与NLL解释.png','已完成来源保持的实际结果与NLL案例')]
                if not teacher_done:
                    root_state = 'running' if teacher_queue.get('status')=='running' else 'waiting' if teacher_queue.get('status')=='waiting_resources' else 'preparing'
                    research = '来源验证已结束；'+teacher_phase+'（0更新）'
                else:
                    research = teacher_phase+'；学生能力目标仍待完成'
                failure = self.read(full_teacher_root/'RUNTIME_ERROR.json')
                if failure:
                    root_state = 'failed'
                    runtime_errors.append({'job':'full_teacher_controller','queue':'teacher_signal','tail':str(failure),'source':self.source(full_teacher_root/'controller.log')})
                for key,job in teacher_queue.get('jobs',{}).items():
                    if job.get('state')=='failed':
                        root_state = 'failed';log=full_teacher_root/'queue/logs'/(key+'.log')
                        runtime_errors.append({'job':key,'queue':'teacher_signal','tail':log.read_text(errors='replace')[-2500:] if log.exists() else '等待日志','source':self.source(log)})
            representative = [c for c in cases if c['status'] == 'score_loss' and c.get('rare')][:3] + [c for c in cases if c['id'] in ('t-49', 't-64', 't-70')]
            for case in representative:
                case['full_image'] = 'data:image/jpeg;base64,' + base64.b64encode(self.images[case['id']].read_bytes()).decode()
            state = {'schema': 1, 'observed_at': now(), 'mode': 'snapshot', 'project': 'RareVLM', 'study': CURRENT.name, 'state': root_state, 'research': research,
                     'controller': {'pid': controller.get('pid'), 'alive': controller_running}, 'runtime_errors': runtime_errors,
                     'completion': completion, 'roles': roles, 'stages': stages, 'gpu': self.gpu(), 'screen': screen, 'analysis': analysis,
                     'failures': failures, 'gates': gate_rows, 'scores': scores, 'history': history, 'sources': sources,
                     'batch_diagnostic': batch_diagnostic,
                     'auxiliary_runs': auxiliary,
                     'independent_validation': independent_validation,
                     'source_holdout_validation': source_holdout_validation,
                     'full_teacher_validation': full_teacher_validation,
                     'regularizer': 'dpo' if preference_study else 'none' if scope_study else 'kl' if kl_study else 'dpo', 'numeric_policy': self.read(CURRENT / 'CONFIG.json').get('policy'),
                     'active_jobs': [{'label': name.replace('_queue', '') + ' / ' + key, **job} for name, qs in observed_queues.items() for key, job in qs.get('jobs', {}).items() if job.get('state') == 'running'] + [{'label': a['name'] + ' / ' + key, **job} for a in auxiliary for key, job in a['jobs'].items() if job.get('state') == 'running'],
                     'dataset': {'reviewed': review.get('reviewed_candidates'), 'supported': review.get('supported'), 'excluded': review.get('not_admitted_or_uncertain'),
                                 'selected': 192 if diversity_study and receipt else 128 if receipt else None, 'unique_targets': receipt.get('training_candidate_unique_targets') if diversity_study else receipt.get('hard128_distinct_targets'), 'strata': receipt.get('hard_length_strata'),
                                 'overlap': receipt.get('train_eval_image_pixel_overlap')},
                     'cases': cases, 'warnings': self.warnings, 'scope': '只读RareVLM当前研究流程与GPU进程；不操作任务、不读取模型权重。'}
            self.cached_state = state; self.cached_at = time.monotonic()
            return state

    @staticmethod
    def internal_count(progress):
        order = ['native_hard', 'literal_validation', 'seen_validation', 'retention']; sizes = [128, 256, 512, 64]
        name = progress.get('cohort')
        return sum(sizes[:order.index(name)]) + progress.get('completed', 0) if name in order else 0


def page(state):
    payload = json.dumps(state, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    template = TEMPLATE.read_text()
    if state.get('regularizer') == 'kd':
        template = template.replace("state.roles.map(r=>`${r.step}/${r.planned}`).join(' · ')", "`${state.roles.reduce((s,r)=>s+r.step,0)}/96`")
        template = template.replace("r.id==='sft'?'#127467':'#316f9e'", "r.color||'#316f9e'")
        template = template.replace('SFT / DPO＋SFT，固定各32步', 'GT / 均匀尾 / 教师KD，固定各32步')
        template = template.replace("['集合／指标','起点','SFT32','DPO＋SFT32']", "['集合／指标','起点','额外GT32','均匀尾32','教师KD32']")
        template = template.replace("['start','sft','dpo_sft']", "['start','hard','uniform','kd']")
        template = template.replace("${g.role==='sft'?'SFT':'DPO＋SFT'} · ${gnames[g.key]}", "教师KD · ${g.key}")
        template = template.replace('value="dpo"', 'value="kd"').replace("loss('mini-dpo','dpo'", "loss('mini-dpo','kd'")
        template = template.replace('偏好QA的DPO', '辅助T²KL（GT臂为T²CE）').replace('DPO损失', '辅助目标')
    if state.get('regularizer') == 'kl':
        for before, after in [('DPO＋SFT', 'SFT＋KL'), ('dpo_sft', 'sft_kl'), ('偏好QA的DPO', '回放QA的KL'), ('DPO损失', 'KL损失'), ('value="dpo"', 'value="kl"'), ('相对原SFT初始', '相对修复后新起点')]:
            template = template.replace(before, after)
    if state.get('regularizer') == 'none':
        template = template.replace('<option value="dpo">偏好QA的DPO</option>', '')
        template = template.replace('DPO损失也在SFT对照中计算，仅其反传权重为0；不同批次难度不同。', '本轮只优化逐QA CE，两组改变适配范围；不同批次难度不同。')
        for before, after in [('DPO＋SFT', '视觉LoRA'), ('SFT', '语言LoRA'), ('dpo_sft', 'visual'), ("'sft'", "'language'"), ('value="sft"', 'value="language"'), ('相对原语言LoRA初始', '相对共同复合起点')]:
            template = template.replace(before, after)
    if state.get('study') in ('vlm_visual_preference_v1', 'vlm_stvqa_replay_v1'):
        for before, after in [('DPO＋SFT32', '视觉CE+DPO32'), ('SFT32', '视觉CE32'), ('DPO＋SFT', '视觉CE+DPO'), ('>SFT<', '>视觉CE控制<'), ("?'SFT':", "?'视觉CE控制':"), ('DPO损失也在SFT对照中计算', 'DPO损失也在CE对照中计算'), ('dpo_sft', 'ce_dpo'), ("'sft'", "'ce'"), ('value="sft"', 'value="ce"')]:
            template = template.replace(before, after)
    if state.get('study') == 'vlm_native_diversity_v1':
        template = template.replace('DPO损失也在SFT对照中计算，仅其反传权重为0；不同批次难度不同。', '两组均反传CE+DPO，独立纠错样本数96/192；相同QA曝光，批次难度不同。')
        template = template.replace('SFT / DPO＋SFT，固定各32步', '重复96 / 覆盖192，固定各32步')
        template = template.replace("['审核准入',`${state.dataset.selected??'—'} / ${state.dataset.reviewed??'—'}`,`暂支持${state.dataset.supported??'—'}；归一化答案${state.dataset.unique_targets??'—'}`]", "['候选纠错池',`${state.dataset.selected??'—'} 题`,`新审核支持${state.dataset.supported??'—'}/${state.dataset.reviewed??'—'}；控制为96题子集`]")
        for before, after in [('DPO＋SFT32', '覆盖192'), ('SFT32', '重复96'), ('DPO＋SFT', '覆盖192'), ('>SFT<', '>重复96<'), ("?'SFT':", "?'重复96':"), ('dpo_sft', 'diverse192'), ("'sft'", "'repeat96'"), ('value="sft"', 'value="repeat96"')]:
            template = template.replace(before, after)
    template = template.replace('点保护要求 ≥ ${f.threshold.toFixed(1)} pp', '${f.cohort===\'transfer_dev\'?\'能力增益要求\':\'点保护要求\'} ≥ ${signed(f.threshold)} pp${f.cohort===\'transfer_dev\'?\'，且区间下界>0\':\'\'}')
    return template.replace('__STATE_JSON__', payload)


def snapshot(collector, render=False):
    state = collector.collect(force=True); OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / 'latest.json', state); html_path = OUT / '状态看板.html'; html_path.write_text(page(state))
    result = {'observed_at': state['observed_at'], 'html': str(html_path), 'state': state['state'], 'research': state['research']}
    if render:
        command = [sys.executable, str(PROJECT / 'tools/rarevlm_monitor_screenshot.py'), str(html_path), str(OUT)]
        env = dict(os.environ, PYTHONPATH='/opt/data/private/.tools/rarevlm-monitor/python', PLAYWRIGHT_BROWSERS_PATH='/opt/data/private/.tools/rarevlm-monitor/browsers')
        completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=90)
        result['render_exit_code'] = completed.returncode; result['render_output'] = completed.stdout[-1500:] + completed.stderr[-1500:]
    atomic_json(OUT / 'SNAPSHOT.json', result); print(json.dumps(result, ensure_ascii=False), flush=True)
    if render and result.get('render_exit_code') != 0:
        raise RuntimeError('Snapshot data saved; browser rendering failed. Do not present an old PNG as current.')


def serve(collector, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args): pass
        def send(self, body, mime='application/json; charset=utf-8', code=200):
            if isinstance(body, str): body = body.encode()
            self.send_response(code); self.send_header('Content-Type', mime); self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store'); self.send_header('X-Content-Type-Options', 'nosniff'); self.end_headers(); self.wfile.write(body)
        def do_GET(self):
            if self.headers.get('Host', '').split(':', 1)[0] not in ('localhost', '127.0.0.1'):
                self.send('local access only', 'text/plain', 403); return
            route = unquote(urlparse(self.path).path)
            try:
                if route in ('/', '/index.html'):
                    self.send(page(collector.collect()), 'text/html; charset=utf-8')
                elif route == '/api/state':
                    state = dict(collector.collect()); state['mode'] = 'live'; self.send(json.dumps(state, ensure_ascii=False))
                elif route.startswith('/api/image/'):
                    path = collector.images.get(route.removeprefix('/api/image/'))
                    if path and path.exists(): self.send(path.read_bytes(), 'image/jpeg')
                    else: self.send('image not found', 'text/plain', 404)
                elif route.startswith('/api/evidence/'):
                    path = collector.artifacts.get(route.removeprefix('/api/evidence/'))
                    if path and path.exists(): self.send(path.read_bytes(), 'text/plain; charset=utf-8')
                    else: self.send('evidence not available', 'text/plain', 404)
                else:
                    self.send('not found', 'text/plain', 404)
            except (BrokenPipeError, ConnectionResetError): pass
            except Exception as error:
                self.send(json.dumps({'error': str(error)}), code=500)
        def do_POST(self): self.send('read-only', 'text/plain', 405)
    collector.collect(); server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    record = {'pid': os.getpid(), 'start_ticks': Path('/proc/self/stat').read_text().split()[21], 'url': f'http://127.0.0.1:{server.server_port}', 'started_at': now(), 'read_only': True, 'run_root': str(CURRENT)}
    atomic_json(OUT / 'SERVER.json', record); print(json.dumps(record), flush=True); server.serve_forever()


def main():
    global CURRENT, ROLE_LABELS
    p = argparse.ArgumentParser(); p.add_argument('--snapshot', action='store_true'); p.add_argument('--render', action='store_true'); p.add_argument('--serve', action='store_true'); p.add_argument('--port', type=int, default=8765); p.add_argument('--run-root'); args = p.parse_args()
    registry = OUT / 'ACTIVE_STUDY.json'
    selected = args.run_root or (json.loads(registry.read_text()).get('run_root') if registry.exists() else None)
    if selected:
        CURRENT = Path(selected).resolve(); CURRENT.relative_to(BASE)
        if not (CURRENT / 'data/MANIFEST.json').exists():
            raise ValueError('This adapter needs a registered native-correction schema; do not treat missing data as zero.')
    if CURRENT.name == 'vlm_native_stable_kl_v1': ROLE_LABELS = {'sft': 'SFT', 'sft_kl': 'SFT＋KL'}
    if CURRENT.name == 'vlm_native_scope_v1': ROLE_LABELS = {'language': '语言LoRA', 'visual': '视觉LoRA'}
    if CURRENT.name == 'vlm_visual_preference_v1': ROLE_LABELS = {'ce': '视觉CE控制', 'ce_dpo': '视觉CE+DPO'}
    if CURRENT.name == 'vlm_stvqa_replay_v1': ROLE_LABELS = {'ce': '整图回放CE', 'ce_dpo': '整图回放CE+DPO'}
    if CURRENT.name == 'vlm_full_teacher_kd_v1': ROLE_LABELS = {'hard': '额外GT', 'uniform': '均匀尾', 'kd': '教师KD'}
    if CURRENT.name == 'vlm_onpolicy_teacher_kd_v1': ROLE_LABELS = {'op_kd': '学生前缀KD'}
    if CURRENT.name == 'vlm_late_visual_dpo_v1': ROLE_LABELS = {'ce_dpo': '后10层视觉DPO'}
    if CURRENT.name == 'vlm_late_rank_matched_dpo_v1': ROLE_LABELS = {'ce_dpo': '后10层r36 DPO'}
    if CURRENT.name == 'vlm_native_diversity_v1':
        ROLE_LABELS = {'repeat96': '重复96（DPO）', 'diverse192': '覆盖192（DPO）'}
        COHORTS['native_hard'] = '原纠错128诊断'
    collector = Collector()
    if args.snapshot: snapshot(collector, args.render)
    if args.serve: serve(collector, args.port)
    if not args.snapshot and not args.serve: p.print_help()


if __name__ == '__main__': main()
