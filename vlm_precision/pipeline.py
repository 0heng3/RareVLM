"""Non-preemptive GPU admission and resumable numerical-path experiment queue."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[1]
PRIVATE = Path('/opt/data/private')
GPU_LOCK_ROOT = PRIVATE/'.tools/gpu-admission'


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def gpu_snapshot():
    query = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    devices = {}
    for parts in csv.reader(query.stdout.splitlines()):
        parts = [x.strip() for x in parts]
        if not parts:
            continue
        index, uuid, name, total, used, utilization = parts
        devices[int(index)] = {'index': int(index), 'uuid': uuid, 'name': name,
                               'memory_total_mib': int(total), 'memory_used_mib': int(used),
                               'free_gib': (int(total)-int(used))/1024,
                               'utilization_percent': int(utilization), 'compute_processes': []}
    query = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    by_uuid = {device['uuid']: device for device in devices.values()}
    for parts in csv.reader(query.stdout.splitlines()):
        parts = [x.strip() for x in parts]
        if not parts or len(parts) != 3:
            continue
        uuid, pid, memory = parts
        if uuid in by_uuid:
            by_uuid[uuid]['compute_processes'].append({'host_pid': int(pid), 'memory_mib': memory})
    return devices


def available(device):
    return not device['compute_processes'] and device['free_gib'] >= 40 and device['utilization_percent'] < 10


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-root', type=Path, required=True)
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=REPO/'configs/vlm_precision_bridge_v1.json')
    parser.add_argument('--poll-seconds', type=int, default=15)
    parser.add_argument('--inspect-only', action='store_true')
    args = parser.parse_args()
    devices = gpu_snapshot()
    config = json.loads(args.config.read_text())
    if args.inspect_only:
        print(json.dumps({role: {'device': devices[index], 'admit_now': available(devices[index])}
                          for role, index in config['gpu_assignment'].items()}, ensure_ascii=False, indent=2))
        return
    root = args.output_root.resolve()
    if not root.is_relative_to(PRIVATE):
        raise ValueError('queue artifacts must remain under /opt/data/private')
    if args.poll_seconds < 10:
        raise ValueError('resource polling must be at least ten seconds apart')
    for role, index in config['gpu_assignment'].items():
        device = devices[index]
        if 'A100' not in device['name'] or device['memory_total_mib'] < 70*1024:
            raise ValueError(f'{role} requires the declared A100 80GB')
    root.mkdir(parents=True, exist_ok=True)
    (root/'logs').mkdir(exist_ok=True)
    owner = (root/'PIPELINE.lock').open('a+')
    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    reference_config = REPO/config['reference_config']
    adapter = Path(config['fixed_adapter_dir'])
    locked = {'schema_version': 'rarevlm.precision_queue.v1',
              'config_sha256': sha_file(args.config), 'reference_config_sha256': sha_file(reference_config),
              'asset_root': str(args.asset_root.resolve()), 'model_root': str(args.model_root.resolve()),
              'model_provenance_sha256': sha_file(args.model_root.parent/'model_provenance.json'),
              'adapter_files_sha256': {p.name: sha_file(p) for p in sorted(adapter.iterdir()) if p.is_file()},
              'controller_sha256': sha_file(Path(__file__)),
              'worker_sha256': sha_file(REPO/'vlm_precision/evaluate.py'),
              'gpu_assignment': config['gpu_assignment'], 'paths': config['paths'], 'batch_size': config['batch_size'],
              'admission': {'minimum_free_gib': 40, 'maximum_utilization_exclusive': 10,
                            'require_zero_compute_processes': True, 'consecutive_checks': 3},
              'process_preemption': False}
    prior = read_json(root/'PIPELINE_INPUT_LOCK.json')
    if prior is not None and prior != locked:
        raise ValueError('queue input/source lock changed; use a new output root')
    write_json(root/'PIPELINE_INPUT_LOCK.json', locked)
    started = now()
    write_json(root/'PID.json', {'pid': os.getpid(), 'started_at': started, 'python': sys.executable})
    GPU_LOCK_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    prior_contention = read_json(root/'RESOURCE_CONTENTION.json') or {}
    previous_counts = prior_contention.get('observations_by_role', {})
    states = {role: {'state': 'waiting_gpu', 'gpu_index': index, 'idle_checks': 0,
                     'contention_observations': previous_counts.get(role, 0)}
              for role, index in config['gpu_assignment'].items()}
    running, logs, gpu_locks = {}, {}, {}
    stop_admitting = False

    def signal_request(signum, frame):
        nonlocal stop_admitting
        # Existing and newly admitted GPU tasks are deliberately left running.
        stop_admitting = True
    signal.signal(signal.SIGTERM, signal_request)
    signal.signal(signal.SIGINT, signal_request)

    def cells():
        result = []
        for role in config['roles']:
            for path in config['paths']:
                directory = root/'runs'/role/path
                evaluation = read_json(directory/'EVALUATION.json')
                result.append({'role': role, 'numerical_path': path, 'complete': evaluation is not None,
                               'score_percent': evaluation.get('score_percent') if evaluation else None,
                               'progress': read_json(directory/'PROGRESS.json')})
        return result

    def status(phase, error=None):
        all_cells = cells()
        value = {'schema_version': 'rarevlm.precision_queue_status.v1', 'phase': phase, 'updated_at': now(),
                 'started_at': started, 'pid': os.getpid(), 'states': states,
                 'completed_cells': sum(item['complete'] for item in all_cells), 'total_cells': len(all_cells),
                 'cells': all_cells, 'resource_contention_by_role': {r: s['contention_observations'] for r,s in states.items()},
                 'role_status': {r: read_json(root/'runs'/r/'STATUS.json') for r in config['roles']},
                 'config_sha256': locked['config_sha256'], 'existing_tasks_untouched': True,
                 'scope': config['scope']}
        if error:
            value['error'] = error
        write_json(root/'STATUS.json', value)

    def all_role_cells_complete(role):
        return all((root/'runs'/role/path/'EVALUATION.json').exists() for path in config['paths'])

    def admit(role, device):
        lock = (GPU_LOCK_ROOT/(device['uuid']+'.lock')).open('a+')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            states[role]['waiting_reason'] = 'cooperative GPU lock held'
            states[role]['idle_checks'] = 0
            return
        # Recheck after the cooperative lock; no process is interrupted if busy.
        current = gpu_snapshot()[device['index']]
        if current['uuid'] != device['uuid'] or not available(current):
            lock.close()
            states[role]['idle_checks'] = 0
            return
        cmd = [sys.executable, '-m', 'vlm_precision.evaluate', '--asset-root', str(args.asset_root),
               '--model-root', str(args.model_root), '--output-root', str(root), '--config', str(args.config),
               '--role', role, '--paths', *config['paths'], '--batch-size', str(config['batch_size'])]
        if role == 'adapted':
            cmd += ['--adapter-dir', config['fixed_adapter_dir']]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=device['uuid'], PYTHONUNBUFFERED='1',
                   PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python')
        for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK'):
            env.pop(key, None)
        log = (root/'logs'/f'{role}_evaluation.log').open('a')
        try:
            process = subprocess.Popen(cmd, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT)
        except BaseException:
            lock.close()
            log.close()
            raise
        running[role], logs[role], gpu_locks[role] = process, log, lock
        states[role].update(state='running', pid=process.pid, admitted_at=now(),
                            admission_snapshot=current, gpu_uuid=device['uuid'], waiting_reason=None)
        print('ADMITTED', role, device['index'], process.pid, now(), flush=True)

    status('waiting_resources')
    try:
        while True:
            devices = gpu_snapshot()
            snapshot = {'time': now(), 'devices': devices, 'role_states': {r:s['state'] for r,s in states.items()}}
            with (root/'RESOURCE_SNAPSHOTS.jsonl').open('a') as stream:
                stream.write(json.dumps(snapshot, ensure_ascii=False)+'\n')
            for role, index in config['gpu_assignment'].items():
                state = states[role]
                device = devices[index]
                state['last_resource_snapshot'] = device
                if role in running:
                    process = running[role]
                    code = process.poll()
                    if code is not None:
                        state.update(state='completed' if code == 0 else 'failed', returncode=code, finished_at=now())
                        logs.pop(role).close()
                        gpu_locks.pop(role).close()
                        del running[role]
                    elif len(device['compute_processes']) > 1:
                        state['contention_observations'] += 1
                        event = {'time': now(), 'role': role, 'processes': device['compute_processes'],
                                 'effect': 'resource timings marked contended; all processes remain running'}
                        with (root/'RESOURCE_CONTENTION.jsonl').open('a') as stream:
                            stream.write(json.dumps(event, ensure_ascii=False)+'\n')
                    continue
                if state['state'] in ('completed', 'failed'):
                    continue
                if all_role_cells_complete(role):
                    state.update(state='completed', finished_at=now(), cached_complete=True)
                    continue
                if stop_admitting:
                    state['state'] = 'admission_stopped'
                    continue
                if available(device):
                    state['idle_checks'] += 1
                    state['waiting_reason'] = f'confirming idle ({state["idle_checks"]}/3)'
                else:
                    state['idle_checks'] = 0
                    state['waiting_reason'] = 'GPU compute process, utilization or memory indicates occupied device'
                if state['idle_checks'] >= 3:
                    admit(role, device)
            write_json(root/'RESOURCE_CONTENTION.json', {
                'updated_at': now(), 'observations_by_role': {r:s['contention_observations'] for r,s in states.items()},
                'timing_interpretation': 'any observed same-GPU co-resident compute process marks role speed results contended; absence of sampled contention is not an exclusivity guarantee'})
            phase = 'running' if running else 'waiting_resources'
            if stop_admitting:
                phase = 'admission_stopped_waiting_existing_workers' if running else 'interrupted'
            status(phase)
            if all(s['state'] in ('completed', 'failed') for s in states.values()):
                break
            if stop_admitting and not running:
                print('ADMISSION STOPPED; no GPU tasks were terminated', now(), flush=True)
                return
            time.sleep(args.poll_seconds)
        status('analysis')
        with (root/'logs/analysis.log').open('a') as log:
            subprocess.run([sys.executable, str(REPO/'tools/analyze_vlm_precision_bridge.py'),
                            '--asset-root', str(args.asset_root), '--probe-root', str(root)],
                           cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
        analysis = read_json(root/'analysis/ANALYSIS.json')
        if any(s['state'] == 'failed' for s in states.values()) or not analysis or not analysis.get('complete'):
            status('failed', error='at least one worker/cell incomplete; other tasks were left running; inspect role logs')
        else:
            status('complete')
            print('PRECISION BRIDGE COMPLETE', now(), flush=True)
    except BaseException as error:
        # Closing a log or cooperative lock does not signal its subprocess.
        status('controller_failed_workers_left_running', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        for log in logs.values():
            log.close()
        for lock in gpu_locks.values():
            lock.close()
        owner.close()


if __name__ == '__main__':
    main()
