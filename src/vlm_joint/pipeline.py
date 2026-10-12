"""Dynamic idle-GPU admission for fixed-checkpoint BF16 visual-budget diagnostics."""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True
from vlm_precision.pipeline import (GPU_LOCK_ROOT, PRIVATE, available,
                                    now, read_json, sha_file, write_json)

REPO = PROJECT_ROOT


def gpu_snapshot():
    """Reject unrecognized resource output; never infer idle from missing rows."""
    query = subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu',
                            '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    devices = {}
    for row in csv.reader(query.stdout.splitlines(), strict=True):
        parts = [x.strip() for x in row]
        if not any(parts):continue
        if len(parts) != 6:raise ValueError('unexpected GPU resource CSV row')
        index,uuid,name,total,used,util = parts
        index,total,used,util = int(index),int(total),int(used),int(util)
        if index in devices or not uuid or total <= 0 or not 0 <= used <= total or not 0 <= util <= 100:
            raise ValueError('invalid or duplicate GPU resource record')
        devices[index] = {'index':index,'uuid':uuid,'name':name,'memory_total_mib':total,
                          'memory_used_mib':used,'free_gib':(total-used)/1024,
                          'utilization_percent':util,'compute_processes':[]}
    if not devices:raise ValueError('GPU resource query returned no devices')
    by_uuid = {d['uuid']:d for d in devices.values()}
    if len(by_uuid) != len(devices):raise ValueError('duplicate GPU UUID')
    query = subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory',
                            '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    for row in csv.reader(query.stdout.splitlines(), strict=True):
        parts = [x.strip() for x in row]
        if not any(parts):continue
        if len(parts) != 3:raise ValueError('unexpected compute-process CSV row')
        uuid,pid,memory = parts
        if uuid not in by_uuid or int(pid) <= 0:raise ValueError('unknown GPU or invalid process in resource query')
        by_uuid[uuid]['compute_processes'].append({'host_pid':int(pid),'memory_mib':memory})
    return devices


def qualified(device):
    return ('A100' in device['name'] and device['memory_total_mib'] >= 70*1024
            and available(device))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-root', required=True, type=Path)
    parser.add_argument('--model-root', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--config', type=Path, default=REPO/'configs/vlm_joint_bf16_v1.json')
    parser.add_argument('--poll-seconds', type=int, default=15)
    parser.add_argument('--inspect-only', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    snapshot = gpu_snapshot()
    if args.inspect_only:
        print(json.dumps({'devices': snapshot, 'qualified_now': [i for i,d in snapshot.items() if qualified(d)],
                          'preferences': config['gpu_preferences'], 'process_preemption': False},
                         ensure_ascii=False, indent=2))
        return
    root = args.output_root.resolve()
    if not root.is_relative_to(PRIVATE) or args.poll_seconds < 10:
        raise ValueError('persistent root and at least ten-second polling required')
    root.mkdir(parents=True, exist_ok=True)
    (root/'logs').mkdir(exist_ok=True)
    owner = (root/'PIPELINE.lock').open('a+')
    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    adapter = Path(config['fixed_adapter_dir'])
    lock = {'schema_version': 'rarevlm.joint_bf16_queue.v1', 'config_sha256': sha_file(args.config),
            'reference_config_sha256': sha_file(REPO/config['reference_config']),
            'asset_root': str(args.asset_root.resolve()), 'model_root': str(args.model_root.resolve()),
            'model_provenance_sha256': sha_file(args.model_root.parent/'model_provenance.json'),
            'adapter_files_sha256': {p.name:sha_file(p) for p in sorted(adapter.iterdir()) if p.is_file()},
            'source_sha256': {name:sha_file(SOURCE_ROOT/name) for name in (
                'vlm_joint/pipeline.py','vlm_joint/evaluate.py','vlm_precision/pipeline.py','rarevlm_paths.py')},
            'roles': config['roles'], 'budgets': config['visual_token_budgets'],
            'batch_size': config['batch_size'], 'preferences': config['gpu_preferences'],
            'admission': 'zero compute processes, >=40GiB free, <10% utilization, three consecutive samples, per-UUID lock',
            'process_preemption': False}
    old_lock = read_json(root/'PIPELINE_INPUT_LOCK.json')
    if old_lock is not None and old_lock != lock:
        raise ValueError('joint queue input/source lock changed; use a new output root')
    write_json(root/'PIPELINE_INPUT_LOCK.json', lock)
    started = now()
    write_json(root/'PID.json', {'pid':os.getpid(),'started_at':started,'python':sys.executable})
    GPU_LOCK_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    previous = read_json(root/'RESOURCE_CONTENTION.json') or {}
    counts = previous.get('observations_by_role', {})
    states = {r:{'state':'waiting_gpu','gpu_index':None,'contention_observations':counts.get(r,0)}
              for r in config['roles']}
    idle_counts, running, logs, gpu_locks = {}, {}, {}, {}
    stop_admitting = False

    def request_stop(signum, frame):
        nonlocal stop_admitting
        stop_admitting = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def cells():
        result = []
        for role in config['roles']:
            for budget in config['visual_token_budgets']:
                directory = root/'runs'/role/f'budget_{budget}'
                evaluation = read_json(directory/'EVALUATION.json')
                result.append({'role':role,'visual_token_budget':budget,'complete':evaluation is not None,
                               'score_percent':evaluation.get('score_percent') if evaluation else None,
                               'progress':read_json(directory/'PROGRESS.json')})
        return result

    def status(phase, error=None):
        all_cells = cells()
        value = {'schema_version':'rarevlm.joint_bf16_status.v1','phase':phase,'updated_at':now(),
                 'started_at':started,'pid':os.getpid(),'states':states,
                 'completed_cells':sum(c['complete'] for c in all_cells),'total_cells':len(all_cells),
                 'cells':all_cells,'idle_confirmations':idle_counts,
                 'role_status':{r:read_json(root/'runs'/r/'STATUS.json') for r in config['roles']},
                 'resource_contention_by_role':{r:s['contention_observations'] for r,s in states.items()},
                 'existing_tasks_untouched':True,'config_sha256':lock['config_sha256'],'scope':config['scope']}
        if error:
            value['error'] = error
        write_json(root/'STATUS.json', value)

    def role_complete(role):
        return all((root/'runs'/role/f'budget_{b}'/'EVALUATION.json').exists()
                   for b in config['visual_token_budgets'])

    def admit(role, device):
        gpu_lock = (GPU_LOCK_ROOT/(device['uuid']+'.lock')).open('a+')
        try:
            fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            gpu_lock.close()
            return False
        try:
            current = gpu_snapshot().get(device['index'])
        except (subprocess.SubprocessError, ValueError, OSError, csv.Error) as error:
            gpu_lock.close()
            idle_counts.clear()
            status('resource_query_failed', error=str(error))
            return False
        if current is None or current['uuid'] != device['uuid'] or not qualified(current):
            gpu_lock.close()
            return False
        cmd = [sys.executable,'-m','vlm_joint.evaluate','--asset-root',str(args.asset_root),
               '--model-root',str(args.model_root),'--output-root',str(root),'--config',str(args.config),
               '--role',role,'--budgets',*map(str,config['visual_token_budgets']),
               '--batch-size',str(config['batch_size'])]
        if role == 'adapted':
            cmd += ['--adapter-dir',config['fixed_adapter_dir']]
        env = dict(os.environ,CUDA_VISIBLE_DEVICES=device['uuid'],PYTHONUNBUFFERED='1',
                   PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python')
        for field in ('RANK','WORLD_SIZE','LOCAL_RANK'):
            env.pop(field,None)
        log = (root/'logs'/f'{role}_evaluation.log').open('a')
        try:
            process = subprocess.Popen(cmd,cwd=REPO,env=env,stdin=subprocess.DEVNULL,
                                       stdout=log,stderr=subprocess.STDOUT,pass_fds=(gpu_lock.fileno(),))
        except BaseException:
            gpu_lock.close();log.close()
            raise
        running[role],logs[role],gpu_locks[role] = process,log,gpu_lock
        states[role].update(state='running',pid=process.pid,gpu_index=device['index'],gpu_uuid=device['uuid'],
                            admitted_at=now(),admission_snapshot=current,waiting_reason=None)
        print('ADMITTED',role,'GPU',device['index'],'pid',process.pid,now(),flush=True)
        return True

    status('waiting_resources')
    try:
        while True:
            for role,process in list(running.items()):
                code = process.poll()
                if code is not None:
                    states[role].update(state='completed' if code == 0 else 'failed',returncode=code,finished_at=now())
                    logs.pop(role).close();gpu_locks.pop(role).close();del running[role]
            try:
                snapshot = gpu_snapshot()
            except (subprocess.SubprocessError, ValueError, OSError, csv.Error) as error:
                idle_counts.clear()
                status('resource_query_failed',error=str(error))
                time.sleep(args.poll_seconds)
                continue
            reserved = {states[r]['gpu_index'] for r in running}
            for index,device in snapshot.items():
                key = device['uuid']
                idle_counts[key] = idle_counts.get(key,0)+1 if index not in reserved and qualified(device) else 0
            with (root/'RESOURCE_SNAPSHOTS.jsonl').open('a') as stream:
                stream.write(json.dumps({'time':now(),'devices':snapshot,'reserved_indices':sorted(reserved)},ensure_ascii=False)+'\n')
            for role in running:
                device = snapshot.get(states[role]['gpu_index'])
                if device is not None and len(device['compute_processes']) > 1:
                    states[role]['contention_observations'] += 1
                    with (root/'RESOURCE_CONTENTION.jsonl').open('a') as stream:
                        stream.write(json.dumps({'time':now(),'role':role,'processes':device['compute_processes'],
                                                 'effect':'timing tagged contended; no process interrupted'})+'\n')
            for role,state in states.items():
                if state['state'] in ('running','completed','failed'):
                    continue
                if role_complete(role):
                    state.update(state='completed',cached_complete=True,finished_at=now())
                    continue
                if stop_admitting:
                    state['state'] = 'admission_stopped'
                    continue
                preferences = config['gpu_preferences'][role]
                candidates = preferences+[i for i in sorted(snapshot) if i not in preferences]
                admitted = False
                for index in candidates:
                    device = snapshot.get(index)
                    if device is None or index in reserved or idle_counts.get(device['uuid'],0) < 3:
                        continue
                    if admit(role,device):
                        reserved.add(index);admitted=True;break
                if not admitted:
                    state['waiting_reason'] = 'waiting for any qualified idle GPU, three checks and cooperative lock; existing tasks preserved'
            write_json(root/'RESOURCE_CONTENTION.json',{'updated_at':now(),
                       'observations_by_role':{r:s['contention_observations'] for r,s in states.items()},
                       'timing_interpretation':'zero sampled contention is not an exclusivity guarantee; historical reference timing labeled separately'})
            phase = 'running' if running else 'waiting_resources'
            if stop_admitting:
                phase = 'admission_stopped_waiting_workers' if running else 'interrupted'
            status(phase)
            if all(s['state'] in ('completed','failed') for s in states.values()):
                break
            if stop_admitting and not running:
                return
            time.sleep(args.poll_seconds)
        script = REPO/'tools/analyze_vlm_joint_bf16.py'
        while True:
            ready = read_json(root/'ANALYSIS_READY.json')
            if ready and script.exists() and ready.get('analysis_source_sha256') == sha_file(script):
                break
            status('waiting_analysis_ready')
            if stop_admitting:
                return
            time.sleep(args.poll_seconds)
        status('analysis')
        with (root/'logs/analysis.log').open('a') as log:
            subprocess.run([sys.executable,str(script),'--asset-root',str(args.asset_root),'--probe-root',str(root),
                            '--reference-root',config['reference_root'],'--nf4-budget-root',config['nf4_budget_root']],
                           cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
        result = read_json(root/'analysis/ANALYSIS.json')
        if any(s['state']=='failed' for s in states.values()) or not result or not result.get('complete'):
            status('failed',error='one or more cells/analysis incomplete; other tasks preserved, inspect logs')
        else:
            status('complete')
            print('JOINT BF16 BUDGET COMPLETE',now(),flush=True)
    except BaseException as error:
        status('controller_failed_workers_left_running',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        for stream in logs.values():stream.close()
        for gpu_lock in gpu_locks.values():gpu_lock.close()
        owner.close()


if __name__ == '__main__':
    main()
