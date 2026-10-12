"""Non-preemptive four-model queue for the frozen BF16 formal benchmark protocol."""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True
from vlm_joint.pipeline import gpu_snapshot, qualified
from vlm_precision.pipeline import GPU_LOCK_ROOT, PRIVATE, now, read_json, sha_file, write_json

REPO = Path(__file__).resolve().parents[1]
RESOURCE_ERRORS = (subprocess.SubprocessError, ValueError, OSError, csv.Error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-root', required=True, type=Path)
    parser.add_argument('--model-root', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--config', type=Path, default=REPO/'configs/vlm_formal_bf16_v1.json')
    parser.add_argument('--poll-seconds', type=int, default=15)
    parser.add_argument('--inspect-only', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    snapshot = gpu_snapshot()
    if args.inspect_only:
        print(json.dumps({'devices':snapshot,'qualified_now':[i for i,d in snapshot.items() if qualified(d)],
                          'model_keys':config['model_keys'],'tasks':config['tasks'],
                          'cells':len(config['model_keys'])*len(config['tasks']),
                          'questions':len(config['model_keys'])*sum(config['expected_questions'].values()),
                          'process_preemption':False},ensure_ascii=False,indent=2))
        return
    root = args.output_root.resolve()
    if not root.is_relative_to(PRIVATE) or args.poll_seconds < 10:
        raise ValueError('persistent output and >=10 second resource polling required')
    root.mkdir(parents=True,exist_ok=True)
    (root/'logs').mkdir(exist_ok=True)
    owner = (root/'PIPELINE.lock').open('a+')
    fcntl.flock(owner,fcntl.LOCK_EX | fcntl.LOCK_NB)
    dev_source = Path(config['dev_selection_source'])
    lock = {'schema_version':'rarevlm.formal_bf16_queue.v1','config_sha256':sha_file(args.config),
            'reference_config_sha256':sha_file(REPO/config['reference_config']),
            'dev_selection_source_sha256':sha_file(dev_source),
            'asset_root':str(args.asset_root.resolve()),'model_root':str(args.model_root.resolve()),
            'model_provenance_sha256':sha_file(args.model_root.parent/'model_provenance.json'),
            'adapter_files_sha256':{key:{p.name:sha_file(p) for p in sorted(Path(directory).iterdir()) if p.is_file()}
                                    for key,directory in config['fixed_adapter_dirs'].items()},
            'source_sha256':{name:sha_file(REPO/name) for name in (
                'vlm_formal/pipeline.py','vlm_formal/evaluate.py','vlm_joint/pipeline.py','vlm_precision/pipeline.py')},
            'model_keys':config['model_keys'],'tasks':config['tasks'],
            'selection':'dev documented before formal; no formal-result-driven configuration changes',
            'process_preemption':False}
    previous = read_json(root/'PIPELINE_INPUT_LOCK.json')
    if previous is not None and previous != lock:
        raise ValueError('formal queue input/source changed; use a new protocol output')
    write_json(root/'PIPELINE_INPUT_LOCK.json',lock)
    started = now()
    write_json(root/'PID.json',{'pid':os.getpid(),'started_at':started,'python':sys.executable})
    GPU_LOCK_ROOT.mkdir(mode=0o700,parents=True,exist_ok=True)
    prior = read_json(root/'RESOURCE_CONTENTION.json') or {}
    counts = prior.get('observations_by_model',{})
    old_status = read_json(root/'STATUS.json') or {}
    old_states = old_status.get('states',{})
    states = {key:{'state':'waiting_gpu','gpu_index':None,
                   'contention_observations':counts.get(key,0)} for key in config['model_keys']}
    for key in states:
        if any((root/'runs'/key/task/'predictions.jsonl').exists() for task in config['tasks']):
            states[key]['resume_gpu_uuid'] = old_states.get(key,{}).get('gpu_uuid')
    idle_counts,running,logs,gpu_locks = {},{},{},{}
    stop_admitting = False

    def request_stop(signum, frame):
        nonlocal stop_admitting
        stop_admitting = True
    signal.signal(signal.SIGTERM,request_stop)
    signal.signal(signal.SIGINT,request_stop)

    def cells():
        result = []
        for key in config['model_keys']:
            for task in config['tasks']:
                directory = root/'runs'/key/task
                evaluation = read_json(directory/'EVALUATION.json')
                result.append({'model_key':key,'task':task,'complete':evaluation is not None,
                               'score_percent':evaluation.get('score_percent') if evaluation else None,
                               'progress':read_json(directory/'PROGRESS.json')})
        return result

    def status(phase,error=None):
        all_cells = cells()
        value = {'schema_version':'rarevlm.formal_bf16_status.v1','phase':phase,'updated_at':now(),
                 'started_at':started,'pid':os.getpid(),'states':states,
                 'completed_cells':sum(c['complete'] for c in all_cells),'total_cells':len(all_cells),
                 'expected_questions':config['expected_new_questions'],'cells':all_cells,
                 'idle_confirmations':idle_counts,
                 'model_status':{key:read_json(root/'runs'/key/'STATUS.json') for key in config['model_keys']},
                 'existing_tasks_untouched':True,'config_sha256':lock['config_sha256'],'scope':config['scope']}
        if error:value['error']=error
        write_json(root/'STATUS.json',value)

    def complete(key):
        return all((root/'runs'/key/task/'EVALUATION.json').exists() for task in config['tasks'])

    def admit(key,device):
        gpu_lock = (GPU_LOCK_ROOT/(device['uuid']+'.lock')).open('a+')
        try:fcntl.flock(gpu_lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            gpu_lock.close();return False
        try:current = gpu_snapshot().get(device['index'])
        except RESOURCE_ERRORS as error:
            gpu_lock.close();idle_counts.clear();status('resource_query_failed',str(error));return False
        if current is None or current['uuid']!=device['uuid'] or not qualified(current):
            gpu_lock.close();return False
        cmd = [sys.executable,'-m','vlm_formal.evaluate','--asset-root',str(args.asset_root),
               '--model-root',str(args.model_root),'--output-root',str(root),'--config',str(args.config),
               '--model-key',key,'--tasks',*config['tasks'],'--batch-size',str(config['batch_size'])]
        if key!='base':cmd += ['--adapter-dir',config['fixed_adapter_dirs'][key]]
        env = dict(os.environ,CUDA_VISIBLE_DEVICES=device['uuid'],PYTHONUNBUFFERED='1',
                   PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python')
        for field in ('RANK','WORLD_SIZE','LOCAL_RANK'):env.pop(field,None)
        log = (root/'logs'/f'{key}_evaluation.log').open('a')
        try:
            process = subprocess.Popen(cmd,cwd=REPO,env=env,stdin=subprocess.DEVNULL,
                                       stdout=log,stderr=subprocess.STDOUT,pass_fds=(gpu_lock.fileno(),))
        except BaseException:
            gpu_lock.close();log.close();raise
        running[key],logs[key],gpu_locks[key] = process,log,gpu_lock
        states[key].update(state='running',pid=process.pid,gpu_index=device['index'],gpu_uuid=device['uuid'],
                           admitted_at=now(),admission_snapshot=current,waiting_reason=None)
        print('ADMITTED',key,'GPU',device['index'],'pid',process.pid,now(),flush=True)
        return True

    status('waiting_resources')
    try:
        while True:
            for key,process in list(running.items()):
                code=process.poll()
                if code is not None:
                    states[key].update(state='completed' if code==0 else 'failed',returncode=code,finished_at=now())
                    logs.pop(key).close();gpu_locks.pop(key).close();del running[key]
            try:snapshot=gpu_snapshot()
            except RESOURCE_ERRORS as error:
                idle_counts.clear();status('resource_query_failed',str(error));time.sleep(args.poll_seconds);continue
            reserved={states[key]['gpu_index'] for key in running}
            for index,device in snapshot.items():
                uuid=device['uuid']
                idle_counts[uuid]=idle_counts.get(uuid,0)+1 if index not in reserved and qualified(device) else 0
            with (root/'RESOURCE_SNAPSHOTS.jsonl').open('a') as stream:
                stream.write(json.dumps({'time':now(),'devices':snapshot,'reserved_indices':sorted(reserved)},ensure_ascii=False)+'\n')
            for key in running:
                device=snapshot.get(states[key]['gpu_index'])
                if device is not None and len(device['compute_processes'])>1:
                    states[key]['contention_observations']+=1
                    current=read_json(root/'runs'/key/'STATUS.json') or {}
                    with (root/'RESOURCE_CONTENTION.jsonl').open('a') as stream:
                        stream.write(json.dumps({'time':now(),'model_key':key,'task':current.get('current_task'),
                                                 'host_processes':device['compute_processes'],
                                                 'effect':'timing tagged contended; no process interrupted'})+'\n')
            for key,state in states.items():
                if state['state'] in ('running','completed','failed'):continue
                if complete(key):state.update(state='completed',cached_complete=True,finished_at=now());continue
                if stop_admitting:state['state']='admission_stopped';continue
                preferences=config['gpu_preferences'][key]
                candidates=preferences+[i for i in sorted(snapshot) if i not in preferences]
                admitted=False
                for index in candidates:
                    device=snapshot.get(index)
                    if device is None or index in reserved or idle_counts.get(device['uuid'],0)<3:continue
                    affinity=state.get('resume_gpu_uuid')
                    if affinity and affinity!=device['uuid']:continue
                    if admit(key,device):reserved.add(index);admitted=True;break
                if not admitted:
                    state['waiting_reason']='waiting for qualified idle GPU and inherited UUID lock; cached model remains on its recorded device'
            write_json(root/'RESOURCE_CONTENTION.json',{'updated_at':now(),
                       'observations_by_model':{key:s['contention_observations'] for key,s in states.items()},
                       'timing_interpretation':'no sampled contention is not an exclusivity guarantee'})
            phase='running' if running else 'waiting_resources'
            if stop_admitting:phase='admission_stopped_waiting_workers' if running else 'interrupted'
            status(phase)
            if all(s['state'] in ('completed','failed') for s in states.values()):break
            if stop_admitting and not running:return
            time.sleep(args.poll_seconds)
        script=REPO/'tools/analyze_vlm_formal_bf16.py'
        while True:
            ready=read_json(root/'ANALYSIS_READY.json')
            if ready and script.exists() and ready.get('analysis_source_sha256')==sha_file(script):break
            status('waiting_analysis_ready')
            if stop_admitting:return
            time.sleep(args.poll_seconds)
        status('analysis')
        with (root/'logs/analysis.log').open('a') as log:
            subprocess.run([sys.executable,str(script),'--asset-root',str(args.asset_root),
                            '--output-root',str(root),'--config',str(args.config)],
                           cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
        analysis=read_json(root/'analysis/ANALYSIS.json')
        if any(s['state']=='failed' for s in states.values()) or not analysis or not analysis.get('complete'):
            status('failed','one or more model/task cells incomplete; other tasks preserved; inspect logs')
        else:
            status('complete');print('FORMAL BF16 COMPLETE',now(),flush=True)
    except BaseException as error:
        status('controller_failed_workers_left_running',f'{type(error).__name__}: {error}');raise
    finally:
        for stream in logs.values():stream.close()
        for gpu_lock in gpu_locks.values():gpu_lock.close()
        owner.close()


if __name__=='__main__':main()
