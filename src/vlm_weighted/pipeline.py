"""Non-preemptive two-condition seed42 training -> internal-dev evaluation -> pilot gate."""
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
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION','python')
from vlm_joint.pipeline import gpu_snapshot,qualified
from vlm_precision.pipeline import GPU_LOCK_ROOT,now,read_json,sha_file,write_json
from vlm_probe.evaluate import content_sha,private_path

REPO=PROJECT_ROOT
CONDITIONS=('uniform','frequency')
RESOURCE_ERRORS=(subprocess.SubprocessError,ValueError,OSError,csv.Error)


def sources():
    names=('vlm_weighted/__init__.py','vlm_weighted/data.py','vlm_weighted/train.py',
        'vlm_weighted/evaluate.py','vlm_weighted/dev_analysis.py','vlm_weighted/pipeline.py',
        'vlm_joint/pipeline.py','vlm_precision/pipeline.py','rarevlm_paths.py')
    return {name:sha_file(SOURCE_ROOT/name) for name in names}


def binding(args):
    return {'schema_version':'rarevlm.weighted_sft.execution_ready.v1',
        'config_sha256':sha_file(args.config),'source_sha256':sources(),
        'asset_root':str(args.asset_root),'model_root':str(args.model_root),
        'model_provenance_sha256':sha_file(args.model_root.parent/'model_provenance.json'),
        'manifest_file_sha256':sha_file(args.asset_root/'prepared/manifest.json'),
        'weight_lock_sha256':sha_file(args.weight_lock),
        'conditions':list(CONDITIONS),'seed':42,'training_world_size':2,
        'scope':'two new full-data final adapters and two internal-dev cells only; no automatic replication or official benchmark',
        'process_preemption':False}


def train_result(root,condition):
    return read_json(root/'runs'/condition/'seed42/train/TRAIN_RESULT.json')


def train_complete(root,condition):
    result=train_result(root,condition)
    return bool(result and result.get('condition')==condition and result.get('seed')==42
        and result.get('completed_updates')==2062 and result.get('actual_questions_seen')==32985)


def eval_complete(root,condition):
    return (root/'runs'/condition/'seed42/dev/EVALUATION.json').exists()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-root',type=Path,required=True)
    parser.add_argument('--model-root',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=REPO/'configs/vlm_weighted_sft_v1.json')
    parser.add_argument('--weight-lock',type=Path)
    parser.add_argument('--poll-seconds',type=int,default=15)
    parser.add_argument('--master-port',type=int,default=29742)
    parser.add_argument('--inspect-only',action='store_true')
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    for name in ('asset_root','model_root','output_root','config'):
        setattr(args,name,private_path(getattr(args,name)))
    args.weight_lock=private_path(args.weight_lock or args.output_root/'WEIGHT_LOCK.json')
    config=read_json(args.config)
    if args.poll_seconds<10 or not 1024<args.master_port<65536:
        raise ValueError('require>=10second poll and unprivileged valid master port')
    args.output_root.mkdir(parents=True,exist_ok=True)
    root=args.output_root
    lock=binding(args)
    if args.prepare_only:
        write_json(root/'EXECUTION_READY.json',lock)
        print(json.dumps({'prepared':True,'config_sha256':lock['config_sha256'],
            'source_binding_sha256':content_sha(lock)},ensure_ascii=False));return
    if args.inspect_only:
        snapshot=gpu_snapshot()
        print(json.dumps({'devices':snapshot,'qualified_now':[i for i,d in snapshot.items() if qualified(d)],
            'training_requires_idle_gpus':2,'training_conditions':list(CONDITIONS),
            'eval_cells':2,'qa_exposures':65970,'updates':4124,'dev_predictions':3234,
            'execution_ready_matches':read_json(root/'EXECUTION_READY.json')==lock,
            'automatic_replication':False,'automatic_formal_eval':False,'process_preemption':False},ensure_ascii=False,indent=2));return
    if read_json(root/'EXECUTION_READY.json')!=lock:
        raise ValueError('missing or changed EXECUTION_READY; complete preflight/source freeze before queue')
    ready=read_json(root/'ANALYSIS_READY.json')
    if not ready or ready.get('analysis_source_sha256')!=sha_file(SOURCE_ROOT/'vlm_weighted/dev_analysis.py'):
        raise ValueError('internal-dev slice/analysis readiness missing or changed')
    previous=read_json(root/'PIPELINE_INPUT_LOCK.json')
    if previous is not None and previous!=lock:
        raise ValueError('resume source/data/config differs; use new protocol output')
    write_json(root/'PIPELINE_INPUT_LOCK.json',lock)
    owner=(root/'PIPELINE.lock').open('a+')
    fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (root/'logs').mkdir(exist_ok=True)
    GPU_LOCK_ROOT.mkdir(parents=True,exist_ok=True,mode=0o700)
    old=read_json(root/'STATUS.json') or {}
    started=old.get('started_at',now())
    write_json(root/'PID.json',{'pid':os.getpid(),'started_at':now(),'python':sys.executable})
    idle_counts,running,logs,gpu_locks={},{},{},{}
    states={f'train:{c}':{'state':'pending'} for c in CONDITIONS}
    states.update({f'eval:{c}':{'state':'pending'} for c in CONDITIONS})
    for key,prior in old.get('states',{}).items():
        if key in states:
            states[key]['prior_gpu_uuids']=prior.get('gpu_uuids')
    admission_stopped=False
    failure=None
    def request_stop(signum,frame):
        nonlocal admission_stopped
        admission_stopped=True
    signal.signal(signal.SIGTERM,request_stop)
    signal.signal(signal.SIGINT,request_stop)

    def status(phase,error=None):
        trains={c:train_result(root,c) for c in CONDITIONS}
        evals={c:read_json(root/'runs'/c/'seed42/dev/EVALUATION.json') for c in CONDITIONS}
        value={'schema_version':'rarevlm.weighted_sft.pipeline_status.v1','phase':phase,
            'updated_at':now(),'started_at':started,'pid':os.getpid(),'states':states,
            'completed_trainings':sum(train_complete(root,c) for c in CONDITIONS),'total_trainings':2,
            'completed_eval_cells':sum(eval_complete(root,c) for c in CONDITIONS),'total_eval_cells':2,
            'expected_training_qa_exposures':65970,'expected_updates':4124,'expected_dev_predictions':3234,
            'training_results':trains,'evaluation_results':evals,
            'training_progress':{c:read_json(root/'runs'/c/'seed42/train/PROGRESS.json') for c in CONDITIONS},
            'eval_progress':{c:read_json(root/'runs'/c/'seed42/dev/PROGRESS.json') for c in CONDITIONS},
            'idle_confirmations':idle_counts,'existing_tasks_untouched':True,
            'config_sha256':lock['config_sha256'],'source_binding_sha256':content_sha(lock),
            'analysis':read_json(root/'analysis/ANALYSIS.json'),'scope':lock['scope']}
        if error:value['error']=error
        write_json(root/'STATUS.json',value)

    def reserve(devices):
        streams=[]
        try:
            for device in sorted(devices,key=lambda d:d['uuid']):
                stream=(GPU_LOCK_ROOT/(device['uuid']+'.lock')).open('a+')
                try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:stream.close();raise
                streams.append(stream)
            current=gpu_snapshot()
            if any(d['index'] not in current or current[d['index']]['uuid']!=d['uuid']
                or not qualified(current[d['index']]) for d in devices):
                for stream in streams:stream.close()
                return None
            return streams
        except RESOURCE_ERRORS:
            for stream in streams:stream.close()
            return None

    def admit(kind,condition,devices):
        key=f'{kind}:{condition}'
        streams=reserve(devices)
        if streams is None:return False
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(d['uuid'] for d in devices),
            PYTHONUNBUFFERED='1',PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python')
        for field in ('RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE','GROUP_RANK','ROLE_RANK'):
            env.pop(field,None)
        common=['--asset-root',str(args.asset_root),'--model-root',str(args.model_root),'--config',str(args.config)]
        if kind=='train':
            cmd=[sys.executable,'-m','torch.distributed.run','--nproc_per_node=2',
                f'--master_port={args.master_port}','--module','vlm_weighted.train',*common,
                '--condition',condition,'--weight-lock',str(args.weight_lock),'--method','qv_r8',
                '--seed','42','--split','train_all','--runtime-world-size','2','--runtime-microbatch','8',
                '--runtime-accumulation','1','--output-dir',str(root/'runs'/condition/'seed42/train')]
        else:
            cmd=[sys.executable,'-m','vlm_weighted.evaluate',*common,'--output-root',str(root),
                '--condition',condition,'--seed','42','--batch-size','4',
                '--adapter-dir',str(root/'runs'/condition/'seed42/train/adapter')]
        log=(root/'logs'/f'{condition}_{kind}.log').open('a')
        try:
            process=subprocess.Popen(cmd,cwd=REPO,env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,pass_fds=tuple(s.fileno() for s in streams))
        except BaseException:
            for stream in streams:stream.close()
            log.close();raise
        running[key],logs[key],gpu_locks[key]=process,log,streams
        states[key].update(state='running',pid=process.pid,admitted_at=now(),
            gpu_indices=[d['index'] for d in devices],gpu_uuids=[d['uuid'] for d in devices],
            admission_snapshots=devices,command=cmd,contention_observations=0)
        print('ADMITTED',key,[d['index'] for d in devices],process.pid,now(),flush=True)
        return True

    status('waiting_resources')
    try:
        while True:
            for key,process in list(running.items()):
                code=process.poll()
                if code is None:continue
                kind,condition=key.split(':',1)
                completed=train_complete(root,condition) if kind=='train' else eval_complete(root,condition)
                state='completed' if code==0 and completed else 'failed'
                states[key].update(state=state,returncode=code,finished_at=now())
                logs.pop(key).close()
                for stream in gpu_locks.pop(key):stream.close()
                del running[key]
                if state=='failed':failure=f'{key} returned{code}; inspect its log; no process was interrupted'
            if failure and not running:status('failed',failure);return
            if all(eval_complete(root,c) for c in CONDITIONS):break
            try:snapshot=gpu_snapshot()
            except RESOURCE_ERRORS as error:
                idle_counts.clear();status('resource_query_failed',str(error));time.sleep(args.poll_seconds);continue
            reserved={index for key in running for index in states[key]['gpu_indices']}
            for index,device in snapshot.items():
                uuid=device['uuid']
                idle_counts[uuid]=idle_counts.get(uuid,0)+1 if index not in reserved and qualified(device) else 0
            with (root/'RESOURCE_SNAPSHOTS.jsonl').open('a') as stream:
                stream.write(json.dumps({'time':now(),'devices':snapshot,'reserved_indices':sorted(reserved)},ensure_ascii=False)+'\n')
            for key in running:
                for index in states[key]['gpu_indices']:
                    device=snapshot.get(index)
                    if device and len(device['compute_processes'])>1:
                        states[key]['contention_observations']+=1
                        with (root/'RESOURCE_CONTENTION.jsonl').open('a') as stream:
                            stream.write(json.dumps({'time':now(),'job':key,'device':device,
                                'effect':'timing tagged; existing jobs unchanged'})+'\n')
            for condition in CONDITIONS:
                if train_complete(root,condition) and f'train:{condition}' not in running:
                    states[f'train:{condition}']['state']='completed'
                if eval_complete(root,condition) and f'eval:{condition}' not in running:
                    states[f'eval:{condition}']['state']='completed'
            candidates=[d for index,d in sorted(snapshot.items()) if index not in reserved and idle_counts.get(d['uuid'],0)>=3]
            if not admission_stopped and not failure:
                pending_train=next((c for c in CONDITIONS if not train_complete(root,c)),None)
                if pending_train is not None:
                    if not running and len(candidates)>=2:
                        prior=states[f'train:{pending_train}'].get('prior_gpu_uuids')
                        devices=candidates[:2]
                        if prior:
                            lookup={d['uuid']:d for d in candidates}
                            devices=[lookup[u] for u in prior] if all(u in lookup for u in prior) else []
                        if len(devices)==2:admit('train',pending_train,devices)
                else:
                    for condition in CONDITIONS:
                        key=f'eval:{condition}'
                        if states[key]['state'] in ('running','completed','failed'):continue
                        affinity=(read_json(root/'runs'/condition/'seed42/EVAL_STATUS.json') or {}).get('hardware',{}).get('uuid')
                        eligible=[d for d in candidates if d['index'] not in reserved and (not affinity or affinity==d['uuid'])]
                        if eligible and admit('eval',condition,[eligible[0]]):reserved.add(eligible[0]['index'])
            phase='training' if any(k.startswith('train:') for k in running) else 'evaluating' if running else 'waiting_resources'
            if admission_stopped:phase='admission_stopped_waiting_workers' if running else 'interrupted'
            status(phase,failure)
            if admission_stopped and not running:return
            time.sleep(args.poll_seconds)
        status('analysis')
        with (root/'logs/dev_analysis.log').open('a') as log:
            subprocess.run([sys.executable,'-m','vlm_weighted.dev_analysis','--asset-root',str(args.asset_root),
                '--output-root',str(root),'--config',str(args.config),'--seed','42'],
                cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
        result=read_json(root/'analysis/ANALYSIS.json')
        if not result or not result.get('complete'):raise ValueError('dev analysis incomplete')
        status('complete');print('WEIGHTED SFT PILOT COMPLETE',result['gate'],now(),flush=True)
    except BaseException as error:
        status('controller_failed_workers_left_running',f'{type(error).__name__}: {error}');raise
    finally:
        for log in logs.values():log.close()
        for streams in gpu_locks.values():
            for stream in streams:stream.close()
        owner.close()


if __name__=='__main__':
    main()
