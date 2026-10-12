"""Non-preemptive four-condition queue; each worker reuses one model for two cells."""
from __future__ import annotations
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
from rarevlm_paths import SOURCE_ROOT
import signal
import subprocess
import sys
import time
sys.dont_write_bytecode=True
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION','python')
from vlm_joint.pipeline import gpu_snapshot,qualified
from vlm_precision.pipeline import GPU_LOCK_ROOT
from vlm_probe.evaluate import content_sha,lock_file,now,private_path,sha_file,write_json
from .evaluate import (CONDITIONS,DEFAULT_CONFIG,REPO,adapter_binding,prepared_binding,
    read_json,source_hashes,validate_config)

GROUPS=(('whole','double_whole'),('text_region','random_matched'))
RESOURCE_ERRORS=(subprocess.SubprocessError,ValueError,OSError,csv.Error)


def binding(args):
    config=validate_config(read_json(args.config))
    prepared,_=prepared_binding(config,args.output_root)
    from vlm_probe.evaluate import repo_path
    adapter=private_path(repo_path(config['fixed_adapter_dir']))
    files,training=adapter_binding(config,adapter,args.model_root)
    sources=source_hashes()
    sources.update({name:sha_file(SOURCE_ROOT/name) for name in
        ('vlm_region/pipeline.py','vlm_region/analysis.py','vlm_joint/pipeline.py','vlm_precision/pipeline.py')})
    return {'schema_version':'rarevlm.region_token.execution_ready.v1','config_sha256':sha_file(args.config),
        'source_sha256':sources,'asset_root':str(args.asset_root),'model_root':str(args.model_root),
        'manifest_file_sha256':sha_file(args.asset_root/'prepared/manifest.json'),
        'model_provenance_sha256':sha_file(args.model_root.parent/'model_provenance.json'),
        'adapter_files_sha256':files,'training_binding':training,'prepared_binding':prepared,
        'condition_groups':[list(g) for g in GROUPS],'batch_size':4,'expected_predictions':6468,
        'admission':'zero compute processes,>=40GiB free,<10percent util,three consecutive snapshots,shared per-UUID inherited lock',
        'process_preemption':False,'scope':'four complete internal-dev cells,fixed final adapter,zero training,no automatic expansion'}


def preflight_ready(root,execution):
    value=read_json(root/'PRECHECK_READY.json')
    if not value or value.get('execution_binding_sha256')!=content_sha(execution): return False
    for condition in CONDITIONS:
        path=root/'preflight'/condition/'PREFLIGHT.json'
        data=read_json(path)
        if not data or data.get('status')!='passed' or value.get('preflight_sha256',{}).get(condition)!=sha_file(path): return False
    return True


def group_complete(root,group,preflight=False):
    return all((read_json(root/'preflight'/c/'PREFLIGHT.json') or {}).get('status')=='passed' for c in group) if preflight else all(
        (read_json(root/'runs'/c/'dev/EVALUATION.json') or {}).get('status')=='completed' for c in group)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-root',type=Path,required=True);parser.add_argument('--model-root',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True);parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--poll-seconds',type=int,default=15);parser.add_argument('--inspect-only',action='store_true')
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args()
    for name in ('asset_root','model_root','output_root','config'):setattr(args,name,private_path(getattr(args,name)))
    if args.poll_seconds<10:raise ValueError('minimum10second nonpreemptive polling')
    root=args.output_root;root.mkdir(parents=True,exist_ok=True)
    if args.inspect_only:
        devices=gpu_snapshot()
        print(json.dumps({'devices':devices,'qualified_now':[i for i,d in devices.items() if qualified(d)],
            'conditions':list(CONDITIONS),'new_predictions':6468,'new_training':0,'process_preemption':False},ensure_ascii=False,indent=2));return
    execution=binding(args)
    if args.prepare_only:
        write_json(root/'EXECUTION_READY.json',execution)
        print(json.dumps({'prepared':True,'binding_sha256':content_sha(execution)},ensure_ascii=False));return
    if read_json(root/'EXECUTION_READY.json')!=execution:raise ValueError('source/config/data freeze missing or changed')
    analysis=read_json(root/'ANALYSIS_READY.json')
    if not analysis or analysis.get('analysis_source_sha256')!=sha_file(SOURCE_ROOT/'vlm_region/analysis.py') or analysis.get('config_sha256')!=sha_file(args.config):
        raise ValueError('CPU analysis readiness missing or changed')
    if not args.preflight_only and not preflight_ready(root,execution):raise ValueError('all4 input-only GPU preflights required before full generation')
    lock_file(root/'PIPELINE_INPUT_LOCK.json',execution)
    owner=(root/'PIPELINE.lock').open('a+');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    GPU_LOCK_ROOT.mkdir(parents=True,exist_ok=True,mode=0o700);(root/'logs').mkdir(exist_ok=True)
    previous=read_json(root/'STATUS.json') or {};started=previous.get('started_at',now())
    worker_states={ '__'.join(g):{'state':'pending','conditions':list(g)} for g in GROUPS}
    for key,state in worker_states.items():
        prior=(read_json(root/'workers'/key/'STATUS.json') or {}).get('hardware',{}).get('uuid')
        if prior:state['prior_gpu_uuid']=prior
    running,streams,logs,idle={},{},{},{}
    stop_admitting=False;failure=None
    def stop(signum,frame):
        nonlocal stop_admitting
        stop_admitting=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    write_json(root/'PID.json',{'pid':os.getpid(),'started_at':now(),'python':sys.executable,'preflight_only':args.preflight_only})

    def status(phase,error=None):
        evaluations={c:read_json(root/'runs'/c/'dev/EVALUATION.json') for c in CONDITIONS}
        value={'schema_version':'rarevlm.region_token.pipeline_status.v1','phase':phase,'started_at':started,
            'updated_at':now(),'pid':os.getpid(),'states':worker_states,'preflight_only':args.preflight_only,
            'completed_cells':sum(e is not None for e in evaluations.values()),'total_cells':4,
            'expected_predictions':6468,'total_trainings':0,'evaluation_results':evaluations,
            'eval_progress':{c:read_json(root/'runs'/c/'dev/PROGRESS.json') for c in CONDITIONS},
            'worker_status':{k:read_json(root/'workers'/k/'STATUS.json') for k in worker_states},
            'idle_confirmations':idle,'existing_tasks_untouched':True,'execution_binding_sha256':content_sha(execution),
            'analysis':read_json(root/'analysis/ANALYSIS.json'),'scope':execution['scope']}
        if error:value['error']=error
        write_json(root/'STATUS.json',value)

    def admit(group,device):
        key='__'.join(group);stream=(GPU_LOCK_ROOT/(device['uuid']+'.lock')).open('a+')
        try:
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            current=gpu_snapshot().get(device['index'])
            if not current or current['uuid']!=device['uuid'] or not qualified(current):stream.close();return False
        except (BlockingIOError,*RESOURCE_ERRORS):stream.close();return False
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=device['uuid'],PYTHONUNBUFFERED='1',PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python')
        for field in ('RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE','GROUP_RANK','ROLE_RANK'):env.pop(field,None)
        command=[sys.executable,'-m','vlm_region.evaluate','--conditions',*group,'--asset-root',str(args.asset_root),
            '--model-root',str(args.model_root),'--output-root',str(root),'--config',str(args.config),'--batch-size','4']
        if args.preflight_only:command.append('--preflight-only')
        log=(root/'logs'/(key+('_preflight' if args.preflight_only else '_eval')+'.log')).open('a')
        try:process=subprocess.Popen(command,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,pass_fds=(stream.fileno(),))
        except BaseException:stream.close();log.close();raise
        running[key],streams[key],logs[key]=process,stream,log
        worker_states[key].update(state='running',pid=process.pid,gpu_index=device['index'],gpu_uuid=device['uuid'],
            admitted_at=now(),admission_snapshot=device,contention_observations=0,command=command)
        print('ADMITTED',key,device['index'],process.pid,now(),flush=True);return True

    status('waiting_resources')
    try:
        while True:
            for key,process in list(running.items()):
                code=process.poll()
                if code is None:continue
                complete=group_complete(root,worker_states[key]['conditions'],args.preflight_only)
                state='completed' if code==0 and complete else 'failed'
                worker_states[key].update(state=state,returncode=code,finished_at=now())
                logs.pop(key).close();streams.pop(key).close();del running[key]
                if state=='failed':failure=f'{key} returned{code};no existing or sibling job was interrupted'
            if failure and not running:status('failed',failure);return
            if not running and all(group_complete(root,g,args.preflight_only) for g in GROUPS):break
            try:snapshot=gpu_snapshot()
            except RESOURCE_ERRORS as error:
                idle.clear();status('resource_query_failed',str(error));time.sleep(args.poll_seconds);continue
            reserved={worker_states[k]['gpu_uuid'] for k in running}
            for device in snapshot.values():
                uuid=device['uuid'];idle[uuid]=idle.get(uuid,0)+1 if uuid not in reserved and qualified(device) else 0
            with (root/'RESOURCE_SNAPSHOTS.jsonl').open('a') as stream:
                stream.write(json.dumps({'time':now(),'devices':snapshot,'reserved_gpu_uuids':sorted(reserved)},ensure_ascii=False)+'\n')
            for key in running:
                device=next((d for d in snapshot.values() if d['uuid']==worker_states[key]['gpu_uuid']),None)
                if device and len(device['compute_processes'])>1:
                    worker_states[key]['contention_observations']+=1
                    with (root/'RESOURCE_CONTENTION.jsonl').open('a') as stream:
                        stream.write(json.dumps({'time':now(),'worker':key,'device':device,'effect':'timing tagged;other jobs untouched'})+'\n')
            if not stop_admitting and not failure:
                for group in GROUPS:
                    key='__'.join(group);state=worker_states[key]
                    if state['state'] in ('running','completed','failed'):continue
                    if group_complete(root,group,args.preflight_only):state['state']='completed';continue
                    affinity=state.get('prior_gpu_uuid')
                    candidates=[d for d in snapshot.values() if d['uuid'] not in reserved and idle.get(d['uuid'],0)>=3 and (not affinity or affinity==d['uuid'])]
                    if candidates and admit(group,sorted(candidates,key=lambda d:d['index'])[0]):reserved.add(state['gpu_uuid'])
            phase='preflight' if args.preflight_only and running else 'evaluating' if running else 'waiting_resources'
            if stop_admitting:phase='admission_stopped_waiting_workers' if running else 'interrupted'
            status(phase,failure)
            if stop_admitting and not running:return
            time.sleep(args.poll_seconds)
        if args.preflight_only:
            ready={'schema_version':'rarevlm.region_token.precheck_ready.v1','completed_at':now(),
                'execution_binding_sha256':content_sha(execution),'preflight_sha256':{c:sha_file(root/'preflight'/c/'PREFLIGHT.json') for c in CONDITIONS},
                'status':'passed','scope':'CPU32/condition plus GPU8/condition;label-free checks;full generation not started'}
            write_json(root/'PRECHECK_READY.json',ready);status('preflight_completed');print('REGION PREFLIGHT COMPLETE',now(),flush=True);return
        status('analysis')
        with (root/'logs/analysis.log').open('a') as log:
            subprocess.run([sys.executable,'-m','vlm_region.analysis','--asset-root',str(args.asset_root),
                '--output-root',str(root),'--config',str(args.config)],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
        result=read_json(root/'analysis/ANALYSIS.json')
        if not result or not result.get('complete'):raise ValueError('automatic analysis incomplete')
        status('complete');print('REGION STUDY COMPLETE',result['gate'],now(),flush=True)
    except BaseException as error:
        status('controller_failed_workers_left_running',f'{type(error).__name__}: {error}');raise
    finally:
        # Child processes inherit UUID locks. Closing these descriptors never
        # kills them or unlocks GPUs while a child still owns the descriptor.
        for log in logs.values():log.close()
        for stream in streams.values():stream.close()
        owner.close()


if __name__=='__main__':main()
