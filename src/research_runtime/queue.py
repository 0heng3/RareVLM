"""Admit each job onto an idle, locked GPU and retain hardware affinity on resume."""
from __future__ import annotations
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from rarevlm_paths import PROJECT_ROOT,SOURCE_ROOT
from vlm_joint.pipeline import gpu_snapshot,qualified
from vlm_precision.pipeline import GPU_LOCK_ROOT
from vlm_probe.evaluate import now,write_json,lock_file,sha_file


def run_queue(root,jobs,identity):
    root.mkdir(parents=True,exist_ok=True);(root/'logs').mkdir(exist_ok=True)
    owner=(root/'PIPELINE.lock').open('a+');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock_file(root/'QUEUE_INPUT_LOCK.json',{'identity':identity,'jobs':jobs,'queue_source_sha256':sha_file(Path(__file__)),
        'admission':'zero compute processes,>=40GiB free,<10% utilization,three15sec samples,shared UUID lock','preemption':False})
    previous=json.loads((root/'STATUS.json').read_text()) if (root/'STATUS.json').exists() else {}
    states=previous.get('jobs',{job['key']:{'state':'waiting_gpu','gpu_uuid':None,'contention_observations':0} for job in jobs})
    for s in states.values():
        if s['state']!='completed':s['state']='waiting_gpu'
    processes={};idle={};failures=previous.get('resource_query_failures',0)
    def status(phase):write_json(root/'STATUS.json',{'status':phase,'updated_at':now(),'controller_pid':os.getpid(),
        'jobs':states,'resource_query_failures':failures,'existing_tasks_untouched':True,'idle_confirmations':idle})
    GPU_LOCK_ROOT.mkdir(mode=0o700,parents=True,exist_ok=True)
    while True:
        for key,(process,stream,log) in list(processes.items()):
            code=process.poll()
            if code is None:continue
            job=next(j for j in jobs if j['key']==key);marker=Path(job['completion_marker'])
            valid=marker.exists() and json.loads(marker.read_text()).get('status') in ['completed','passed']
            states[key].update(state='completed' if code==0 and valid else 'failed',exit_code=code,ended_at=now())
            stream.close();log.close();del processes[key]
        if any(s['state']=='failed' for s in states.values()):status('failed');raise RuntimeError('research worker failed;other jobs not killed')
        if all(s['state']=='completed' for s in states.values()):status('completed');return
        try:devices=gpu_snapshot()
        except Exception:
            failures+=1;idle.clear();status('resource_query_failed');time.sleep(15);continue
        reserved={states[key]['gpu_uuid'] for key in processes}
        for device in devices.values():idle[device['uuid']]=idle.get(device['uuid'],0)+1 if qualified(device) and device['uuid'] not in reserved else 0
        with (root/'RESOURCE_SNAPSHOTS.jsonl').open('a') as f:f.write(json.dumps({'time':now(),'devices':devices,'reserved':list(reserved)})+'\n')
        for key in processes:
            device=next((d for d in devices.values() if d['uuid']==states[key]['gpu_uuid']),None)
            if device is None or len(device['compute_processes'])>1:states[key]['contention_observations']+=1
        for job in jobs:
            key=job['key'];s=states[key]
            if s['state']!='waiting_gpu':continue
            affinity=s.get('gpu_uuid');choices=[d for d in devices.values() if idle.get(d['uuid'],0)>=3 and d['uuid'] not in reserved and (not affinity or affinity==d['uuid'])]
            if not choices:continue
            device=sorted(choices,key=lambda d:d['index'])[0];stream=(GPU_LOCK_ROOT/(device['uuid']+'.lock')).open('a+')
            try:
                fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
                if not qualified(gpu_snapshot()[device['index']]):stream.close();continue
            except BlockingIOError:stream.close();continue
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=device['uuid'],PYTHONPATH=str(SOURCE_ROOT),PYTHONDONTWRITEBYTECODE='1',
                PYTHONUNBUFFERED='1',PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python')
            for field in ['RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE','GROUP_RANK','ROLE_RANK']:env.pop(field,None)
            log=(root/'logs'/(key+'.log')).open('a')
            try:process=subprocess.Popen(job['command'],cwd=PROJECT_ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,pass_fds=(stream.fileno(),))
            except BaseException:stream.close();log.close();raise
            processes[key]=(process,stream,log);s.update(state='running',pid=process.pid,gpu_uuid=device['uuid'],gpu_index=device['index'],
                admitted_at=now(),admission_snapshot=device)
            reserved.add(device['uuid']);print('ADMITTED',key,device['index'],process.pid,flush=True)
        status('running' if processes else 'waiting_resources');time.sleep(15)
