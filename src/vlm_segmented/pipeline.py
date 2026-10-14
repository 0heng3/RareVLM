"""Wait for idle GPUs, run same-batch attention contrasts, then rescore all cells."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from rarevlm_paths import PROJECT_ROOT,SOURCE_ROOT
from vlm_joint.pipeline import gpu_snapshot,qualified
from vlm_precision.pipeline import GPU_LOCK_ROOT
from vlm_probe.evaluate import write_json,sha_file,lock_file,now
from .prepare import DEFAULT_CONFIG,read_context


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG);args=parser.parse_args()
    cfg,root,plan,executor=read_context(args.config)
    owner=(root/'PIPELINE.lock').open('a+');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    GPU_LOCK_ROOT.mkdir(mode=0o700,parents=True,exist_ok=True);(root/'logs').mkdir(exist_ok=True)
    lock_file(root/'PIPELINE_INPUT_LOCK.json',{'config_sha256':sha_file(args.config),
        'executor_sha256':sha_file(root/'SEGMENTED_EXECUTOR_LOCK.json'),'process_preemption':False})
    old=json.loads((root/'STATUS.json').read_text()) if (root/'STATUS.json').exists() else {}
    groups=old.get('groups',{g:{'state':'waiting_gpu','gpu_uuid':None,'contention_observations':0} for g in cfg['groups']})
    for g,s in groups.items():
        ws=root/'workers'/g/'STATUS.json'
        if ws.exists() and json.loads(ws.read_text())['status']=='completed':s['state']='completed'
        else:s['state']='waiting_gpu'
    processes={};idle={};failures=old.get('resource_query_failures',0)
    def status(phase):write_json(root/'STATUS.json',{'status':phase,'updated_at':now(),'groups':groups,
        'controller_pid':os.getpid(),'resource_query_failures':failures,'existing_tasks_untouched':True,'idle_confirmations':idle})
    while True:
        for g,(p,stream,log) in list(processes.items()):
            code=p.poll()
            if code is None:continue
            ws=json.loads((root/'workers'/g/'STATUS.json').read_text())
            groups[g].update(state='completed' if code==0 and ws['status']=='completed' else 'failed',exit_code=code)
            stream.close();log.close();del processes[g]
        if any(s['state']=='failed' for s in groups.values()):status('failed');return 1
        if all(s['state']=='completed' for s in groups.values()):break
        try:devices=gpu_snapshot()
        except Exception as error:
            failures+=1;idle.clear();status('resource_query_failed');time.sleep(15);continue
        reserved={groups[g]['gpu_uuid'] for g in processes}
        for d in devices.values():idle[d['uuid']]=idle.get(d['uuid'],0)+1 if qualified(d) and d['uuid'] not in reserved else 0
        with (root/'RESOURCE_SNAPSHOTS.jsonl').open('a') as f:f.write(json.dumps({'time':now(),'devices':devices,'reserved':list(reserved)})+'\n')
        for g in processes:
            d=next((d for d in devices.values() if d['uuid']==groups[g]['gpu_uuid']),None)
            if d is None or len(d['compute_processes'])>1:groups[g]['contention_observations']+=1
        for g,s in groups.items():
            if s['state']!='waiting_gpu':continue
            affinity=s.get('gpu_uuid')
            lock_path=root/'workers'/g/'INPUT_LOCK.json'
            if not affinity and lock_path.exists():affinity=json.loads(lock_path.read_text())['hardware']['uuid']
            choices=[d for d in devices.values() if idle.get(d['uuid'],0)>=3 and d['uuid'] not in reserved and (not affinity or d['uuid']==affinity)]
            if not choices:continue
            d=sorted(choices,key=lambda d:d['index'])[0];stream=(GPU_LOCK_ROOT/(d['uuid']+'.lock')).open('a+')
            try:
                fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
                if not qualified(gpu_snapshot()[d['index']]):stream.close();continue
            except BlockingIOError:stream.close();continue
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=d['uuid'],PYTHONPATH=str(SOURCE_ROOT),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python')
            for key in ('RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE','GROUP_RANK','ROLE_RANK'):env.pop(key,None)
            log=(root/'logs'/(g+'.log')).open('a')
            command=[sys.executable,'-m','vlm_segmented.evaluate','--group',g,'--config',str(args.config)]
            p=subprocess.Popen(command,cwd=PROJECT_ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,pass_fds=(stream.fileno(),))
            processes[g]=(p,stream,log);s.update(state='running',pid=p.pid,gpu_uuid=d['uuid'],gpu_index=d['index'],admitted_at=now(),admission_snapshot=d)
            reserved.add(d['uuid']);print('ADMITTED',g,d['index'],p.pid,flush=True)
        status('running' if processes else 'waiting_resources');time.sleep(15)
    status('analyzing')
    from .analysis import analyze
    result=analyze(args.config);status('completed');print('STUDY COMPLETE',result['gates'],flush=True)
    return 0


if __name__=='__main__':sys.exit(main())
