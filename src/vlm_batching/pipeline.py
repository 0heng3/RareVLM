"""Admit idle GPUs without disturbing existing work; complete and analyze both groups."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.dont_write_bytecode=True
from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT
from vlm_joint.pipeline import gpu_snapshot, qualified
from vlm_precision.pipeline import GPU_LOCK_ROOT
from vlm_probe.evaluate import now,write_json,sha_file,lock_file
from .prepare import DEFAULT_CONFIG,read_plan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    args=parser.parse_args();cfg,root,plan=read_plan(args.config)
    (root/'logs').mkdir(exist_ok=True)
    owner=(root/'PIPELINE.lock').open('a+')
    fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    GPU_LOCK_ROOT.mkdir(mode=0o700,parents=True,exist_ok=True)
    lock_file(root/'PIPELINE_INPUT_LOCK.json',{'config_sha256':sha_file(args.config),
        'plan_sha256':sha_file(root/'PLAN.json'),'source_hashes':plan['source_hashes'],
        'groups':cfg['groups'],'process_preemption':False,
        'admission':'zero compute processes,>=40GiB free,<10% utilization,three15sec snapshots,shared UUID lock'})
    write_json(root/'PID.json',{'pid':os.getpid(),'started_at':now(),'python':sys.executable})
    prior=json.loads((root/'STATUS.json').read_text()) if (root/'STATUS.json').exists() else {}
    states=prior.get('groups',{})
    for g in cfg['groups']:
        if g not in states:states[g]={'state':'waiting_gpu','gpu_uuid':None,'contention_observations':0}
        ws=root/'workers'/g/'STATUS.json'
        if ws.exists() and json.loads(ws.read_text()).get('status')=='completed':states[g]['state']='completed'
        elif states[g]['state']!='completed':states[g]['state']='waiting_gpu'
    handles={};idle={};query_failures=prior.get('resource_query_failures',0)
    def status(phase):
        write_json(root/'STATUS.json',{'status':phase,'updated_at':now(),'groups':states,
            'resource_query_failures':query_failures,'idle_confirmations':idle,
            'existing_tasks_untouched':True,'plan_sha256':sha_file(root/'PLAN.json')})
    while True:
        for g,(process,stream,log) in list(handles.items()):
            code=process.poll()
            if code is None:continue
            ws=json.loads((root/'workers'/g/'STATUS.json').read_text())
            states[g].update(state='completed' if code==0 and ws['status']=='completed' else 'failed',
                exit_code=code,ended_at=now())
            log.close();stream.close();del handles[g]
        if any(s['state']=='failed' for s in states.values()):
            status('failed');return 1
        if all(s['state']=='completed' for s in states.values()):break
        try:devices=gpu_snapshot()
        except Exception as error:
            query_failures+=1;idle.clear();status('resource_query_failed')
            print('RESOURCE QUERY FAILED',repr(error),flush=True);time.sleep(15);continue
        reserved={states[g]['gpu_uuid'] for g in handles}
        for d in devices.values():
            uid=d['uuid'];idle[uid]=idle.get(uid,0)+1 if uid not in reserved and qualified(d) else 0
        with (root/'RESOURCE_SNAPSHOTS.jsonl').open('a') as f:
            f.write(json.dumps({'time':now(),'devices':devices,'reserved_gpu_uuids':list(reserved)})+'\n')
        for g in handles:
            d=next((d for d in devices.values() if d['uuid']==states[g]['gpu_uuid']),None)
            if d is None or len(d['compute_processes'])>1:
                states[g]['contention_observations']+=1
        for g,s in states.items():
            if s['state']!='waiting_gpu':continue
            affinity=s.get('gpu_uuid')
            candidates=[d for d in devices.values() if d['uuid'] not in reserved and idle.get(d['uuid'],0)>=3
                        and (not affinity or d['uuid']==affinity)]
            if not candidates:continue
            d=sorted(candidates,key=lambda d:d['index'])[0]
            stream=(GPU_LOCK_ROOT/(d['uuid']+'.lock')).open('a+')
            try:
                fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
                fresh=gpu_snapshot().get(d['index'])
                if not fresh or not qualified(fresh) or fresh['uuid']!=d['uuid']:
                    stream.close();continue
            except BlockingIOError:stream.close();continue
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=d['uuid'],PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',
                PYTHONPATH=str(SOURCE_ROOT),PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python')
            for name in ('RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE','GROUP_RANK','ROLE_RANK'):env.pop(name,None)
            log=(root/'logs'/(g+'.log')).open('a')
            cmd=[sys.executable,'-m','vlm_batching.evaluate','--group',g,'--config',str(args.config)]
            try:
                proc=subprocess.Popen(cmd,cwd=PROJECT_ROOT,env=env,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,pass_fds=(stream.fileno(),))
            except BaseException:stream.close();log.close();raise
            handles[g]=(proc,stream,log);s.update(state='running',pid=proc.pid,gpu_uuid=d['uuid'],
                gpu_index=d['index'],admitted_at=now(),admission_snapshot=d)
            reserved.add(d['uuid']);print('ADMITTED',g,d['index'],proc.pid,flush=True)
        status('running' if handles else 'waiting_resources');time.sleep(15)
    status('analyzing')
    from .analysis import analyze
    result=analyze(args.config)
    status('completed');print('STUDY COMPLETE',result['gates'],flush=True)
    return 0


if __name__=='__main__':sys.exit(main())
