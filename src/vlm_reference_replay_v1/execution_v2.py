import argparse
from pathlib import Path
import sys
from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import sha_file,write_json
from .common import ROOT,read
from .study import verify,condition
from .analysis_v2 import analyze,render

JOB_TO_MODEL={'plain_transfer':'plain','soft_kl':'soft_kl','hard_replay':'hard_replay'}


def jobs():
    return [{'key':key,'command':[sys.executable,'-u','-m','vlm_reference_replay_v1.execution_v2','--role',key],
             'completion_marker':str(ROOT/'runs'/model/'COMPLETE.json')} for key,model in JOB_TO_MODEL.items()]


def prepare_execution():
    verify()
    if (ROOT/'EXECUTION_V2_LOCK.json').exists():raise FileExistsError('Execution already frozen')
    specification=jobs()
    if len(specification)!=3 or specification[0]['completion_marker']!=str(ROOT/'runs/plain/COMPLETE.json'):
        raise ValueError('Fixed plain reference dispatch differs')
    write_json(ROOT/'DISPATCH_V2_REVIEW.json',{'status':'passed','task_to_model':JOB_TO_MODEL,'new_trainable_roles':['soft_kl','hard_replay'],
               'fixed_plain_has_no_training':True,'completion_markers':[q['completion_marker'] for q in specification],'gpu_updates':0})
    paths=[Path(__file__),Path(__file__).with_name('analysis_v2.py'),PROJECT_ROOT/'tools/run_reference_replay_v2.py',
           PROJECT_ROOT/'docs/VLM_REFERENCE_REPLAY_V1_EXECUTION_V2.md',ROOT/'DISPATCH_V2_REVIEW.json']
    write_json(ROOT/'EXECUTION_V2_LOCK.json',{'version':2,'original_plan_sha256':sha_file(ROOT/'PLAN.json'),
               'files':{str(p):sha_file(p) for p in paths},'scientific_design_changed':False})
    render()


def verify_execution():
    verify();lock=read(ROOT/'EXECUTION_V2_LOCK.json')
    if lock['original_plan_sha256']!=sha_file(ROOT/'PLAN.json'):raise ValueError('Original plan changed')
    for path,digest in lock['files'].items():
        if sha_file(path)!=digest:raise ValueError('Execution v2 source changed: '+path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-execution',action='store_true');parser.add_argument('--role',choices=JOB_TO_MODEL);args=parser.parse_args()
    if args.prepare_execution:prepare_execution();print('EXECUTION_V2_PREPARED',flush=True)
    elif args.role:
        verify_execution();condition(JOB_TO_MODEL[args.role])
    else:
        from research_runtime.queue import run_queue
        verify_execution();run_queue(ROOT/'queue_v2',jobs(),{'plan_sha256':sha_file(ROOT/'PLAN.json'),
                    'execution_v2_sha256':sha_file(ROOT/'EXECUTION_V2_LOCK.json')})
        result=analyze();print('COMPLETED_REFERENCE_REPLAY',result.get('passed',result['status']),flush=True)


if __name__=='__main__':main()
