import argparse
import sys
from pathlib import Path
from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import sha_file,write_json
from .common import ROOT,ROLES,read,actor
from .study import verify,analyze
from .evaluation_v2 import evaluate


def prepare_execution():
    verify()
    if (ROOT/'EXECUTION_V2_LOCK.json').exists():raise FileExistsError('Execution v2 already frozen')
    paths=[Path(__file__),Path(__file__).with_name('evaluation_v2.py'),PROJECT_ROOT/'tools/run_merger_transfer_v2.py',
           PROJECT_ROOT/'docs/VLM_MERGER_TRANSFER_V1_EXECUTION_V2.md',ROOT/'EVAL_V2_DIFF_REVIEW.json',ROOT/'INPUT_METADATA_REVIEW.json']
    write_json(ROOT/'EXECUTION_V2_LOCK.json',{'version':2,'original_plan_sha256':sha_file(ROOT/'PLAN.json'),
               'files':{str(path):sha_file(path) for path in paths},'scientific_protocol_changed':False,
               'new_model_or_optimizer_calls_during_repair':0,'only_reader_difference':'optional retention frequency_slice'})


def verify_execution():
    plan=verify();lock=read(ROOT/'EXECUTION_V2_LOCK.json')
    if lock['original_plan_sha256']!=sha_file(ROOT/'PLAN.json'):raise ValueError('Original plan changed')
    for path,digest in lock['files'].items():
        if sha_file(path)!=digest:raise ValueError('Execution v2 changed: '+path)
    return plan


def condition(role):
    from vlm_human_sft.evaluate import lora_hash
    plan=verify_execution()
    if role=='candidate':
        from .train import train
        model,processor,result=train(role)
        if lora_hash(model)!=result['final_lora_sha256']:raise ValueError('Candidate final adapter differs')
    else:
        cfg=plan['models'][role];model,processor,_=actor(adapter=cfg['adapter'])
        if lora_hash(model)!=cfg['lora_sha256']:raise ValueError('Reference fixed adapter differs')
        result={'status':'fixed_reference','optimizer_updates':0,'lora_sha256':cfg['lora_sha256']}
    evaluate(model,processor,role)
    write_json(ROOT/'runs'/role/'COMPLETE.json',{'status':'completed','execution_version':2,'model':result})
    if role=='candidate':write_json(ROOT/'runs'/role/'train/PROGRESS.json',{'status':'completed','completed_updates':64,'role':role})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-execution',action='store_true');parser.add_argument('--role',choices=ROLES);args=parser.parse_args()
    if args.prepare_execution:
        prepare_execution();print('EXECUTION_V2_PREPARED',flush=True)
    elif args.role:condition(args.role)
    else:
        from research_runtime.queue import run_queue
        verify_execution()
        run_queue(ROOT/'queue_v2',[{'key':role,'command':[sys.executable,'-u','-m','vlm_merger_transfer_v1.execution_v2','--role',role],
                   'completion_marker':str(ROOT/'runs'/role/'COMPLETE.json')} for role in ROLES],
                  {'original_plan_sha256':sha_file(ROOT/'PLAN.json'),'execution_v2_lock_sha256':sha_file(ROOT/'EXECUTION_V2_LOCK.json')})
        print('COMPLETED_MERGER_TRANSFER',analyze()['passed'],flush=True)


if __name__=='__main__':main()
