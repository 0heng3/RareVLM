"""Eval-only engineering recovery with a distinct queue and effective-source lock."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime,timezone
from .lock import (REPO,PROTOCOL,REVIEW_SHA,source_hashes,sha_file,read,verify_files,
                   verify_lock,assert_original_inactive)
from .compat import content_sha,legacy_anchor_view

ARMS=('teacher_ce32','gt_ce32')


def now():return datetime.now(timezone.utc).isoformat()
def write(file,value):
    file=Path(file);file.parent.mkdir(parents=True,exist_ok=True)
    temp=file.with_name(file.name+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(file)
def immutable(file,value):
    file=Path(file)
    if file.exists():
        if read(file)!=value:raise ValueError(f'immutable recovery metadata differs: {file}')
    else:write(file,value)


def cpu_contracts(config_file,recovery_root):
    config_file,recovery_root=Path(config_file).resolve(),Path(recovery_root).resolve()
    config=read(config_file);root=Path(config['output_root']);original=read(root/'STUDY_INPUT_LOCK.json')
    if (recovery_root/'RECOVERY_INPUT_LOCK.json').exists():raise ValueError('CPU receipt cannot be replaced after recovery lock')
    receipt=recovery_root/'CPU_CONTRACTS.json';log=recovery_root/'cpu_contracts.log';xml=recovery_root/'cpu_contracts.xml'
    if any(file.exists() for file in (receipt,log,xml)):raise ValueError('existing CPU evidence retained; no silent overwrite')
    recovery_root.mkdir(parents=True,exist_ok=True)
    before=source_hashes();science_before={name:sha_file(name) for name in original['source_sha256']}
    if science_before!=original['source_sha256']:raise ValueError('original89 scientific sources changed')
    command=[sys.executable,'-m','pytest','-q','tests/test_object_behavior_recovery.py','--junitxml',str(xml)]
    env=dict(os.environ,PYTHONPATH=str(REPO/'src'),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    begun=time.monotonic()
    with log.open('xb') as output:result=subprocess.run(command,cwd=REPO,env=env,stdout=output,stderr=subprocess.STDOUT)
    after=source_hashes();science_after={name:sha_file(name) for name in original['source_sha256']}
    value={'schema_version':'rarevlm.object_behavior_replay.recovery_cpu.v1','status':'passed' if result.returncode==0 and before==after and science_before==science_after else 'failed',
        'command':command,'returncode':result.returncode,'seconds':time.monotonic()-begun,
        'original_study_input_lock_sha256':sha_file(root/'STUDY_INPUT_LOCK.json'),
        'recovery_source_sha256_before':before,'recovery_source_sha256_after':after,
        'scientific_source_sha256_before':science_before,'scientific_source_sha256_after':science_after,
        'files_sha256':{str(file):sha_file(file) for file in (log,xml,REPO/'tests/test_object_behavior_recovery.py') if file.exists()},
        'new_model_forwards':0,'gpu_operations':0,'optimizer_updates':0,'goal_complete':False}
    write(receipt,value)
    if value['status']!='passed':raise ValueError('recovery CPU contracts failed; original evidence retained')
    return value


def prepare(config_file,recovery_root,cpu_receipt,failure_archive):
    from vlm_object_behavior_replay.study import verify_study
    config_file,recovery_root=Path(config_file).resolve(),Path(recovery_root).resolve()
    config=read(config_file);root=Path(config['output_root']);assert_original_inactive(config_file)
    if recovery_root!=root/'recovery_v1':raise ValueError('exact independent recovery_v1 namespace required')
    original_file=root/'STUDY_INPUT_LOCK.json';original=verify_study(original_file,config_file)
    cpu_receipt=Path(cpu_receipt).resolve();cpu=read(cpu_receipt)
    if (cpu.get('status')!='passed' or cpu.get('schema_version')!='rarevlm.object_behavior_replay.recovery_cpu.v1'
        or cpu['recovery_source_sha256_before']!=source_hashes() or cpu['recovery_source_sha256_after']!=source_hashes()
        or cpu['scientific_source_sha256_before']!=original['source_sha256'] or cpu['scientific_source_sha256_after']!=original['source_sha256']
        or cpu['original_study_input_lock_sha256']!=sha_file(original_file)):
        raise ValueError('recovery CPU receipt/source identity differs')
    verify_files(cpu['files_sha256'])
    review_file=root/'independent_review/TRAINING_COMPLETION_REVIEW.json'
    if sha_file(review_file)!=REVIEW_SHA:raise ValueError('root-approved completed Final review differs')
    manual=root/'independent_review/TRAINING_COMPLETION_RECOVERY.json'
    if not manual.is_file():raise ValueError('manual full-Final review recovery receipt required')
    archive=Path(failure_archive).resolve()
    if not archive.is_relative_to(root/'independent_review') or not archive.is_dir():raise ValueError('original failure archive required')
    archive_files=[file for file in archive.rglob('*') if file.is_file()]
    if not archive_files or not any(file.suffix=='.log' for file in archive_files):raise ValueError('failure archive lacks original logs')
    old_root=Path(config['historical_coverage_study_root']);old_train=old_root/'runs/wide32/seed42/train'
    anchor_file=old_train/'TRAIN_RESULT.json'
    if original['files_sha256'].get(str(anchor_file))!=sha_file(anchor_file):raise ValueError('historical result not original-STUDY-bound')
    anchor=read(anchor_file);old_schedule=read(old_train/'SCHEDULE.json');plan=read(root/'prepared/seed42/PLAN.json')
    files={str(cpu_receipt):sha_file(cpu_receipt),**cpu['files_sha256'],str(review_file):sha_file(review_file),str(manual):sha_file(manual),
        str(old_train/'SCHEDULE.json'):sha_file(old_train/'SCHEDULE.json'),str(anchor_file):sha_file(anchor_file)}
    files.update({str(file):sha_file(file) for file in archive_files})
    proofs={}
    for arm in ARMS:
        directory=root/'runs'/arm/'seed42/train';result=read(directory/'TRAIN_RESULT.json')
        if result.get('status')!='completed' or result.get('completed_updates')!=181:raise ValueError('both complete Final181 required')
        _,proofs[arm]=legacy_anchor_view(anchor,old_schedule,read(directory/'SCHEDULE.json'),plan['schedule'],result)
        selected=[directory/name for name in ('TRAIN_RESULT.json','TRAIN_INPUT_LOCK.json','STATIC_INPUT_LOCK.json','SCHEDULE.json','TRAINED_INPUT_CONTRACTS.json')]
        selected.extend(file for file in (directory/'adapter').iterdir() if file.is_file())
        files.update({str(file):sha_file(file) for file in selected})
    for folder,pattern in [(root/'prepared/seed42','*'),(root/'teacher_replies/seed42','MANIFEST.json'),(root/'teacher_replies/seed42','TEACHER_REPLIES_LOCK.json')]:
        files.update({str(file):sha_file(file) for file in folder.glob(pattern) if file.is_file() and file.suffix in ('.json','.jsonl')})
    previous_queue=root/'queues/pilot42/STATUS.json';files[str(previous_queue)]=sha_file(previous_queue)
    queue=read(previous_queue);affinity={arm:queue['jobs'][f'train_eval_{arm}_seed42']['gpu_uuid'] for arm in ARMS}
    for arm in ARMS:
        worker=root/'workers'/f'train_eval_{arm}_seed42'
        files.update({str(file):sha_file(file) for file in worker.rglob('*') if file.is_file() and file.suffix in ('.json','.log')})
    identity={'schema_version':'rarevlm.object_behavior_replay.recovery_input.v1','scope':'completed_seed42_Finals_to_full_internal_pilot_only',
        'config_file':str(config_file),'config_file_sha256':sha_file(config_file),
        'original_study_input_lock_file':str(original_file),'original_study_input_lock_sha256':sha_file(original_file),
        'original_scientific_source_sha256':original['source_sha256'],'original_scientific_files_sha256':original['files_sha256'],
        'recovery_source_sha256':source_hashes(),'files_sha256':dict(sorted(files.items())),
        'historical_missing_field_proofs':proofs,'gpu_affinity_uuid':affinity,'failure_archive':str(archive),
        'new_training_or_teacher_generation_allowed':False,'old_worker_receipts_or_status_rewritten':False,
        'mandatory_later_stages':['teacher_ce32 seeds0/200 with original full teacher/training protocol','three-seed internal screen',
            'full4372 external confirmation','full9000 POPE confirmation','root independent final acceptance'],
        'later_automatic_controller_implemented':False,'goal_complete':False}
    lock_file=recovery_root/'RECOVERY_INPUT_LOCK.json';immutable(lock_file,identity)
    verify_lock(lock_file,config_file)
    from .contracts import adapter_binding
    bindings={}
    for arm in ARMS:
        _,bindings[arm]=adapter_binding(config,arm,42,root/'runs'/arm/'seed42/train/adapter',Path(config['model_root']),
            config_file=config_file,recovery_lock=lock_file)
    receipt={'schema_version':'rarevlm.object_behavior_replay.recovery_preparation.v1','status':'passed',
        'recovery_input_lock_sha256':sha_file(lock_file),'full_original_training_and_remaining_adapter_guards_passed':True,
        'bindings':bindings,'gpu_operations':0,'new_model_forwards':0,'optimizer_updates':0,'goal_complete':False}
    immutable(recovery_root/'PREPARATION.json',receipt)
    return identity


def read_worker(file,lock_file):
    value=read(file)
    if (value.get('schema_version')!='rarevlm.object_behavior_replay.recovery_worker.v1' or value.get('status')!='completed'
        or value.get('recovery_input_lock_sha256')!=sha_file(lock_file)):raise ValueError('recovery worker receipt identity differs')
    verify_files(value['artifact_files_sha256'])
    if value.get('outcome')=='engineering_rejected':
        from vlm_object_behavior_replay.study import call_reason
        evidence=value['engineering_evidence_files_sha256'];verify_files(evidence)
        attempts=[read(name) for name in evidence if name.endswith('.call.json')]
        if len(attempts)!=1 or call_reason(attempts[0],evidence)!=value['engineering_reason']:
            raise ValueError('recovery engineering rejection lacks the original matching attempt evidence')
    elif value.get('outcome')!='passed':raise ValueError('unknown recovery worker outcome')
    return value


def worker(config_file,recovery_root,condition):
    from vlm_joint.evaluate import visible_gpu_binding
    from vlm_object_behavior_replay.study import subprocess_call,EngineeringRejected
    from vlm_object_behavior_replay.contracts import validate_gpu_admission
    from .contracts import adapter_binding,read_cell
    config_file=Path(config_file).resolve();recovery_root=Path(recovery_root).resolve();lock_file=recovery_root/'RECOVERY_INPUT_LOCK.json'
    lock=verify_lock(lock_file,config_file);config=read(config_file);root=Path(config['output_root'])
    assert_original_inactive(config_file)
    if condition not in ARMS:raise ValueError('only two complete pilot42 Finals can be evaluated')
    hardware=visible_gpu_binding()
    if hardware['uuid']!=lock['gpu_affinity_uuid'][condition]:raise ValueError('original role UUID affinity changed')
    name=f'recovery_pilot_{condition}_seed42';place=recovery_root/'workers'/name
    identity={'schema_version':'rarevlm.object_behavior_replay.recovery_worker_input.v1','role':name,'condition':condition,'seed':42,
        'hardware':hardware,'recovery_input_lock_sha256':sha_file(lock_file),'effective_recovery_source_sha256':lock['recovery_source_sha256'],
        'training_allowed':False,'teacher_generation_allowed':False}
    immutable(place/'INPUT_LOCK.json',identity)
    if (place/'COMPLETE.json').exists():return read_worker(place/'COMPLETE.json',lock_file)
    artifacts={str(place/'INPUT_LOCK.json'):sha_file(place/'INPUT_LOCK.json')};outcome='passed';reason=None;evidence={}
    try:
        adapter=root/'runs'/condition/'seed42/train/adapter'
        adapter_binding(config,condition,42,adapter,Path(config['model_root']),config_file=config_file,recovery_lock=lock_file)
        for task in ('object','text'):
            admission=place/'admissions'/f'{task}.json'
            value={'schema_version':'rarevlm.object_behavior_replay.gpu_admission.v1','status':'admitted','condition':condition,
                'seed':42,'task':task,'role':f'train_eval_{condition}_seed42','gpu_uuid':hardware['uuid'],
                'config_sha256':sha_file(config_file),'config_file_sha256':sha_file(config_file),
                'study_input_lock_path':str(root/'STUDY_INPUT_LOCK.json'),'study_input_lock_sha256':sha_file(root/'STUDY_INPUT_LOCK.json'),
                'recovery_worker_role':name,'recovery_input_lock_sha256':sha_file(lock_file),
                'effective_recovery_source_sha256':lock['recovery_source_sha256']}
            immutable(admission,value);validate_gpu_admission(config_file,admission,condition=condition,seed=42,task=task,gpu_uuid=hardware['uuid'])
            artifacts[str(admission)]=sha_file(admission)
            command=[sys.executable,'-m','vlm_object_behavior_recovery_v1.evaluate','--config',str(config_file),'--condition',condition,
                '--seed','42','--task',task,'--adapter-dir',str(adapter),'--output-root',str(root),'--admission',str(admission),
                '--recovery-lock',str(lock_file)]
            subprocess_call(place,name,artifacts,task,command,root/'preflight'/condition/'seed42'/task/'PREFLIGHT.json')
            cell=read_cell(root,condition,42,task,config_file,config,recovery_lock=lock_file)
            if not cell['complete']:raise ValueError('full original task did not complete')
            for kind in ('predictions','evaluation','input_lock'):
                artifacts[cell['provenance'][kind+'_file']]=cell['provenance'][kind+'_sha256']
    except EngineeringRejected as error:outcome='engineering_rejected';reason=error.reason;evidence=error.evidence
    except BaseException as error:
        write(place/'PROGRESS.json',{'status':'execution_or_contract_failure','error':str(error),'updated_at':now(),
            'scientific_failure':False,'new_optimizer_updates':0});raise
    artifacts.update({str(file):sha_file(file) for file in place.rglob('*') if file.is_file()
        and file.suffix in ('.json','.log') and file.name not in ('PROGRESS.json','COMPLETE.json')})
    receipt={'schema_version':'rarevlm.object_behavior_replay.recovery_worker.v1','status':'completed','outcome':outcome,
        'role':name,'condition':condition,'seed':42,'gpu_uuid':hardware['uuid'],'completed_at':now(),
        'recovery_input_lock_sha256':sha_file(lock_file),'effective_recovery_source_sha256':lock['recovery_source_sha256'],
        'artifact_files_sha256':artifacts,'engineering_reason':reason,'engineering_evidence_files_sha256':evidence,
        'new_optimizer_updates':0,'new_teacher_generation_questions':0,'new_training_preflights':0,'goal_complete':False}
    immutable(place/'COMPLETE.json',receipt);write(place/'PROGRESS.json',{'status':'completed','outcome':outcome,'updated_at':now()})
    return receipt


def run(config_file,recovery_root):
    from research_runtime.queue import run_queue
    from .analysis import analyze
    config_file=Path(config_file).resolve();recovery_root=Path(recovery_root).resolve();config=read(config_file);root=Path(config['output_root'])
    owner=(recovery_root/'CONTROLLER.lock').open('a+');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    original_owner=(root/'STUDY.lock').open('r');fcntl.flock(original_owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock_file=recovery_root/'RECOVERY_INPUT_LOCK.json';begun=time.monotonic()
    try:
        lock=verify_lock(lock_file,config_file);assert_original_inactive(config_file)
        preparation=read(recovery_root/'PREPARATION.json')
        if preparation.get('status')!='passed' or preparation['recovery_input_lock_sha256']!=sha_file(lock_file):raise ValueError('passed CPU preparation required')
        if (recovery_root/'COMPLETION.json').exists():
            result=read(recovery_root/'COMPLETION.json')
            if (result.get('schema_version')!='rarevlm.object_behavior_replay.recovery_completion.v1'
                or result.get('recovery_input_lock_sha256')!=sha_file(lock_file)
                or result.get('effective_recovery_source_sha256')!=lock['recovery_source_sha256']):
                raise ValueError('completed recovery identity differs')
            verify_files(result['worker_receipts_sha256']);verify_files(result['analysis_files_sha256']);return result
        queue=recovery_root/'queues/pilot42';jobs=[]
        for arm in ARMS:
            name=f'recovery_pilot_{arm}_seed42';marker=recovery_root/'workers'/name/'COMPLETE.json'
            if marker.exists():read_worker(marker,lock_file)
            command=[sys.executable,'-m','vlm_object_behavior_recovery_v1.study','--worker','--config',str(config_file),
                '--recovery-root',str(recovery_root),'--condition',arm]
            jobs.append({'key':name,'command':command,'completion_marker':str(marker)})
        if not (queue/'STATUS.json').exists():
            write(queue/'STATUS.json',{'status':'waiting_resources','jobs':{f'recovery_pilot_{arm}_seed42':
                {'state':'waiting_gpu','gpu_uuid':lock['gpu_affinity_uuid'][arm],'contention_observations':0} for arm in ARMS}})
        identity={'recovery_input_lock_sha256':sha_file(lock_file),'original_study_input_lock_sha256':lock['original_study_input_lock_sha256'],
            'effective_recovery_source_sha256':lock['recovery_source_sha256'],'prepared_receipt_sha256':sha_file(recovery_root/'PREPARATION.json')}
        write(recovery_root/'STATUS.json',{'status':'pilot_eval_running','controller_pid':os.getpid(),'updated_at':now(),'goal_complete':False})
        run_queue(queue,jobs,identity)
        receipts=[read_worker(Path(job['completion_marker']),lock_file) for job in jobs]
        if any(value['outcome']!='passed' for value in receipts):status='engineering_rejected';analysis_files={};screen=None
        else:
            screen,file=analyze(config_file,root,[42],'screen',recovery_lock=lock_file)
            if screen['status']!='complete':raise ValueError('all four complete pilot tasks required')
            status='pilot_passed_pending_original_replication' if screen['gate']['passed'] else 'completed_negative'
            analysis_files={str(file):sha_file(file)}
        result={'schema_version':'rarevlm.object_behavior_replay.recovery_completion.v1','status':status,'trial_status':status,
            'completed_at':now(),'candidate':'teacher_ce32','control_can_be_selected':False,'goal_complete':False,
            'original_study_input_lock_sha256':lock['original_study_input_lock_sha256'],'recovery_input_lock_sha256':sha_file(lock_file),
            'effective_recovery_source_sha256':lock['recovery_source_sha256'],'analysis_files_sha256':analysis_files,
            'worker_receipts_sha256':{job['completion_marker']:sha_file(job['completion_marker']) for job in jobs},
            'new_optimizer_updates':0,'new_teacher_generation_questions':0,'new_training_preflights':0,
            'formal_evaluation_questions':3918 if screen is not None else None,
            'evaluation_engineering_generations':'original evaluator preflights retained; see per-cell PREFLIGHT records, separate from formal3918',
            'recovery_controller_session_seconds':time.monotonic()-begun,'scientific_failure':status=='completed_negative',
            'later_automatic_controller_implemented':False,'mandatory_later_stages':lock['mandatory_later_stages']}
        immutable(recovery_root/'COMPLETION.json',result);write(recovery_root/'STATUS.json',result)
        return result
    except BaseException as error:
        write(recovery_root/'STATUS.json',{'status':'execution_or_contract_failure','error':str(error),'updated_at':now(),
            'scientific_failure':False,'goal_complete':False});raise
    finally:original_owner.close();owner.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--recovery-root',type=Path,required=True)
    action=parser.add_mutually_exclusive_group(required=True)
    for flag in ('check-cpu','prepare-only','run','worker'):action.add_argument('--'+flag,action='store_true')
    parser.add_argument('--cpu-receipt',type=Path);parser.add_argument('--failure-archive',type=Path)
    parser.add_argument('--condition',choices=ARMS)
    args=parser.parse_args()
    if not args.config.is_absolute() or not args.recovery_root.is_absolute():raise ValueError('absolute paths required')
    if args.check_cpu:result=cpu_contracts(args.config,args.recovery_root)
    elif args.prepare_only:
        if args.cpu_receipt is None or args.failure_archive is None:parser.error('--prepare-only requires --cpu-receipt and --failure-archive')
        result=prepare(args.config,args.recovery_root,args.cpu_receipt,args.failure_archive)
        result={'status':'prepared','recovery_input_lock_sha256':sha_file(args.recovery_root/'RECOVERY_INPUT_LOCK.json'),'goal_complete':False}
    elif args.worker:
        if args.condition is None:parser.error('--worker requires --condition')
        result=worker(args.config,args.recovery_root,args.condition)
    else:result=run(args.config,args.recovery_root)
    print(json.dumps({'status':result.get('status'),'goal_complete':False},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
