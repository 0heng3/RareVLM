"""Independent coverage queue; only wide32 can pass the frozen four endpoints."""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_key]='2'
import argparse
import fcntl
import json
import re
from pathlib import Path
import sys
import time
from rarevlm_paths import PROJECT_ROOT,SOURCE_ROOT
from research_runtime.queue import run_queue
from vlm_joint.evaluate import visible_gpu_binding
from vlm_object_kl_study.worker import call as subprocess_call,EngineeringRejected
from vlm_object_kl_study.common import call_reason
from vlm_object_kl_eval.read import verify_historical_study
from vlm_object_study.run import validate_quality_admission
from vlm_object_train.data import read_training_rows
from vlm_probe.evaluate import content_sha,lock_file,now,package_versions,read_json,sha_file,write_json
from .contracts import (CANDIDATE,CONTROL,TRAINED_CONDITIONS,adapter_binding,path,source_hashes,read_cell,
    external_admission,load_data,verify_source_data,validate_condition)
from .analysis import analyze,validate_settings
from .evaluate import validate_config as validate_evaluation

OPERATIONS=('preflight','train_eval','external','pope')


def verify_files(files):
    for name, digest in files.items():
        if not Path(name).is_file() or sha_file(Path(name)) != digest:
            raise ValueError(f"locked KL study file changed: {name}")

def worker_dir(root, name):
    if not re.fullmatch(r"[a-z0-9_]+", name):
        raise ValueError("unsafe role")
    return root / "workers" / name

def eval_arguments(config_file, root, condition, seed, task, adapter, admission, prior=None):
    args = [sys.executable, "-m", "vlm_object_replay_coverage.evaluate", "--config", str(config_file),
        "--condition", condition, "--seed", str(seed), "--task", task, "--adapter-dir", str(adapter),
        "--output-root", str(root), "--admission", str(admission)]
    if condition == CANDIDATE and task in ("external", "pope"):
        if prior is None:
            raise ValueError("candidate confirmation needs passing prior phase")
        args.extend(["--screen-analysis" if task == "external" else "--external-analysis", str(prior)])
    return args

def read_receipt(file, identity_sha):
    value = read_json(file)
    if (value.get("schema_version") != "rarevlm.object_replay_coverage.worker.v1" or value.get("status") != "completed"
        or value.get("study_input_lock_sha256") != identity_sha):
        raise ValueError("KL worker completion identity differs")
    verify_files(value["artifact_files_sha256"])
    if value["outcome"] == "engineering_rejected":
        evidence = value["engineering_evidence_files_sha256"]
        verify_files(evidence)
        attempts = [read_json(Path(name)) for name in evidence if name.endswith(".call.json")]
        if len(attempts) != 1 or call_reason(attempts[0], evidence) != value["engineering_reason"]:
            raise ValueError("engineering rejection lacks original matching evidence")
    elif value["outcome"] != "passed":
        raise ValueError("unknown KL worker outcome")
    return value

def task_admission(directory, config_file, condition, seed, task, hardware):
    file = directory / "admissions" / f"{task}.json"
    lock_file(file, {"schema_version": "rarevlm.object_replay_coverage.gpu_admission.v1", "status": "admitted",
        "condition": condition, "seed": seed, "task": task, "gpu_uuid": hardware["uuid"],
        "config_sha256": sha_file(config_file)})
    return file

def worker_run(args):
    config_file = args.config.resolve()
    config = read_json(config_file)
    root = path(config, "output_root")
    if args.output_root.resolve() != root:
        raise ValueError("KL worker output differs")
    verify_study(args.study_lock, config_file)
    name = role(args.operation, args.condition, args.seed)
    place = worker_dir(root, name)
    hardware = visible_gpu_binding()
    identity = {"schema_version": "rarevlm.object_replay_coverage.worker_input.v1", "role": name,
        "operation": args.operation, "condition": args.condition, "seed": args.seed,
        "study_input_lock_sha256": sha_file(args.study_lock), "config_sha256": sha_file(config_file),
        "hardware": hardware, "prior_phase_sha256": sha_file(args.screen_analysis or args.external_analysis) if args.screen_analysis or args.external_analysis else None}
    lock_file(place / "INPUT_LOCK.json", identity)
    if (place / "COMPLETE.json").exists():
        return read_receipt(place / "COMPLETE.json", sha_file(args.study_lock))
    artifacts = {str(place / "INPUT_LOCK.json"): sha_file(place / "INPUT_LOCK.json")}
    def evaluate(task, adapter, prior=None):
        admission = task_admission(place, config_file, args.condition, args.seed, task, hardware)
        artifacts[str(admission)] = sha_file(admission)
        subprocess_call(place, name, artifacts, task, eval_arguments(config_file, root, args.condition, args.seed, task, adapter, admission, prior),
            root / "preflight" / args.condition / f"seed{args.seed}" / task / "PREFLIGHT.json")
        cell = read_cell(root, args.condition, args.seed, task, config_file, config)
        if not cell["complete"]: raise ValueError("evaluator exited without full completed cell")
        for kind in ("predictions", "evaluation", "input_lock"):
            artifacts[cell["provenance"][f"{kind}_file"]] = cell["provenance"][f"{kind}_sha256"]
    outcome, failure, evidence = "passed", None, {}
    try:
        if args.operation in ("preflight", "train_eval"):
            training_root = root / "engineering_runs" if args.operation == "preflight" else root
            train_dir = training_root / "runs" / args.condition / f"seed{args.seed}/train"
            if args.operation == "preflight" or not (train_dir / "TRAIN_RESULT.json").exists():
                subprocess_call(place, name, artifacts, "engineering_preflight" if args.operation == "preflight" else "training",
                    train_arguments(config_file, training_root, args.condition, args.seed, args.operation == "preflight"),
                    train_dir / "PREFLIGHT.json", train_dir / "teacher_cache/BUILD_FAILURE.json")
            artifacts.update(local_training_artifacts(train_dir))
            if args.operation == "train_eval":
                adapter = train_dir / "adapter"
                files, binding = adapter_binding(config, args.condition, args.seed, adapter, path(config, "model_root"), config_file=config_file)
                artifacts[str(train_dir / "TRAIN_RESULT.json")] = binding["sha256"]
                for file, digest in files.items(): artifacts[str(adapter / file)] = digest
                evaluate("object", adapter)
                evaluate("text", adapter)
        elif args.operation == "external":
            evaluate("external", root / "runs" / CANDIDATE / f"seed{args.seed}/train/adapter", args.screen_analysis)
        elif args.operation == "pope":
            evaluate("pope", root / "runs" / CANDIDATE / f"seed{args.seed}/train/adapter", args.external_analysis)
    except EngineeringRejected as error:
        outcome, failure, evidence = "engineering_rejected", error.reason, error.evidence
    except BaseException as error:
        write_json(place / "PROGRESS.json", {"status": "execution_or_contract_failure", "error_type": type(error).__name__,
            "error": str(error), "scientific_failure": False, "updated_at": now()})
        raise
    receipt = {"schema_version": "rarevlm.object_replay_coverage.worker.v1", "status": "completed", "outcome": outcome,
        "engineering_reason": failure, "engineering_evidence_files_sha256": evidence,
        "role": name, "operation": args.operation, "condition": args.condition, "seed": args.seed,
        "completed_at": now(), "gpu_uuid": hardware["uuid"], "study_input_lock_sha256": sha_file(args.study_lock),
        "artifact_files_sha256": artifacts, "automatic_runtime_fallback": False, "existing_tasks_untouched": True}
    write_json(place / "COMPLETE.json", receipt)
    write_json(place / "PROGRESS.json", {"status": "completed", "outcome": outcome, "updated_at": now()})
    return receipt


def local_training_artifacts(directory):
    files={str(f):sha_file(f) for pattern in ('*.json','*.jsonl') for f in directory.glob(pattern)}
    for child in directory.iterdir():
        if child.is_dir() and child.name!='adapter' and not child.name.startswith('checkpoint_'):
            for file in child.rglob('*'):
                if file.is_file() and file.suffix in ('.pt','.json'):files[str(file)]=sha_file(file)
    return files


def role(operation,condition,seed):
    validate_condition(condition,seed)
    if (operation not in OPERATIONS or (operation=='preflight' and seed!=42)
        or (condition==CONTROL and operation in ('external','pope'))):
        raise ValueError('control is pilot-only; confirmation belongs only to wide32')
    return f'{operation}_{condition}_seed{seed}'


def train_arguments(config_file,root,condition,seed,preflight=False):
    validate_condition(condition,seed)
    values=[sys.executable,'-m','vlm_object_replay_coverage.train','--config',str(config_file),
        '--condition',condition,'--seed',str(seed),'--output-root',str(root)]
    if preflight:values.append('--preflight')
    elif (root/'runs'/condition/f'seed{seed}/train/LATEST.json').exists():values.append('--resume')
    return values


def job(config_file,root,operation,condition,seed,prior=None):
    name=role(operation,condition,seed)
    args=[sys.executable,'-m','vlm_object_replay_coverage.study','--worker','--config',str(config_file),
        '--output-root',str(root),'--study-lock',str(root/'STUDY_INPUT_LOCK.json'),
        '--operation',operation,'--condition',condition,'--seed',str(seed)]
    if prior is not None:args.extend(['--screen-analysis' if operation=='external' else '--external-analysis',str(prior)])
    return {'key':name,'command':args,'completion_marker':str(worker_dir(root,name)/'COMPLETE.json')}


def study_sources():
    return {str(PROJECT_ROOT/name if name.startswith('tools/') else SOURCE_ROOT/name):digest
            for name,digest in source_hashes().items()}


def verify_study(file,config_file):
    lock=read_json(file)
    if (lock.get('schema_version')!='rarevlm.object_replay_coverage.study_input.v1'
        or lock.get('config_file_sha256')!=sha_file(config_file)
        or lock.get('source_sha256')!=study_sources() or lock.get('packages')!=package_versions()):
        raise ValueError('coverage study source/config/runtime identity changed')
    verify_files(lock['source_sha256']);verify_files(lock['files_sha256'])
    return lock


def file_map(paths):
    result={}
    for file in paths:
        file=Path(file).resolve()
        if not file.exists():raise ValueError(f'required coverage source missing: {file}')
        if file.is_file():result[str(file)]=sha_file(file)
    return result


def validate_registry(config):
    if (config.get('schema_version')!='rarevlm.object_replay_coverage.train.v1'
        or config.get('candidates')!=[CANDIDATE] or config.get('controls')!=[CONTROL]
        or config.get('pilot_seed')!=42 or config.get('replication_seeds')!=[0,200]
        or config.get('all_seeds')!=[0,42,200]):
        raise ValueError('only wide32 may expand; repeat8x4 stays exploratory seed42')
    for name in ('source_kl_config','source_kl_root','gradient_diagnostic_root','external_initial_reference',
                 'teacher_reference_study_root','study_protocol','cpu_contracts','output_root'):
        path(config,name)
    if path(config,'teacher_reference_study_root')!=path(config,'source_kl_root'):
        raise ValueError('control teacher must reference the original KL bank')
    if config.get('teacher_reference_condition')!='object_kl' or config.get('standalone_reference_scope')!='engineering_first_update_only':
        raise ValueError('fixed original teacher and first32-only standalone scope required')
    if path(config,'output_root') in (path(config,'source_kl_root'),path(config,'gradient_diagnostic_root')):
        raise ValueError('coverage outputs must be independent')


def prepare(config_file):
    from .train import validate_config as validate_training
    from vlm_object_kl_study.common import verify_study as verify_original_kl
    config_file=Path(config_file).resolve();config=read_json(config_file)
    validate_registry(config);validate_training(config);validate_evaluation(config);validate_settings(config)
    validate_quality_admission(read_json(path(config,'data_admission')))
    external_binding=external_admission(config)
    cpu=read_json(path(config,'cpu_contracts'))
    if cpu.get('status')!='passed':raise ValueError('new complete CPU contracts required before registration/GPU')
    verify_files(cpu['files_sha256']);verify_files(cpu['scientific_source_sha256'])
    sources=study_sources()
    for name,digest in sources.items():
        if ('vlm_object_replay_coverage/' in name or name.endswith('run_object_replay_coverage_study.py')) and cpu['scientific_source_sha256'].get(name)!=digest:
            raise ValueError('CPU receipt must bind all current coverage scientific source')
    objects,replays,data_binding=read_training_rows(config)
    old_root,old_config_file=path(config,'source_kl_root'),path(config,'source_kl_config')
    old_config=read_json(old_config_file);old=verify_original_kl(old_root/'STUDY_INPUT_LOCK.json',old_config_file)
    completed=read_json(old_root/'COMPLETION.json')
    if (completed.get('status')!='completed_negative' or completed.get('goal_complete') is not False
        or completed.get('study_input_lock_sha256')!=sha_file(old_root/'STUDY_INPUT_LOCK.json')):
        raise ValueError('original negative KL study must remain complete and frozen')
    verify_files(completed['analysis_files_sha256'])
    for name in ('asset_root','object_data_root','train_manifest','val_manifest','val_class_balanced_manifest',
                 'replay_manifest','pope_data_root','reference_config','model_root','initial_adapter','dataset_protocol',
                 'external_data_root','external_manifest','external_class_balanced_manifest','external_dataset_protocol'):
        if path(config,name)!=path(old_config,name):raise ValueError(f'fixed source data/model path differs: {name}')
    files={}
    for name in ('source_sha256','protocol_files_sha256','data_files_sha256','model_files_sha256',
                 'initializer_files_sha256','historical_files_sha256'):files.update(old[name])
    files.update(file_map([old_config_file,old_root/'STUDY_INPUT_LOCK.json',old_root/'STUDY.json',old_root/'COMPLETION.json']))
    old_train=old_root/'runs/object_kl/seed42/train'
    files.update(file_map([*old_train.glob('*.json'),*old_train.glob('adapter/*'),*old_train.glob('teacher_cache/*')]))
    diagnostic=path(config,'gradient_diagnostic_root')
    diag=read_json(diagnostic/'COMPLETION.json');recovery=read_json(diagnostic/'RECOVERY.json')
    if (diag.get('status')!='completed_diagnostic' or diag.get('goal_complete') is not False
        or diag.get('performance_evaluation') is not False or recovery.get('exact_frozen_aggregate_checks_passed') is not True
        or diag.get('study_input_lock_sha256')!=sha_file(diagnostic/'STUDY_INPUT_LOCK.json')):
        raise ValueError('descriptive gradient diagnostic must be complete, not performance success')
    verify_files(diag['artifact_files_sha256']);verify_files(recovery['protected_files_sha256'])
    files.update(diag['artifact_files_sha256']);files.update(recovery['protected_files_sha256'])
    files.update(file_map([diagnostic/'COMPLETION.json',diagnostic/'RECOVERY.json',diagnostic/'STUDY_INPUT_LOCK.json']))
    root=path(config,'output_root')
    historical={}
    for task in ('object','text','external','pope'):
        cell=read_cell(root,'initial',42,task,config_file,config)
        if not cell['complete']:raise ValueError('all complete historical initial predictions must be available')
        for kind in ('predictions','evaluation','input_lock'):
            file=cell['provenance'][kind+'_file'];historical[file]=cell['provenance'][kind+'_sha256']
        historical.update(file_map([root/'runs/initial/seed42'/task/'HISTORICAL_REFERENCE.json']))
    files.update(historical);files.update(cpu['files_sha256']);files.update(cpu['scientific_source_sha256'])
    new_files=[config_file,*[path(config,name) for name in ('study_protocol','dataset_protocol','external_dataset_protocol',
        'reference_config','cpu_contracts','data_admission','evaluation_data_admission','train_manifest','val_manifest',
        'val_class_balanced_manifest','replay_manifest','external_manifest','external_class_balanced_manifest',
        'external_semantic_review','external_frozen_audit','external_cpu_completion')]]
    new_files.extend(Path(value) for value in config.get('extra_lock_files',[]))
    files.update(file_map(new_files));verify_files(files);verify_files(sources)
    identity={'schema_version':'rarevlm.object_replay_coverage.study_input.v1','config_path':str(config_file),
        'config_file_sha256':sha_file(config_file),'config_content_sha256':content_sha(config),'source_sha256':sources,
        'files_sha256':dict(sorted(files.items())),'packages':package_versions(),'training_data_binding':data_binding,
        'external_data_admission':external_binding,'dataset_metadata':{task:load_data(config,task)[3] for task in ('object','text','external','pope')},
        'train_counts':{'object_questions':len(objects),'replay_pool_questions':len(replays),'updates':(len(objects)+15)//16,
                        'replay_exposures':((len(objects)+15)//16)*32},
        'candidate':CANDIDATE,'exploratory_control':CONTROL,'equal_teacher_compute_claim':False}
    lock_file(root/'STUDY_INPUT_LOCK.json',identity)
    lock_file(root/'STUDY.json',{'schema_version':'rarevlm.object_replay_coverage.study.v1',
        'study_input_lock_sha256':sha_file(root/'STUDY_INPUT_LOCK.json'),'candidate':CANDIDATE,'control':CONTROL,
        'control_only_pilot_seed42':True,'control_can_be_selected':False,'all_initial_baselines_historical':True,
        'stages':['both_engineering_first32','both_full_pilot_internal','wide32_pilot_screen','wide32_seeds0_200',
                  'wide32_three_seed_internal','wide32_full_external','wide32_full_POPE'],
        'family':'one candidate four endpoints alpha=.05/4; exploratory control not eligible',
        'gpu_policy':'shared queue three15sec idle >=40GiB util<10% zero compute UUID lock same-role resume',
        'memory_cap_gib':65,'automatic_runtime_fallback':False,'equal_teacher_compute_claim':False,'goal_complete':False})
    return config,root,identity


def queue_stage(root,name,jobs,identity):
    digest=sha_file(root/'STUDY_INPUT_LOCK.json');queue=root/'queues'/name
    previous=read_json(queue/'STATUS.json') if (queue/'STATUS.json').exists() else {}
    for item in jobs:
        file=Path(item['completion_marker'])
        if file.exists():read_receipt(file,digest)
        elif previous.get('jobs',{}).get(item['key'],{}).get('state')=='completed':
            raise ValueError('queue completion receipt missing')
    run_queue(queue,jobs,identity)
    return [read_receipt(Path(item['completion_marker']),digest) for item in jobs]


def finish(root,status,phase,analyses,receipts,started):
    preflight_history=list((root/'engineering_runs/runs').glob('*/seed*/train/PREFLIGHT_HISTORY.jsonl'))
    preflight_history.extend((root/'runs').glob('*/seed*/train/PREFLIGHT_HISTORY.jsonl'))
    preflight_attempts=[]
    for file in sorted(preflight_history):
        for line in file.read_text().splitlines():
            value=json.loads(line)
            preflight_attempts.append({'file':str(file),'status':value['status'],
                **{key:value.get(key) for key in ('reference_scope','discarded_engineering_optimizer_updates',
                    'formal_optimizer_updates','initial_reference_parity_questions',
                    'after_temporary_update_reference_probe_questions','engineering_actor_forward_backward_questions',
                    'actual_new_teacher_forward_questions','historical_teacher_bank_questions')}})
    result={'schema_version':'rarevlm.object_replay_coverage.trial.v1','status':status,'trial_status':status,
        'completed_at':now(),'last_phase':phase,'candidate':CANDIDATE,'control_can_be_selected':False,
        'goal_complete':False,'root_independent_prediction_review_required':True,
        'confirmation_passed':status=='confirmation_passed_pending_independent_review',
        'scientific_failure':status=='completed_negative','study_input_lock_sha256':sha_file(root/'STUDY_INPUT_LOCK.json'),
        'analysis_files_sha256':{str(f):sha_file(f) for f in analyses.values()},
        'worker_receipts_sha256':{str(worker_dir(root,r['role'])/'COMPLETE.json'):sha_file(worker_dir(root,r['role'])/'COMPLETE.json') for r in receipts},
        'current_controller_session_seconds':time.monotonic()-started,'existing_tasks_untouched':True,
        'equal_teacher_compute_claim':False}
    result['preflight_accounting']={'planned_pilot_discarded_engineering_optimizer_updates':4,
        'actual_discarded_engineering_optimizer_updates_all_recorded_attempts':sum(
            row['discarded_engineering_optimizer_updates'] for row in preflight_attempts),
        'recorded_attempts':preflight_attempts,'source_files_sha256':file_map(preflight_history),
        'scope':'engineering discarded steps separate from formal181 updates/arm; history includes any retries and replication preflights'}
    write_json(root/'COMPLETION.json',result);write_json(root/'STATUS.json',result)
    if result['confirmation_passed']:write_json(root/'TRIAL_EVIDENCE.json',result)
    return result


def run_study(config_file,prepare_only=False):
    config_file=Path(config_file).resolve();root=path(read_json(config_file),'output_root')
    root.mkdir(parents=True,exist_ok=True)
    owner=(root/'STUDY.lock').open('a+');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.monotonic();analyses={};receipts=[]
    def queue(name,jobs):
        write_json(root/'STATUS.json',{'status':name,'updated_at':now(),'controller_pid':os.getpid(),
            'goal_complete':False,'existing_tasks_untouched':True})
        values=queue_stage(root,name,jobs,identity);receipts.extend(values)
        return all(value['outcome']=='passed' for value in values)
    def close(status,phase):
        verify_study(root/'STUDY_INPUT_LOCK.json',config_file)
        return finish(root,status,phase,analyses,receipts,started)
    try:
        config,root,identity=prepare(config_file)
        if prepare_only:return {'status':'prepared','goal_complete':False}
        if (root/'COMPLETION.json').exists():
            value=read_json(root/'COMPLETION.json')
            if value['study_input_lock_sha256']!=sha_file(root/'STUDY_INPUT_LOCK.json'):raise ValueError('completed coverage lock changed')
            verify_files(value['analysis_files_sha256']);verify_files(value['worker_receipts_sha256'])
            for role_name in (Path(f).parent.name for f in value['worker_receipts_sha256']):
                read_receipt(worker_dir(root,role_name)/'COMPLETE.json',sha_file(root/'STUDY_INPUT_LOCK.json'))
            return value
        if not queue('engineering_preflight',[job(config_file,root,'preflight',arm,42) for arm in TRAINED_CONDITIONS]):
            return close('engineering_rejected','engineering_preflight')
        if not queue('pilot42',[job(config_file,root,'train_eval',arm,42) for arm in TRAINED_CONDITIONS]):
            return close('engineering_rejected','pilot42')
        pilot,file=analyze(config_file,root,[42],'screen');analyses['pilot']=file
        if pilot['status']!='complete':raise ValueError('both full pilot arms must complete before screen')
        if not pilot['gate']['passed']:return close('completed_negative','pilot_internal_screen')
        if not queue('replication',[job(config_file,root,'train_eval',CANDIDATE,seed) for seed in (0,200)]):
            return close('engineering_rejected','replication')
        screen,file=analyze(config_file,root,[0,42,200],'screen',pilot_file=analyses['pilot']);analyses['internal']=file
        if screen['status']!='complete':raise ValueError('wide32 full three-seed internal cells required')
        if not screen['gate']['passed']:return close('completed_negative','three_seed_internal_screen')
        if not queue('external_confirmation',[job(config_file,root,'external',CANDIDATE,seed,file) for seed in (0,42,200)]):
            return close('engineering_rejected','external_confirmation')
        external,file=analyze(config_file,root,[0,42,200],'external',screen_file=analyses['internal']);analyses['external']=file
        if external['status']!='complete':raise ValueError('all wide32 external full cells required')
        if not external['gate']['passed']:return close('completed_negative','external_confirmation')
        if not queue('pope_confirmation',[job(config_file,root,'pope',CANDIDATE,seed,file) for seed in (0,42,200)]):
            return close('engineering_rejected','pope_confirmation')
        confirmed,file=analyze(config_file,root,[0,42,200],'confirm',external_file=analyses['external']);analyses['pope']=file
        if confirmed['status']!='complete':raise ValueError('all wide32 full original POPE cells required')
        return close('confirmation_passed_pending_independent_review' if confirmed['gate']['passed'] else 'completed_negative','pope_confirmation')
    except BaseException as error:
        write_json(root/'STATUS.json',{'status':'execution_or_contract_failure','updated_at':now(),
            'error_type':type(error).__name__,'error':str(error),'scientific_failure':False,
            'goal_complete':False,'existing_tasks_untouched':True})
        raise
    finally:owner.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-only',action='store_true');mode.add_argument('--run',action='store_true')
    mode.add_argument('--worker',action='store_true')
    parser.add_argument('--output-root',type=Path);parser.add_argument('--study-lock',type=Path)
    parser.add_argument('--operation',choices=OPERATIONS);parser.add_argument('--condition',choices=TRAINED_CONDITIONS)
    parser.add_argument('--seed',type=int,choices=(0,42,200));parser.add_argument('--screen-analysis',type=Path)
    parser.add_argument('--external-analysis',type=Path)
    args=parser.parse_args()
    if args.worker:
        if any(getattr(args,name) is None for name in ('output_root','study_lock','operation','condition','seed')):
            parser.error('worker requires output-root/study-lock/operation/condition/seed')
        result=worker_run(args)
    else:result=run_study(args.config,args.prepare_only)
    print(result['status'],flush=True)


if __name__=='__main__':main()
