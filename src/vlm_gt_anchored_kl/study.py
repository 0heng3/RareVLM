"""One generated-prefix KL candidate through pilot, replication, external and POPE."""
PARENT_COPY_PROVENANCE = {'source': 'src/vlm_object_replay_coverage/study.py', 'sha256': 'ed25fb5339ac5315a934e082f57a7c65e5bb439c90c1463d1bbee59bcc44c301', 'scope': 'independent adapted copy; frozen original globals and bytes are unchanged'}
COPY_PROVENANCE = {'source': '/opt/data/private/longtail_vlm/rarevlm_pilot/packages/RareVLM_minimal_20261001/src/vlm_object_behavior_replay/study.py', 'sha256': 'f5a5960f45096c0bd1618dc7e1a259a1e10063fbca860d4e5700bbee0e575c83', 'scope': 'new full-stage queue copy; only embedded new-objective preflight, historical reference control, direct locks'}
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_key]='2'
import argparse
import fcntl
import hashlib
import subprocess
import json
import re
from pathlib import Path
import sys
import time
from rarevlm_paths import PROJECT_ROOT,SOURCE_ROOT
from research_runtime.queue import run_queue
from vlm_joint.evaluate import visible_gpu_binding
from vlm_object_kl_study.worker import snapshot
from vlm_object_kl_study.common import reason as original_engineering_reason
from .lock import verify_parent_proof,verify_study,study_sources
from vlm_object_study.run import validate_quality_admission
from .data import load_rows
from vlm_probe.evaluate import content_sha,lock_file,now,package_versions,read_json,sha_file,write_json
from .contracts import (CANDIDATE,CONTROL,TRAINED_CONDITIONS,adapter_binding,path,source_hashes,read_cell,
    external_admission,load_data,verify_source_data,validate_condition,validate_inherited_admissions,create_inherited_admissions)
from .analysis import analyze,validate_settings
from .evaluate import validate_config as validate_evaluation
from vlm_joint.evaluate import model_metadata_binding

OPERATIONS=('train_eval','external','pope')


def verify_files(files):
    for name, digest in files.items():
        if not Path(name).is_file() or sha_file(Path(name)) != digest:
            raise ValueError(f"locked behavior study file changed: {name}")

def worker_dir(root, name):
    if not re.fullmatch(r"[a-z0-9_]+", name):
        raise ValueError("unsafe role")
    return root / "workers" / name

def eval_arguments(config_file, root, condition, seed, task, adapter, admission, prior=None):
    args = [sys.executable, "-m", "vlm_gt_anchored_kl.evaluate", "--config", str(config_file),
        "--condition", condition, "--seed", str(seed), "--task", task, "--adapter-dir", str(adapter),
        "--output-root", str(root), "--admission", str(admission)]
    if condition == CANDIDATE and task in ("external", "pope"):
        if prior is None:
            raise ValueError("candidate confirmation needs passing prior phase")
        args.extend(["--screen-analysis" if task == "external" else "--external-analysis", str(prior)])
    return args

def read_receipt(file, identity_sha):
    value = read_json(file)
    if (value.get("schema_version") != "rarevlm.gt_anchored_kl.worker.v1" or value.get("status") != "completed"
        or value.get("study_input_lock_sha256") != identity_sha):
        raise ValueError("behavior worker completion identity differs")
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

def task_admission(directory,config_file,condition,seed,task,hardware,*,shard=None,prompt=None):
    from .contracts import validate_gpu_admission
    root=path(read_json(config_file),'output_root');study=root/'STUDY_INPUT_LOCK.json'
    role_name=role('teacher','initial',seed,shard) if task=='teacher_rollout' else role('train_eval' if task in ('object','text') else task,condition,seed)
    file=directory/'admissions'/('teacher_shard'+str(shard)+'.json' if task=='teacher_rollout' else task+'.json')
    value={'schema_version':'rarevlm.gt_anchored_kl.gpu_admission.v1','status':'admitted',
        'condition':condition,'seed':seed,'task':task,'role':role_name,'gpu_uuid':hardware['uuid'],
        'config_sha256':sha_file(config_file),'config_file_sha256':sha_file(config_file),
        'study_input_lock_path':str(study),'study_input_lock_sha256':sha_file(study)}
    if task=='teacher_rollout':
        if prompt is None:raise ValueError('teacher admission requires complete prompt CPU manifest')
        value.update(shard=shard,plan_file_sha256=prompt['plan_file_sha256'],prompt_inputs_file_sha256=prompt['prompt_inputs_file_sha256'])
    lock_file(file,value)
    validate_gpu_admission(config_file,file,condition=condition,seed=seed,task=task,gpu_uuid=hardware['uuid'],
        shard=shard,plan_file_sha256=value.get('plan_file_sha256'),prompt_inputs_file_sha256=value.get('prompt_inputs_file_sha256'))
    return file

def worker_run(args):
    config_file=args.config.resolve();config=read_json(config_file);root=path(config,'output_root')
    if args.output_root.resolve()!=root:raise ValueError('behavior worker output differs')
    verify_study(args.study_lock,config_file,check_weights=True)
    prior_binding=worker_prior(config_file,root,args.operation,args.seed,args.screen_analysis or args.external_analysis)
    name=role(args.operation,args.condition,args.seed,args.shard);place=worker_dir(root,name);hardware=visible_gpu_binding()
    identity={'schema_version':'rarevlm.gt_anchored_kl.worker_input.v1','role':name,
        'operation':args.operation,'condition':args.condition,'seed':args.seed,'shard':args.shard,
        'study_input_lock_sha256':sha_file(args.study_lock),'config_sha256':sha_file(config_file),'hardware':hardware,
        'prior_phase_sha256':sha_file(args.screen_analysis or args.external_analysis) if args.screen_analysis or args.external_analysis else None}
    identity.update(prior_binding)
    if args.operation=='train_eval':
        from .inputs import verify_prepared_training_inputs
        prepared=verify_prepared_training_inputs(config,args.seed,args.condition)
        identity['prepared_training_inputs_file_sha256']=prepared['manifest_file_sha256']
        identity['teacher_replies_manifest_file_sha256']=prepared['manifest']['teacher_replies_manifest_file_sha256']
        identity['teacher_replies_lock_file_sha256']=prepared['manifest']['teacher_replies_lock_file_sha256']
    lock_file(place/'INPUT_LOCK.json',identity)
    if (place/'COMPLETE.json').exists():return read_receipt(place/'COMPLETE.json',sha_file(args.study_lock))
    artifacts={str(place/'INPUT_LOCK.json'):sha_file(place/'INPUT_LOCK.json')}
    def evaluate(task,adapter,prior=None):
        admission=task_admission(place,config_file,args.condition,args.seed,task,hardware);artifacts[str(admission)]=sha_file(admission)
        subprocess_call(place,name,artifacts,task,eval_arguments(config_file,root,args.condition,args.seed,task,adapter,admission,prior),
            root/'preflight'/args.condition/f'seed{args.seed}'/task/'PREFLIGHT.json')
        cell=read_cell(root,args.condition,args.seed,task,config_file,config)
        if not cell['complete']:raise ValueError('evaluator exited without full completed cell')
        for kind in ('predictions','evaluation','input_lock'):
            artifacts[cell['provenance'][kind+'_file']]=cell['provenance'][kind+'_sha256']
    outcome,failure,evidence='passed',None,{}
    try:
        if args.operation=='train_eval':
            training_root=root
            train_dir=training_root/'runs'/args.condition/f'seed{args.seed}/train'
            if not (train_dir/'TRAIN_RESULT.json').exists():
                subprocess_call(place,name,artifacts,'training',
                    train_arguments(config_file,training_root,args.condition,args.seed),train_dir/'PREFLIGHT.json')
            artifacts.update(local_training_artifacts(train_dir))
            if args.operation=='train_eval':
                adapter=train_dir/'adapter';files,binding=adapter_binding(config,args.condition,args.seed,adapter,path(config,'model_root'),config_file=config_file)
                artifacts[str(train_dir/'TRAIN_RESULT.json')]=binding['sha256']
                for file,digest in files.items():artifacts[str(adapter/file)]=digest
                evaluate('object',adapter);evaluate('text',adapter)
        elif args.operation=='external':evaluate('external',root/'runs'/CANDIDATE/f'seed{args.seed}/train/adapter',args.screen_analysis)
        elif args.operation=='pope':evaluate('pope',root/'runs'/CANDIDATE/f'seed{args.seed}/train/adapter',args.external_analysis)
    except EngineeringRejected as error:outcome,failure,evidence='engineering_rejected',error.reason,error.evidence
    except BaseException as error:
        write_json(place/'PROGRESS.json',{'status':'execution_or_contract_failure','error_type':type(error).__name__,
            'error':str(error),'scientific_failure':False,'updated_at':now()});raise
    receipt={'schema_version':'rarevlm.gt_anchored_kl.worker.v1','status':'completed','outcome':outcome,
        'engineering_reason':failure,'engineering_evidence_files_sha256':evidence,'role':name,'operation':args.operation,
        'condition':args.condition,'seed':args.seed,'shard':args.shard,'completed_at':now(),'gpu_uuid':hardware['uuid'],
        'study_input_lock_sha256':sha_file(args.study_lock),'artifact_files_sha256':artifacts,
        'automatic_runtime_fallback':False,'existing_tasks_untouched':True}
    write_json(place/'COMPLETE.json',receipt);write_json(place/'PROGRESS.json',{'status':'completed','outcome':outcome,'updated_at':now()})
    return receipt

def local_training_artifacts(directory):
    # Behavior replay has no probability bank. Full new checkpoint masters remain resume artifacts.
    return {str(file):sha_file(file) for file in directory.rglob('*') if file.is_file()
        and file.suffix in ('.json','.jsonl','.pt','.safetensors')}


def role(operation,condition,seed,shard=None):
    validate_condition(condition,seed)
    if shard is not None or operation not in OPERATIONS or (condition==CONTROL and operation!='train_eval'):raise ValueError('registered GT-only phase required; control pilot-only')
    return f'{operation}_{condition}_seed{seed}'

def worker_prior(config_file,root,operation,seed,prior=None):
    if operation=='train_eval' and seed in (0,200):
        from .analysis import passing_pilot
        file=root/'analysis/pilot_seed42/screen/ANALYSIS.json';passing_pilot(file,config_file)
        return {'passing_pilot_file':str(file),'passing_pilot_sha256':sha_file(file)}
    if operation in ('external','pope'):
        if prior is None:raise ValueError('confirmation worker requires the unchanged passing three-seed prior phase')
        from .contracts import phase_admission
        phase_admission(prior,config_file,'screen' if operation=='external' else 'external')
    return {}


def train_arguments(config_file,root,condition,seed):
    validate_condition(condition,seed)
    values=[sys.executable,'-m','vlm_gt_anchored_kl.train','--config',str(config_file),
        '--condition',condition,'--seed',str(seed),'--output-root',str(root)]
    if (root/'runs'/condition/f'seed{seed}/train/LATEST.json').exists():values.append('--resume')
    return values


def job(config_file,root,operation,condition,seed,prior=None,*,shard=None):
    name=role(operation,condition,seed,shard)
    args=[sys.executable,'-m','vlm_gt_anchored_kl.study','--worker','--config',str(config_file),
        '--output-root',str(root),'--study-lock',str(root/'STUDY_INPUT_LOCK.json'),
        '--operation',operation,'--condition',condition,'--seed',str(seed)]
    if shard is not None:args.extend(['--shard',str(shard)])
    if prior is not None:args.extend(['--screen-analysis' if operation=='external' else '--external-analysis',str(prior)])
    return {'key':name,'command':args,'completion_marker':str(worker_dir(root,name)/'COMPLETE.json')}








def file_map(paths):
    result={}
    for file in paths:
        file=Path(file).resolve()
        if not file.exists():raise ValueError(f'required coverage source missing: {file}')
        if file.is_file():result[str(file)]=sha_file(file)
    return result








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





def run_study(config_file,prepare_only=False):
    config_file=Path(config_file).resolve();root=path(read_json(config_file),'output_root');root.mkdir(parents=True,exist_ok=True)
    owner=(root/'STUDY.lock').open('a+');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.monotonic();analyses={};receipts=[];phase='static_preparation'
    def queue(name,jobs):
        write_json(root/'STATUS.json',{'status':name,'updated_at':now(),'controller_pid':os.getpid(),'goal_complete':False,'existing_tasks_untouched':True})
        values=queue_stage(root,name,jobs,identity);receipts.extend(values)
        return all(value['outcome']=='passed' for value in values)
    def close(status,current):
        verify_study(root/'STUDY_INPUT_LOCK.json',config_file)
        return finish(root,status,current,analyses,receipts,started)
    try:
        config,root,identity=prepare(config_file)
        if prepare_only:return {'status':'prepared','goal_complete':False,'full_prompt_preparation_run':False,'GPU_operations':0}
        if (root/'COMPLETION.json').exists():
            value=read_json(root/'COMPLETION.json')
            if value['study_input_lock_sha256']!=sha_file(root/'STUDY_INPUT_LOCK.json'):raise ValueError('completed behavior lock changed')
            verify_files(value['analysis_files_sha256']);verify_files(value['worker_receipts_sha256']);verify_files(value['dynamic_input_files_sha256'])
            for file in value['worker_receipts_sha256']:read_receipt(Path(file),sha_file(root/'STUDY_INPUT_LOCK.json'))
            return value
        from .inputs import prepare_training_inputs,verify_prepared_training_inputs
        phase='accepted_parent_seed42_input_reference'
        for condition in TRAINED_CONDITIONS:
            prepare_training_inputs(config,42,condition);verify_prepared_training_inputs(config,42,condition)
        phase='pilot42'
        if not queue(phase,[job(config_file,root,'train_eval',condition,42) for condition in TRAINED_CONDITIONS]):return close('engineering_rejected',phase)
        pilot,file=analyze(config_file,root,[42],'screen');analyses['pilot']=file
        if pilot['status']!='complete':raise ValueError('full new candidate and immutable historical references required before pilot screen')
        if not pilot['gate']['passed']:return close('completed_negative','pilot_internal_screen')
        for seed in (0,200):
            prepare_training_inputs(config,seed,CANDIDATE);verify_prepared_training_inputs(config,seed,CANDIDATE)
        phase='replication'
        if not queue(phase,[job(config_file,root,'train_eval',CANDIDATE,seed) for seed in (0,200)]):return close('engineering_rejected',phase)
        screen,file=analyze(config_file,root,[0,42,200],'screen',pilot_file=analyses['pilot']);analyses['internal']=file
        if screen['status']!='complete':raise ValueError('full three-seed internal cells required')
        if not screen['gate']['passed']:return close('completed_negative','three_seed_internal_screen')
        phase='external_confirmation'
        if not queue(phase,[job(config_file,root,'external',CANDIDATE,seed,file) for seed in (0,42,200)]):return close('engineering_rejected',phase)
        external,file=analyze(config_file,root,[0,42,200],'external',screen_file=analyses['internal']);analyses['external']=file
        if external['status']!='complete':raise ValueError('all candidate external full cells required')
        if not external['gate']['passed']:return close('completed_negative',phase)
        phase='pope_confirmation'
        if not queue(phase,[job(config_file,root,'pope',CANDIDATE,seed,file) for seed in (0,42,200)]):return close('engineering_rejected',phase)
        confirmed,file=analyze(config_file,root,[0,42,200],'confirm',external_file=analyses['external']);analyses['pope']=file
        if confirmed['status']!='complete':raise ValueError('all candidate full original POPE cells required')
        return close('confirmation_passed_pending_independent_review' if confirmed['gate']['passed'] else 'completed_negative',phase)
    except BaseException as error:
        write_json(root/'STATUS.json',{'status':'execution_or_contract_failure','phase':phase,'updated_at':now(),
            'error_type':type(error).__name__,'error':str(error),'scientific_failure':False,'goal_complete':False,'existing_tasks_untouched':True});raise
    finally:owner.close()







class EngineeringRejected(RuntimeError):
    def __init__(self,reason,evidence):super().__init__(reason);self.reason=reason;self.evidence=evidence


def engineering_reason(log,preflight=None):
    found=re.search(r'(?:reference|formal_training|training|preflight|teacher_generation|teacher_rollout|rollout)_memory_cap_exceeded',log)
    if found:return found.group(0)
    return original_engineering_reason(log,preflight)


def call_reason(record,evidence):
    if record.get('schema_version')!='rarevlm.gt_anchored_kl.call.v1' or not record.get('returncode'):
        raise ValueError('engineering evidence must identify current failed invocation')
    file=Path(record['log_file']);data=file.read_bytes();start,end=record['log_start_byte'],record['log_end_byte']
    if evidence.get(str(file))!=sha_file(file) or not 0<=start<=end==len(data):raise ValueError('invocation log bytes differ')
    segment=data[start:end]
    if hashlib.sha256(segment).hexdigest()!=record['log_segment_sha256']:raise ValueError('current invocation log segment differs')
    receipt=None
    for kind,item in record['fresh_failure_receipts'].items():
        if (kind!='preflight' or item['before']==item['after'] or item['after'] is None
            or evidence.get(item['file'])!=item['after']['sha256']):raise ValueError('failure receipt is not fresh and bound')
        receipt=read_json(item['file'])
    return engineering_reason(segment.decode(errors='replace'),receipt)


def subprocess_call(place,name,artifacts,phase,command,preflight=None):
    log=place/'logs'/f'{phase}.log';log.parent.mkdir(parents=True,exist_ok=True)
    old=log.stat().st_size if log.exists() else 0;before=snapshot(preflight)
    write_json(place/'PROGRESS.json',{'status':phase,'role':name,'command':command,'updated_at':now()})
    with log.open('a') as stream:completed=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=False)
    data=log.read_bytes();after=snapshot(preflight);fresh={}
    if completed.returncode and after is not None and before!=after:fresh['preflight']={'file':str(preflight),'before':before,'after':after}
    record={'schema_version':'rarevlm.gt_anchored_kl.call.v1','phase':phase,'command':command,'returncode':completed.returncode,
        'completed_at':now(),'log_file':str(log),'log_file_sha256':sha_file(log),'log_start_byte':old,'log_end_byte':len(data),
        'log_segment_sha256':hashlib.sha256(data[old:]).hexdigest(),'fresh_failure_receipts':fresh}
    file=place/'attempts'/f'{phase}-{old}-{time.time_ns()}.call.json';write_json(file,record)
    evidence={str(log):sha_file(log),str(file):sha_file(file)};evidence.update({item['file']:item['after']['sha256'] for item in fresh.values()})
    reason=call_reason(record,evidence) if completed.returncode else None
    if reason:raise EngineeringRejected(reason,evidence)
    if completed.returncode:raise subprocess.CalledProcessError(completed.returncode,command)
    artifacts.update(evidence)
    if preflight is not None and preflight.exists():artifacts[str(preflight)]=sha_file(preflight)









def validate_registry(config):
    if (config.get('schema_version')!='rarevlm.gt_anchored_kl.train.v1'
        or config.get('candidates')!=[CANDIDATE] or config.get('exploratory_control')!=CONTROL
        or config.get('control_reference_only') is not False or config.get('control_can_be_selected') is not False
        or config.get('replay_objective')!='forward_kl_plus_GT_CE'):
        raise ValueError('one new full-KL candidate and immutable CE reference required')
    if config.get('pilot_seed')!=42 or config.get('all_seeds')!=[0,42,200] or config.get('replication_seeds')!=[0,200]:
        raise ValueError('registered pilot/replication seeds differ')
    policy={'min_free_gib':40,'max_utilization_percent_exclusive':10,'zero_compute_processes':True,
        'idle_observations':3,'observation_interval_seconds':15,'uuid_lock':True,'same_role_same_uuid':True,
        'terminate_existing_tasks':False,'baseline_generation_allowed':False,
        'only_embedded_new_objective_training_preflight':True,'later_stage_success_scope_required':True}
    if config.get('execution_policy')!=policy:raise ValueError('resource/progression policy differs')
    for name in ('model_root','initial_adapter','reference_config','asset_root','train_manifest','val_manifest',
        'val_class_balanced_manifest','external_manifest','external_class_balanced_manifest','study_protocol','cpu_contracts',
        'output_root','data_admission','evaluation_data_admission','reference_data_admission','reference_evaluation_data_admission',
        'parent_config_file','parent_output_root','teacher_reply_root'):path(config,name)


def validate_cpu_contract_receipt(cpu,sources,config_file):
    if (cpu.get('schema_version')!='rarevlm.gt_anchored_kl.cpu_contracts.v1' or cpu.get('status')!='passed'
        or cpu.get('config_file_sha256')!=sha_file(config_file)
        or cpu.get('source_sha256_before')!=sources or cpu.get('source_sha256_after')!=sources
        or cpu.get('parent_proof_sha256_before')!=cpu.get('parent_proof_sha256_after')
        or not cpu.get('run_input_files_sha256_before')
        or cpu.get('run_input_files_sha256_before')!=cpu.get('run_input_files_sha256_after')
        or cpu.get('returncode')!=0 or cpu.get('new_model_forwards')!=0 or cpu.get('gpu_operations')!=0):
        raise ValueError('complete new CPU/source/config receipt required')
    if not cpu.get('files_sha256'):raise ValueError('CPU log/XML/test byte bindings required')
    expected={str(PROJECT_ROOT/'tests'/name) for name in ('test_gt_anchored_kl_inputs.py','test_gt_anchored_kl_train.py','test_gt_anchored_kl_study.py')}
    config=read_json(config_file);expected.update((str(Path(config_file).resolve()),str(path(config,'study_protocol'))))
    if set(cpu['run_input_files_sha256_before'])!=expected:raise ValueError('all three tests/config/protocol must be bound before execution')
    verify_files(cpu['run_input_files_sha256_before'])
    verify_files(cpu['files_sha256'])


def check_cpu(config_file):
    config_file=Path(config_file).resolve();config=read_json(config_file);root=path(config,'output_root');root.mkdir(parents=True,exist_ok=True)
    receipt=path(config,'cpu_contracts');log=root/'cpu_contracts.log';xml=root/'cpu_contracts.xml'
    if (root/'STUDY_INPUT_LOCK.json').exists() or any(file.exists() for file in (receipt,log,xml)):
        raise ValueError('CPU/study evidence exists; preserve it rather than overwrite')
    sources=study_sources();proof=verify_parent_proof(config)
    tests=[PROJECT_ROOT/'tests'/name for name in ('test_gt_anchored_kl_inputs.py','test_gt_anchored_kl_train.py','test_gt_anchored_kl_study.py')]
    if any(not file.is_file() for file in tests):raise ValueError('all three new test modules required')
    run_inputs=[*tests,config_file,path(config,'study_protocol')]
    inputs_before={str(file):sha_file(file) for file in run_inputs}
    command=[sys.executable,'-m','pytest','-q',*[str(file) for file in tests],'--junitxml',str(xml)]
    begun=time.monotonic();env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        PYTHONPATH=str(SOURCE_ROOT),PYTHONDONTWRITEBYTECODE='1')
    with log.open('xb') as stream:result=subprocess.run(command,cwd=PROJECT_ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
    after=study_sources();parent_after=verify_parent_proof(config)
    inputs_after={str(file):sha_file(file) for file in run_inputs}
    value={'schema_version':'rarevlm.gt_anchored_kl.cpu_contracts.v1','status':'passed' if result.returncode==0 and sources==after and proof==parent_after and inputs_before==inputs_after else 'failed',
        'config_file_sha256':sha_file(config_file),'command':command,'returncode':result.returncode,'wall_seconds':time.monotonic()-begun,
        'source_sha256_before':sources,'source_sha256_after':after,'parent_proof_sha256_before':proof,'parent_proof_sha256_after':parent_after,
        'run_input_files_sha256_before':inputs_before,'run_input_files_sha256_after':inputs_after,
        'files_sha256':{str(file):sha_file(file) for file in [log,xml,*run_inputs] if file.exists()},
        'new_model_forwards':0,'gpu_operations':0,'optimizer_updates':0,'goal_complete':False}
    write_json(receipt,value)
    if value['status']!='passed':raise ValueError('new combined CPU contracts failed; original evidence retained')
    return value


def historical_registry(config,parent_proof):
    parent=path(config,'parent_output_root');analysis=read_json(parent/'analysis/pilot_seed42/screen/ANALYSIS.json')
    parent_study=read_json(parent/'STUDY_INPUT_LOCK.json');sources={};metadata={}
    for task in ('object','text','external','pope'):
        if task in ('object','text'):sources[f'initial_42_{task}']=analysis['source_data'][f'initial_42_{task}']
        else:
            file=parent/'runs/initial/seed42'/task/'HISTORICAL_REFERENCE.json'
            if parent_study['files_sha256'].get(str(file))!=sha_file(file):raise ValueError('parent historical baseline declaration changed')
            declaration=read_json(file)
            if declaration.get('condition')!='initial' or declaration.get('task')!=task or declaration.get('new_generation_questions')!=0:raise ValueError('only accepted initial baselines')
            sources[f'initial_42_{task}']=declaration['source_data'];metadata[str(file)]=sha_file(file)
    verify_source_data(sources)
    for source in sources.values():
        metadata[source['evaluation_file']]=source['evaluation_sha256'];metadata[source['input_lock_file']]=source['input_lock_sha256']
    return sources,metadata

def prepare(config_file):
    from .train import validate_config as validate_training
    from .contracts import historical_cell
    config_file=Path(config_file).resolve();config=read_json(config_file);root=path(config,'output_root');root.mkdir(parents=True,exist_ok=True)
    validate_registry(config);validate_training(config);validate_evaluation(config);validate_settings(config)
    create_inherited_admissions(config)
    data_binding=validate_inherited_admissions(config);external_binding=external_admission(config)
    sources=study_sources();cpu=read_json(path(config,'cpu_contracts'));validate_cpu_contract_receipt(cpu,sources,config_file)
    parent_proof=verify_parent_proof(config);historical,history_metadata=historical_registry(config,parent_proof)
    # Current model metadata/weights are direct dependencies, independently checked.
    reference=read_json(path(config,'reference_config'));model_root=path(config,'model_root')
    model=model_metadata_binding(model_root,reference)
    weights={str(model_root/name):digest for name,digest in model['model_weight_provenance_sha256'].items()}
    initial_files,initial_binding=adapter_binding(config,'initial',42,path(config,'initial_adapter'),model_root,config_file=config_file)
    weights.update({str(path(config,'initial_adapter')/name):digest for name,digest in initial_files.items()})
    metadata={**parent_proof,**history_metadata,**cpu['files_sha256']}
    names=('reference_config','cpu_contracts','study_protocol','dataset_protocol','data_admission','evaluation_data_admission',
        'reference_data_admission','reference_evaluation_data_admission','train_manifest','val_manifest','val_class_balanced_manifest',
        'replay_manifest','external_manifest','external_class_balanced_manifest','external_dataset_protocol','external_semantic_review',
        'external_frozen_audit','external_cpu_completion')
    metadata.update(file_map([config_file,*(path(config,name) for name in names),Path(initial_binding['path']),model_root.parent/'model_provenance.json']))
    metadata.update(file_map([file for file in model_root.iterdir() if file.is_file() and file.suffix not in ('.safetensors','.bin','.pt')]))
    verify_files(metadata);verify_files(weights);verify_files(sources)
    identity={'schema_version':'rarevlm.gt_anchored_kl.study_input.v1','config_path':str(config_file),
        'config_file_sha256':sha_file(config_file),'config_content_sha256':content_sha(config),'source_sha256':sources,
        'metadata_files_sha256':dict(sorted(metadata.items())),'weight_files_sha256':dict(sorted(weights.items())),
        'packages':package_versions(),'parent_proof_files_sha256':parent_proof,'historical_sources':historical,
        'accepted_parent_source_sha256':read_json(path(config,'parent_output_root')/'STUDY_INPUT_LOCK.json')['source_sha256'],
        'training_data_binding':data_binding,'external_data_admission':external_binding,
        'dataset_metadata':{task:load_data(config,task)[3] for task in ('object','text','external','pope')},
        'initial_adapter_files_sha256':initial_files,'initial_training_binding':initial_binding,
        'candidate':CANDIDATE,'control_reference_only':False,'teacher_probability_bank_read':False,
        'historical_recursive_source_or_weight_verification':False,
        'dynamic_teacher_and_training_inputs_locked_in_later_stage':True,'goal_complete':False}
    lock_file(root/'STUDY_INPUT_LOCK.json',identity);lock_file(root/'STUDY.json',{'schema_version':'rarevlm.gt_anchored_kl.study.v1',
        'study_input_lock_sha256':sha_file(root/'STUDY_INPUT_LOCK.json'),'candidate':CANDIDATE,
        'protocol_sha256':sha_file(path(config,'study_protocol')),'goal_complete':False})
    for task in ('object','text','external','pope'):historical_cell(root,'initial',task,config_file,config)
    return config,root,identity


def teacher_generation_accounting(root):
    return {'new_generation_questions_all_seeds':0,'scope':'original GT supervision only at every seed; independent reference forwards accounted separately','new_rollout_execution_records':[]}

def finish(root,status,phase,analyses,receipts,started):
    attempts=[];histories=sorted((root/'runs').glob('*/seed*/train/PREFLIGHT_HISTORY.jsonl'))
    for file in histories:
        for line in file.read_text().splitlines():attempts.append({'file':str(file),**json.loads(line)})
    train_results={str(file):read_json(file) for file in sorted((root/'runs').glob('*/seed*/train/TRAIN_RESULT.json'))}
    dynamic=[*root.glob('prepared/seed*/*.json'),*root.glob('prepared/seed*/*.jsonl'),
        *root.glob('teacher_replies/seed*/MANIFEST.json'),*root.glob('teacher_replies/seed*/TEACHER_REPLIES_LOCK.json')]
    result={'schema_version':'rarevlm.gt_anchored_kl.trial.v1','status':status,'trial_status':status,'completed_at':now(),
        'last_phase':phase,'candidate':CANDIDATE,'control_reference_only':False,'control_can_be_selected':False,
        'goal_complete':False,'root_independent_prediction_review_required':True,
        'confirmation_passed':status=='confirmation_passed_pending_independent_review','scientific_failure':status=='completed_negative',
        'study_input_lock_sha256':sha_file(root/'STUDY_INPUT_LOCK.json'),
        'analysis_files_sha256':{str(file):sha_file(file) for file in analyses.values()},
        'worker_receipts_sha256':{str(worker_dir(root,item['role'])/'COMPLETE.json'):sha_file(worker_dir(root,item['role'])/'COMPLETE.json') for item in receipts},
        'dynamic_input_files_sha256':file_map(dynamic),'current_controller_session_seconds':time.monotonic()-started,
        'existing_tasks_untouched':True,'equal_compute_claim':False,'teacher_probability_bank_read':False,
        'training_result_files_sha256':{file:sha_file(file) for file in train_results},
        'formal_training_updates':sum(item['completed_updates'] for item in train_results.values()),
        'formal_reference_forward_questions':sum(item['formal_reference_forward_questions'] for item in train_results.values()),
        'reference_forward_seconds':sum(item.get('reference_forward_seconds',0.) for item in train_results.values()),
        'preflight_accounting':{'planned_pilot_discarded_engineering_optimizer_updates':2,
            'actual_discarded_engineering_optimizer_updates_all_recorded_attempts':sum(item.get('discarded_engineering_optimizer_updates',0) for item in attempts),
            'recorded_attempts':attempts,'source_files_sha256':file_map(histories)},
        'teacher_generation_accounting':teacher_generation_accounting(root)}
    write_json(root/'COMPLETION.json',result);write_json(root/'STATUS.json',result)
    if result['confirmation_passed']:write_json(root/'TRIAL_EVIDENCE.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    action=parser.add_mutually_exclusive_group(required=True)
    for mode in ('check-cpu','prepare-only','run','worker'):action.add_argument('--'+mode,action='store_true')
    parser.add_argument('--output-root',type=Path);parser.add_argument('--study-lock',type=Path)
    parser.add_argument('--operation',choices=OPERATIONS);parser.add_argument('--condition',choices=TRAINED_CONDITIONS)
    parser.add_argument('--seed',type=int,choices=(0,42,200));parser.add_argument('--shard',type=int,choices=(0,1))
    parser.add_argument('--screen-analysis',type=Path);parser.add_argument('--external-analysis',type=Path)
    args=parser.parse_args()
    if not args.config.is_absolute():raise ValueError('absolute config path required')
    if args.check_cpu:result=check_cpu(args.config)
    elif args.worker:
        if any(getattr(args,name) is None for name in ('output_root','study_lock','operation','condition','seed')):parser.error('worker requires full registered identity')
        result=worker_run(args)
    else:result=run_study(args.config,args.prepare_only)
    print(json.dumps({'status':result.get('status'),'goal_complete':False},ensure_ascii=False),flush=True)


if __name__=='__main__':main()






