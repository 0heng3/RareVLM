"""Fixed coordinate shrink through pilot, conditional new producers, external and POPE."""
PARENT_UPSTREAM_COPY_PROVENANCE = {'source': 'src/vlm_object_replay_coverage/study.py', 'sha256': 'ed25fb5339ac5315a934e082f57a7c65e5bb439c90c1463d1bbee59bcc44c301', 'scope': 'independent adapted copy; frozen original globals and bytes are unchanged'}
COPY_PROVENANCE = {'source': '/opt/data/private/longtail_vlm/rarevlm_pilot/packages/RareVLM_minimal_20261001/src/vlm_object_behavior_replay/study.py', 'sha256': 'f5a5960f45096c0bd1618dc7e1a259a1e10063fbca860d4e5700bbee0e575c83', 'scope': 'new full-stage queue copy; only embedded new-objective preflight, historical reference control, direct locks'}
COPY_PROVENANCE = {'source': '/opt/data/private/longtail_vlm/rarevlm_pilot/packages/RareVLM_minimal_20261001/src/vlm_generated_prefix_kl/study.py', 'sha256': 'b05271d338ea2f4c4ca2afcc2b57fd537e76e3fc670360592ae8e332e7d1c311', 'scope': 'independent coordinate-shrink copy; original modules/globals are untouched'}
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
from .lock import verify_parent_proof,verify_study,study_sources,verify_producer_reference,prepare_historical_producer_reference
from vlm_object_study.run import validate_quality_admission
from .data import load_rows,prepare_generation_inputs,load_prompt_inputs,finalize_teacher_replies
from vlm_probe.evaluate import content_sha,lock_file,now,package_versions,read_json,sha_file,write_json
from .contracts import (CANDIDATE,CONTROL,TRAINED_CONDITIONS,adapter_binding,path,source_hashes,read_cell,producer_reference_for_seed,
    external_admission,load_data,verify_source_data,validate_condition,validate_inherited_admissions,create_inherited_admissions)
from .analysis import analyze,validate_settings
from .evaluate import validate_config as validate_evaluation
from . import CONFIG_SCHEMA,SCHEMA
from vlm_joint.evaluate import model_metadata_binding

OPERATIONS=('teacher','producer','eval','external','pope')


def verify_files(files):
    for name, digest in files.items():
        if not Path(name).is_file() or sha_file(Path(name)) != digest:
            raise ValueError(f"locked behavior study file changed: {name}")

def worker_dir(root, name):
    if not re.fullmatch(r"[a-z0-9_]+", name):
        raise ValueError("unsafe role")
    return root / "workers" / name

def eval_arguments(config_file, root, condition, seed, task, adapter, admission, prior=None):
    args = [sys.executable, "-m", "vlm_lora_coordinate_shrink.evaluate", "--config", str(config_file),
        "--condition", condition, "--seed", str(seed), "--task", task, "--adapter-dir", str(adapter),
        "--output-root", str(root), "--admission", str(admission)]
    if condition == CANDIDATE and task in ("external", "pope"):
        if prior is None:
            raise ValueError("candidate confirmation needs passing prior phase")
        args.extend(["--screen-analysis" if task == "external" else "--external-analysis", str(prior)])
    return args

def read_receipt(file, identity_sha):
    value = read_json(file)
    if (value.get("schema_version") != "rarevlm.lora_coordinate_shrink.worker.v1" or value.get("status") != "completed"
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
    role_name=role('teacher','initial',seed,shard) if task=='teacher_rollout' else role('eval' if task in ('object','text') else task,condition,seed)
    file=directory/'admissions'/('teacher_shard'+str(shard)+'.json' if task=='teacher_rollout' else task+'.json')
    value={'schema_version':'rarevlm.lora_coordinate_shrink.gpu_admission.v1','status':'admitted',
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




def local_training_artifacts(directory):
    # Behavior replay has no probability bank. Full new checkpoint masters remain resume artifacts.
    return {str(file):sha_file(file) for file in directory.rglob('*') if file.is_file()
        and file.suffix in ('.json','.jsonl','.pt','.safetensors')}











def job(config_file,root,operation,condition,seed,prior=None,*,shard=None):
    name=role(operation,condition,seed,shard)
    args=[sys.executable,'-m','vlm_lora_coordinate_shrink.study','--worker','--config',str(config_file),
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














class EngineeringRejected(RuntimeError):
    def __init__(self,reason,evidence):super().__init__(reason);self.reason=reason;self.evidence=evidence


def engineering_reason(log,preflight=None):
    found=re.search(r'(?:reference|formal_training|training|preflight|teacher_generation|teacher_rollout|rollout)_memory_cap_exceeded',log)
    if found:return found.group(0)
    return original_engineering_reason(log,preflight)


def call_reason(record,evidence):
    if record.get('schema_version')!='rarevlm.lora_coordinate_shrink.call.v1' or not record.get('returncode'):
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
    record={'schema_version':'rarevlm.lora_coordinate_shrink.call.v1','phase':phase,'command':command,'returncode':completed.returncode,
        'completed_at':now(),'log_file':str(log),'log_file_sha256':sha_file(log),'log_start_byte':old,'log_end_byte':len(data),
        'log_segment_sha256':hashlib.sha256(data[old:]).hexdigest(),'fresh_failure_receipts':fresh}
    file=place/'attempts'/f'{phase}-{old}-{time.time_ns()}.call.json';write_json(file,record)
    evidence={str(log):sha_file(log),str(file):sha_file(file)};evidence.update({item['file']:item['after']['sha256'] for item in fresh.values()})
    reason=call_reason(record,evidence) if completed.returncode else None
    if reason:raise EngineeringRejected(reason,evidence)
    if completed.returncode:raise subprocess.CalledProcessError(completed.returncode,command)
    artifacts.update(evidence)
    if preflight is not None and preflight.exists():artifacts[str(preflight)]=sha_file(preflight)


def teacher_arguments(config_file,root,seed,shard,admission):
    return [sys.executable,'-m','vlm_lora_coordinate_shrink.rollout','--config',str(config_file),
        '--seed',str(seed),'--shard',str(shard),'--output-root',str(root),'--admission',str(admission)]





def validate_teacher_completion(value,*,seed,shard,gpu_uuid,config_file_sha256,study_input_lock_sha256,initial_master_sha256):
    expected={'schema_version':'rarevlm.lora_coordinate_shrink.teacher_shard.complete.v1','status':'completed',
        'seed':seed,'shard':shard,'role':f'teacher_seed{seed}_shard{shard}','producer_uuid':gpu_uuid,
        'config_file_sha256':config_file_sha256,'study_input_lock_sha256':study_input_lock_sha256,
        'formal_saved_generation_questions':2896,'formal_batches_completed':724,'preflight_saved_generation_questions':8,
        'teacher_initial_master_sha256_before':initial_master_sha256,'teacher_initial_master_sha256_after':initial_master_sha256,
        'optimizer_updates':0,'new_probability_cache':False}
    for key,item in expected.items():
        if value.get(key)!=item:raise ValueError(f'teacher complete receipt differs: {key}')
    for before,after in (('frozen_base_sha256_before','frozen_base_sha256_after'),
        ('teacher_native_lora_sha256_before','teacher_native_lora_sha256_after'),
        ('all_frozen_parameters_sha256_before','all_frozen_parameters_sha256_after')):
        if not value.get(before) or value[before]!=value.get(after):raise ValueError(f'teacher complete native/frozen parameters differ: {before}')
    inference={'numerical_path':'original_bf16','autocast':False,'native_lora_dtype':'torch.bfloat16',
        'adapter_autocast_dtype':False,'explicit_lora_bf16_conversion':True,'optimizer_updates':0}
    if any(value.get('teacher_inference',{}).get(key)!=item for key,item in inference.items()):
        raise ValueError('teacher complete must retain baseline native BF16 including LoRA, no autocast')
    return value

















def teacher_generation_accounting(root):
    records=[]
    for file in sorted((root/'teacher_replies').glob('seed*/shard*/EXECUTION.jsonl')):
        for line in file.read_text().splitlines():records.append({'file':str(file),**json.loads(line)})
    return {'seed42_new_generation_questions':0,'seed42_parent_reference_questions':5792,
        'scope':'seed42 accepted reference counted as historical; fresh0/200 actual journals only; no probability bank',
        'new_rollout_execution_records':records,'journal_files_sha256':file_map(sorted((root/'teacher_replies').glob('seed*/shard*/EXECUTION.jsonl')))}


def role(operation,condition,seed,shard=None):
    if operation=='teacher':
        if condition!='initial' or seed not in (0,200) or shard not in (0,1):raise ValueError('only fresh conditional replication teacher roles are allowed')
        return f'teacher_seed{seed}_shard{shard}'
    validate_condition(condition,seed)
    if operation not in OPERATIONS or shard is not None or (operation=='producer' and seed==42):
        raise ValueError('seed42 is historical producer/export only; no raw training role')
    return f'{operation}_{condition}_seed{seed}'


def worker_prior(config_file,root,operation,seed,prior=None):
    if operation in ('teacher','producer','eval') and seed in (0,200):
        from .analysis import passing_pilot
        file=root/'analysis/pilot_seed42/screen/ANALYSIS.json';passing_pilot(file,config_file)
        return {'passing_new_half_pilot_file':str(file),'passing_new_half_pilot_sha256':sha_file(file)}
    if operation in ('external','pope'):
        if prior is None:raise ValueError('full passing three-seed prior phase required')
        from .contracts import phase_admission
        phase_admission(prior,config_file,'screen' if operation=='external' else 'external')
    return {}


def train_arguments(config_file,root,condition,seed):
    validate_condition(condition,seed)
    if seed==42:raise ValueError('historical seed42 producer cannot be retrained')
    args=[sys.executable,'-m','vlm_lora_coordinate_shrink.train','--config',str(config_file),
        '--condition',condition,'--seed',str(seed),'--output-root',str(root)]
    if (root/'producer_runs'/condition/f'seed{seed}/train/LATEST.json').exists():args.append('--resume')
    return args


def ensure_export(config_file,config,seed):
    from .export import export_adapter,verify_export
    from .contracts import export_directory
    reference=producer_reference_for_seed(config,seed);output=export_directory(config,seed)
    export_adapter(config_file,seed,Path(reference['producer_result_file']),Path(reference['producer_adapter_dir']),output,producer_reference=reference)
    verified=verify_export(config_file,seed,output,producer_reference=reference)
    if verified.get('status')!='verified_cpu_export':raise ValueError('fixed shrink CPU export verification failed')
    return verified


def worker_run(args):
    from .inputs import verify_prepared_training_inputs,verify_producer_training_result
    config_file=args.config.resolve();config=read_json(config_file);root=path(config,'output_root')
    if args.output_root.resolve()!=root:raise ValueError('new worker output root differs')
    name=role(args.operation,args.condition,args.seed,args.shard);place=worker_dir(root,name)
    verify_study(args.study_lock,config_file,check_weights=True)
    prior=worker_prior(config_file,root,args.operation,args.seed,args.screen_analysis or args.external_analysis)
    hardware=visible_gpu_binding()
    identity={'schema_version':'rarevlm.lora_coordinate_shrink.worker_input.v1','role':name,'operation':args.operation,
        'condition':args.condition,'seed':args.seed,'shard':args.shard,'hardware':hardware,
        'config_sha256':sha_file(config_file),'study_input_lock_sha256':sha_file(args.study_lock),**prior}
    prompt=None
    if args.operation=='teacher':
        prompt,_,_=load_prompt_inputs(config,args.seed)
        identity['prompt_manifest_file_sha256']=sha_file(root/'prepared'/f'seed{args.seed}'/'PROMPT_MANIFEST.json')
    elif args.operation=='producer':
        prepared=verify_prepared_training_inputs(config,args.seed,args.condition)
        identity['prepared_training_inputs_file_sha256']=prepared['manifest_file_sha256']
        identity['artifact_role']='unshrunk_producer'
    else:
        _,binding=adapter_binding(config,args.condition,args.seed,root/'runs'/CANDIDATE/f'seed{args.seed}/shrink/adapter',path(config,'model_root'),config_file=config_file)
        identity.update(adapter_export_sha256=binding['sha256'],producer_result_file_sha256=binding['producer_result_file_sha256'])
    lock_file(place/'INPUT_LOCK.json',identity)
    if (place/'COMPLETE.json').exists():return read_receipt(place/'COMPLETE.json',sha_file(args.study_lock))
    artifacts={str(place/'INPUT_LOCK.json'):sha_file(place/'INPUT_LOCK.json')};outcome='passed';failure=None;evidence={}
    def evaluate(task,prior_file=None):
        adapter=root/'runs'/CANDIDATE/f'seed{args.seed}/shrink/adapter'
        admission=task_admission(place,config_file,CANDIDATE,args.seed,task,hardware);artifacts[str(admission)]=sha_file(admission)
        subprocess_call(place,name,artifacts,task,eval_arguments(config_file,root,CANDIDATE,args.seed,task,adapter,admission,prior_file),
            root/'preflight'/CANDIDATE/f'seed{args.seed}'/task/'PREFLIGHT.json')
        cell=read_cell(root,CANDIDATE,args.seed,task,config_file,config)
        if not cell['complete']:raise ValueError('full exported-candidate evaluation did not complete')
        for kind in ('predictions','evaluation','input_lock'):
            artifacts[cell['provenance'][kind+'_file']]=cell['provenance'][kind+'_sha256']
    try:
        if args.operation=='teacher':
            admission=task_admission(place,config_file,'initial',args.seed,'teacher_rollout',hardware,shard=args.shard,prompt=prompt)
            artifacts[str(admission)]=sha_file(admission);shard_dir=path(config,'teacher_reply_root')/f'seed{args.seed}'/f'shard{args.shard}'
            subprocess_call(place,name,artifacts,'teacher_rollout',teacher_arguments(config_file,root,args.seed,args.shard,admission),shard_dir/'PREFLIGHT.json')
            complete=read_json(shard_dir/'COMPLETE.json')
            validate_teacher_completion(complete,seed=args.seed,shard=args.shard,gpu_uuid=hardware['uuid'],config_file_sha256=sha_file(config_file),
                study_input_lock_sha256=sha_file(args.study_lock),initial_master_sha256=config['teacher_generation']['initial_runtime_master_sha256'])
            verify_files(complete['artifact_files_sha256']);artifacts.update(complete['artifact_files_sha256'])
            artifacts[str(shard_dir/'COMPLETE.json')]=sha_file(shard_dir/'COMPLETE.json')
        elif args.operation=='producer':
            directory=root/'producer_runs'/CANDIDATE/f'seed{args.seed}'/'train'
            if not (directory/'TRAIN_RESULT.json').exists():
                subprocess_call(place,name,artifacts,'producer_training',train_arguments(config_file,root,CANDIDATE,args.seed),directory/'PREFLIGHT.json')
            verify_producer_training_result(config,directory,condition=CANDIDATE,seed=args.seed)
            artifacts.update(local_training_artifacts(directory))
        elif args.operation=='eval':evaluate('object');evaluate('text')
        elif args.operation=='external':evaluate('external',args.screen_analysis)
        elif args.operation=='pope':evaluate('pope',args.external_analysis)
    except EngineeringRejected as error:outcome='engineering_rejected';failure=error.reason;evidence=error.evidence
    except BaseException as error:
        write_json(place/'PROGRESS.json',{'status':'execution_or_contract_failure','error':str(error),'scientific_failure':False,'updated_at':now()});raise
    receipt={'schema_version':'rarevlm.lora_coordinate_shrink.worker.v1','status':'completed','outcome':outcome,'role':name,
        'operation':args.operation,'condition':args.condition,'seed':args.seed,'shard':args.shard,'gpu_uuid':hardware['uuid'],
        'completed_at':now(),'study_input_lock_sha256':sha_file(args.study_lock),'artifact_files_sha256':artifacts,
        'engineering_reason':failure,'engineering_evidence_files_sha256':evidence,'existing_tasks_untouched':True,'automatic_runtime_fallback':False}
    lock_file(place/'COMPLETE.json',receipt);write_json(place/'PROGRESS.json',{'status':'completed','outcome':outcome,'updated_at':now()})
    return receipt


def validate_registry(config):
    if (config.get('schema_version')!=CONFIG_SCHEMA or config.get('producer_train_schema_version')!=SCHEMA
        or config.get('candidates')!=[CANDIDATE] or config.get('exploratory_control')!=CONTROL
        or config.get('control_can_be_selected') is not False or config.get('control_reference_only') is not True
        or config.get('seed42_new_training_allowed') is not False):
        raise ValueError('one coordinate-shrunk candidate, historical raw producer and schema separation required')
    if config.get('pilot_seed')!=42 or config.get('all_seeds')!=[0,42,200] or config.get('replication_seeds')!=[0,200]:raise ValueError('registered seed axes differ')
    if config.get('coordinate_shrink',{}).get('alpha')!=.5:raise ValueError('fixed0.5 only; no alpha search')
    expected={'min_free_gib':40,'max_utilization_percent_exclusive':10,'zero_compute_processes':True,'idle_observations':3,
        'observation_interval_seconds':15,'uuid_lock':True,'same_role_same_uuid':True,'terminate_existing_tasks':False,
        'baseline_generation_allowed':False,'only_embedded_new_objective_training_preflight':True,'later_stage_success_scope_required':True}
    if config.get('execution_policy')!=expected:raise ValueError('registered GPU and progression policy differs')
    for key in ('output_root','initial_adapter','model_root','study_protocol','cpu_contracts','parent_output_root','parent_config_file',
        'producer_reference_root','producer_reference_config','data_admission','evaluation_data_admission'):path(config,key)


def historical_registry(config,parent_proof):
    producer_root=path(config,'producer_reference_root');producer_study=read_json(producer_root/'STUDY_INPUT_LOCK.json')
    producer_analysis=read_json(producer_root/'analysis/pilot_seed42/screen/ANALYSIS.json')
    sources={f'initial_42_{task}':producer_study['historical_sources'][f'initial_42_{task}'] for task in ('object','text','external','pope')}
    for task in ('object','text'):sources[f'{CONTROL}_42_{task}']=producer_analysis['source_data'][f'{CONTROL}_42_{task}']
    verify_source_data(sources);metadata={}
    for source in sources.values():
        metadata[source['evaluation_file']]=source['evaluation_sha256'];metadata[source['input_lock_file']]=source['input_lock_sha256']
    return sources,metadata


def prepare(config_file):
    from .train import validate_config as validate_training
    from .contracts import historical_cell
    config_file=Path(config_file).resolve();config=read_json(config_file);root=path(config,'output_root');root.mkdir(parents=True,exist_ok=True)
    validate_registry(config);validate_training(config);validate_evaluation(config);validate_settings(config)
    create_inherited_admissions(config);data_binding=validate_inherited_admissions(config);external_binding=external_admission(config)
    sources=study_sources();cpu=read_json(path(config,'cpu_contracts'));validate_cpu_contract_receipt(cpu,sources,config_file)
    parent_proof=verify_parent_proof(config);reference=prepare_historical_producer_reference(config)
    historical,history_metadata=historical_registry(config,parent_proof)
    model_root=path(config,'model_root');model=model_metadata_binding(model_root,read_json(path(config,'reference_config')))
    weights={str(model_root/name):digest for name,digest in model['model_weight_provenance_sha256'].items()}
    initial_files,initial_binding=adapter_binding(config,'initial',42,path(config,'initial_adapter'),model_root,config_file=config_file)
    weights.update({str(path(config,'initial_adapter')/name):digest for name,digest in initial_files.items()})
    weights.update({str(Path(reference['producer_adapter_dir'])/name):digest for name,digest in reference['producer_adapter_files_sha256'].items()})
    href=root/'producer_runs'/CANDIDATE/'seed42/HISTORICAL_PRODUCER_REFERENCE.json'
    metadata={**parent_proof,**config['producer_reference_files_sha256'],**history_metadata,**cpu['files_sha256'],str(href):sha_file(href)}
    names=('reference_config','cpu_contracts','study_protocol','dataset_protocol','data_admission','evaluation_data_admission',
        'reference_data_admission','reference_evaluation_data_admission','train_manifest','val_manifest','val_class_balanced_manifest','replay_manifest',
        'external_manifest','external_class_balanced_manifest','external_dataset_protocol','external_semantic_review','external_frozen_audit','external_cpu_completion')
    metadata.update(file_map([config_file,*(path(config,key) for key in names),Path(initial_binding['path']),model_root.parent/'model_provenance.json']))
    metadata.update(file_map([file for file in model_root.iterdir() if file.is_file() and file.suffix not in ('.safetensors','.bin','.pt')]))
    # The big R teacher payload is already bound by accepted N.parentProof;
    # normal metadata validation must not reread it or any old probability bank.
    large_parent=Path(config['seed42_input_reference']['teacher_reply_manifest_file']);metadata.pop(str(large_parent),None)
    verify_files(metadata);verify_files(weights);verify_files(sources)
    producer_study=read_json(path(config,'producer_reference_root')/'STUDY_INPUT_LOCK.json')
    identity={'schema_version':'rarevlm.lora_coordinate_shrink.study_input.v1','config_path':str(config_file),
        'config_file_sha256':sha_file(config_file),'config_content_sha256':content_sha(config),'source_sha256':sources,
        'metadata_files_sha256':dict(sorted(metadata.items())),'weight_files_sha256':dict(sorted(weights.items())),'packages':package_versions(),
        'parent_proof_files_sha256':parent_proof,'producer_reference_files_sha256':config['producer_reference_files_sha256'],
        'historical_producer_reference':reference,'accepted_parent_source_sha256':read_json(path(config,'parent_output_root')/'STUDY_INPUT_LOCK.json')['source_sha256'],
        'accepted_producer_source_sha256':producer_study['source_sha256'],'historical_sources':historical,
        'training_data_binding':data_binding,'external_data_admission':external_binding,'dataset_metadata':{task:load_data(config,task)[3] for task in ('object','text','external','pope')},
        'candidate':CANDIDATE,'control_reference_only':True,'seed42_new_training_allowed':False,'teacher_probability_bank_read':False,
        'historical_recursive_source_or_weight_verification':False,'goal_complete':False}
    lock_file(root/'STUDY_INPUT_LOCK.json',identity);lock_file(root/'STUDY.json',{'schema_version':'rarevlm.lora_coordinate_shrink.study.v1',
        'study_input_lock_sha256':sha_file(root/'STUDY_INPUT_LOCK.json'),'candidate':CANDIDATE,'protocol_sha256':sha_file(path(config,'study_protocol')),'goal_complete':False})
    for task in ('object','text','external','pope'):historical_cell(root,'initial',task,config_file,config)
    for task in ('object','text'):historical_cell(root,CONTROL,task,config_file,config)
    export=ensure_export(config_file,config,42)
    lock_file(root/'PREPARATION.json',{'schema_version':'rarevlm.lora_coordinate_shrink.preparation.v1','status':'passed',
        'study_input_lock_sha256':sha_file(root/'STUDY_INPUT_LOCK.json'),'historical_producer_reference_sha256':sha_file(href),
        'seed42_export_manifest_file':export['export_manifest_file'],'seed42_export_manifest_file_sha256':export['export_manifest_file_sha256'],
        'seed42_adapter_files_sha256':export['adapter_files_sha256'],'new_training_updates':0,'new_teacher_generation_questions':0,'gpu_operations':0})
    return config,root,identity


def finish(root,status,phase,analyses,receipts,started):
    results={str(file):read_json(file) for file in sorted((root/'producer_runs'/CANDIDATE).glob('seed*/train/TRAIN_RESULT.json'))}
    if any(item['seed']==42 for item in results.values()):raise ValueError('seed42 cannot acquire a new producer TRAIN_RESULT')
    histories=sorted((root/'producer_runs'/CANDIDATE).glob('seed*/train/PREFLIGHT_HISTORY.jsonl'));attempts=[]
    for file in histories:
        for line in file.read_text().splitlines():attempts.append({'file':str(file),**json.loads(line)})
    manifests=sorted((root/'runs'/CANDIDATE).glob('seed*/shrink/EXPORT_MANIFEST.json'))
    dynamic=[*root.glob('prepared/seed*/*.json'),*root.glob('prepared/seed*/*.jsonl'),
        *root.glob('teacher_replies/seed*/MANIFEST.json'),*root.glob('teacher_replies/seed*/TEACHER_REPLIES_LOCK.json')]
    value={'schema_version':'rarevlm.lora_coordinate_shrink.trial.v1','status':status,'trial_status':status,'last_phase':phase,'completed_at':now(),
        'candidate':CANDIDATE,'control_can_be_selected':False,'control_reference_only':True,'confirmation_passed':status=='confirmation_passed_pending_independent_review',
        'goal_complete':False,'root_independent_prediction_review_required':True,'scientific_failure':status=='completed_negative',
        'study_input_lock_sha256':sha_file(root/'STUDY_INPUT_LOCK.json'),'analysis_files_sha256':{str(file):sha_file(file) for file in analyses.values()},
        'worker_receipts_sha256':{str(worker_dir(root,item['role'])/'COMPLETE.json'):sha_file(worker_dir(root,item['role'])/'COMPLETE.json') for item in receipts},
        'dynamic_input_files_sha256':file_map(dynamic),'export_manifest_files_sha256':file_map(manifests),
        'historical_seed42_producer_training_updates':181,'new_seed42_training_updates':0,'new_producer_training_updates':sum(item['completed_updates'] for item in results.values()),
        'new_producer_reference_forward_questions':sum(item['formal_reference_forward_questions'] for item in results.values()),
        'discarded_new_producer_engineering_updates':sum(item.get('discarded_engineering_optimizer_updates',0) for item in attempts),
        'new_producer_training_result_files_sha256':{file:sha_file(file) for file in results},'preflight_history_files_sha256':file_map(histories),
        'teacher_generation_accounting':teacher_generation_accounting(root),'current_controller_session_seconds':time.monotonic()-started,
        'CPU_export_is_not_training':True,'existing_tasks_untouched':True}
    write_json(root/'COMPLETION.json',value);write_json(root/'STATUS.json',value)
    if value['confirmation_passed']:write_json(root/'TRIAL_EVIDENCE.json',value)
    return value


def run_study(config_file,prepare_only=False):
    config_file=Path(config_file).resolve();root=path(read_json(config_file),'output_root');root.mkdir(parents=True,exist_ok=True)
    owner=(root/'STUDY.lock').open('a+');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.monotonic();analyses={};receipts=[];phase='static_preparation'
    def queue(name,jobs):
        write_json(root/'STATUS.json',{'status':name,'updated_at':now(),'controller_pid':os.getpid(),'goal_complete':False})
        current=queue_stage(root,name,jobs,identity);receipts.extend(current);return all(item['outcome']=='passed' for item in current)
    def close(status,stage):
        verify_study(root/'STUDY_INPUT_LOCK.json',config_file);return finish(root,status,stage,analyses,receipts,started)
    try:
        config,root,identity=prepare(config_file)
        if prepare_only:return {'status':'prepared','new_seed42_training_updates':0,'GPU_operations':0,'goal_complete':False}
        if (root/'COMPLETION.json').exists():
            existing=read_json(root/'COMPLETION.json')
            if existing['study_input_lock_sha256']!=sha_file(root/'STUDY_INPUT_LOCK.json'):raise ValueError('completed study source lock differs')
            for key in ('analysis_files_sha256','worker_receipts_sha256','dynamic_input_files_sha256','export_manifest_files_sha256'):verify_files(existing[key])
            for file in existing['worker_receipts_sha256']:read_receipt(Path(file),sha_file(root/'STUDY_INPUT_LOCK.json'))
            return existing
        phase='pilot42'
        if not queue(phase,[job(config_file,root,'eval',CANDIDATE,42)]):return close('engineering_rejected',phase)
        pilot,file=analyze(config_file,root,[42],'screen');analyses['pilot']=file
        if pilot['status']!='complete':raise ValueError('full shrink candidate plus historical references required')
        if not pilot['gate']['passed']:return close('completed_negative','pilot_internal_screen')
        for seed in (0,200):prepare_generation_inputs(config,seed)
        phase='teacher_generation_seeds0_200'
        if not queue(phase,[job(config_file,root,'teacher','initial',seed,shard=shard) for seed in (0,200) for shard in (0,1)]):return close('engineering_rejected',phase)
        from .inputs import prepare_training_inputs,verify_prepared_training_inputs
        for seed in (0,200):
            finalize_teacher_replies(config,seed);prepare_training_inputs(config,seed,CANDIDATE);verify_prepared_training_inputs(config,seed,CANDIDATE)
        phase='producer_replication'
        if not queue(phase,[job(config_file,root,'producer',CANDIDATE,seed) for seed in (0,200)]):return close('engineering_rejected',phase)
        for seed in (0,200):ensure_export(config_file,config,seed)
        phase='replication_evaluation'
        if not queue(phase,[job(config_file,root,'eval',CANDIDATE,seed) for seed in (0,200)]):return close('engineering_rejected',phase)
        screen,file=analyze(config_file,root,[0,42,200],'screen',pilot_file=analyses['pilot']);analyses['internal']=file
        if screen['status']!='complete':raise ValueError('complete three-seed shrink screen required')
        if not screen['gate']['passed']:return close('completed_negative','three_seed_internal_screen')
        phase='external_confirmation'
        if not queue(phase,[job(config_file,root,'external',CANDIDATE,seed,file) for seed in (0,42,200)]):return close('engineering_rejected',phase)
        external,file=analyze(config_file,root,[0,42,200],'external',screen_file=analyses['internal']);analyses['external']=file
        if external['status']!='complete':raise ValueError('complete full4372 external for each seed required')
        if not external['gate']['passed']:return close('completed_negative',phase)
        phase='pope_confirmation'
        if not queue(phase,[job(config_file,root,'pope',CANDIDATE,seed,file) for seed in (0,42,200)]):return close('engineering_rejected',phase)
        confirmed,file=analyze(config_file,root,[0,42,200],'confirm',external_file=analyses['external']);analyses['pope']=file
        if confirmed['status']!='complete':raise ValueError('complete full9000 POPE for each seed required')
        return close('confirmation_passed_pending_independent_review' if confirmed['gate']['passed'] else 'completed_negative',phase)
    except BaseException as error:
        write_json(root/'STATUS.json',{'status':'execution_or_contract_failure','phase':phase,'error':str(error),
            'updated_at':now(),'scientific_failure':False,'goal_complete':False});raise
    finally:owner.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    action=parser.add_mutually_exclusive_group(required=True)
    for mode in ('check-cpu','prepare-only','run','worker'):action.add_argument('--'+mode,action='store_true')
    parser.add_argument('--output-root',type=Path);parser.add_argument('--study-lock',type=Path);parser.add_argument('--operation',choices=OPERATIONS)
    parser.add_argument('--condition',choices=('initial',CANDIDATE));parser.add_argument('--seed',type=int,choices=(0,42,200));parser.add_argument('--shard',type=int,choices=(0,1))
    parser.add_argument('--screen-analysis',type=Path);parser.add_argument('--external-analysis',type=Path);args=parser.parse_args()
    if not args.config.is_absolute():raise ValueError('absolute config path required')
    if args.check_cpu:result=check_cpu(args.config)
    elif args.worker:
        if any(getattr(args,key) is None for key in ('output_root','study_lock','operation','condition','seed')):parser.error('full worker identity required')
        result=worker_run(args)
    else:result=run_study(args.config,args.prepare_only)
    print(json.dumps({'status':result.get('status'),'goal_complete':False},ensure_ascii=False),flush=True)


if __name__=='__main__':main()

















def validate_cpu_contract_receipt(cpu,sources,config_file):
    if (cpu.get('schema_version')!='rarevlm.lora_coordinate_shrink.cpu_contracts.v1' or cpu.get('status')!='passed'
        or cpu.get('config_file_sha256')!=sha_file(config_file)
        or cpu.get('source_sha256_before')!=sources or cpu.get('source_sha256_after')!=sources
        or cpu.get('parent_proof_sha256_before')!=cpu.get('parent_proof_sha256_after')
        or not cpu.get('producer_reference_before')
        or cpu.get('producer_reference_before')!=cpu.get('producer_reference_after')
        or not cpu.get('run_input_files_sha256_before')
        or cpu.get('run_input_files_sha256_before')!=cpu.get('run_input_files_sha256_after')
        or cpu.get('returncode')!=0 or cpu.get('new_model_forwards')!=0 or cpu.get('gpu_operations')!=0):
        raise ValueError('complete new CPU/source/config receipt required')
    if not cpu.get('files_sha256'):raise ValueError('CPU log/XML/test byte bindings required')
    expected={str(PROJECT_ROOT/'tests'/name) for name in ('test_lora_coordinate_shrink_inputs.py','test_lora_coordinate_shrink_train.py','test_lora_coordinate_shrink_export.py','test_lora_coordinate_shrink_study.py')}
    config=read_json(config_file);expected.update((str(Path(config_file).resolve()),str(path(config,'study_protocol'))))
    if cpu['producer_reference_before']!=verify_producer_reference(config,check_weights=False):
        raise ValueError('CPU receipt accepted producer reference differs')
    if set(cpu['run_input_files_sha256_before'])!=expected:raise ValueError('all four tests/config/protocol must be bound before execution')
    verify_files(cpu['run_input_files_sha256_before'])
    verify_files(cpu['files_sha256'])

def check_cpu(config_file):
    config_file=Path(config_file).resolve();config=read_json(config_file);root=path(config,'output_root');root.mkdir(parents=True,exist_ok=True)
    receipt=path(config,'cpu_contracts');log=root/'cpu_contracts.log';xml=root/'cpu_contracts.xml'
    if (root/'STUDY_INPUT_LOCK.json').exists() or any(file.exists() for file in (receipt,log,xml)):
        raise ValueError('CPU/study evidence exists; preserve it rather than overwrite')
    sources=study_sources();proof=verify_parent_proof(config);producer=verify_producer_reference(config,check_weights=False)
    tests=[PROJECT_ROOT/'tests'/name for name in ('test_lora_coordinate_shrink_inputs.py','test_lora_coordinate_shrink_train.py','test_lora_coordinate_shrink_export.py','test_lora_coordinate_shrink_study.py')]
    if any(not file.is_file() for file in tests):raise ValueError('all four new test modules required')
    run_inputs=[*tests,config_file,path(config,'study_protocol')]
    inputs_before={str(file):sha_file(file) for file in run_inputs}
    command=[sys.executable,'-m','pytest','-q',*[str(file) for file in tests],'--junitxml',str(xml)]
    begun=time.monotonic();env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        PYTHONPATH=str(SOURCE_ROOT),PYTHONDONTWRITEBYTECODE='1')
    with log.open('xb') as stream:result=subprocess.run(command,cwd=PROJECT_ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
    after=study_sources();parent_after=verify_parent_proof(config);producer_after=verify_producer_reference(config,check_weights=False)
    inputs_after={str(file):sha_file(file) for file in run_inputs}
    value={'schema_version':'rarevlm.lora_coordinate_shrink.cpu_contracts.v1','status':'passed' if result.returncode==0 and sources==after and proof==parent_after and producer==producer_after and inputs_before==inputs_after else 'failed',
        'config_file_sha256':sha_file(config_file),'command':command,'returncode':result.returncode,'wall_seconds':time.monotonic()-begun,
        'source_sha256_before':sources,'source_sha256_after':after,'parent_proof_sha256_before':proof,'parent_proof_sha256_after':parent_after,'producer_reference_before':producer,'producer_reference_after':producer_after,
        'run_input_files_sha256_before':inputs_before,'run_input_files_sha256_after':inputs_after,
        'files_sha256':{str(file):sha_file(file) for file in [log,xml,*run_inputs] if file.exists()},
        'new_model_forwards':0,'gpu_operations':0,'optimizer_updates':0,'goal_complete':False}
    write_json(receipt,value)
    if value['status']!='passed':raise ValueError('new combined CPU contracts failed; original evidence retained')
    return value

def teacher_generation_accounting(root):
    records=[]
    for file in sorted((root/'teacher_replies').glob('seed*/shard*/EXECUTION.jsonl')):
        for line in file.read_text().splitlines():records.append({'file':str(file),**json.loads(line)})
    return {'seed42_new_generation_questions':0,'seed42_parent_reference_questions':5792,
        'scope':'seed42 accepted reference counted as historical; fresh0/200 actual journals only; no probability bank',
        'new_rollout_execution_records':records,'journal_files_sha256':file_map(sorted((root/'teacher_replies').glob('seed*/shard*/EXECUTION.jsonl')))}
