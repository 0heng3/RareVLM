"""Independent behavior replay queue; teacher targets precede both CE pilot arms."""
COPY_PROVENANCE = {'source': 'src/vlm_object_replay_coverage/study.py', 'sha256': 'ed25fb5339ac5315a934e082f57a7c65e5bb439c90c1463d1bbee59bcc44c301', 'scope': 'independent adapted copy; frozen original globals and bytes are unchanged'}
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
from vlm_object_kl_eval.read import verify_historical_study
from vlm_object_study.run import validate_quality_admission
from .data import load_rows,prepare_generation_inputs,load_prompt_inputs,finalize_teacher_replies
from vlm_probe.evaluate import content_sha,lock_file,now,package_versions,read_json,sha_file,write_json
from .contracts import (CANDIDATE,CONTROL,TRAINED_CONDITIONS,adapter_binding,path,source_hashes,read_cell,
    external_admission,load_data,verify_source_data,validate_condition,validate_inherited_admissions,create_inherited_admissions)
from .analysis import analyze,validate_settings
from .evaluate import validate_config as validate_evaluation

OPERATIONS=('teacher','preflight','train_eval','external','pope')


def verify_files(files):
    for name, digest in files.items():
        if not Path(name).is_file() or sha_file(Path(name)) != digest:
            raise ValueError(f"locked behavior study file changed: {name}")

def worker_dir(root, name):
    if not re.fullmatch(r"[a-z0-9_]+", name):
        raise ValueError("unsafe role")
    return root / "workers" / name

def eval_arguments(config_file, root, condition, seed, task, adapter, admission, prior=None):
    args = [sys.executable, "-m", "vlm_object_behavior_replay.evaluate", "--config", str(config_file),
        "--condition", condition, "--seed", str(seed), "--task", task, "--adapter-dir", str(adapter),
        "--output-root", str(root), "--admission", str(admission)]
    if condition == CANDIDATE and task in ("external", "pope"):
        if prior is None:
            raise ValueError("candidate confirmation needs passing prior phase")
        args.extend(["--screen-analysis" if task == "external" else "--external-analysis", str(prior)])
    return args

def read_receipt(file, identity_sha):
    value = read_json(file)
    if (value.get("schema_version") != "rarevlm.object_behavior_replay.worker.v1" or value.get("status") != "completed"
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
    value={'schema_version':'rarevlm.object_behavior_replay.gpu_admission.v1','status':'admitted',
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
    verify_study(args.study_lock,config_file)
    name=role(args.operation,args.condition,args.seed,args.shard);place=worker_dir(root,name);hardware=visible_gpu_binding()
    identity={'schema_version':'rarevlm.object_behavior_replay.worker_input.v1','role':name,
        'operation':args.operation,'condition':args.condition,'seed':args.seed,'shard':args.shard,
        'study_input_lock_sha256':sha_file(args.study_lock),'config_sha256':sha_file(config_file),'hardware':hardware,
        'prior_phase_sha256':sha_file(args.screen_analysis or args.external_analysis) if args.screen_analysis or args.external_analysis else None}
    if args.operation=='teacher':
        prompt,plan,_=load_prompt_inputs(config,args.seed)
        identity['prompt_manifest_file_sha256']=sha_file(root/'prepared'/f'seed{args.seed}'/'PROMPT_MANIFEST.json')
    elif args.operation in ('preflight','train_eval'):
        from .inputs import verify_prepared_training_inputs
        prepared=verify_prepared_training_inputs(config,args.seed,args.condition)
        identity['prepared_training_inputs_file_sha256']=prepared['manifest_file_sha256']
        identity['teacher_replies_manifest_file_sha256']=sha_file(path(config,'teacher_reply_root')/f'seed{args.seed}'/'MANIFEST.json')
        identity['teacher_replies_lock_file_sha256']=sha_file(path(config,'teacher_reply_root')/f'seed{args.seed}'/'TEACHER_REPLIES_LOCK.json')
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
        if args.operation=='teacher':
            admission=task_admission(place,config_file,'initial',args.seed,'teacher_rollout',hardware,shard=args.shard,prompt=prompt)
            artifacts[str(admission)]=sha_file(admission)
            shard_dir=path(config,'teacher_reply_root')/f'seed{args.seed}'/f'shard{args.shard}'
            subprocess_call(place,name,artifacts,'teacher_rollout',teacher_arguments(config_file,root,args.seed,args.shard,admission),shard_dir/'PREFLIGHT.json')
            complete=read_json(shard_dir/'COMPLETE.json')
            validate_teacher_completion(complete,seed=args.seed,shard=args.shard,gpu_uuid=hardware['uuid'],
                config_file_sha256=sha_file(config_file),study_input_lock_sha256=sha_file(args.study_lock),
                initial_master_sha256=config['teacher_generation']['initial_runtime_master_sha256'])
            verify_files(complete['artifact_files_sha256']);artifacts.update(complete['artifact_files_sha256'])
            artifacts[str(shard_dir/'COMPLETE.json')]=sha_file(shard_dir/'COMPLETE.json')
        elif args.operation in ('preflight','train_eval'):
            training_root=root/'engineering_runs' if args.operation=='preflight' else root
            train_dir=training_root/'runs'/args.condition/f'seed{args.seed}/train'
            if args.operation=='preflight' or not (train_dir/'TRAIN_RESULT.json').exists():
                subprocess_call(place,name,artifacts,'engineering_preflight' if args.operation=='preflight' else 'training',
                    train_arguments(config_file,training_root,args.condition,args.seed,args.operation=='preflight'),train_dir/'PREFLIGHT.json')
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
    receipt={'schema_version':'rarevlm.object_behavior_replay.worker.v1','status':'completed','outcome':outcome,
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
    if operation=='teacher':
        if condition!='initial' or seed not in (0,42,200) or shard not in (0,1):raise ValueError('invalid teacher shard role')
        return f'teacher_seed{seed}_shard{shard}'
    validate_condition(condition,seed)
    if (operation not in OPERATIONS or shard is not None or (operation=='preflight' and seed!=42)
        or (condition==CONTROL and operation in ('external','pope'))):raise ValueError('GT CE is pilot-only; teacher CE is sole candidate')
    return f'{operation}_{condition}_seed{seed}'


def train_arguments(config_file,root,condition,seed,preflight=False):
    validate_condition(condition,seed)
    values=[sys.executable,'-m','vlm_object_behavior_replay.train','--config',str(config_file),
        '--condition',condition,'--seed',str(seed),'--output-root',str(root)]
    if preflight:values.append('--preflight-only')
    elif (root/'runs'/condition/f'seed{seed}/train/LATEST.json').exists():values.append('--resume')
    return values


def job(config_file,root,operation,condition,seed,prior=None,*,shard=None):
    name=role(operation,condition,seed,shard)
    args=[sys.executable,'-m','vlm_object_behavior_replay.study','--worker','--config',str(config_file),
        '--output-root',str(root),'--study-lock',str(root/'STUDY_INPUT_LOCK.json'),
        '--operation',operation,'--condition',condition,'--seed',str(seed)]
    if shard is not None:args.extend(['--shard',str(shard)])
    if prior is not None:args.extend(['--screen-analysis' if operation=='external' else '--external-analysis',str(prior)])
    return {'key':name,'command':args,'completion_marker':str(worker_dir(root,name)/'COMPLETE.json')}


def study_sources():
    return {str(PROJECT_ROOT/name if name.startswith('tools/') else SOURCE_ROOT/name):digest
            for name,digest in source_hashes().items()}


def verify_study(file,config_file):
    lock=read_json(file)
    if (lock.get('schema_version')!='rarevlm.object_behavior_replay.study_input.v1'
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
    if (config.get('schema_version')!='rarevlm.object_behavior_replay.train.v1'
        or config.get('candidates')!=[CANDIDATE] or config.get('exploratory_control')!=CONTROL
        or config.get('control_can_be_selected') is not False or config.get('pilot_seed')!=42
        or config.get('replication_seeds')!=[0,200] or config.get('all_seeds')!=[0,42,200]):
        raise ValueError('only teacher CE may expand; GT CE stays exploratory seed42')
    for name in ('historical_coverage_study_config','historical_coverage_study_root','external_initial_reference',
                 'teacher_reply_root','study_protocol','cpu_contracts','output_root','reference_data_admission',
                 'reference_evaluation_data_admission'):
        path(config,name)
    if path(config,'output_root')==path(config,'historical_coverage_study_root'):
        raise ValueError('behavior outputs must be independent')
    if path(config,'teacher_reply_root')!=path(config,'output_root')/'teacher_replies':
        raise ValueError('teacher replies must belong to the new output root')
    expected={'shards':2,'batch_size':4,'max_new_tokens':32,'do_sample':False,'use_cache':True,'repetition_penalty':1.05,
        'preflight_questions':8,'plan_questions_per_seed':5792,'qa_per_shard':2896,'microbatches_per_shard':724,
        'partition':'contiguous_complete_microbatches','actual_generated_ids_required':True,
        'retokenize_teacher_answers':False,'eos_token_ids':[151645,151643],'pad_token_id':151643,
        'chat_template_eos_token_id':151645,'post_chat_eos_template_ids':[198],
        'non_chat_eos_append_template':False,'capped_append_eos_or_template':False,
        'keep_empty_content_when_real_eos':True,'filter_or_resample_teacher_answers':False,
        'initial_runtime_master_sha256':'6a44359406c53261b254c3a19393c63187b34af09baec539d6b235ca148c0fa7',
        'teacher_inference_parameter_precision':'bf16_all_floating_including_lora','teacher_inference_autocast':False,
        'initial_master_sha256_scope':'canonical_FP32_representation_of_BF16_initial_LoRA; actual_actor_master_FP32; teacher_native_inference_BF16'}
    if config.get('teacher_generation')!=expected:raise ValueError('fixed teacher generation/termination contract differs')
    if config.get('preflight_memory_cap_gib')!=65:raise ValueError('65GiB hard cap without fallback required')
    return config


def prepare(config_file):
    from .train import validate_config as validate_training
    from vlm_joint.evaluate import model_metadata_binding
    config_file=Path(config_file).resolve();config=read_json(config_file)
    validate_registry(config);validate_training(config);validate_evaluation(config);validate_settings(config)
    create_inherited_admissions(config)
    data_binding=validate_inherited_admissions(config);external_binding=external_admission(config)
    cpu=read_json(path(config,'cpu_contracts'))
    sources=study_sources();validate_cpu_contract_receipt(cpu,sources)
    verify_files(cpu['files_sha256']);verify_files(cpu['scientific_source_sha256'])
    objects,replays=load_rows(config)
    old_root,old_config_file=path(config,'historical_coverage_study_root'),path(config,'historical_coverage_study_config')
    old_config=read_json(old_config_file);old=read_json(old_root/'STUDY_INPUT_LOCK.json');completed=read_json(old_root/'COMPLETION.json')
    if (old.get('schema_version')!='rarevlm.object_replay_coverage.study_input.v1'
        or old.get('config_file_sha256')!=sha_file(old_config_file)
        or completed.get('status')!='completed_negative' or completed.get('goal_complete') is not False
        or completed.get('study_input_lock_sha256')!=sha_file(old_root/'STUDY_INPUT_LOCK.json')):
        raise ValueError('original coverage must remain completed negative')
    verify_files(old['source_sha256']);verify_files(completed['analysis_files_sha256'])
    for name in ('asset_root','object_data_root','train_manifest','val_manifest','val_class_balanced_manifest',
        'replay_manifest','pope_data_root','reference_config','model_root','initial_adapter','dataset_protocol',
        'external_data_root','external_manifest','external_class_balanced_manifest','external_dataset_protocol','external_initial_reference'):
        if path(config,name)!=path(old_config,name):raise ValueError(f'fixed source data/model path differs: {name}')
    root=path(config,'output_root');files=file_map([old_config_file,old_root/'STUDY_INPUT_LOCK.json',old_root/'STUDY.json',old_root/'COMPLETION.json'])
    files.update(completed['analysis_files_sha256'])
    old_analysis_file=old_root/'analysis/pilot_seed42/screen/ANALYSIS.json'
    old_analysis=read_json(old_analysis_file);old_train=old_root/'runs/wide32/seed42/train'
    if old_analysis['source_data']['wide32_42_object']['training_result_sha256']!=sha_file(old_train/'TRAIN_RESULT.json'):
        raise ValueError('old fixed stream anchor no longer matches original complete analysis')
    old_train_result=read_json(old_train/'TRAIN_RESULT.json')
    if old_train_result['trained_input_contracts_file_sha256']!=sha_file(old_train/'TRAINED_INPUT_CONTRACTS.json'):
        raise ValueError('old standard actual-input stream bytes changed')
    files.update(file_map([old_train/'TRAIN_RESULT.json',old_train/'TRAINED_INPUT_CONTRACTS.json']))
    # Do not inherit old flat files_sha256: it contains full-vocabulary probability .pt banks.
    model_root=path(config,'model_root');model=model_metadata_binding(model_root,read_json(path(config,'reference_config')))
    files.update(file_map([model_root/name for name in model['model_weight_provenance_sha256']]))
    files.update(file_map([file for file in model_root.iterdir() if file.is_file()]))
    files.update(file_map([model_root.parent/'model_provenance.json',*path(config,'initial_adapter').glob('*')]))
    initial_files,initial_binding=adapter_binding(config,'initial',42,path(config,'initial_adapter'),model_root,config_file=config_file)
    files.update(file_map([initial_binding['path']]))
    for task in ('object','text','external','pope'):
        cell=read_cell(root,'initial',42,task,config_file,config)
        if not cell['complete']:raise ValueError('all four complete original baselines required')
        for kind in ('predictions','evaluation','input_lock'):
            files[cell['provenance'][kind+'_file']]=cell['provenance'][kind+'_sha256']
        files.update(file_map([root/'runs/initial/seed42'/task/'HISTORICAL_REFERENCE.json']))
    files.update(cpu['files_sha256']);files.update(cpu['scientific_source_sha256'])
    new_files=[config_file,*[path(config,name) for name in ('study_protocol','dataset_protocol','external_dataset_protocol',
        'reference_config','cpu_contracts','data_admission','evaluation_data_admission','reference_data_admission',
        'reference_evaluation_data_admission','train_manifest','val_manifest','val_class_balanced_manifest','replay_manifest',
        'external_manifest','external_class_balanced_manifest','external_semantic_review','external_frozen_audit','external_cpu_completion')]]
    new_files.extend(Path(value) for value in config.get('extra_lock_files',[]));files.update(file_map(new_files))
    for task in ('object','text','external','pope'):
        metadata=load_data(config,task)[3]
        if 'data_manifest_path' in metadata:files.update(file_map([metadata['data_manifest_path']]))
    if any('/teacher_cache/' in name and name.endswith('.pt') for name in files):
        raise ValueError('new behavior study must not inherit old probability-bank tensors')
    verify_files(files);verify_files(sources)
    identity={'schema_version':'rarevlm.object_behavior_replay.study_input.v1','config_path':str(config_file),
        'config_file_sha256':sha_file(config_file),'config_content_sha256':content_sha(config),'source_sha256':sources,
        'files_sha256':dict(sorted(files.items())),'packages':package_versions(),'training_data_binding':data_binding,
        'external_data_admission':external_binding,'dataset_metadata':{task:load_data(config,task)[3] for task in ('object','text','external','pope')},
        'train_counts':{'object_questions':len(objects),'replay_pool_questions':len(replays),'updates':(len(objects)+15)//16,
                        'replay_exposures':((len(objects)+15)//16)*32},'candidate':CANDIDATE,'exploratory_control':CONTROL,
        'teacher_probability_bank_read':False,'dynamic_teacher_and_training_inputs_locked_in_later_stage':True}
    lock_file(root/'STUDY_INPUT_LOCK.json',identity)
    lock_file(root/'STUDY.json',{'schema_version':'rarevlm.object_behavior_replay.study.v1','study_input_lock_sha256':sha_file(root/'STUDY_INPUT_LOCK.json'),
        'candidate':CANDIDATE,'control':CONTROL,'control_only_pilot_seed42':True,'control_can_be_selected':False,
        'stages':['prompt_cpu_per_seed','two_teacher_shards','teacher_merge_and_target_cpu_lock','both_engineering_preflight',
            'both_full_pilot_internal','teacher_ce32_pilot_screen','teacher_ce32_seed0_200_generation_and_training',
            'teacher_ce32_three_seed_internal','teacher_ce32_full_external','teacher_ce32_full_POPE'],
        'family':'one candidate four endpoints alpha=.05/4; exploratory control not eligible',
        'gpu_policy':'shared queue three15sec idle >=40GiB util<10% zero compute UUID lock same-role resume',
        'memory_cap_gib':65,'automatic_runtime_fallback':False,'teacher_references_are_factual_ground_truth':False,
        'equal_compute_claim':False,'goal_complete':False})
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
    histories=list((root/'engineering_runs/runs').glob('*/seed*/train/PREFLIGHT_HISTORY.jsonl'))+list((root/'runs').glob('*/seed*/train/PREFLIGHT_HISTORY.jsonl'))
    attempts=[]
    for file in sorted(histories):
        for line in file.read_text().splitlines():attempts.append({'file':str(file),**json.loads(line)})
    teacher_journals=sorted((root/'teacher_replies').glob('seed*/shard*/EXECUTION.jsonl'))
    teacher_execution=[]
    for file in teacher_journals:
        for line in file.read_text().splitlines():teacher_execution.append({'file':str(file),**json.loads(line)})
    dynamic=[*root.glob('prepared/seed*/*.json'),*root.glob('prepared/seed*/*.jsonl'),
        *root.glob('teacher_replies/seed*/MANIFEST.json'),*root.glob('teacher_replies/seed*/TEACHER_REPLIES_LOCK.json')]
    result={'schema_version':'rarevlm.object_behavior_replay.trial.v1','status':status,'trial_status':status,
        'completed_at':now(),'last_phase':phase,'candidate':CANDIDATE,'control_can_be_selected':False,
        'goal_complete':False,'root_independent_prediction_review_required':True,
        'confirmation_passed':status=='confirmation_passed_pending_independent_review','scientific_failure':status=='completed_negative',
        'study_input_lock_sha256':sha_file(root/'STUDY_INPUT_LOCK.json'),'analysis_files_sha256':{str(file):sha_file(file) for file in analyses.values()},
        'worker_receipts_sha256':{str(worker_dir(root,item['role'])/'COMPLETE.json'):sha_file(worker_dir(root,item['role'])/'COMPLETE.json') for item in receipts},
        'dynamic_input_files_sha256':file_map(dynamic),'current_controller_session_seconds':time.monotonic()-started,'existing_tasks_untouched':True,
        'equal_compute_claim':False,'teacher_probability_bank_read':False,
        'preflight_accounting':{'planned_pilot_discarded_engineering_optimizer_updates':4,
            'actual_discarded_engineering_optimizer_updates_all_recorded_attempts':sum(item.get('discarded_engineering_optimizer_updates',0) for item in attempts),
            'recorded_attempts':attempts,'source_files_sha256':file_map(histories)},
        'teacher_generation_accounting':teacher_generation_accounting(root)}
    write_json(root/'COMPLETION.json',result);write_json(root/'STATUS.json',result)
    if result['confirmation_passed']:write_json(root/'TRIAL_EVIDENCE.json',result)
    return result


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
    def prepare_prompts(seed):
        verify_study(root/'STUDY_INPUT_LOCK.json',config_file)
        write_json(root/'STATUS.json',{'status':'prompt_cpu_preparation','seed':seed,'updated_at':now(),'goal_complete':False})
        return prepare_generation_inputs(config,seed)
    def targets(seed,conditions):
        from .inputs import prepare_training_inputs,verify_prepared_training_inputs
        finalize_teacher_replies(config,seed)
        for condition in conditions:
            prepare_training_inputs(config,seed,condition);verify_prepared_training_inputs(config,seed,condition)
    try:
        config,root,identity=prepare(config_file)
        if prepare_only:return {'status':'prepared','goal_complete':False,'full_prompt_preparation_run':False,'GPU_operations':0}
        if (root/'COMPLETION.json').exists():
            value=read_json(root/'COMPLETION.json')
            if value['study_input_lock_sha256']!=sha_file(root/'STUDY_INPUT_LOCK.json'):raise ValueError('completed behavior lock changed')
            verify_files(value['analysis_files_sha256']);verify_files(value['worker_receipts_sha256']);verify_files(value['dynamic_input_files_sha256'])
            for file in value['worker_receipts_sha256']:read_receipt(Path(file),sha_file(root/'STUDY_INPUT_LOCK.json'))
            return value
        prepare_prompts(42);phase='teacher_generation_seed42'
        if not queue(phase,[job(config_file,root,'teacher','initial',42,shard=shard) for shard in (0,1)]):return close('engineering_rejected',phase)
        targets(42,TRAINED_CONDITIONS);phase='engineering_preflight'
        if not queue(phase,[job(config_file,root,'preflight',condition,42) for condition in TRAINED_CONDITIONS]):return close('engineering_rejected',phase)
        phase='pilot42'
        if not queue(phase,[job(config_file,root,'train_eval',condition,42) for condition in TRAINED_CONDITIONS]):return close('engineering_rejected',phase)
        pilot,file=analyze(config_file,root,[42],'screen');analyses['pilot']=file
        if pilot['status']!='complete':raise ValueError('both full pilot arms must complete before screen')
        if not pilot['gate']['passed']:return close('completed_negative','pilot_internal_screen')
        for seed in (0,200):prepare_prompts(seed)
        phase='teacher_generation_seeds0_200'
        if not queue(phase,[job(config_file,root,'teacher','initial',seed,shard=shard) for seed in (0,200) for shard in (0,1)]):return close('engineering_rejected',phase)
        for seed in (0,200):targets(seed,(CANDIDATE,))
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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-only',action='store_true');mode.add_argument('--run',action='store_true')
    mode.add_argument('--worker',action='store_true')
    parser.add_argument('--output-root',type=Path);parser.add_argument('--study-lock',type=Path)
    parser.add_argument('--operation',choices=OPERATIONS);parser.add_argument('--condition',choices=('initial',*TRAINED_CONDITIONS))
    parser.add_argument('--shard',type=int,choices=(0,1))
    parser.add_argument('--seed',type=int,choices=(0,42,200));parser.add_argument('--screen-analysis',type=Path)
    parser.add_argument('--external-analysis',type=Path)
    args=parser.parse_args()
    if args.worker:
        if any(getattr(args,name) is None for name in ('output_root','study_lock','operation','condition','seed')):
            parser.error('worker requires output-root/study-lock/operation/condition/seed')
        result=worker_run(args)
    else:result=run_study(args.config,args.prepare_only)
    print(result['status'],flush=True)





class EngineeringRejected(RuntimeError):
    def __init__(self,reason,evidence):super().__init__(reason);self.reason=reason;self.evidence=evidence


def engineering_reason(log,preflight=None):
    found=re.search(r'(?:formal_training|preflight|teacher_generation|teacher_rollout|rollout)_memory_cap_exceeded',log)
    if found:return found.group(0)
    return original_engineering_reason(log,preflight)


def call_reason(record,evidence):
    if record.get('schema_version')!='rarevlm.object_behavior_replay.call.v1' or not record.get('returncode'):
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
    record={'schema_version':'rarevlm.object_behavior_replay.call.v1','phase':phase,'command':command,'returncode':completed.returncode,
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
    return [sys.executable,'-m','vlm_object_behavior_replay.rollout','--config',str(config_file),
        '--seed',str(seed),'--shard',str(shard),'--output-root',str(root),'--admission',str(admission)]


def teacher_generation_accounting(root):
    from .rollout import execution_accounting
    journals=sorted((root/'teacher_replies').glob('seed*/shard*/EXECUTION.jsonl'))
    per_shard={str(file.parent):execution_accounting(file.parent) for file in journals}
    keys=('actual_generation_started_QA_all_sessions','actual_generation_completed_QA_all_sessions',
        'actual_formal_generation_questions','actual_preflight_generation_questions','actual_total_generation_questions',
        'generation_call_starts_all_sessions','generation_call_completions_all_sessions','unfinished_generation_call_attempts',
        'actual_model_load_starts_all_sessions','actual_model_load_completions_all_sessions',
        'formal_generation_seconds_all_completed_calls','preflight_generation_seconds_all_completed_calls')
    return {**{key:sum(value[key] for value in per_shard.values()) for key in keys},
        'per_shard_actual_accounting':per_shard,'journal_files_sha256':file_map(journals),
        'planned_per_seed_formal_questions':5792,'planned_per_seed_engineering_questions':16,
        'scope':'actual per-call journals include all sessions/retries; started calls are separate from completed QA; plans are not actual calls; no teacher probabilities'}


def validate_teacher_completion(value,*,seed,shard,gpu_uuid,config_file_sha256,study_input_lock_sha256,initial_master_sha256):
    expected={'schema_version':'rarevlm.object_behavior_replay.teacher_shard.complete.v1','status':'completed',
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


def validate_cpu_contract_receipt(cpu,sources):
    counts=cpu.get('counts',{})
    if (cpu.get('schema_version')!='rarevlm.object_behavior_replay.cpu_contracts.v1' or cpu.get('status')!='passed'
        or cpu.get('source_unchanged_during_tests') is not True or cpu.get('new_GPU_operations')!=0
        or cpu.get('new_pretrained_model_forwards')!=0 or cpu.get('return_code')!=0
        or type(counts.get('tests')) is not int or counts['tests']<=0
        or any(counts.get(key)!=0 for key in ('failures','errors','skipped'))):
        raise ValueError('complete successful new CPU contracts required before registration/GPU')
    if any(cpu.get(key)!=sources for key in ('scientific_source_sha256','scientific_source_sha256_before','scientific_source_sha256_after')):
        raise ValueError('CPU receipt must bind exact full scientific source before and after tests')
    return cpu


if __name__=='__main__':main()
