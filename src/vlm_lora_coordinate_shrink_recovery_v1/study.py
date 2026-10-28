"""Versioned full-stage recovery; old Object and failed execution evidence stay immutable."""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_key]='2'
import argparse
import fcntl
import gc
import json
from pathlib import Path
import subprocess
import sys
import time
from rarevlm_paths import PROJECT_ROOT,SOURCE_ROOT
from research_runtime.queue import run_queue
from vlm_joint.evaluate import visible_gpu_binding,model_metadata_binding
from vlm_lora_coordinate_shrink import CANDIDATE,CONTROL
from vlm_lora_coordinate_shrink import study as original
from vlm_lora_coordinate_shrink import evaluate as original_evaluation
from vlm_lora_coordinate_shrink.contracts import (path,directory,adapter_binding,producer_reference_for_seed,
    validate_condition,validate_inherited_admissions,external_admission,source_hashes,FIXED,load_data)
from vlm_lora_coordinate_shrink.data import prepare_generation_inputs,load_prompt_inputs,finalize_teacher_replies
from vlm_lora_coordinate_shrink.lock import verify_study
from vlm_probe.evaluate import (read_json,sha_file,write_json,lock_file,now,content_sha,package_versions,
    processor_for,verify_dev_images)
from .contracts import (read_cell,recovery_file,verify_context,recovery_fields,verify_recovery_fields,phase_admission)
from .analysis import analyze,passing_pilot
from .compat import binding_identity_equal
from .lock import prepare_lock,verify_recovery,sources

OPERATIONS=original.OPERATIONS
EngineeringRejected=original.EngineeringRejected
subprocess_call=original.subprocess_call
train_arguments=original.train_arguments
teacher_arguments=original.teacher_arguments
validate_teacher_completion=original.validate_teacher_completion
local_training_artifacts=original.local_training_artifacts
verify_files=original.verify_files
file_map=original.file_map


def recovery_root(config_file):
    return path(read_json(config_file),'output_root')/'recovery_v1'


def role(operation,condition,seed,shard=None):
    return 'recovery_v1_'+original.role(operation,condition,seed,shard)


def worker_dir(root,name):
    return original.worker_dir(root/'recovery_v1',name)


def worker_prior(config_file,root,operation,seed,prior=None):
    verify_context(config_file)
    if operation in ('teacher','producer','eval') and seed in (0,200):
        file=root/'analysis/pilot_seed42/screen/ANALYSIS.json';passing_pilot(file,config_file)
        return {'passing_recovery_half_pilot_file':str(file),'passing_recovery_half_pilot_sha256':sha_file(file)}
    if operation in ('external','pope'):
        if prior is None:raise ValueError('full passing three-seed recovery prior phase required')
        phase_admission(prior,config_file,'screen' if operation=='external' else 'external')
    return {}


def task_admission(place,config_file,condition,seed,task,hardware,*,shard=None,prompt=None):
    from vlm_lora_coordinate_shrink.contracts import validate_gpu_admission
    root=path(read_json(config_file),'output_root');context=verify_context(config_file)
    operation='teacher' if task=='teacher_rollout' else 'eval' if task in ('object','text') else task
    numeric=original.role(operation,condition,seed,shard)
    if seed==42 and operation=='eval' and hardware['uuid']!=context['original_eval_gpu_uuid']:
        raise ValueError('recovery pilot must preserve original numeric role GPU UUID')
    value={'schema_version':'rarevlm.lora_coordinate_shrink.gpu_admission.v1','status':'admitted',
        'condition':condition,'seed':seed,'task':task,'role':numeric,'gpu_uuid':hardware['uuid'],
        'config_sha256':sha_file(config_file),'config_file_sha256':sha_file(config_file),
        'study_input_lock_path':str(root/'STUDY_INPUT_LOCK.json'),'study_input_lock_sha256':sha_file(root/'STUDY_INPUT_LOCK.json'),
        'recovery_worker_role':role(operation,condition,seed,shard),**recovery_fields(config_file)}
    if task=='teacher_rollout':
        if prompt is None:raise ValueError('teacher admission needs complete CPU prompt inputs')
        value.update(shard=shard,plan_file_sha256=prompt['plan_file_sha256'],prompt_inputs_file_sha256=prompt['prompt_inputs_file_sha256'])
    file=place/'admissions'/('teacher_shard'+str(shard)+'.json' if task=='teacher_rollout' else task+'.json')
    lock_file(file,value)
    validate_gpu_admission(config_file,file,condition=condition,seed=seed,task=task,gpu_uuid=hardware['uuid'],
        shard=shard,plan_file_sha256=value.get('plan_file_sha256'),prompt_inputs_file_sha256=value.get('prompt_inputs_file_sha256'))
    verify_recovery_fields(read_json(file),config_file)
    return file


def eval_arguments(config_file,root,condition,seed,task,adapter,admission,prior=None):
    if seed==42 and task=='object':raise ValueError('completed original Object must never be generated again')
    args=[sys.executable,'-m','vlm_lora_coordinate_shrink_recovery_v1.study','--evaluate',
        '--config',str(config_file),'--output-root',str(root),'--recovery-lock',str(recovery_file(config_file)),
        '--condition',condition,'--seed',str(seed),'--task',task,'--adapter-dir',str(adapter),'--admission',str(admission)]
    if task in ('external','pope'):
        if prior is None:raise ValueError('complete passing prior phase required')
        args.extend(['--screen-analysis' if task=='external' else '--external-analysis',str(prior)])
    return args


def job(config_file,root,operation,condition,seed,prior=None,*,shard=None):
    name=role(operation,condition,seed,shard)
    command=[sys.executable,'-m','vlm_lora_coordinate_shrink_recovery_v1.study','--worker','--config',str(config_file),
        '--output-root',str(root),'--study-lock',str(root/'STUDY_INPUT_LOCK.json'),
        '--recovery-lock',str(recovery_file(config_file)),'--operation',operation,'--condition',condition,'--seed',str(seed)]
    if shard is not None:command.extend(['--shard',str(shard)])
    if prior is not None:command.extend(['--screen-analysis' if operation=='external' else '--external-analysis',str(prior)])
    return {'key':name,'command':command,'completion_marker':str(worker_dir(root,name)/'COMPLETE.json')}


def read_receipt(file,config_file):
    value=read_json(file);verify_recovery_fields(value,config_file)
    if value.get('schema_version')!='rarevlm.lora_coordinate_shrink.recovery_worker.v1' or value.get('status')!='completed':
        raise ValueError('complete recovery worker identity missing')
    verify_files(value['artifact_files_sha256'])
    if value['outcome']=='engineering_rejected':
        evidence=value['engineering_evidence_files_sha256'];verify_files(evidence)
        attempts=[read_json(name) for name in evidence if name.endswith('.call.json')]
        if len(attempts)!=1 or original.call_reason(attempts[0],evidence)!=value['engineering_reason']:
            raise ValueError('engineering rejection needs matching fresh invocation evidence')
    elif value['outcome']!='passed':raise ValueError('unknown recovery worker outcome')
    if value['operation']=='eval' and value['seed']==42 and value['gpu_uuid']!=verify_context(config_file)['original_eval_gpu_uuid']:
        raise ValueError('recovery pilot completion GPU UUID changed')
    return value


def queue_stage(root,name,jobs,identity,config_file):
    verify_context(config_file);queue=root/'recovery_v1/queues'/name
    for item in jobs:
        if Path(item['completion_marker']).exists():read_receipt(Path(item['completion_marker']),config_file)
    status_file=queue/'STATUS.json'
    if name=='pilot42':
        expected=identity['original_eval_gpu_uuid']
        if status_file.exists():
            state=read_json(status_file)
            if set(state.get('jobs',{}))!={item['key'] for item in jobs} or any(s.get('gpu_uuid')!=expected for s in state['jobs'].values()):
                raise ValueError('pinned original pilot numeric GPU UUID changed')
        else:
            write_json(status_file,{'status':'waiting_resources','jobs':{item['key']:{'state':'waiting_gpu','gpu_uuid':expected,'contention_observations':0} for item in jobs}})
    run_queue(queue,jobs,identity)
    return [read_receipt(Path(item['completion_marker']),config_file) for item in jobs]


def ensure_export(config_file,config,seed):
    verify_context(config_file)
    if seed==42:raise ValueError('historical accepted export must never be recreated')
    return original.ensure_export(config_file,config,seed)


def prepare_producer_inputs(config_file,config,seed):
    from vlm_lora_coordinate_shrink.inputs import prepare_training_inputs,verify_prepared_training_inputs
    verify_context(config_file)
    finalize_teacher_replies(config,seed)
    prepare_training_inputs(config,seed,CANDIDATE)
    return verify_prepared_training_inputs(config,seed,CANDIDATE)


def resume_input_lock(file,fresh):
    """Retain saved raw timing and exact content seal after stable identity validation."""
    if not file.exists():return fresh
    saved=read_json(file)
    if not binding_identity_equal(saved['training_binding'],fresh['training_binding']):
        raise ValueError('completed candidate export/producer identity changed')
    value=dict(fresh);value['training_binding']=saved['training_binding']
    # lock_file enforces every remaining field via its exact canonical content seal.
    lock_file(file,value)
    return value


COPY_PROVENANCE={'study_source':'src/vlm_lora_coordinate_shrink/study.py','study_sha256':'cac6748a9392ec4cfc7c373af2075754b732193bc607f7390022342856383b34','evaluation_source':'src/vlm_lora_coordinate_shrink/evaluate.py','evaluation_sha256':'4e7d7723d8ed935f0dab08a51ab2d8b6358e504e397fb30e21797547c678fc82','scope':'explicit independent full-stage copy; frozen helpers and numeric operations unchanged'}

def worker_run(args):
    from vlm_lora_coordinate_shrink.inputs import verify_prepared_training_inputs,verify_producer_training_result
    config_file=args.config.resolve();config=read_json(config_file);root=path(config,'output_root')
    if args.output_root.resolve()!=root:raise ValueError('new worker output root differs')
    name=role(args.operation,args.condition,args.seed,args.shard);place=worker_dir(root,name)
    context=verify_context(config_file)
    if args.recovery_lock.resolve()!=recovery_file(config_file):raise ValueError('worker recovery lock path differs')
    verify_study(args.study_lock,config_file,check_weights=True)
    prior=worker_prior(config_file,root,args.operation,args.seed,args.screen_analysis or args.external_analysis)
    hardware=visible_gpu_binding()
    identity={'schema_version':'rarevlm.lora_coordinate_shrink.recovery_worker_input.v1','role':name,'operation':args.operation,
        'condition':args.condition,'seed':args.seed,'shard':args.shard,'hardware':hardware,
        'config_sha256':sha_file(config_file),'study_input_lock_sha256':sha_file(args.study_lock),**recovery_fields(config_file),**prior}
    if args.operation=='eval' and args.seed==42 and hardware['uuid']!=context['original_eval_gpu_uuid']:
        raise ValueError('original pilot GPU UUID must be retained')
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
    if (place/'COMPLETE.json').exists():return read_receipt(place/'COMPLETE.json',config_file)
    artifacts={str(place/'INPUT_LOCK.json'):sha_file(place/'INPUT_LOCK.json')};outcome='passed';failure=None;evidence={}
    def evaluate(task,prior_file=None):
        adapter=root/'runs'/CANDIDATE/f'seed{args.seed}/shrink/adapter'
        existing=root/'runs'/CANDIDATE/f'seed{args.seed}'/task/'EVALUATION.json'
        if not existing.exists():
            if args.seed==42 and task=='object':raise ValueError('original completed Object missing; never regenerate')
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
    receipt={'schema_version':'rarevlm.lora_coordinate_shrink.recovery_worker.v1','status':'completed','outcome':outcome,'role':name,
        'operation':args.operation,'condition':args.condition,'seed':args.seed,'shard':args.shard,'gpu_uuid':hardware['uuid'],
        'completed_at':now(),'study_input_lock_sha256':sha_file(args.study_lock),**recovery_fields(config_file),'artifact_files_sha256':artifacts,
        'engineering_reason':failure,'engineering_evidence_files_sha256':evidence,'existing_tasks_untouched':True,'automatic_runtime_fallback':False}
    lock_file(place/'COMPLETE.json',receipt);write_json(place/'PROGRESS.json',{'status':'completed','outcome':outcome,'updated_at':now()})
    return receipt



def evaluate_cell(args):
    from vlm_object_eval import evaluate as generation
    from vlm_precision.evaluate import load_path
    verify_context(args.config)
    if args.recovery_lock.resolve()!=recovery_file(args.config):raise ValueError('evaluator recovery lock differs')
    if args.seed==42 and args.task=='object':raise ValueError('original Object regeneration prohibited')
    verify_study(args.output_root/'STUDY_INPUT_LOCK.json',args.config,check_weights=True)
    config = read_json(args.config)
    original_evaluation.validate_config(config)
    external_admission(config)
    if args.output_root.resolve() != path(config, "output_root"):
        raise ValueError("output root differs from predeclared KL study")
    model_root, reference_file = path(config, "model_root"), path(config, "reference_config")
    reference = read_json(reference_file)
    import torch
    torch.set_num_threads(2)
    files, binding = adapter_binding(config, args.condition, args.seed, args.adapter_dir, model_root, config_file=args.config)
    if args.task in ('external','pope'):
        phase_admission(args.screen_analysis if args.task=='external' else args.external_analysis,args.config,'screen' if args.task=='external' else 'external')
    progress_binding = original_evaluation.progression(args, config, binding)
    rows, images, image_root, metadata = load_data(config, args.task)
    model_metadata = model_metadata_binding(model_root, reference)
    for name, digest in model_metadata["model_weight_provenance_sha256"].items():
        if sha_file(model_root / name) != digest:
            raise ValueError("current original base weight bytes differ")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or not torch.cuda.is_available():
        raise ValueError("single external-queue-admitted CUDA worker required")
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda", 0)
    props = torch.cuda.get_device_properties(device)
    if "A100" not in props.name or props.total_memory < 70 * 2**30:
        raise ValueError("admitted A10080GB required")
    hardware = {**visible_gpu_binding(), "name": props.name, "total_memory_bytes": props.total_memory,
        "compute_capability": [props.major, props.minor], "torch_cuda_version": torch.version.cuda}
    from vlm_lora_coordinate_shrink.contracts import validate_gpu_admission
    admission = validate_gpu_admission(args.config, args.admission, condition=args.condition,
        seed=args.seed, task=args.task, gpu_uuid=hardware["uuid"])
    verify_recovery_fields(admission,args.config)
    context=verify_context(args.config)
    if args.seed==42 and hardware['uuid']!=context['original_eval_gpu_uuid']:raise ValueError('pilot numeric GPU UUID changed')
    lock = {"schema_version": "rarevlm.lora_coordinate_shrink.eval_input.v1", **FIXED, **model_metadata, **metadata,
        "condition": args.condition, "seed": args.seed, "task": args.task,
        "study_config_sha256": sha_file(args.config), "study_config_content_sha256": content_sha(config),
        "reference_config_sha256": sha_file(reference_file), "source_sha256": source_hashes(),
        "packages": package_versions(), "hardware": hardware, "adapter_dir": str(args.adapter_dir.resolve()),
        "adapter_files_sha256": files, "training_binding": binding, "prompt_suffix": generation.suffix(args.task),
        "current_original_weight_bytes_verified": True, "head_last_position_only": True,
        "explicit_lora_bf16_conversion": True, "gpu_admission_sha256": sha_file(args.admission),
        "input": "original image+question+fixed task suffix; no reference/label/target/OCR", **progress_binding,**recovery_fields(args.config)}
    place = directory(args.output_root, args.condition, args.seed, args.task)
    status = {"condition": args.condition, "seed": args.seed, "task": args.task, "phase": "loading_model", "started_at": now()}
    def update(count=None, state=None):
        if count is not None: status["completed_questions"] = count
        if state is not None: status["cell_status"] = state
        status["updated_at"] = now()
        write_json(place / "STATUS.json", status)
    model, hook = None, None
    try:
        update()
        started = time.monotonic()
        model, dtypes = load_path(model_root, reference, device, "original_bf16", args.adapter_dir)
        status["model_load_seconds"] = time.monotonic() - started
        lock["actual_parameter_dtypes"] = dtypes
        from vlm_object_train.train import tensor_hash
        lora=[(name,p) for name,p in model.named_parameters() if 'lora_' in name]
        if len(lora)!=144 or sum(p.numel() for _,p in lora)!=1843200 or any(p.dtype!=torch.bfloat16 for _,p in lora):
            raise ValueError('actual native inference LoRA layout/dtype differs')
        native=tensor_hash(lora);promoted=tensor_hash((name,p.float()) for name,p in lora)
        if native!=binding['exported_bf16_tensor_sha256'] or promoted!=binding['exported_bf16_round_fp32_tensor_sha256']:
            raise ValueError('actual native loaded LoRA bytes differ from verified CPU export conversion')
        lock['actual_native_bf16_tensor_sha256']=native
        lock['actual_native_bf16_round_fp32_tensor_sha256']=promoted
        if int(model.config.image_token_id) != 151655:
            raise ValueError("image token ID differs")
        hook = model.get_output_embeddings().register_forward_pre_hook(lambda module, inputs: (inputs[0][:, -1:, :],))
        processor = processor_for(model_root, reference, 1024)
        status["verify_export_current_CPU_wall_seconds"]=binding["export_verification"]["verify_CPU_wall_seconds"]
        lock=resume_input_lock(place/"INPUT_LOCK.json",lock)
        lock_file(place / "INPUT_LOCK.json", lock)
        status["phase"] = "validating_images"; update()
        status["image_validation"] = verify_dev_images(rows, images, image_root)
        status["phase"] = "engineering_preflight"; update()
        generation.run_preflight(model, processor, args, lock, rows, images, image_root, device)
        if not args.preflight_only:
            status["phase"] = "evaluating"; update()
            result = generation.run_cell(model, processor, args, lock, rows, images, image_root, device, update)
            result.update(schema_version="rarevlm.lora_coordinate_shrink.eval_cell.v1", scope="fixed coordinate-shrunk export; real unshrunk producer bound separately; no confirmation selection")
            result.update(recovery_fields(args.config))
            write_json(place / "EVALUATION.json", result)
        status.update(phase="preflight_completed" if args.preflight_only else "completed", completed_at=now()); update()
    except BaseException as error:
        status.update(phase="failed", error_type=type(error).__name__, error=str(error)); update()
        raise
    finally:
        if hook is not None: hook.remove()
        del model
        gc.collect()
        torch.cuda.empty_cache()


TEST_NAMES=('test_lora_coordinate_shrink_recovery_compat.py','test_lora_coordinate_shrink_recovery_lock.py',
    'test_lora_coordinate_shrink_recovery_study.py')


def run_inputs(config_file):
    config=read_json(config_file)
    return [*(PROJECT_ROOT/'tests'/name for name in TEST_NAMES),Path(config_file).resolve(),path(config,'study_protocol'),
        PROJECT_ROOT/'docs/VLM_LORA_COORDINATE_SHRINK_RECOVERY_V1_PROTOCOL.md']


def validate_cpu_receipt(receipt,config_file):
    current=sources();expected_inputs=file_map(run_inputs(config_file))
    if (receipt.get('schema_version')!='rarevlm.lora_coordinate_shrink.recovery_cpu.v1'
        or receipt.get('status')!='passed' or receipt.get('returncode')!=0
        or receipt.get('source_sha256_before')!=current or receipt.get('source_sha256_after')!=current
        or receipt.get('run_input_files_sha256_before')!=expected_inputs
        or receipt.get('run_input_files_sha256_after')!=expected_inputs
        or any(type(receipt.get(key)) is not int or receipt[key]!=0 for key in ('new_model_forwards','gpu_operations','optimizer_updates'))):
        raise ValueError('complete unchanged new recovery CPU/source/input receipt required')
    original_map=read_json(path(read_json(config_file),'output_root')/'STUDY_INPUT_LOCK.json')['source_sha256']
    if receipt.get('original_source_sha256_before')!=original_map or receipt.get('original_source_sha256_after')!=original_map:
        raise ValueError('original96 scientific source bytes changed during recovery contracts')
    verify_files(original_map);verify_files(receipt['files_sha256'])


def check_cpu(config_file):
    config_file=Path(config_file).resolve();work=recovery_root(config_file);work.mkdir(parents=True,exist_ok=True)
    receipt=work/'CPU_CONTRACTS.json';log=work/'cpu_contracts.log';xml=work/'cpu_contracts.xml'
    if any(file.exists() for file in (receipt,log,xml,work/'RECOVERY_INPUT_LOCK.json')):
        raise ValueError('recovery CPU evidence exists; preserve rather than overwrite')
    before=sources();inputs_before=file_map(run_inputs(config_file))
    original_before=read_json(path(read_json(config_file),'output_root')/'STUDY_INPUT_LOCK.json')['source_sha256'];verify_files(original_before)
    command=[sys.executable,'-m','pytest','-q',*[str(PROJECT_ROOT/'tests'/name) for name in TEST_NAMES],'--junitxml',str(xml)]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        PYTHONPATH=str(SOURCE_ROOT),PYTHONDONTWRITEBYTECODE='1')
    begun=time.monotonic()
    with log.open('xb') as stream:completed=subprocess.run(command,cwd=PROJECT_ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
    after=sources();inputs_after=file_map(run_inputs(config_file))
    original_after=read_json(path(read_json(config_file),'output_root')/'STUDY_INPUT_LOCK.json')['source_sha256'];verify_files(original_after)
    value={'schema_version':'rarevlm.lora_coordinate_shrink.recovery_cpu.v1',
        'status':'passed' if completed.returncode==0 and before==after and inputs_before==inputs_after and original_before==original_after else 'failed',
        'command':command,'returncode':completed.returncode,'wall_seconds':time.monotonic()-begun,
        'source_sha256_before':before,'source_sha256_after':after,
        'original_source_sha256_before':original_before,'original_source_sha256_after':original_after,
        'run_input_files_sha256_before':inputs_before,'run_input_files_sha256_after':inputs_after,
        'files_sha256':file_map([log,xml,*run_inputs(config_file)]),
        'new_model_forwards':0,'gpu_operations':0,'optimizer_updates':0,'goal_complete':False}
    write_json(receipt,value)
    if value['status']!='passed':raise ValueError('new recovery CPU contracts failed; evidence retained')
    validate_cpu_receipt(value,config_file)
    return value


def prepare(config_file):
    config_file=Path(config_file).resolve();config=read_json(config_file);root=path(config,'output_root')
    original.validate_registry(config);original_evaluation.validate_config(config)
    cpu=recovery_root(config_file)/'CPU_CONTRACTS.json';validate_cpu_receipt(read_json(cpu),config_file)
    identity=prepare_lock(config_file,cpu);verify_recovery(identity,config_file)
    return config,root,identity


def finish(root,status,phase,analyses,receipts,started,config_file):
    verify_context(config_file);work=root/'recovery_v1'
    results={str(file):read_json(file) for file in sorted((root/'producer_runs'/CANDIDATE).glob('seed*/train/TRAIN_RESULT.json'))}
    if any(item['seed']==42 for item in results.values()):raise ValueError('no new seed42 producer is allowed')
    dynamic=[*root.glob('prepared/seed*/*.json'),*root.glob('prepared/seed*/*.jsonl'),
        *root.glob('teacher_replies/seed*/MANIFEST.json'),*root.glob('teacher_replies/seed*/TEACHER_REPLIES_LOCK.json')]
    histories=sorted((root/'producer_runs'/CANDIDATE).glob('seed*/train/PREFLIGHT_HISTORY.jsonl'));attempts=[]
    for file in histories:
        for line in file.read_text().splitlines():attempts.append(json.loads(line))
    pilot_formal={};pilot_preflight={}
    for task in ('object','text'):
        result=root/'runs'/CANDIDATE/f'seed42/{task}/EVALUATION.json'
        value=read_json(result) if result.exists() else {}
        pilot_formal[task]=value.get('questions',0) if value.get('status')=='completed' else 0
        pref=root/'preflight'/CANDIDATE/f'seed42/{task}/PREFLIGHT.json'
        saved=read_json(pref) if pref.exists() else {}
        pilot_preflight[task]=len(saved.get('records',[]))
    value={'schema_version':'rarevlm.lora_coordinate_shrink.recovery_trial.v1','status':status,'trial_status':status,
        'last_phase':phase,'completed_at':now(),'candidate':CANDIDATE,'control_can_be_selected':False,
        'goal_complete':False,'confirmation_passed':status=='confirmation_passed_pending_independent_review',
        'scientific_failure':status=='completed_negative','root_independent_prediction_review_required':True,
        'original_failure_preserved':True,'study_input_lock_sha256':sha_file(root/'STUDY_INPUT_LOCK.json'),**recovery_fields(config_file),
        'analysis_files_sha256':file_map(analyses.values()),
        'worker_receipts_sha256':file_map([worker_dir(root,item['role'])/'COMPLETE.json' for item in receipts]),
        'dynamic_input_files_sha256':file_map(dynamic),
        'export_manifest_files_sha256':file_map(sorted((root/'runs'/CANDIDATE).glob('seed*/shrink/EXPORT_MANIFEST.json'))),
        'new_seed42_training_updates':0,'new_seed42_teacher_generation_questions':0,'new_seed42_CPU_exports':0,
        'pilot_formal_generation_questions_total':sum(pilot_formal.values()),
        'pilot_eval_preflight_questions_total':sum(pilot_preflight.values()),
        'original_object_formal_questions_reused':342,'original_object_preflight_questions_reused':8,
        'recovery_pilot_formal_generation_questions':pilot_formal['text'],
        'recovery_pilot_eval_preflight_questions':pilot_preflight['text'],
        'pilot_generation_accounting_scope':'saved completed EVALUATION counts and saved preflight records; interrupted unsaved attempts are not inferred',
        'new_producer_training_updates':sum(item['completed_updates'] for item in results.values()),
        'new_producer_reference_forward_questions':sum(item['formal_reference_forward_questions'] for item in results.values()),
        'discarded_new_producer_engineering_updates':sum(item.get('discarded_engineering_optimizer_updates',0) for item in attempts),
        'new_producer_training_result_files_sha256':file_map(results),'preflight_history_files_sha256':file_map(histories),
        'teacher_generation_accounting':original.teacher_generation_accounting(root),
        'current_recovery_controller_session_seconds':time.monotonic()-started,'existing_tasks_untouched':True}
    write_json(work/'COMPLETION.json',value);write_json(work/'STATUS.json',value)
    if value['confirmation_passed']:write_json(work/'TRIAL_EVIDENCE.json',value)
    return value


def run_study(config_file,prepare_only=False):
    config_file=Path(config_file).resolve();root=path(read_json(config_file),'output_root');work=recovery_root(config_file);work.mkdir(parents=True,exist_ok=True)
    owner=(work/'CONTROLLER.lock').open('a+');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.monotonic();analyses={};receipts=[];phase='recovery_preparation'
    def queue(name,jobs):
        verify_context(config_file)
        write_json(work/'STATUS.json',{'status':name,'updated_at':now(),'controller_pid':os.getpid(),'goal_complete':False,**recovery_fields(config_file)})
        current=queue_stage(root,name,jobs,identity,config_file);receipts.extend(current);return all(item['outcome']=='passed' for item in current)
    def close(status,stage):return finish(root,status,stage,analyses,receipts,started,config_file)
    try:
        config,root,identity=prepare(config_file)
        if prepare_only:return {'status':'prepared','gpu_operations':0,'new_seed42_training_updates':0,'goal_complete':False}
        if (work/'COMPLETION.json').exists():
            existing=read_json(work/'COMPLETION.json');verify_recovery_fields(existing,config_file)
            for key in ('analysis_files_sha256','worker_receipts_sha256','dynamic_input_files_sha256','export_manifest_files_sha256'):verify_files(existing[key])
            for file in existing['worker_receipts_sha256']:read_receipt(Path(file),config_file)
            return existing
        phase='pilot42'
        if not queue(phase,[job(config_file,root,'eval',CANDIDATE,42)]):return close('engineering_rejected',phase)
        pilot,file=analyze(config_file,root,[42],'screen');analyses['pilot']=file
        if pilot['status']!='complete':raise ValueError('original Object plus complete new Text and historical references required')
        if not pilot['gate']['passed']:return close('completed_negative','pilot_internal_screen')
        for seed in (0,200):verify_context(config_file);prepare_generation_inputs(config,seed)
        phase='teacher_generation_seeds0_200'
        if not queue(phase,[job(config_file,root,'teacher','initial',seed,shard=shard) for seed in (0,200) for shard in (0,1)]):return close('engineering_rejected',phase)
        for seed in (0,200):
            prepare_producer_inputs(config_file,config,seed)
        phase='producer_replication'
        if not queue(phase,[job(config_file,root,'producer',CANDIDATE,seed) for seed in (0,200)]):return close('engineering_rejected',phase)
        for seed in (0,200):ensure_export(config_file,config,seed)
        phase='replication_evaluation'
        if not queue(phase,[job(config_file,root,'eval',CANDIDATE,seed) for seed in (0,200)]):return close('engineering_rejected',phase)
        screen,file=analyze(config_file,root,[0,42,200],'screen',pilot_file=analyses['pilot']);analyses['internal']=file
        if screen['status']!='complete':raise ValueError('complete three-seed recovery screen required')
        if not screen['gate']['passed']:return close('completed_negative','three_seed_internal_screen')
        phase='external_confirmation'
        if not queue(phase,[job(config_file,root,'external',CANDIDATE,seed,file) for seed in (0,42,200)]):return close('engineering_rejected',phase)
        external,file=analyze(config_file,root,[0,42,200],'external',screen_file=analyses['internal']);analyses['external']=file
        if external['status']!='complete':raise ValueError('all three full4372 external cells required')
        if not external['gate']['passed']:return close('completed_negative',phase)
        phase='pope_confirmation'
        if not queue(phase,[job(config_file,root,'pope',CANDIDATE,seed,file) for seed in (0,42,200)]):return close('engineering_rejected',phase)
        confirmed,file=analyze(config_file,root,[0,42,200],'confirm',external_file=analyses['external']);analyses['pope']=file
        if confirmed['status']!='complete':raise ValueError('all three full9000 POPE cells required')
        return close('confirmation_passed_pending_independent_review' if confirmed['gate']['passed'] else 'completed_negative',phase)
    except BaseException as error:
        write_json(work/'STATUS.json',{'status':'execution_or_contract_failure','phase':phase,'error':str(error),
            'scientific_failure':False,'goal_complete':False,'updated_at':now()});raise
    finally:owner.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    mode=parser.add_mutually_exclusive_group(required=True)
    for name in ('check-cpu','prepare-only','run','worker','evaluate'):mode.add_argument('--'+name,action='store_true')
    for name in ('output-root','study-lock','recovery-lock','screen-analysis','external-analysis','adapter-dir','admission'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--operation',choices=OPERATIONS);parser.add_argument('--condition',choices=('initial',CANDIDATE))
    parser.add_argument('--seed',type=int,choices=(0,42,200));parser.add_argument('--shard',type=int,choices=(0,1))
    parser.add_argument('--task',choices=('object','text','external','pope'));parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args()
    if not args.config.is_absolute():raise ValueError('absolute original config required')
    if args.check_cpu:result=check_cpu(args.config)
    elif args.evaluate:
        if any(getattr(args,key) is None for key in ('output_root','recovery_lock','condition','seed','task','adapter_dir','admission')):parser.error('full evaluator identity required')
        evaluate_cell(args);result={'status':'completed','goal_complete':False}
    elif args.worker:
        if any(getattr(args,key) is None for key in ('output_root','study_lock','recovery_lock','operation','condition','seed')):parser.error('full worker identity required')
        result=worker_run(args)
    else:result=run_study(args.config,args.prepare_only)
    print(json.dumps({'status':result.get('status'),'goal_complete':False},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
