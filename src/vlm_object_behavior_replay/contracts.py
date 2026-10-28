"""Replay-coverage-specific identities, actual-input readers, and historical refs."""
COPY_PROVENANCE = {'source': 'src/vlm_object_replay_coverage/contracts.py', 'sha256': 'b13e120853d94afca7a40b080a39d625a7dd9ab1e6ac596d531f03f70a2e0be2', 'scope': 'independent adapted copy; frozen original globals and bytes are unchanged'}
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_key]='2'
from pathlib import Path
import json
import math
import re
from rarevlm_paths import PROJECT_ROOT,SOURCE_ROOT
from vlm_object_eval import evaluate as generation
from vlm_posttrain_eval import evaluate as public_text
from vlm_posttrain_eval.analysis import read_cell as original_text_cell
from vlm_posttrain_pope.evaluate import read_cell as original_pope_cell
from vlm_object_kl_eval.contracts import (FIXED,path,canonical_rows,external_admission as frozen_external_admission,external_data,load_data,validate_parameter_budget)
from vlm_object_kl_eval.read import verify_historical_study,verify_source_data
from vlm_probe.evaluate import content_sha,lock_file,package_versions,processor_for,private_path,read_json,sha_file

CANDIDATE='teacher_ce32'
CONTROL='gt_ce32'
TRAINED_CONDITIONS=(CANDIDATE,CONTROL)
CONDITIONS=('initial',*TRAINED_CONDITIONS)
TASKS=('object','text','external','pope')

def source_hashes():
    from vlm_object_kl_eval.contracts import source_hashes as inherited_sources
    from .train import source_hashes as training_sources
    result=inherited_sources();result.update(training_sources())
    for file in sorted(Path(__file__).parent.glob('*.py')):
        result[str(file.relative_to(SOURCE_ROOT))]=sha_file(file)
    for name in ('run_object_behavior_replay_study.py','check_object_behavior_replay_contracts.py'):
        file=PROJECT_ROOT/'tools'/name
        if file.exists():result[str(file.relative_to(PROJECT_ROOT))]=sha_file(file)
    return dict(sorted(result.items()))

def directory(root,condition,seed,task):
    return root/'runs'/condition/f'seed{seed}'/task


def validate_condition(condition,seed,task=None):
    if condition not in TRAINED_CONDITIONS or seed not in (0,42,200):
        raise ValueError('unknown coverage arm/order seed')
    if condition==CONTROL and (seed!=42 or task not in (None,'object','text')):
        raise ValueError('gt_ce32 is a pilot42 internal control, never a candidate')


def training_expectations(config,condition,seed,files,initial_files,model_root,config_file):
    validate_condition(condition,seed)
    n=len(read_json(path(config,'train_manifest'))['rows']);updates=math.ceil(n/16)
    return {'schema_version':'rarevlm.object_behavior_replay.train.v1','condition':condition,'seed':seed,
        'status':'completed','config_sha256':content_sha(config),'config_file_sha256':sha_file(config_file),
        'adapter_files_sha256':files,'initial_adapter_files_sha256':initial_files,
        'initial_adapter_dir':str(path(config,'initial_adapter')),'trainable_parameters':1843200,
        'training_microbatch_questions':4,'training_accumulation':4,'object_effective_batch':16,
        'replay_questions_per_update':32,'actual_object_exposures':n,'actual_replay_exposures':updates*32,
        'actual_questions_seen':n+updates*32,'planned_updates':updates,'completed_updates':updates,'skipped_updates':0,
        'numerical_path':'original_bf16','visual_token_budget':1024,'lora_parameter_precision':'fp32',
        'frozen_parameters_unchanged':True,'replay_lambda':1.,
        'teacher_probability_cache_used':False,'teacher_reply_generation_new_training_calls':0,
        'train_manifest_file_sha256':sha_file(path(config,'train_manifest')),
        'model_provenance_sha256':sha_file(model_root.parent/'model_provenance.json')}


def validate_teacher_accounting(train,condition,exposures):
    if train.get('replay_coverage',{}).get('unique_questions_consumed')!=exposures:
        raise ValueError('both CE arms must consume the same full unique replay stream')
    if any(train.get(key) not in (None,False,0) for key in ('teacher_probability_cache_used','teacher_full_vocabulary')):
        raise ValueError('behavior replay must not use a teacher probability bank')
    if condition not in TRAINED_CONDITIONS:raise ValueError('unknown behavior arm')


def adapter_binding(config,condition,seed,adapter_dir,model_root,*,config_file):
    from .train import source_hashes as training_sources
    from .inputs import verify_training_result
    initial=path(config,'initial_adapter')
    initial_files,initial_binding=public_text.adapter_binding(config,'initial',42,initial,model_root)
    adapter_dir=private_path(adapter_dir)
    if condition=='initial':
        if seed!=42 or adapter_dir!=initial:raise ValueError('one shared fixed SFT42 initializer required')
        return initial_files,{**initial_binding,'shared_initial_checkpoint':True}
    validate_condition(condition,seed)
    train_dir=path(config,'output_root')/'runs'/condition/f'seed{seed}'/'train'
    if adapter_dir!=train_dir/'adapter':raise ValueError('only this behavior run Final adapter is eligible')
    files=generation.adapter_files(adapter_dir);file=train_dir/'TRAIN_RESULT.json';train=read_json(file)
    expected=training_expectations(config,condition,seed,files,initial_files,model_root,config_file)
    expected['source_sha256']=training_sources()
    for key,value in expected.items():
        if train.get(key)!=value:raise ValueError(f'behavior training binding differs: {key}')
    validate_parameter_budget(train)
    if train.get('initial_adapter_tensor_sha256')!=config['teacher_generation']['initial_runtime_master_sha256']:
        raise ValueError('behavior initializer FP32 master numerical path differs')
    # Historical wide32 is provenance only: no old optimizer/probability tensor is opened.
    anchor_file=path(config,'historical_coverage_study_root')/'runs/wide32/seed42/train/TRAIN_RESULT.json'
    anchor=read_json(anchor_file)
    for key in ('initial_adapter_tensor_sha256','trainable_names','frozen_parameter_sha256_before',
                'frozen_parameter_sha256_after','train_manifest_file_sha256'):
        if train.get(key)!=anchor.get(key):raise ValueError(f'behavior base/initial/data numerical path differs: {key}')
    validate_legacy_stream_alignment(train,anchor,condition,seed)
    validate_teacher_accounting(train,condition,train['actual_replay_exposures'])
    prepared=verify_training_result(config,train_dir,condition=condition,seed=seed)
    return files,{'path':str(file),'sha256':sha_file(file),'condition':condition,'training_seed':seed,
        'initial_train_result_sha256':initial_binding['sha256'],'initial_adapter_files_sha256':initial_files,
        'initial_adapter_tensor_sha256':train['initial_adapter_tensor_sha256'],'behavior_training_inputs':prepared,
        'scope':'fixed Final behavior replay; GT CE is pilot control; target content/length/compute may differ'}


def historical_external(root,config_file,config,processor=None):
    return historical_initial(root,'external',config_file,config,processor)


def historical_initial(root,task,config_file,config,processor=None):
    """Reuse the already-qualified raw baseline, without old tensor-bank reads."""
    old_root=path(config,'historical_coverage_study_root')
    old_config_file=path(config,'historical_coverage_study_config')
    old_config=read_json(old_config_file);old_study=read_json(old_root/'STUDY_INPUT_LOCK.json')
    completed=read_json(old_root/'COMPLETION.json')
    if (old_study.get('schema_version')!='rarevlm.object_replay_coverage.study_input.v1'
        or old_study.get('config_file_sha256')!=sha_file(old_config_file)
        or completed.get('status')!='completed_negative' or completed.get('goal_complete') is not False
        or completed.get('study_input_lock_sha256')!=sha_file(old_root/'STUDY_INPUT_LOCK.json')):
        raise ValueError('historical coverage provenance must remain completed negative')
    for name in ('model_root','initial_adapter','reference_config','asset_root','pope_data_root',
                 'val_manifest','external_manifest','external_class_balanced_manifest','external_initial_reference'):
        if path(config,name)!=path(old_config,name):raise ValueError(f'historical baseline path differs: {name}')
    declaration_file=old_root/'runs/initial/seed42'/task/'HISTORICAL_REFERENCE.json'
    if old_study['files_sha256'].get(str(declaration_file))!=sha_file(declaration_file):
        raise ValueError('original historical declaration no longer matches frozen coverage study')
    original=read_json(declaration_file);provenance=original['source_data']
    if (original.get('condition')!='initial' or original.get('seed')!=42 or original.get('task')!=task
        or original.get('new_generation_questions')!=0):raise ValueError('wrong original historical baseline')
    for kind in ('predictions','evaluation','input_lock'):
        file=private_path(provenance[kind+'_file']);digest=provenance[kind+'_sha256']
        if old_study['files_sha256'].get(str(file))!=digest or sha_file(file)!=digest:
            raise ValueError(f'historical baseline raw bytes differ: {kind}')
    lock=read_json(provenance['input_lock_file']);evaluation=read_json(provenance['evaluation_file'])
    raw=Path(provenance['predictions_file']).read_bytes()
    records=parse_complete_history(raw)
    refs,images,_,metadata=load_data(config,task)
    selected_images=validate_historical_cohort_scope(records,refs,images,metadata,evaluation,lock,task)
    evaluation_view,evaluation_view_proof=historical_evaluation_view(evaluation,task)
    if (evaluation.get('status')!='completed' or len(records)!=len(refs)
        or len({row['question_id'] for row in records})!=len(refs)
        or len({row['image_id'] for row in records})!=len(selected_images)
        or evaluation.get('predictions_sha256')!=provenance['predictions_sha256']
        or evaluation.get('input_lock_sha256')!=content_sha(lock)):
        raise ValueError('historical full baseline coverage/hash differs')
    count=evaluation['n'] if task=='text' else evaluation['questions']
    if count!=len(refs) or evaluation['unique_images']!=len(selected_images):
        raise ValueError('historical full task denominator differs')
    for key,value in FIXED.items():
        if lock.get(key)!=value:raise ValueError(f'historical original generation setting differs: {key}')
    for row,ref in zip(records,refs):
        if (row['question_id']!=ref['question_id'] or row['image_id']!=ref['image_id']
            or row['question_sha256']!=content_sha(ref['question'])
            or row['question_prompt_sha256']!=content_sha(ref['question']+generation.suffix(task))
            or row.get('input_truncated') is not False or row.get('input_lock_sha256')!=content_sha(lock)
            or not isinstance(row.get('prediction'),str) or not isinstance(row.get('generated_ids'),list)
            or content_sha(row['generated_ids'])!=row.get('generated_ids_sha256')
            or type(row.get('score')) not in (int,float) or not math.isfinite(row['score'])):
            raise ValueError('historical per-item qid/prompt/input/generated-ID identity differs')
        if task!='text' and any(row.get(key)!=ref[key] for key in ('label','category')):
            raise ValueError('historical binary label/category differs')
    summary=generation.summary(records,task)
    if not math.isclose(evaluation_view['score_percent'],summary['score_percent'],rel_tol=0,abs_tol=1e-10):
        raise ValueError('historical frozen score summary differs')
    if task!='text':validate_legacy_metric_subset(evaluation['metrics'],summary['metrics'])
    initial_files,binding=adapter_binding(config,'initial',42,path(config,'initial_adapter'),path(config,'model_root'),config_file=config_file)
    if lock['adapter_files_sha256']!=initial_files or lock['training_binding']['sha256']!=binding['sha256']:
        raise ValueError('historical initializer actual adapter byte binding differs')
    canonical,proof=canonical_rows(records,task)
    declaration={'schema_version':'rarevlm.object_behavior_replay.historical_reference.v1','condition':'initial',
        'seed':42,'task':task,'new_generation_questions':0,'source_data':provenance,
        'original_declaration_file':str(declaration_file),'original_declaration_sha256':sha_file(declaration_file),
        'historical_study_root':str(old_root),'historical_config_file':str(old_config_file),
        'historical_config_sha256':sha_file(old_config_file),'historical_study_input_lock_sha256':sha_file(old_root/'STUDY_INPUT_LOCK.json'),
        'new_study_config_sha256':sha_file(config_file),'original_source_and_hardware_lock_retained':True,
        'text_category_canonicalization':proof,'scope':'qualified complete raw historical reference; no new generation/re-encoding or old probability-bank read'}
    if evaluation_view_proof['changed_fields']:declaration['evaluation_view_provenance']=evaluation_view_proof
    lock_file(directory(root,'initial',42,task)/'HISTORICAL_REFERENCE.json',declaration)
    return {'complete':True,'condition':'initial','seed':42,'task':task,'rows':canonical,'lock':lock,
        'evaluation':evaluation_view,'canonicalization':proof,'comparison_metadata':metadata,
        'historical_reference':declaration,'provenance':provenance}

def read_cell(root, condition, seed, task, config_file, config, processor=None):
    place = directory(root, condition, seed, task)
    if condition == "initial":
        return historical_initial(root, task, config_file, config, processor)
    result_file = place / "EVALUATION.json"
    if not result_file.exists():
        return {"complete": False, "condition": condition, "seed": seed, "task": task, "path": str(place)}
    lock, result = read_json(place / "INPUT_LOCK.json"), read_json(result_file)
    rows, images, _, metadata = load_data(config, task)
    expected = {**FIXED, **metadata, "condition": condition, "seed": seed, "task": task,
        "schema_version": "rarevlm.object_behavior_replay.eval_input.v1", "study_config_sha256": sha_file(config_file),
        "study_config_content_sha256": content_sha(config), "source_sha256": source_hashes(),
        "packages": package_versions(), "prompt_suffix": generation.suffix(task)}
    if task == "external":
        expected.update(external_admission(config))
    for key, value in expected.items():
        if lock.get(key) != value:
            raise ValueError(f"completed replay-coverage cell identity differs: {key}")
    files, binding = adapter_binding(config, condition, seed, Path(lock["adapter_dir"]),
        path(config, "model_root"), config_file=config_file)
    if files != lock["adapter_files_sha256"] or binding != lock["training_binding"]:
        raise ValueError("completed replay-coverage training/teacher/adapter bytes changed")
    if task in ("external", "pope") and condition != "initial":
        previous_file = Path(lock["prior_phase_analysis_file"])
        previous = phase_admission(previous_file, config_file, "screen" if task == "external" else "external")
        if sha_file(previous_file) != lock["prior_phase_analysis_sha256"]:
            raise ValueError("prior phase analysis byte binding changed")
        source_task = "text" if task == "external" else "external"
        if previous["source_data"][f"{condition}_{seed}_{source_task}"]["training_result_sha256"] != binding["sha256"]:
            raise ValueError("confirmation adapter differs from its passing prior phase")
    if processor is None:
        processor = processor_for(path(config, "model_root"), read_json(path(config, "reference_config")), 1024)
    predictions = place / "predictions.jsonl"
    records, repaired = generation.complete_cache(predictions, rows, images, lock, processor, repair=False)
    if (len(records) != len(rows) or repaired or result.get("status") != "completed"
        or result.get("schema_version") != "rarevlm.object_behavior_replay.eval_cell.v1"
        or result.get("input_lock_sha256") != content_sha(lock)
        or result.get("predictions_sha256") != sha_file(predictions)
        or any(result.get(key) != value for key, value in generation.summary(records, task).items())):
        raise ValueError("completed replay-coverage full-task score/count/hash differs")
    canonical, proof = canonical_rows(records, task)
    return {"complete": True, "condition": condition, "seed": seed, "task": task, "rows": canonical,
        "lock": lock, "evaluation": result, "canonicalization": proof,
        "provenance": {"predictions_file": str(predictions), "predictions_sha256": sha_file(predictions),
            "evaluation_file": str(result_file), "evaluation_sha256": sha_file(result_file),
            "input_lock_file": str(place / "INPUT_LOCK.json"), "input_lock_sha256": sha_file(place / "INPUT_LOCK.json"),
            "training_result_sha256": binding["sha256"]}}

def phase_admission(file, config_file, phase):
    file = Path(file).resolve()
    value = read_json(file)
    if (value.get('candidate') != CANDIDATE
        or value.get("schema_version") != f"rarevlm.object_behavior_replay.{phase}.v1" or value.get("status") != "complete"
        or value.get("config_sha256") != sha_file(config_file) or value.get("source_sha256") != source_hashes()
        or value.get("seeds") != [0, 42, 200] or value.get("gate", {}).get("passed") is not True):
        raise ValueError(f"{phase} phase must pass full three-seed criteria before progression")
    verify_source_data(value["source_data"])
    if phase == "external":
        screen = phase_admission(value["screen_analysis_path"], config_file, "screen")
        if sha_file(Path(value["screen_analysis_path"])) != value["screen_analysis_sha256"]:
            raise ValueError("internal screen changed after external confirmation")
    return value


def validate_inherited_admissions(config):
    """New alias files retain the exact old qualified evidence, without re-review."""
    result={}
    definitions=[('data_admission','reference_data_admission',
        ('status','quality_gate_passed','scope','automatic_diagnostic_bypass_allowed','train_manifest_sha256','dataset_protocol_sha')),
        ('evaluation_data_admission','reference_evaluation_data_admission',
        ('schema_version','status','quality_gate_passed','scope','automatic_diagnostic_bypass_allowed',
         'external_manifest_sha256','external_class_balanced_manifest_sha256','external_dataset_protocol_sha256',
         'frozen_audit_sha256','semantic_review_sha256','cpu_completion_sha256'))]
    for name,reference,fields in definitions:
        current_file,reference_file=path(config,name),path(config,reference)
        if current_file==reference_file:raise ValueError('new study requires an explicit inherited-admission alias')
        current,original=read_json(current_file),read_json(reference_file)
        if (current.get('source_admission_path')!=str(reference_file)
            or current.get('source_admission_sha256')!=sha_file(reference_file)):
            raise ValueError('inherited admission original byte binding differs')
        for field in fields:
            if field not in original or current.get(field)!=original[field]:
                raise ValueError(f'inherited admission scientific field differs: {name}/{field}')
        if (current.get('status')!='admitted' or current.get('quality_gate_passed') is not True
            or current.get('automatic_diagnostic_bypass_allowed') is not False):
            raise ValueError('qualified factual data admission required; no diagnostic bypass')
        expected_scope='annotation_supervised_object_sft' if name=='data_admission' else 'object_external_confirmation'
        if current.get('scope')!=expected_scope:raise ValueError('inherited admission task scope differs')
        result[name+'_sha256']=sha_file(current_file);result[reference+'_sha256']=sha_file(reference_file)
    training=read_json(path(config,'data_admission'))
    if (training['train_manifest_sha256']!=sha_file(path(config,'train_manifest'))
        or training['dataset_protocol_sha']!=sha_file(path(config,'dataset_protocol'))):
        raise ValueError('inherited object train/protocol bytes differ')
    return result


def external_admission(config):
    validate_inherited_admissions(config)
    return frozen_external_admission(config)


def verify_locked_files(files):
    for name,digest in files.items():
        if not Path(name).is_file() or sha_file(Path(name))!=digest:
            raise ValueError(f'locked behavior study file changed: {name}')


def study_sources():
    return {str(PROJECT_ROOT/name if name.startswith('tools/') else SOURCE_ROOT/name):digest
            for name,digest in source_hashes().items()}


def verify_study(file,config_file):
    file,config_file=private_path(file),private_path(config_file);lock=read_json(file)
    if (lock.get('schema_version')!='rarevlm.object_behavior_replay.study_input.v1'
        or lock.get('config_file_sha256')!=sha_file(config_file)
        or lock.get('source_sha256')!=study_sources() or lock.get('packages')!=package_versions()):
        raise ValueError('behavior study source/config/runtime identity changed')
    verify_locked_files(lock['source_sha256']);verify_locked_files(lock['files_sha256'])
    return lock


def validate_gpu_admission(config_file,admission_file,*,condition,seed,task,gpu_uuid,shard=None,
                           plan_file_sha256=None,prompt_inputs_file_sha256=None):
    config_file=private_path(config_file);config=read_json(config_file);root=path(config,'output_root')
    study_file=root/'STUDY_INPUT_LOCK.json';verify_study(study_file,config_file)
    admission=read_json(private_path(admission_file))
    if task=='teacher_rollout':
        if condition!='initial' or seed not in (0,42,200) or shard not in (0,1):raise ValueError('invalid teacher shard role')
        expected_role=f'teacher_seed{seed}_shard{shard}'
    else:
        validate_condition(condition,seed,task)
        expected_role=f'{"train_eval" if task in ("object","text") else task}_{condition}_seed{seed}'
    expected={'schema_version':'rarevlm.object_behavior_replay.gpu_admission.v1','status':'admitted',
        'condition':condition,'seed':seed,'task':task,'role':expected_role,'gpu_uuid':gpu_uuid,
        'config_sha256':sha_file(config_file),'config_file_sha256':sha_file(config_file),
        'study_input_lock_path':str(study_file),'study_input_lock_sha256':sha_file(study_file)}
    if task=='teacher_rollout':
        if plan_file_sha256 is None or prompt_inputs_file_sha256 is None:raise ValueError('teacher requires frozen actual prompt stream')
        expected.update(shard=shard,plan_file_sha256=plan_file_sha256,prompt_inputs_file_sha256=prompt_inputs_file_sha256)
        prepared=root/'prepared'/f'seed{seed}'
        if plan_file_sha256!=sha_file(prepared/'PLAN.json') or prompt_inputs_file_sha256!=sha_file(prepared/'PROMPT_INPUTS.jsonl'):
            raise ValueError('teacher admitted prompt actual bytes differ')
        if seed!=42:
            from .analysis import passing_pilot
            passing_pilot(root/'analysis/pilot_seed42/screen/ANALYSIS.json',config_file)
    for name,value in expected.items():
        if admission.get(name)!=value:raise ValueError(f'behavior GPU admission differs: {name}')
    return admission


def validate_legacy_stream_alignment(train,anchor,condition,seed):
    """Compare the old standard projection; full supervised rows keep new fields."""
    if seed!=42:return
    for key in ('object_order_sha256','object_microbatch_order_sha256','replay_order_sha256','replay_microbatch_order_sha256'):
        if train.get(key)!=anchor.get(key):raise ValueError(f'seed42 fixed QA stream differs: {key}')
    if train.get('object_legacy_input_contract_sha256')!=anchor.get('object_input_contract_sha256'):
        raise ValueError('seed42 old object standard-input projection differs')
    if condition==CONTROL and train.get('replay_legacy_input_contract_sha256')!=anchor.get('replay_input_contract_sha256'):
        raise ValueError('GT control old standard replay projection differs')


def parse_complete_history(raw):
    if not raw.endswith(bytes([10])):raise ValueError('historical complete JSONL has interrupted tail')
    rows=[json.loads(line) for line in raw.splitlines()]
    if not rows or any(not isinstance(row,dict) or 'question_id' not in row for row in rows):
        raise ValueError('historical JSONL requires explicit nonempty per-item records')
    return rows


def validate_legacy_metric_subset(original,current):
    """Retain every old official field; new first-answer diagnostics are additional."""
    if isinstance(original,dict):
        if not isinstance(current,dict):raise ValueError('historical metric structure differs')
        for key,value in original.items():
            if key not in current:raise ValueError(f'historical official metric missing: {key}')
            validate_legacy_metric_subset(value,current[key])
    elif original!=current:raise ValueError('historical official metric value differs')


def create_inherited_admissions(config):
    """Clean setup copies qualified evidence, never invents or overwrites admission."""
    root=path(config,'output_root')
    original_train=read_json(path(config,'reference_data_admission'))
    if (original_train.get('status')!='admitted' or original_train.get('quality_gate_passed') is not True
        or original_train.get('scope')!='annotation_supervised_object_sft'
        or original_train.get('automatic_diagnostic_bypass_allowed') is not False
        or original_train.get('train_manifest_sha256')!=sha_file(path(config,'train_manifest'))
        or original_train.get('dataset_protocol_sha')!=sha_file(path(config,'dataset_protocol'))):
        raise ValueError('original qualified object admission cannot be inherited')
    # The frozen verifier checks original external schema/evidence/raw byte contracts.
    frozen_external_admission({**config,'evaluation_data_admission':str(path(config,'reference_evaluation_data_admission'))})
    aliases={}
    for name,reference,filename in (('data_admission','reference_data_admission','DATA_ADMISSION.json'),
        ('evaluation_data_admission','reference_evaluation_data_admission','EVALUATION_DATA_ADMISSION.json')):
        file=path(config,name);original_file=path(config,reference)
        if file!=root/filename or file==original_file:raise ValueError('admission alias must be a new-root file')
        if (root/'STUDY_INPUT_LOCK.json').exists() and not file.exists():
            raise ValueError('frozen study admission is missing; no silent recreation')
        original=read_json(original_file);value=dict(original)
        previous={key:original[key] for key in ('source_admission_path','source_admission_sha256') if key in original}
        if previous:value['inherited_original_source_provenance']=previous
        value.update(source_admission_path=str(original_file),source_admission_sha256=sha_file(original_file))
        aliases[file]=value
    # lock_file accepts identical existing bytes/content and rejects differences without replacement.
    for file,value in aliases.items():lock_file(file,value)
    return validate_inherited_admissions(config)


def validate_historical_cohort_scope(records,refs,image_lookup,metadata,evaluation,lock,task):
    """Image lookup catalogs may span all splits; counts belong to selected refs."""
    selected={row['image_id'] for row in refs};predicted={row['image_id'] for row in records}
    if not selected or predicted!=selected or not selected.issubset(image_lookup):
        raise ValueError('historical selected image cohort differs or lacks lookup members')
    questions,images=len(refs),len(selected)
    if 'questions' not in metadata or 'unique_images' not in metadata:
        raise ValueError('historical canonical metadata lacks selected scope')
    required_question='n' if task=='text' else 'questions'
    if required_question not in evaluation or 'unique_images' not in evaluation:
        raise ValueError('historical evaluation lacks original declared scope')
    if required_question not in lock or ('expected_images' if task=='text' else 'unique_images') not in lock:
        raise ValueError('historical input lock lacks original declared scope')
    for name,scope in (('metadata',metadata),('evaluation',evaluation),('input_lock',lock)):
        for key in ('questions','n','expected_questions'):
            if key in scope and (type(scope[key]) is not int or scope[key]!=questions):
                raise ValueError(f'historical selected question scope differs: {name}/{key}')
        for key in ('unique_images','expected_images'):
            if key in scope and (type(scope[key]) is not int or scope[key]!=images):
                raise ValueError(f'historical selected image scope differs: {name}/{key}')
    return selected


def historical_evaluation_view(original,task):
    """Only old POPE lacks a percent field; original JSON bytes stay authoritative."""
    view=dict(original);changed=[];origin='original.score_percent'
    if 'score_percent' not in original:
        if task!='pope':raise ValueError('missing historical percent is allowed only for original POPE schema')
        overall=original.get('metrics',{}).get('overall',{});accuracy=overall.get('accuracy')
        if type(accuracy) not in (int,float) or not math.isfinite(accuracy) or not 0<=accuracy<=1:
            raise ValueError('historical POPE official accuracy fraction missing or invalid')
        if 'n' in overall and overall['n']!=original['questions']:
            raise ValueError('historical POPE official denominator differs')
        view['score_percent']=100*accuracy;changed=['score_percent'];origin='original.metrics.overall.accuracy * 100'
    value=view['score_percent']
    if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=100:
        raise ValueError('historical score percent is invalid')
    restored={key:value for key,value in view.items() if key not in changed}
    if restored!=original:raise ValueError('historical evaluation normalization changed original fields')
    return view,{'task':task,'changed_fields':changed,'score_percent_source':origin,'in_memory_only':True,
        'original_evaluation_content_sha256':content_sha(original),
        'all_original_fields_sha256_after':content_sha(restored),'original_fields_unchanged':True}
