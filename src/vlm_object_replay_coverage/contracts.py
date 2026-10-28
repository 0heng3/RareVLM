"""Replay-coverage-specific identities, actual-input readers, and historical refs."""
from pathlib import Path
import math
from rarevlm_paths import PROJECT_ROOT,SOURCE_ROOT
from vlm_object_eval import evaluate as generation
from vlm_posttrain_eval import evaluate as public_text
from vlm_posttrain_eval.analysis import read_cell as original_text_cell
from vlm_posttrain_pope.evaluate import read_cell as original_pope_cell
from vlm_object_kl_eval.contracts import (FIXED,path,canonical_rows,external_admission,external_data,load_data,validate_parameter_budget)
from vlm_object_kl_eval.read import verify_historical_study,verify_source_data
from vlm_probe.evaluate import content_sha,lock_file,package_versions,processor_for,private_path,read_json,sha_file

CANDIDATE='wide32'
CONTROL='repeat8x4'
TRAINED_CONDITIONS=(CANDIDATE,CONTROL)
CONDITIONS=('initial',*TRAINED_CONDITIONS)
TASKS=('object','text','external','pope')

def source_hashes():
    from vlm_object_kl_eval.contracts import source_hashes as inherited_sources
    from .train import source_hashes as training_sources
    result=inherited_sources();result.update(training_sources())
    for file in Path(__file__).parent.glob('*.py'):result[str(file.relative_to(SOURCE_ROOT))]=sha_file(file)
    file=PROJECT_ROOT/'tools/run_object_replay_coverage_study.py'
    if file.exists():result[str(file.relative_to(PROJECT_ROOT))]=sha_file(file)
    return dict(sorted(result.items()))

def directory(root,condition,seed,task):
    return root/'runs'/condition/f'seed{seed}'/task


def validate_condition(condition,seed,task=None):
    if condition not in TRAINED_CONDITIONS or seed not in (0,42,200):
        raise ValueError('unknown coverage arm/order seed')
    if condition==CONTROL and (seed!=42 or task not in (None,'object','text')):
        raise ValueError('repeat8x4 is a pilot42 internal control, never a candidate')


def training_expectations(config,condition,seed,files,initial_files,model_root,config_file):
    validate_condition(condition,seed)
    n=len(read_json(path(config,'train_manifest'))['rows']);updates=math.ceil(n/16)
    return {'schema_version':'rarevlm.object_replay_coverage.train.v1','condition':condition,'seed':seed,
        'status':'completed','config_sha256':content_sha(config),'config_file_sha256':sha_file(config_file),
        'adapter_files_sha256':files,'initial_adapter_files_sha256':initial_files,
        'initial_adapter_dir':str(path(config,'initial_adapter')),'trainable_parameters':1843200,
        'training_microbatch_questions':4,'training_accumulation':4,'object_effective_batch':16,
        'replay_questions_per_update':32,'actual_object_exposures':n,'actual_replay_exposures':updates*32,
        'actual_questions_seen':n+updates*32,'planned_updates':updates,'completed_updates':updates,'skipped_updates':0,
        'numerical_path':'original_bf16','visual_token_budget':1024,'lora_parameter_precision':'fp32',
        'frozen_parameters_unchanged':True,'kl_direction':'initial_to_actor','kl_temperature':1.,
        'teacher_logprob_dtype':'float32','teacher_full_vocabulary':True,'replay_hard_ce_backward':False,
        'replay_lambda':1.,'train_manifest_file_sha256':sha_file(path(config,'train_manifest')),
        'model_provenance_sha256':sha_file(model_root.parent/'model_provenance.json')}


def validate_teacher_accounting(source,condition,exposures,unique):
    expected={'schema_version':'rarevlm.object_replay_coverage.teacher_source.v1',
        'kind':'fresh_full_fp32' if condition==CANDIDATE else 'immutable_repeated_reference',
        'scope':'formal_full_stream','actual_new_teacher_forward_questions':exposures if condition==CANDIDATE else 0,
        'historical_teacher_bank_questions':0 if condition==CANDIDATE else unique,
        'actor_replay_questions_planned':exposures,'actor_replay_unique_questions_planned':unique,
        'source_teacher_unique_questions_referenced':unique}
    if condition not in TRAINED_CONDITIONS or any(source.get(key)!=value for key,value in expected.items()):
        raise ValueError('authoritative teacher source/compute/coverage differs')
    for key,value in expected.items():
        if type(value) is int and type(source[key]) is not int:raise ValueError('teacher QA counts must be integers')


def adapter_binding(config,condition,seed,adapter_dir,model_root,*,config_file):
    from .reference import verify_teacher_source
    from .train import source_hashes as training_sources
    initial=path(config,'initial_adapter')
    initial_files,initial_binding=public_text.adapter_binding(config,'initial',42,initial,model_root)
    adapter_dir=private_path(adapter_dir)
    if condition=='initial':
        if seed!=42 or adapter_dir!=initial:raise ValueError('one shared fixed SFT42 initializer required')
        return initial_files,{**initial_binding,'shared_initial_checkpoint':True}
    validate_condition(condition,seed)
    expected_dir=path(config,'output_root')/'runs'/condition/f'seed{seed}'/'train'
    if adapter_dir!=expected_dir/'adapter':raise ValueError('only this coverage run Final adapter is eligible')
    files=generation.adapter_files(adapter_dir);file=expected_dir/'TRAIN_RESULT.json';train=read_json(file)
    expected=training_expectations(config,condition,seed,files,initial_files,model_root,config_file)
    expected['source_sha256']=training_sources()
    for key,value in expected.items():
        if train.get(key)!=value:raise ValueError(f'coverage training binding differs: {key}')
    validate_parameter_budget(train)
    anchor_file=path(config,'source_kl_root')/'runs/object_kl/seed42/train/TRAIN_RESULT.json'
    anchor=read_json(anchor_file)
    for key in ('initial_adapter_tensor_sha256','trainable_names','frozen_parameter_sha256_before',
                'frozen_parameter_sha256_after','train_manifest_file_sha256'):
        if train.get(key)!=anchor.get(key):raise ValueError(f'coverage base/initial/data numerical path differs: {key}')
    if seed==42:
        for key in ('object_order_sha256','object_microbatch_order_sha256','object_input_contract_sha256'):
            if train.get(key)!=anchor.get(key):raise ValueError(f'original seed42 object stream differs: {key}')
    actual_file=expected_dir/'TRAINED_INPUT_CONTRACTS.json';actual=read_json(actual_file)
    if (sha_file(actual_file)!=train.get('trained_input_contracts_file_sha256')
        or len(actual['object'])!=train['actual_object_exposures'] or len(actual['replay'])!=train['actual_replay_exposures']
        or content_sha(actual['object'])!=train['object_input_contract_sha256']
        or content_sha(actual['replay'])!=train['replay_input_contract_sha256']):
        raise ValueError('coverage actual full training input contracts differ')
    unique=train['actual_replay_exposures'] if condition==CANDIDATE else train['completed_updates']*8
    if train.get('replay_coverage',{}).get('unique_questions_consumed')!=unique:
        raise ValueError('wide32/repeated8 true unique replay coverage differs')
    teacher=verify_teacher_source(train,expected_dir,config)
    source=train['teacher_source']
    validate_teacher_accounting(source,condition,train['actual_replay_exposures'],unique)
    return files,{'path':str(file),'sha256':sha_file(file),'condition':condition,'training_seed':seed,
        'initial_train_result_sha256':initial_binding['sha256'],'initial_adapter_files_sha256':initial_files,
        'initial_adapter_tensor_sha256':train['initial_adapter_tensor_sha256'],'teacher_source':teacher,
        'scope':'fixed Final coverage study; repeat8x4 is pilot exploratory control; teacher compute not equal'}


def historical_external(root,config_file,config,processor=None):
    from vlm_object_kl_eval.read import read_cell as original_kl_cell
    old_root,old_config_file=path(config,'source_kl_root'),path(config,'source_kl_config')
    old_config=read_json(old_config_file)
    for name in ('model_root','initial_adapter','reference_config','asset_root','pope_data_root'):
        if path(config,name)!=path(old_config,name):raise ValueError(f'historical external model/processor/data path differs: {name}')
    if path(config,'external_initial_reference')!=old_root/'runs/initial/seed42/external':
        raise ValueError('external initial reference must be the exact original full KL baseline directory')
    verify_historical_study(old_root,old_config_file)
    before={str(f):sha_file(f) for f in (path(config,'external_initial_reference')/name for name in ('EVALUATION.json','INPUT_LOCK.json','predictions.jsonl'))}
    cell=original_kl_cell(old_root,'initial',42,'external',old_config_file,old_config,processor=processor)
    _,_,_,metadata=load_data(config,'external')
    for key,value in metadata.items():
        if cell['lock'].get(key)!=value:raise ValueError(f'historical external exact data differs: {key}')
    files,binding=adapter_binding(config,'initial',42,path(config,'initial_adapter'),path(config,'model_root'),config_file=config_file)
    if (not cell['complete'] or cell['lock']['adapter_files_sha256']!=files
        or cell['lock']['training_binding']['sha256']!=binding['sha256']):
        raise ValueError('historical external initial checkpoint differs')
    if any(sha_file(Path(f))!=digest for f,digest in before.items()):raise ValueError('historical external bytes changed')
    declaration={'schema_version':'rarevlm.object_replay_coverage.historical_reference.v1','condition':'initial','seed':42,
        'task':'external','new_generation_questions':0,'historical_study_root':str(old_root),
        'historical_config_file':str(old_config_file),'historical_config_sha256':sha_file(old_config_file),
        'historical_study_input_lock_sha256':sha_file(old_root/'STUDY_INPUT_LOCK.json'),
        'new_study_config_sha256':sha_file(config_file),'source_data':cell['provenance'],
        'original_source_and_hardware_lock_retained':True,
        'scope':'complete original4372 baseline reused; no new generation, independent replicate, blind or pretraining-disjoint claim'}
    lock_file(directory(root,'initial',42,'external')/'HISTORICAL_REFERENCE.json',declaration)
    return {**cell,'historical_reference':declaration,'comparison_metadata':metadata}


def historical_initial(root, task, config_file, config, processor=None):
    if task == "external":
        return historical_external(root,config_file,config,processor)
    prefix = "historical_object_study" if task == "object" else "reference_study"
    old_root, old_config_file = path(config, prefix + "_root"), path(config, prefix + "_config")
    old_config = read_json(old_config_file)
    verify_historical_study(old_root, old_config_file)
    for name in ("model_root", "initial_adapter", "reference_config", "asset_root", "pope_data_root"):
        if path(config, name) != generation.config_path(old_config, name):
            raise ValueError(f"historical/new model or fixed data path differs: {name}")
    rows, images, image_root, metadata = load_data(config, task)
    processor = processor or processor_for(path(config, "model_root"), read_json(path(config, "reference_config")), 1024)
    if task == "object":
        if path(config, "val_manifest") != path(old_config, "val_manifest"):
            raise ValueError("historical object342 source differs from fixed internal data")
        cell = generation.read_cell(old_root, "initial", 42, "object", old_config_file, old_config, processor=processor)
        old_task = "object"
    elif task == "text":
        cell = original_text_cell(old_root, "initial", 42, rows, images, old_config_file, old_config, processor)
        old_task = "dev"
    elif task == "pope":
        cell = original_pope_cell(old_root, "initial", 42, old_config_file, old_config)
        old_task = "pope"
    else:
        raise ValueError("unknown historical task")
    if not cell["complete"]:
        raise ValueError("historical initializer full task is incomplete")
    files, binding = adapter_binding(config, "initial", 42, path(config, "initial_adapter"),
        path(config, "model_root"), config_file=config_file)
    if cell["lock"]["adapter_files_sha256"] != files or cell["lock"]["training_binding"]["sha256"] != binding["sha256"]:
        raise ValueError("historical actual checkpoint differs from fixed strong initial")
    old_directory = old_root / "runs/initial/seed42" / old_task
    provenance = {"predictions_file": str(old_directory / "predictions.jsonl"),
        "predictions_sha256": sha_file(old_directory / "predictions.jsonl"),
        "evaluation_file": str(old_directory / "EVALUATION.json"),
        "evaluation_sha256": sha_file(old_directory / "EVALUATION.json"),
        "input_lock_file": str(old_directory / "INPUT_LOCK.json"),
        "input_lock_sha256": sha_file(old_directory / "INPUT_LOCK.json"),
        "training_result_sha256": binding["sha256"], "historical_reference": True}
    canonical, proof = canonical_rows(cell["rows"], task)
    declaration = {"schema_version": "rarevlm.object_replay_coverage.historical_reference.v1", "condition": "initial",
        "seed": 42, "task": task, "new_generation_questions": 0, "original_source_and_hardware_lock_retained": True,
        "historical_study_root": str(old_root), "historical_config_file": str(old_config_file),
        "historical_config_sha256": sha_file(old_config_file),
        "historical_study_input_lock_sha256": sha_file(old_root / "STUDY_INPUT_LOCK.json"),
        "new_study_config_sha256": sha_file(config_file), "source_data": provenance,
        "text_category_canonicalization": proof,
        "scope": "original complete historical predictions; exact byte checks; no new run/model/independent replicate claim"}
    lock_file(directory(root, "initial", 42, task) / "HISTORICAL_REFERENCE.json", declaration)
    return {**cell, "task": task, "rows": canonical, "canonicalization": proof,
        "historical_reference": declaration, "comparison_metadata": metadata, "provenance": provenance}

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
        "schema_version": "rarevlm.object_replay_coverage.eval_input.v1", "study_config_sha256": sha_file(config_file),
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
        or result.get("schema_version") != "rarevlm.object_replay_coverage.eval_cell.v1"
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
        or value.get("schema_version") != f"rarevlm.object_replay_coverage.{phase}.v1" or value.get("status") != "complete"
        or value.get("config_sha256") != sha_file(config_file) or value.get("source_sha256") != source_hashes()
        or value.get("seeds") != [0, 42, 200] or value.get("gate", {}).get("passed") is not True):
        raise ValueError(f"{phase} phase must pass full three-seed criteria before progression")
    verify_source_data(value["source_data"])
    if phase == "external":
        screen = phase_admission(value["screen_analysis_path"], config_file, "screen")
        if sha_file(Path(value["screen_analysis_path"])) != value["screen_analysis_sha256"]:
            raise ValueError("internal screen changed after external confirmation")
    return value
