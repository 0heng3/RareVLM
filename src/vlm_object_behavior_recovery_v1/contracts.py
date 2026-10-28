"""Explicit adapter/reader copies, with one proven historical missing-field view."""
from pathlib import Path
from vlm_object_behavior_replay import contracts as frozen
from vlm_object_behavior_replay.contracts import (path,public_text,generation,private_path,read_json,sha_file,
    training_expectations,validate_parameter_budget,validate_legacy_stream_alignment,validate_teacher_accounting,
    validate_condition,directory,FIXED,load_data,external_admission,source_hashes,package_versions,content_sha,
    phase_admission,processor_for,canonical_rows)
from .compat import legacy_anchor_view
from .lock import verify_lock

COPY_PROVENANCE = {'copied_from': '/opt/data/private/longtail_vlm/rarevlm_pilot/packages/RareVLM_minimal_20261001/src/vlm_object_behavior_replay/contracts.py', 'copied_source_sha256': '859a675b3904dcf7d1a116d4eaf287ddcd20cf70093e63765403bfa92dc761c1', 'adapted_functions': ['adapter_binding', 'read_cell'], 'change': 'explicit recovery lock/source metadata and proven missing historical field; remaining original guard logic retained'}

def adapter_binding(config,condition,seed,adapter_dir,model_root,*,config_file,recovery_lock):
    from vlm_object_behavior_replay.train import source_hashes as training_sources
    from vlm_object_behavior_replay.inputs import verify_training_result
    recovery=verify_lock(recovery_lock,config_file)
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
    proof=None
    if seed==42:
        if recovery['files_sha256'].get(str(anchor_file))!=sha_file(anchor_file):raise ValueError('historical result not recovery-locked')
        anchor,proof=legacy_anchor_view(anchor,read_json(anchor_file.parent/'SCHEDULE.json'),
            read_json(train_dir/'SCHEDULE.json'),read_json(path(config,'output_root')/'prepared/seed42/PLAN.json')['schedule'],train)
    for key in ('initial_adapter_tensor_sha256','trainable_names','frozen_parameter_sha256_before',
                'frozen_parameter_sha256_after','train_manifest_file_sha256'):
        if train.get(key)!=anchor.get(key):raise ValueError(f'behavior base/initial/data numerical path differs: {key}')
    validate_legacy_stream_alignment(train,anchor,condition,seed)
    validate_teacher_accounting(train,condition,train['actual_replay_exposures'])
    prepared=verify_training_result(config,train_dir,condition=condition,seed=seed)
    return files,{'path':str(file),'sha256':sha_file(file),'condition':condition,'training_seed':seed,
        'initial_train_result_sha256':initial_binding['sha256'],'initial_adapter_files_sha256':initial_files,
        'initial_adapter_tensor_sha256':train['initial_adapter_tensor_sha256'],'behavior_training_inputs':prepared,
        'scope':'fixed Final behavior replay; GT CE is pilot control; target content/length/compute may differ',
        'versioned_engineering_recovery':{'recovery_input_lock_sha256':sha_file(Path(recovery_lock)),
            'effective_recovery_source_sha256':recovery['recovery_source_sha256'],'historical_serialization_proof':proof}}

def read_cell(root, condition, seed, task, config_file, config, processor=None,*,recovery_lock):
    recovery=verify_lock(recovery_lock,config_file)
    place = directory(root, condition, seed, task)
    if condition == "initial":
        return frozen.historical_initial(root, task, config_file, config, processor)
    result_file = place / "EVALUATION.json"
    if not result_file.exists():
        return {"complete": False, "condition": condition, "seed": seed, "task": task, "path": str(place)}
    lock, result = read_json(place / "INPUT_LOCK.json"), read_json(result_file)
    rows, images, _, metadata = load_data(config, task)
    expected = {**FIXED, **metadata, "condition": condition, "seed": seed, "task": task,
        "schema_version": "rarevlm.object_behavior_replay.eval_input.v1", "study_config_sha256": sha_file(config_file),
        "study_config_content_sha256": content_sha(config), "source_sha256": source_hashes(),
        "packages": package_versions(), "prompt_suffix": generation.suffix(task)}
    expected.update(recovery_input_lock_file=str(Path(recovery_lock).resolve()),
        recovery_input_lock_sha256=sha_file(Path(recovery_lock)),effective_recovery_source_sha256=recovery['recovery_source_sha256'])
    if task == "external":
        expected.update(external_admission(config))
    for key, value in expected.items():
        if lock.get(key) != value:
            raise ValueError(f"completed replay-coverage cell identity differs: {key}")
    files, binding = adapter_binding(config, condition, seed, Path(lock["adapter_dir"]),
        path(config, "model_root"), config_file=config_file,recovery_lock=recovery_lock)
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
