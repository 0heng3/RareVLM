"""Complete versioned reader; only the precise verified-export timing leaf is nonidentity."""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_key]='2'
from pathlib import Path
from vlm_lora_coordinate_shrink.contracts import (CANDIDATE,CONTROL,FIXED,load_data,path,source_hashes,
    directory,validate_condition,validate_historical_request,historical_cell,adapter_binding,external_admission,
    canonical_rows,verify_source_data,generation)
from vlm_probe.evaluate import read_json,sha_file,content_sha,package_versions,processor_for,private_path
from .compat import binding_identity_equal
from .lock import verify_recovery,sources


def recovery_file(config_file):
    return path(read_json(config_file),'output_root')/'recovery_v1/RECOVERY_INPUT_LOCK.json'


def verify_context(config_file):
    return verify_recovery(recovery_file(config_file),config_file)


def recovery_fields(config_file):
    file=recovery_file(config_file);value=verify_context(config_file)
    return {'recovery_input_lock_path':str(file),'recovery_input_lock_sha256':sha_file(file),
        'recovery_source_sha256':value['recovery_source_sha256']}


def verify_recovery_fields(value,config_file):
    expected=recovery_fields(config_file)
    if any(value.get(key)!=item for key,item in expected.items()):
        raise ValueError('mandatory effective recovery source/input-lock identity differs')
    return value


def phase_admission(file,config_file,phase):
    verify_context(config_file)
    value=read_json(private_path(file));verify_recovery_fields(value,config_file)
    if (value.get('schema_version')!=f'rarevlm.lora_coordinate_shrink.{phase}.v1' or value.get('status')!='complete'
        or value.get('candidate')!=CANDIDATE or value.get('seeds')!=[0,42,200]
        or value.get('config_sha256')!=sha_file(config_file) or value.get('source_sha256')!=source_hashes()
        or value.get('gate',{}).get('passed') is not True):
        raise ValueError('unchanged full three-seed recovery phase must pass before confirmation')
    verify_source_data(value['source_data'])
    if phase=='external':
        phase_admission(value['screen_analysis_path'],config_file,'screen')
        if sha_file(value['screen_analysis_path'])!=value['screen_analysis_sha256']:raise ValueError('internal screen raw bytes changed')
    return value


COPY_PROVENANCE={'source':'src/vlm_lora_coordinate_shrink/contracts.py','sha256':'d4be2719cb741959fc0b9cbb78d056da15f6d6fc82ebc05e30f57685caaa319d','scope':'complete reader copy; exact timing leaf compatibility only'}

def read_cell(root, condition, seed, task, config_file, config, processor=None):
    if condition in ('initial',CONTROL):validate_historical_request(condition,seed,task)
    else:validate_condition(condition,seed,task)
    recovery=verify_context(config_file)
    place = directory(root, condition, seed, task)
    if condition in ("initial",CONTROL):
        return historical_cell(root,condition,task,config_file,config)
    result_file = place / "EVALUATION.json"
    if not result_file.exists():
        return {"complete": False, "condition": condition, "seed": seed, "task": task, "path": str(place)}
    lock, result = read_json(place / "INPUT_LOCK.json"), read_json(result_file)
    if not (seed==42 and task=="object"):
        verify_recovery_fields(lock,config_file);verify_recovery_fields(result,config_file)
    rows, images, _, metadata = load_data(config, task)
    expected = {**FIXED, **metadata, "condition": condition, "seed": seed, "task": task,
        "schema_version": "rarevlm.lora_coordinate_shrink.eval_input.v1", "study_config_sha256": sha_file(config_file),
        "study_config_content_sha256": content_sha(config), "source_sha256": source_hashes(),
        "packages": package_versions(), "prompt_suffix": generation.suffix(task)}
    if task == "external":
        expected.update(external_admission(config))
    for key, value in expected.items():
        if lock.get(key) != value:
            raise ValueError(f"completed replay-coverage cell identity differs: {key}")
    files, binding = adapter_binding(config, condition, seed, Path(lock["adapter_dir"]),
        path(config, "model_root"), config_file=config_file)
    if (lock.get('actual_native_bf16_tensor_sha256')!=binding['exported_bf16_tensor_sha256']
        or lock.get('actual_native_bf16_round_fp32_tensor_sha256')!=binding['exported_bf16_round_fp32_tensor_sha256']):
        raise ValueError('saved actual native inference tensor identity differs from verified export')
    if files != lock["adapter_files_sha256"] or not binding_identity_equal(lock["training_binding"],binding):
        raise ValueError("completed replay-coverage training/teacher/adapter bytes changed")
    if task in ("external", "pope") and condition != "initial":
        previous_file = Path(lock["prior_phase_analysis_file"])
        previous = phase_admission(previous_file, config_file, "screen" if task == "external" else "external")
        if sha_file(previous_file) != lock["prior_phase_analysis_sha256"]:
            raise ValueError("prior phase analysis byte binding changed")
        source_task = "text" if task == "external" else "external"
        if previous["source_data"][f"{condition}_{seed}_{source_task}"].get("adapter_export_sha256") != binding["sha256"]:
            raise ValueError("confirmation adapter differs from its passing prior phase")
    if processor is None:
        processor = processor_for(path(config, "model_root"), read_json(path(config, "reference_config")), 1024)
    predictions = place / "predictions.jsonl"
    records, repaired = generation.complete_cache(predictions, rows, images, lock, processor, repair=False)
    if (len(records) != len(rows) or repaired or result.get("status") != "completed"
        or result.get("schema_version") != "rarevlm.lora_coordinate_shrink.eval_cell.v1"
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
            "training_result_sha256": binding["producer_result_file_sha256"],
            "producer_result_file":binding["producer_result_file"],"adapter_export_sha256":binding["sha256"],
            "export_manifest_file":binding["path"],"model_artifact_role":"coordinate_shrunk_candidate"}}
