"""Strict new cells and explicit original-byte historical initializer references."""
from pathlib import Path

from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT
from vlm_joint.evaluate import model_metadata_binding
from vlm_object_eval import evaluate as generation
from vlm_posttrain_eval.analysis import read_cell as original_text_cell
from vlm_posttrain_pope.evaluate import read_cell as original_pope_cell
from vlm_probe.evaluate import content_sha, lock_file, package_versions, processor_for, read_json, sha_file
from .contracts import (FIXED, adapter_binding, canonical_rows, external_admission, load_data, path, source_hashes)


def verify_files(files):
    for name, expected in files.items():
        if sha_file(Path(name)) != expected:
            raise ValueError(f"bound historical/input file changed: {name}")


def verify_historical_study(root, config_file):
    lock = read_json(root / "STUDY_INPUT_LOCK.json")
    digest = lock.get("config_file_sha256", lock.get("config_sha256"))
    if digest != sha_file(config_file):
        raise ValueError("historical study config changed")
    if "source_sha256" in lock:
        verify_files(lock["source_sha256"])
        for key in ("protocol_files_sha256", "data_files_sha256", "model_files_sha256", "initializer_files_sha256"):
            verify_files(lock[key])
    else:
        for name, expected in lock["sources"].items():
            file = PROJECT_ROOT / name if name.startswith("tools/") else SOURCE_ROOT / name
            if sha_file(file) != expected:
                raise ValueError("historical public diagnostic source changed")
        verify_files(lock["data_sha256"])
    return lock


def directory(root, condition, seed, task):
    return root / "runs" / condition / f"seed{seed}" / task


def historical_initial(root, task, config_file, config, processor=None):
    if task == "external":
        raise ValueError("new external initial must be generated, never historical")
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
    declaration = {"schema_version": "rarevlm.object_kl.historical_reference.v1", "condition": "initial",
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
    if condition == "initial" and task != "external":
        return historical_initial(root, task, config_file, config, processor)
    result_file = place / "EVALUATION.json"
    if not result_file.exists():
        return {"complete": False, "condition": condition, "seed": seed, "task": task, "path": str(place)}
    lock, result = read_json(place / "INPUT_LOCK.json"), read_json(result_file)
    rows, images, _, metadata = load_data(config, task)
    expected = {**FIXED, **metadata, "condition": condition, "seed": seed, "task": task,
        "schema_version": "rarevlm.object_kl.eval_input.v1", "study_config_sha256": sha_file(config_file),
        "study_config_content_sha256": content_sha(config), "source_sha256": source_hashes(),
        "packages": package_versions(), "prompt_suffix": generation.suffix(task)}
    if task == "external":
        expected.update(external_admission(config))
    for key, value in expected.items():
        if lock.get(key) != value:
            raise ValueError(f"completed KL cell identity differs: {key}")
    files, binding = adapter_binding(config, condition, seed, Path(lock["adapter_dir"]),
        path(config, "model_root"), config_file=config_file)
    if files != lock["adapter_files_sha256"] or binding != lock["training_binding"]:
        raise ValueError("completed KL training/teacher/adapter bytes changed")
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
        or result.get("schema_version") != "rarevlm.object_kl.eval_cell.v1"
        or result.get("input_lock_sha256") != content_sha(lock)
        or result.get("predictions_sha256") != sha_file(predictions)
        or any(result.get(key) != value for key, value in generation.summary(records, task).items())):
        raise ValueError("completed KL full-task score/count/hash differs")
    canonical, proof = canonical_rows(records, task)
    return {"complete": True, "condition": condition, "seed": seed, "task": task, "rows": canonical,
        "lock": lock, "evaluation": result, "canonicalization": proof,
        "provenance": {"predictions_file": str(predictions), "predictions_sha256": sha_file(predictions),
            "evaluation_file": str(result_file), "evaluation_sha256": sha_file(result_file),
            "input_lock_file": str(place / "INPUT_LOCK.json"), "input_lock_sha256": sha_file(place / "INPUT_LOCK.json"),
            "training_result_sha256": binding["sha256"]}}


def verify_source_data(sources):
    if not sources:
        raise ValueError("phase analysis source_data is missing")
    for source in sources.values():
        for kind in ("predictions", "evaluation", "input_lock"):
            if sha_file(Path(source[f"{kind}_file"])) != source[f"{kind}_sha256"]:
                raise ValueError("prior-phase prediction/evaluation/input bytes changed")


def phase_admission(file, config_file, phase):
    file = Path(file).resolve()
    value = read_json(file)
    if (value.get("schema_version") != f"rarevlm.object_kl.{phase}.v1" or value.get("status") != "complete"
        or value.get("config_sha256") != sha_file(config_file) or value.get("source_sha256") != source_hashes()
        or value.get("seeds") != [0, 42, 200] or value.get("gate", {}).get("passed") is not True):
        raise ValueError(f"{phase} phase must pass full three-seed criteria before progression")
    verify_source_data(value["source_data"])
    if phase == "external":
        screen = phase_admission(value["screen_analysis_path"], config_file, "screen")
        if sha_file(Path(value["screen_analysis_path"])) != value["screen_analysis_sha256"]:
            raise ValueError("internal screen changed after external confirmation")
    return value
