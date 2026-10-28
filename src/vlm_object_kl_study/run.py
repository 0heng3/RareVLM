"""Locked KL pilot → three-seed internal → new external → full POPE confirmation."""
import argparse
import fcntl
import os
from pathlib import Path
import time

from research_runtime.queue import run_queue
from vlm_joint.evaluate import model_metadata_binding
from vlm_object_kl.train import validate_config as validate_training
from vlm_object_kl_eval.analysis import analyze, validate_settings
from vlm_object_kl_eval.contracts import CANDIDATE, adapter_binding, external_admission, load_data, path
from vlm_object_kl_eval.evaluate import validate_config as validate_evaluation
from vlm_object_kl_eval.read import read_cell, verify_historical_study
from vlm_object_train.data import read_training_rows
from vlm_object_study.run import validate_quality_admission
from vlm_probe.evaluate import content_sha, lock_file, now, package_versions, read_json, sha_file, write_json
from .common import job, read_receipt, source_hashes, verify_files, verify_study

REQUIRED_PATHS = ("output_root", "asset_root", "object_data_root", "train_manifest", "val_manifest",
    "val_class_balanced_manifest", "replay_manifest", "pope_data_root", "reference_config", "reference_study_root",
    "reference_study_config", "historical_object_study_root", "historical_object_study_config", "model_root",
    "initial_adapter", "data_admission", "dataset_protocol", "study_protocol", "cpu_contracts", "external_data_root",
    "external_manifest", "external_class_balanced_manifest", "external_dataset_protocol", "evaluation_data_admission",
    "external_semantic_review", "external_frozen_audit", "external_cpu_completion")


def files(paths):
    result = {}
    for file in sorted(set(Path(item).resolve() for item in paths)):
        if not file.exists(): raise ValueError(f"required KL lock file missing: {file}")
        if file.is_file(): result[str(file)] = sha_file(file)
    return result


def prepare(config_file):
    config_file = Path(config_file).resolve()
    config = read_json(config_file)
    for name in REQUIRED_PATHS: path(config, name)
    validate_training(config); validate_evaluation(config); validate_settings(config)
    if (config.get("candidates") != [CANDIDATE] or config.get("pilot_seed") != 42
        or config.get("replication_seeds") != [0, 200] or config.get("all_seeds") != [0, 42, 200]):
        raise ValueError("one KL candidate and fixed pilot/replication axes required")
    validate_quality_admission(read_json(path(config, "data_admission")))
    external_binding = external_admission(config)
    cpu_receipt = read_json(path(config, "cpu_contracts"))
    if cpu_receipt.get("status") != "passed":
        raise ValueError("complete new CPU contract receipt required before prepare or any GPU")
    objects, replays, training_binding = read_training_rows(config)
    root = path(config, "output_root")
    root.mkdir(parents=True, exist_ok=True)
    historical_files = {}
    for prefix in ("historical_object_study", "reference_study"):
        old_root, old_config = path(config, prefix + "_root"), path(config, prefix + "_config")
        verify_historical_study(old_root, old_config)
        historical_files.update(files([old_root / "STUDY_INPUT_LOCK.json", old_config]))
    # Original full baselines are validated/read, never rewritten or rebranded.
    for task in ("object", "text", "pope"):
        cell = read_cell(root, "initial", 42, task, config_file, config)
        if not cell["complete"]: raise ValueError("historical full initializer unavailable")
        for kind in ("predictions", "evaluation", "input_lock"):
            historical_files[cell["provenance"][f"{kind}_file"]] = cell["provenance"][f"{kind}_sha256"]
    old_train = path(config, "historical_object_study_root") / "runs/object_replay/seed42/train"
    historical_files.update(files([old_train / "TRAIN_RESULT.json", old_train / "TRAINED_INPUT_CONTRACTS.json",
        old_train / "SCHEDULE.json", *old_train.glob("adapter/*")]))
    model_root = path(config, "model_root")
    model_binding = model_metadata_binding(model_root, read_json(path(config, "reference_config")))
    for name, digest in model_binding["model_weight_provenance_sha256"].items():
        if sha_file(model_root / name) != digest: raise ValueError("current base-model bytes changed")
    initial = path(config, "initial_adapter")
    _, initial_binding = adapter_binding(config, "initial", 42, initial, model_root, config_file=config_file)
    data_paths = [path(config, name) for name in ("train_manifest", "val_manifest", "val_class_balanced_manifest",
        "replay_manifest", "data_admission", "evaluation_data_admission", "cpu_contracts", "external_manifest",
        "external_class_balanced_manifest", "external_semantic_review", "external_frozen_audit", "external_cpu_completion")]
    data_paths.extend(path(config, "external_data_root").glob("*.json"))
    data_paths.append(path(config, "external_data_root") / "ENCODING_CHECKS.jsonl")
    data_paths.extend(path(config, "object_data_root").glob("*.json"))
    data_paths.extend(path(config, "pope_data_root").glob("*.json"))
    data_paths.append(path(config, "pope_data_root") / "official_evaluate.py")
    data_paths.extend(Path(value) for value in config.get("extra_lock_files", []))
    metadata = {task: load_data(config, task)[3] for task in ("object", "text", "external", "pope")}
    identity = {"schema_version": "rarevlm.object_kl.study_input.v1", "config_path": str(config_file),
        "config_file_sha256": sha_file(config_file), "config_content_sha256": content_sha(config),
        "source_sha256": source_hashes(config), "packages": package_versions(),
        "protocol_files_sha256": files([path(config, name) for name in ("study_protocol", "dataset_protocol",
            "external_dataset_protocol", "reference_config", "historical_object_study_config", "reference_study_config")]),
        "data_files_sha256": files(data_paths),
        "model_files_sha256": files([*model_root.glob("*"), model_root.parent / "model_provenance.json"]),
        "initializer_files_sha256": files([*initial.glob("*"), Path(initial_binding["path"])]),
        "historical_files_sha256": historical_files, "training_data_binding": training_binding,
        "external_data_admission": external_binding, "dataset_metadata": metadata,
        "train_counts": {"object_questions": len(objects), "replay_pool_questions": len(replays),
            "updates": (len(objects) + 15) // 16, "replay_exposures": ((len(objects) + 15) // 16) * 8},
        "teacher_cache_contract": {"initial": "fixed rounded-BF16→FP32 SFT42", "dtype": "float32",
            "full_vocabulary": True, "temperature": 1., "direction": "initial_to_actor", "manifest_frozen_before_training": True}}
    verify_files(identity["source_sha256"])
    lock_file(root / "STUDY_INPUT_LOCK.json", identity)
    lock_file(root / "STUDY.json", {"schema_version": "rarevlm.object_kl.study.v1",
        "study_input_lock_sha256": sha_file(root / "STUDY_INPUT_LOCK.json"), "candidate": CANDIDATE,
        "stages": ["standalone_KL_engineering", "pilot42_train_internal_and_initial_external_parallel",
            "pilot_internal_screen", "seed0_200_training_internal", "three_seed_internal_screen",
            "three_candidate_full_external_confirmation", "three_candidate_full_original_POPE_confirmation"],
        "initial_external_score_used_for_internal_selection": False, "external_counts_dynamic_from_full_manifest": True,
        "family": "one candidate four preregistered endpoints alpha=.05/4; not all-history FWER",
        "gpu_policy": "shared run_queue, three15sec idle/40GiB/zero compute/<10%,UUID lock and same-role resume",
        "preflight_memory_cap_gib": 65, "automatic_runtime_fallback": False,
        "goal_complete": False, "root_independent_actual_prediction_review_required": True})
    return config, root, identity


def queue_stage(root, name, jobs, identity):
    marker_sha = sha_file(root / "STUDY_INPUT_LOCK.json")
    queue_root = root / "queues" / name
    previous = read_json(queue_root / "STATUS.json") if (queue_root / "STATUS.json").exists() else {}
    for item in jobs:
        file = Path(item["completion_marker"])
        if file.exists(): read_receipt(file, marker_sha)
        elif previous.get("jobs", {}).get(item["key"], {}).get("state") == "completed":
            raise ValueError("queue marked complete but receipt missing")
    if jobs: run_queue(queue_root, jobs, identity)
    return [read_receipt(Path(item["completion_marker"]), marker_sha) for item in jobs]


def finish(root, status, phase, analyses, receipts, started):
    result = {"schema_version": "rarevlm.object_kl.trial.v1", "status": status, "trial_status": status,
        "completed_at": now(), "last_phase": phase, "goal_complete": False,
        "root_independent_prediction_review_required": True,
        "confirmation_passed": status == "confirmation_passed_pending_independent_review",
        "scientific_failure": status == "completed_negative",
        "study_input_lock_sha256": sha_file(root / "STUDY_INPUT_LOCK.json"),
        "analysis_files_sha256": {str(file): sha_file(file) for file in analyses.values()},
        "engineering_outcomes": [{key: receipt.get(key) for key in ("role", "outcome", "engineering_reason")} for receipt in receipts],
        "current_controller_session_seconds": time.monotonic() - started, "existing_tasks_untouched": True}
    write_json(root / "COMPLETION.json", result); write_json(root / "STATUS.json", result)
    if result["confirmation_passed"]: write_json(root / "TRIAL_EVIDENCE.json", result)
    return result


def run(config_file, prepare_only=False):
    config_file = Path(config_file).resolve()
    root = path(read_json(config_file), "output_root")
    root.mkdir(parents=True, exist_ok=True)
    owner = (root / "STUDY.lock").open("a+")
    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.monotonic()
    analyses, receipts = {}, []
    def status(phase):
        write_json(root / "STATUS.json", {"status": phase, "updated_at": now(), "controller_pid": os.getpid(),
            "goal_complete": False, "existing_tasks_untouched": True})
    def queue(name, jobs):
        status(name)
        latest = queue_stage(root, name, jobs, identity)
        receipts.extend(latest)
        return all(receipt["outcome"] == "passed" for receipt in latest)
    try:
        config, root, identity = prepare(config_file)
        if prepare_only: return {"status": "prepared", "goal_complete": False}
        if (root / "COMPLETION.json").exists():
            complete = read_json(root / "COMPLETION.json")
            if complete["study_input_lock_sha256"] != sha_file(root / "STUDY_INPUT_LOCK.json"):
                raise ValueError("completed KL study lock changed")
            verify_files(complete["analysis_files_sha256"])
            return complete
        if not queue("engineering_preflight", [job(config_file, root, "preflight", CANDIDATE, 42)]):
            return finish(root, "engineering_rejected", "engineering_preflight", analyses, receipts, started)
        if not queue("pilot42", [job(config_file, root, "train_eval", CANDIDATE, 42),
            job(config_file, root, "external_initial", "initial", 42)]):
            return finish(root, "engineering_rejected", "pilot42", analyses, receipts, started)
        status("pilot_internal_screen")
        pilot, pilot_file = analyze(config_file, root, [42], "screen")
        if pilot["status"] != "complete": raise ValueError("pilot queue left incomplete full internal evaluations")
        analyses["pilot"] = pilot_file
        if not pilot["gate"]["passed"]:
            return finish(root, "completed_negative", "pilot_internal_screen", analyses, receipts, started)
        if not queue("replication", [job(config_file, root, "train_eval", CANDIDATE, seed) for seed in (0, 200)]):
            return finish(root, "engineering_rejected", "replication", analyses, receipts, started)
        status("three_seed_internal_screen")
        screen, screen_file = analyze(config_file, root, [0, 42, 200], "screen", pilot_file=pilot_file)
        if screen["status"] != "complete": raise ValueError("replication queue left incomplete internal evaluations")
        analyses["internal"] = screen_file
        if not screen["gate"]["passed"]:
            return finish(root, "completed_negative", "three_seed_internal_screen", analyses, receipts, started)
        if not queue("external_confirmation", [job(config_file, root, "external", CANDIDATE, seed, screen_file) for seed in (0, 42, 200)]):
            return finish(root, "engineering_rejected", "external_confirmation", analyses, receipts, started)
        status("external_confirmation_analysis")
        external, external_file = analyze(config_file, root, [0, 42, 200], "external", screen_file=screen_file)
        if external["status"] != "complete": raise ValueError("external full four-cell evaluations incomplete")
        analyses["external"] = external_file
        if not external["gate"]["passed"]:
            return finish(root, "completed_negative", "external_confirmation", analyses, receipts, started)
        if not queue("pope_confirmation", [job(config_file, root, "pope", CANDIDATE, seed, external_file) for seed in (0, 42, 200)]):
            return finish(root, "engineering_rejected", "pope_confirmation", analyses, receipts, started)
        status("pope_confirmation_analysis")
        confirmation, file = analyze(config_file, root, [0, 42, 200], "confirm", external_file=external_file)
        if confirmation["status"] != "complete": raise ValueError("full original9000POPE evaluations incomplete")
        analyses["pope"] = file
        verify_study(root / "STUDY_INPUT_LOCK.json", config_file)
        state = "confirmation_passed_pending_independent_review" if confirmation["gate"]["passed"] else "completed_negative"
        return finish(root, state, "pope_confirmation", analyses, receipts, started)
    except BaseException as error:
        write_json(root / "STATUS.json", {"status": "execution_or_contract_failure", "updated_at": now(),
            "error_type": type(error).__name__, "error": str(error), "scientific_failure": False,
            "goal_complete": False, "existing_tasks_untouched": True})
        raise
    finally:
        owner.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    print(run(args.config, args.prepare_only)["status"], flush=True)


if __name__ == "__main__":
    main()
