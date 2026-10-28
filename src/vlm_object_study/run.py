"""Preregistered object adaptation: engineering, pilot, replication, one POPE arm."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import time

from research_runtime.queue import run_queue
from vlm_joint.evaluate import model_metadata_binding
from vlm_object_eval.analysis import analyze, validate_analysis_config
from vlm_object_eval.evaluate import (CANDIDATES, adapter_binding, config_path, load_data,
    read_cell, validate_config as validate_evaluation)
from vlm_object_train.data import read_training_rows
from vlm_object_train.train import validate_config as validate_training
from vlm_probe.evaluate import content_sha, lock_file, now, package_versions, private_path, read_json, sha_file, write_json
from .common import (engineering_admission, job, read_receipt, source_hashes,
    verified_files, verify_study)

REQUIRED_PATHS = ("output_root", "asset_root", "object_data_root", "train_manifest", "val_manifest",
    "val_class_balanced_manifest", "replay_manifest", "pope_data_root", "reference_config",
    "reference_study_root", "reference_study_config", "model_root", "initial_adapter",
    "data_admission", "dataset_protocol", "study_protocol", "cpu_contracts")


def validate_study_config(config):
    for key in REQUIRED_PATHS:
        value = config.get(key)
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError(f"study requires an explicit absolute path: {key}")
        private_path(value)
    if (config.get("pilot_seed") != 42 or config.get("replication_seeds") != [0, 200]
        or config.get("all_seeds") != [0, 42, 200] or config.get("candidates") != list(CANDIDATES)):
        raise ValueError("fixed two-arm pilot42 then order-seed0/200 replication required")
    if config.get("preflight_memory_cap_gib", 65) != 65:
        raise ValueError("preflight memory cap is fixed at65GiB")
    validate_training(config)
    reference = read_json(private_path(config["reference_config"]))
    validate_evaluation(config, reference)
    validate_analysis_config(config)
    if not (private_path(config["asset_root"]) / "prepared/manifest.json").is_file():
        raise ValueError("asset_root must be original vlm_deep_v2 root; evaluator appends prepared/manifest.json")
    return config


def file_map(paths):
    result = {}
    for path in sorted(set(map(private_path, paths))):
        if not path.exists():
            raise ValueError(f"required study lock file missing: {path}")
        if path.is_file():
            result[str(path)] = sha_file(path)
    return result


def validate_quality_admission(admission):
    flags = [admission[key] for key in ("quality_gate_passed", "factual_quality_gate_passed") if key in admission]
    if (admission.get("status") != "admitted" or not flags or any(flag is not True for flag in flags)
        or "diagnostic" in str(admission.get("scope", "")).lower()
        or admission.get("automatic_diagnostic_bypass_allowed", False) is not False):
        raise ValueError("factual data quality must pass; failed-data diagnostic bypass is forbidden")
    return admission


def prepare(config_file):
    config_file = private_path(config_file)
    cfg = validate_study_config(read_json(config_file))
    root = private_path(cfg["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    validate_quality_admission(read_json(private_path(cfg["data_admission"])))
    receipt = private_path(cfg["cpu_contracts"])
    if not receipt.is_file() or read_json(receipt).get("status") != "passed":
        raise ValueError("CPU_CONTRACTS receipt is required before study preparation/admission")
    objects, replays, data_binding = read_training_rows(cfg)
    object_rows, object_images, _, object_metadata = load_data(cfg, "object")
    text_rows, _, _, text_metadata = load_data(cfg, "text")
    pope_rows, _, _, pope_metadata = load_data(cfg, "pope")
    initial_files, initializer = adapter_binding(cfg, "initial", 42,
        private_path(cfg["initial_adapter"]), private_path(cfg["model_root"]))
    model_root = private_path(cfg["model_root"])
    model_binding = model_metadata_binding(model_root, read_json(private_path(cfg["reference_config"])))
    for name, expected in model_binding["model_weight_provenance_sha256"].items():
        if sha_file(model_root / name) != expected:
            raise ValueError(f"current base-model weight bytes differ: {name}")
    data_paths = [private_path(cfg[key]) for key in ("train_manifest", "val_manifest", "val_class_balanced_manifest",
        "replay_manifest", "data_admission", "cpu_contracts")]
    data_paths.extend(sorted(private_path(cfg["object_data_root"]).glob("*.json")))
    data_paths.extend(sorted(private_path(cfg["pope_data_root"]).glob("*.json")))
    data_paths.append(private_path(cfg["pope_data_root"]) / "official_evaluate.py")
    data_paths.append(private_path(cfg["reference_study_root"]) / "STUDY_INPUT_LOCK.json")
    for value in cfg.get("extra_lock_files", []):
        if not Path(value).is_absolute():
            raise ValueError("extra_lock_files must use absolute paths")
        data_paths.append(private_path(value))
    identity = {"schema_version": "rarevlm.object_presence.study_input.v1", "config_path": str(config_file),
        "config_file_sha256": sha_file(config_file), "config_content_sha256": content_sha(cfg),
        "source_sha256": source_hashes(), "packages": package_versions(),
        "protocol_files_sha256": file_map([private_path(cfg[key]) for key in ("study_protocol", "dataset_protocol", "reference_config", "reference_study_config")]),
        "data_files_sha256": file_map(data_paths),
        "model_files_sha256": file_map([*model_root.glob("*"), model_root.parent / "model_provenance.json"]),
        "initializer_files_sha256": file_map([*private_path(cfg["initial_adapter"]).glob("*"), Path(initializer["path"])]),
        "model_binding": model_binding, "initializer": initializer, "initial_adapter_files_sha256": initial_files,
        "training_data_binding": data_binding,
        "split_counts": {"object_train_questions": len(objects), "object_train_images": len({r["image_id"] for r in objects}),
            "replay_pool_questions": len(replays), "object_val_questions": len(object_rows),
            "object_val_images": len(object_images), "text_dev_questions": len(text_rows), "pope_questions": len(pope_rows)},
        "evaluation_data_bindings": {"object": object_metadata, "text": text_metadata, "pope": pope_metadata}}
    lock_file(root / "STUDY_INPUT_LOCK.json", identity)
    plan = {"schema_version": "rarevlm.object_presence.study.v1", "study_input_lock_sha256": sha_file(root / "STUDY_INPUT_LOCK.json"),
        "candidates": list(CANDIDATES), "pilot_seed": 42, "replication_seeds": [0, 200],
        "stages": ["standalone_engineering_preflights", "pilot42_full_train_object_text", "pilot_screen",
            "pilot_eligible_arms_seed0_200_full_train_object_text", "three_seed_internal_selection",
            "one_selected_candidate_three_seed_full_pope", "confirmation_analysis"],
        "historical_initial_reuse": {"text": True, "pope": True, "new_generation_questions": 0},
        "gpu_admission": "shared research_runtime.queue; three15sec idle observations; >=40GiB free,<10%util,zero compute; UUID lock; same-role resume affinity",
        "selection": "aggregate internal full-object accuracy descending, Text descending, fixed condition name ascending; never POPE",
        "memory_cap_gib": 65, "automatic_microbatch_fallback": False,
        "goal_completion": "root independent review only; controller never marks the user goal complete"}
    lock_file(root / "STUDY.json", plan)
    return cfg, root, identity


def queue_stage(root, phase, jobs, identity):
    lock_sha = sha_file(root / "STUDY_INPUT_LOCK.json")
    queue_root = root / "queues" / phase
    previous = read_json(queue_root / "STATUS.json") if (queue_root / "STATUS.json").exists() else {}
    for item in jobs:
        marker = Path(item["completion_marker"])
        if marker.exists():
            read_receipt(marker, lock_sha)
        elif previous.get("jobs", {}).get(item["key"], {}).get("state") == "completed":
            raise ValueError("queue says completed but worker receipt is missing")
    if jobs:
        run_queue(queue_root, jobs, identity)
    return [Path(item["completion_marker"]) for item in jobs]


def complete(root, status, phase, *, analyses=None, engineering=None, started=None):
    analyses = analyses or {}
    evidence = {str(path): sha_file(path) for path in analyses.values() if path is not None}
    result = {"schema_version": "rarevlm.object_presence.trial.v1", "status": status,
        "trial_status": status, "completed_at": now(), "last_phase": phase,
        "goal_complete": False, "root_independent_review_required": True,
        "confirmation_passed": status == "confirmation_passed_pending_independent_review",
        "study_input_lock_sha256": sha_file(root / "STUDY_INPUT_LOCK.json"),
        "analysis_files_sha256": evidence, "engineering_admission_files": engineering or [],
        "current_controller_session_seconds": time.monotonic() - started if started is not None else None,
        "existing_tasks_untouched": True, "scientific_failure": status == "completed_negative"}
    write_json(root / "COMPLETION.json", result)
    write_json(root / "STATUS.json", result)
    if result["confirmation_passed"]:
        write_json(root / "TRIAL_EVIDENCE.json", result)
    return result


def run(config_file, *, prepare_only=False):
    config_file = private_path(config_file)
    cfg_raw = read_json(config_file)
    root = private_path(cfg_raw["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    owner = (root / "STUDY.lock").open("a+")
    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.monotonic()
    def status(phase, **extra):
        write_json(root / "STATUS.json", {"status": phase, "updated_at": now(), "controller_pid": os.getpid(),
            "existing_tasks_untouched": True, "goal_complete": False, **extra})
    try:
        cfg, root, identity = prepare(config_file)
        status("prepared")
        if prepare_only:
            return {"status": "prepared", "study_input_lock_sha256": sha_file(root / "STUDY_INPUT_LOCK.json")}
        if (root / "COMPLETION.json").exists():
            result = read_json(root / "COMPLETION.json")
            if result.get("study_input_lock_sha256") != sha_file(root / "STUDY_INPUT_LOCK.json"):
                raise ValueError("completed trial study identity changed")
            verified_files(result["analysis_files_sha256"])
            write_json(root / "STATUS.json", result)
            return result
        status("standalone_engineering_preflights")
        preflight_receipts = queue_stage(root, "engineering_preflights",
            [job(config_file, root, "preflight", name, 42) for name in CANDIDATES], identity)
        preflight_exclusion, preflight_state = engineering_admission(root, "preflight", config_file, preflight_receipts)
        active = [name for name in CANDIDATES if name not in preflight_state["rejected_candidates"]]
        engineering_files = [str(preflight_exclusion)]
        if not active:
            return complete(root, "engineering_rejected", "standalone_engineering_preflights", engineering=engineering_files, started=started)
        status("pilot42_train_and_evaluate", active_candidates=active)
        # Read historical Text through the old validator; zero GPU and no recast run.
        reference_text = read_cell(root, "initial", 42, "text", config_file, cfg)
        if not reference_text["complete"]:
            raise ValueError("fixed historical initial Text is unavailable")
        pilot_jobs = [job(config_file, root, "train_eval", name, 42) for name in active]
        pilot_jobs.append(job(config_file, root, "object_initial", "initial", 42))
        pilot_receipts = queue_stage(root, "pilot42", pilot_jobs, identity)
        pilot_exclusion, pilot_state = engineering_admission(root, "pilot", config_file, [*preflight_receipts, *pilot_receipts])
        engineering_files.append(str(pilot_exclusion))
        if all(name in pilot_state["rejected_candidates"] for name in CANDIDATES):
            return complete(root, "engineering_rejected", "pilot42", engineering=engineering_files, started=started)
        status("pilot42_screen")
        pilot, pilot_file = analyze(config_file, root, [42], "screen", engineering_file=pilot_exclusion)
        if pilot["status"] != "complete":
            raise ValueError("pilot queue completed without all full validation predictions")
        analyses = {"pilot_screen": pilot_file}
        eligible = pilot["eligible_candidates"]
        if not eligible:
            return complete(root, "completed_negative", "pilot42_screen", analyses=analyses, engineering=engineering_files, started=started)
        status("replication_train_and_evaluate", pilot_eligible_candidates=eligible)
        replication_receipts = queue_stage(root, "replication", [job(config_file, root, "train_eval", name, seed)
            for name in eligible for seed in (0, 200)], identity)
        replication_exclusion, replication_state = engineering_admission(root, "replication", config_file,
            [*preflight_receipts, *pilot_receipts, *replication_receipts])
        engineering_files.append(str(replication_exclusion))
        if all(name in replication_state["rejected_candidates"] for name in eligible):
            return complete(root, "engineering_rejected", "replication", analyses=analyses, engineering=engineering_files, started=started)
        status("three_seed_internal_selection")
        screen, screen_file = analyze(config_file, root, [0, 42, 200], "screen", pilot_file=pilot_file,
            engineering_file=replication_exclusion)
        if screen["status"] != "complete":
            raise ValueError("replication queue completed without all full validation predictions")
        analyses["replicated_screen"] = screen_file
        selected = screen["selected_candidate"]
        if selected is None:
            return complete(root, "completed_negative", "three_seed_internal_selection", analyses=analyses, engineering=engineering_files, started=started)
        status("selected_candidate_pope_confirmation", selected_candidate=selected)
        reference_pope = read_cell(root, "initial", 42, "pope", config_file, cfg)
        if not reference_pope["complete"]:
            raise ValueError("fixed historical initial POPE is unavailable")
        pope_receipts = queue_stage(root, "pope_confirmation", [job(config_file, root, "pope", selected, seed, screen_file)
            for seed in (0, 42, 200)], identity)
        pope_exclusion, pope_state = engineering_admission(root, "pope_confirmation", config_file, pope_receipts)
        engineering_files.append(str(pope_exclusion))
        if pope_state["rejected_candidates"]:
            return complete(root, "engineering_rejected", "pope_confirmation", analyses=analyses, engineering=engineering_files, started=started)
        status("confirm_analysis", selected_candidate=selected)
        confirmation, confirmation_file = analyze(config_file, root, [0, 42, 200], "confirm", screen_file=screen_file)
        if confirmation["status"] != "complete":
            raise ValueError("POPE queue completed without all full predictions")
        analyses["confirmation"] = confirmation_file
        verify_study(root / "STUDY_INPUT_LOCK.json", config_file)
        final_status = "confirmation_passed_pending_independent_review" if confirmation["true_improvement_goal_achieved"] else "completed_negative"
        return complete(root, final_status, "confirm_analysis", analyses=analyses, engineering=engineering_files, started=started)
    except BaseException as error:
        status("execution_or_contract_failure", error_type=type(error).__name__, error=str(error),
            scientific_failure=False)
        raise
    finally:
        owner.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true", help="CPU locking/admission validation only; no queue or GPU launch")
    args = parser.parse_args()
    result = run(args.config, prepare_only=args.prepare_only)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
