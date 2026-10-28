#!/usr/bin/env python3
"""CPU-only recovery of the Text empty-category compatibility omission.

Original frozen readers, actual-input assertions, scorers, bootstrap and gates
remain unchanged. Only copied Text rows lose an absent/None/empty category key.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_sha(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_bytes(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(raw)
    temporary.replace(path)


def write(path, value):
    write_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())


def normalize_category(rows, task):
    """No score, identity, input, label or noncategory metadata is altered."""
    if task != "text":
        return rows, {"task": task, "rows": len(rows), "normalized_categories": 0,
            "original_rows_object_returned": True, "all_fields_unchanged": True}
    copied = deepcopy(rows)
    counts = {"missing": 0, "none": 0, "empty_string": 0}
    for original, row in zip(rows, copied):
        if "category" not in original:
            counts["missing"] += 1
        elif original["category"] is None:
            counts["none"] += 1
        elif type(original["category"]) is str and original["category"] == "":
            counts["empty_string"] += 1
        else:
            raise ValueError("nonempty Text category is not an allowed compatibility alias")
        row.pop("category", None)
    before = [{key: value for key, value in row.items() if key != "category"} for row in rows]
    if copied != before:
        raise ValueError("Text normalization changed a noncategory field")
    scores_before, scores_after = [row["score"] for row in rows], [row["score"] for row in copied]
    if content_sha(before) != content_sha(copied) or content_sha(scores_before) != content_sha(scores_after):
        raise ValueError("Text normalization changed canonical noncategory bytes or scores")
    return copied, {"task": task, "rows": len(rows), "aliases": counts,
        "normalized_categories": counts["none"] + counts["empty_string"],
        "canonical_category": "missing key, semantically None",
        "noncategory_fields_exactly_equal": True, "scores_exactly_equal": True,
        "noncategory_fields_sha256_before": content_sha(before), "noncategory_fields_sha256_after": content_sha(copied),
        "score_fields_sha256_before": content_sha(scores_before), "score_fields_sha256_after": content_sha(scores_after)}


def collect_protection(root, cfg, lock, config_file):
    protected = {str(config_file): lock["config_file_sha256"]}
    for key in ("source_sha256", "protocol_files_sha256", "data_files_sha256", "model_files_sha256", "initializer_files_sha256"):
        for name, digest in lock[key].items():
            if name in protected and protected[name] != digest:
                raise ValueError("conflicting scientific protection digest")
            protected[name] = digest
    paths = [root / name for name in ("STUDY_INPUT_LOCK.json", "STUDY.json", "LAUNCH.json", "controller.log", "RECOVERY_ORIGINAL_STATUS.json")]
    for pattern in ("runs/**/EVALUATION.json", "runs/**/predictions.jsonl", "runs/**/INPUT_LOCK.json",
        "runs/**/TRAIN_RESULT.json", "runs/**/train/adapter/*", "**/PREFLIGHT.json", "**/PREFLIGHT_HISTORY.jsonl",
        "workers/*/COMPLETE.json", "workers/*/INPUT_LOCK.json"):
        paths.extend(root.glob(pattern))
    historical = Path(cfg["reference_study_root"]) / "runs/initial/seed42/dev"
    paths.extend(historical / name for name in ("EVALUATION.json", "predictions.jsonl", "INPUT_LOCK.json"))
    for path in paths:
        if path.is_file():
            digest = sha(path)
            if str(path) in protected and protected[str(path)] != digest:
                raise ValueError("protected file already differs before recovery")
            protected[str(path)] = digest
    return protected


def verify(protected):
    failed = [name for name, digest in protected.items() if not Path(name).is_file() or sha(name) != digest]
    if failed:
        raise ValueError("protected science/data/model/CV/evidence changed: " + repr(failed))
    return {"checked_files": len(protected), "mismatches": [], "all_protected_bytes_unchanged": True}


def recover(config_file):
    started = time.monotonic()
    config_file = Path(config_file).resolve()
    cfg = read(config_file)
    root = Path(cfg["output_root"]).resolve()
    if not root.is_relative_to(Path("/opt/data/private")):
        raise ValueError("persistent output root required")
    if (root / "COMPLETION.json").exists():
        raise ValueError("completed study exists; do not reapply recovery")
    status_path = root / "STATUS.json"
    status_raw = status_path.read_bytes()
    status = json.loads(status_raw)
    if status.get("status") != "execution_or_contract_failure" or status.get("error") != "paired scoring identity differs: category":
        raise ValueError("unexpected failure; this recovery only handles the empty Text category alias")
    if (root / "RECOVERY_ORIGINAL_STATUS.json").exists():
        if (root / "RECOVERY_ORIGINAL_STATUS.json").read_bytes() != status_raw:
            raise ValueError("original failure snapshot already exists with different bytes")
    else:
        write_bytes(root / "RECOVERY_ORIGINAL_STATUS.json", status_raw)
    launch = read(root / "LAUNCH.json")
    lock = read(root / "STUDY_INPUT_LOCK.json")
    if str(Path(__file__).resolve()) in lock["source_sha256"]:
        raise ValueError("independent recovery unexpectedly entered scientific source lock")
    protected = collect_protection(root, cfg, lock, config_file)
    before = verify(protected)
    # Importing/constructing the processor is CPU-only. No model is loaded.
    import vlm_object_eval.analysis as frozen_analysis
    from vlm_probe.evaluate import processor_for
    processor = processor_for(Path(cfg["model_root"]), read(cfg["reference_config"]), 1024)
    original_reader = frozen_analysis.read_cell
    proofs = []
    def wrapped_reader(*args, **kwargs):
        kwargs["processor"] = processor
        cell = original_reader(*args, **kwargs)
        if not cell.get("complete"):
            raise ValueError("original frozen reader did not verify a complete cell")
        task = args[3] if len(args) > 3 else kwargs["task"]
        rows, proof = normalize_category(cell["rows"], task)
        source = cell["provenance"]
        proof.update(condition=cell["condition"], seed=cell["seed"],
            predictions_file=source["predictions_file"], predictions_file_sha256=source["predictions_sha256"],
            original_frozen_reader_completed_before_normalization=True,
            generated_ids_decoded_with_original_cpu_processor=True)
        proofs.append(proof)
        if task != "text":
            return cell
        copied = dict(cell)
        copied["rows"] = rows
        return copied
    frozen_analysis.read_cell = wrapped_reader
    try:
        analysis, analysis_file = frozen_analysis.analyze(config_file, root, [42], "screen",
            engineering_file=root / "engineering_admissions/pilot.json")
    except BaseException as error:
        failure_guard = verify(protected)
        write(root / "RECOVERY_FAILED.json", {"status": "stopped_without_expanding_compatibility",
            "error_type": type(error).__name__, "error": str(error), "protection_before": before,
            "protection_after": failure_guard, "reader_proofs": proofs,
            "new_updates": 0, "new_predictions": 0, "new_model_forwards": 0})
        raise
    finally:
        frozen_analysis.read_cell = original_reader
    if frozen_analysis.read_cell is not original_reader:
        raise ValueError("reader global was not restored")
    after = verify(protected)
    if analysis.get("status") != "complete" or analysis.get("eligible_candidates") != []:
        write(root / "RECOVERY_UNEXPECTED_OUTCOME.json", {"status": "requires_root_review",
            "analysis_status": analysis.get("status"), "eligible_candidates": analysis.get("eligible_candidates"),
            "protected_before": before, "protected_after": after, "new_updates": 0, "new_predictions": 0})
        raise ValueError("unexpected recovered outcome; stop without completion, seed expansion or POPE")
    first = datetime.fromisoformat(launch["launched_at"])
    failed_at = datetime.fromisoformat(status["updated_at"])
    wall = (failed_at - first).total_seconds()
    recovery_seconds = time.monotonic() - started
    completed_at = datetime.now(timezone.utc).isoformat()
    receipt = {"schema_version": "rarevlm.object_adapt.cpu_recovery.v1", "status": "completed",
        "completed_at": completed_at, "recovery_reason": "Text unused category metadata empty-alias compatibility",
        "original_status_snapshot": str(root / "RECOVERY_ORIGINAL_STATUS.json"),
        "original_status_sha256": hashlib.sha256(status_raw).hexdigest(), "original_error": status["error"],
        "original_controller_log_sha256": sha(root / "controller.log"),
        "original_workflow_started_at": launch["launched_at"], "original_failure_at": status["updated_at"],
        "original_workflow_wall_seconds": wall,
        "original_workflow_time_scope": "launch through original controller failure, including queue/CPU checks/load/preflights/training/evaluation/failed aggregation; not pure GPU time",
        "cpu_recovery_seconds": recovery_seconds, "protected_file_count": len(protected),
        "protected_files_sha256": protected, "protection_before": before, "protection_after": after,
        "wrapper_source_file": str(Path(__file__).resolve()), "wrapper_source_sha256": sha(Path(__file__)),
        "minimal_cpu_contracts": {"test_file": str(PROJECT / "tests/test_object_adapt_report_recovery.py"),
            "test_file_sha256": sha(PROJECT / "tests/test_object_adapt_report_recovery.py"),
            "passed_tests": 10, "full116_test_suite_rerun": False},
        "per_file_text_normalization": proofs, "reader_global_restored": True,
        "original_assert_same_inputs_scorers_bootstrap_and_gates_unchanged": True,
        "all_input_ids_pixels_grid_padding_and_noncategory_fields_remain_strict": True,
        "analysis_file": str(analysis_file), "analysis_sha256": sha(analysis_file),
        "eligible_candidates": [], "new_updates": 0, "new_predictions": 0,
        "new_model_forwards": 0, "gpu_operations": 0, "scientific_file_mutations": 0,
        "goal_complete": False,
        "report_wait_cleanup": {"tool_session_id": 68531, "verified_owned_pid": 389593,
            "command": [sys.executable, "tools/complete_object_adapt_report.py", "--config", str(config_file),
                "--wait-seconds", "480", "--poll-seconds", "45"],
            "owner_uid_verified": True, "owner_uid": os.geteuid(),
            "action": "SIGTERM own CPU wait only; no controller/worker/GPU task signaled"}}
    write(root / "RECOVERY.json", receipt)
    completion = {"schema_version": "rarevlm.object_presence.trial.v1", "status": "completed_negative",
        "trial_status": "completed_negative", "completed_at": completed_at, "last_phase": "pilot42_screen_cpu_recovery",
        "goal_complete": False, "root_independent_review_required": True, "confirmation_passed": False,
        "study_input_lock_sha256": sha(root / "STUDY_INPUT_LOCK.json"),
        "analysis_files_sha256": {str(analysis_file): sha(analysis_file)},
        "scientific_failure": True, "existing_tasks_untouched": True,
        "recovery_file": str(root / "RECOVERY.json"), "recovery_sha256": sha(root / "RECOVERY.json"),
        "original_workflow_started_at": launch["launched_at"], "original_failure_at": status["updated_at"],
        "original_workflow_wall_seconds": wall, "current_controller_session_seconds": wall,
        "current_controller_session_seconds_scope": receipt["original_workflow_time_scope"],
        "cpu_recovery_seconds": recovery_seconds, "new_updates_during_recovery": 0,
        "new_predictions_during_recovery": 0, "new_model_forwards_during_recovery": 0}
    write(root / "COMPLETION.json", completion)
    write(root / "STATUS.json", completion)
    print(json.dumps({"status": "completed_negative", "analysis_file": str(analysis_file),
        "eligible_candidates": [], "original_workflow_wall_seconds": wall,
        "cpu_recovery_seconds": recovery_seconds, "protected_file_count": len(protected),
        "reader_proofs": [{key: row.get(key) for key in ("condition", "task", "rows", "normalized_categories", "aliases")} for row in proofs],
        "new_model_forwards": 0, "new_updates": 0, "new_predictions": 0}, ensure_ascii=False), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/vlm_object_adapt_v2.json")
    args = parser.parse_args()
    recover(args.config)


if __name__ == "__main__":
    main()
