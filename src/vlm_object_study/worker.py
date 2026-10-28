"""One externally admitted role; preserve UUID affinity and complete artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from vlm_joint.evaluate import visible_gpu_binding
from vlm_object_eval.evaluate import adapter_binding, config_path, read_cell
from vlm_probe.evaluate import lock_file, now, private_path, read_json, sha_file, write_json
from .common import (engineering_reason, eval_arguments, read_receipt, role_key,
    train_arguments, verify_study, worker_directory)


class EngineeringRejected(RuntimeError):
    def __init__(self, reason, evidence):
        super().__init__(reason)
        self.reason, self.evidence = reason, evidence


def task_admission(directory, config_file, condition, seed, task, hardware):
    path = directory / "admissions" / f"{task}.json"
    lock_file(path, {"schema_version": "rarevlm.object_presence.gpu_admission.v1", "status": "admitted",
        "condition": condition, "seed": seed, "task": task, "gpu_uuid": hardware["uuid"],
        "config_sha256": sha_file(config_file)})
    return path


def run(args):
    config_file, root, lock_path = map(private_path, (args.config, args.output_root, args.study_lock))
    cfg = read_json(config_file)
    if root != private_path(cfg["output_root"]):
        raise ValueError("worker output root differs from locked config")
    verify_study(lock_path, config_file)
    role = role_key(args.operation, args.condition, args.seed)
    directory = worker_directory(root, role)
    hardware = visible_gpu_binding()
    identity = {"schema_version": "rarevlm.object_presence.worker_input.v1", "role": role,
        "operation": args.operation, "condition": args.condition, "seed": args.seed,
        "config_sha256": sha_file(config_file), "study_input_lock_sha256": sha_file(lock_path),
        "gpu_uuid": hardware["uuid"], "hardware": hardware,
        "screen_analysis_sha256": sha_file(args.screen_analysis) if args.screen_analysis else None}
    lock_file(directory / "INPUT_LOCK.json", identity)
    complete = directory / "COMPLETE.json"
    if complete.exists():
        saved = read_receipt(complete, sha_file(lock_path))
        if saved.get("worker_input_lock_sha256") != sha_file(directory / "INPUT_LOCK.json"):
            raise ValueError("completed worker UUID/input lock changed")
        return saved
    artifacts = {str(directory / "INPUT_LOCK.json"): sha_file(directory / "INPUT_LOCK.json")}
    logs, history = [], []
    def call(phase, command, preflight_file=None):
        log = directory / "logs" / f"{phase}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        logs.append(log)
        write_json(directory / "PROGRESS.json", {"status": phase, "updated_at": now(),
            "role": role, "gpu_uuid": hardware["uuid"], "command": command, "log": str(log)})
        print("WORKER_STAGE", role, phase, flush=True)
        old_size = log.stat().st_size if log.exists() else 0
        with log.open("a") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
        preflight = read_json(preflight_file) if preflight_file is not None and preflight_file.exists() else None
        text = log.read_bytes()[old_size:].decode(errors="replace")
        reason = engineering_reason(text, preflight)
        if reason:
            evidence = {str(log): sha_file(log)}
            if preflight_file is not None and preflight_file.exists():
                evidence[str(preflight_file)] = sha_file(preflight_file)
            raise EngineeringRejected(reason, evidence)
        if result.returncode:
            raise subprocess.CalledProcessError(result.returncode, command)
        history.append({"phase": phase, "log": str(log), "log_sha256": sha_file(log)})
        artifacts[str(log)] = sha_file(log)
        if preflight_file is not None and preflight_file.exists():
            artifacts[str(preflight_file)] = sha_file(preflight_file)
    def evaluate(task, adapter):
        admission = task_admission(directory, config_file, args.condition, args.seed, task, hardware)
        artifacts[str(admission)] = sha_file(admission)
        preflight = root / "preflight" / args.condition / f"seed{args.seed}" / task / "PREFLIGHT.json"
        call(task, eval_arguments(config_file, root, args.condition, args.seed, task, adapter, admission,
            args.screen_analysis), preflight)
        cell = read_cell(root, args.condition, args.seed, task, config_file, cfg)
        if not cell["complete"]:
            raise ValueError("successful evaluator did not write a complete cell")
        for kind in ("predictions", "evaluation", "input_lock"):
            artifacts[cell["provenance"][f"{kind}_file"]] = cell["provenance"][f"{kind}_sha256"]
    outcome, rejection, rejection_evidence = "passed", None, {}
    try:
        if args.operation in ("preflight", "train_eval"):
            training_root = root / "engineering_runs" if args.operation == "preflight" else root
            training_dir = training_root / "runs" / args.condition / f"seed{args.seed}" / "train"
            train_result = training_dir / "TRAIN_RESULT.json"
            if args.operation == "preflight" or not train_result.exists():
                call("standalone_preflight" if args.operation == "preflight" else "training",
                    train_arguments(config_file, training_root, args.condition, args.seed,
                        preflight=args.operation == "preflight"), training_dir / "PREFLIGHT.json")
            if args.operation == "train_eval":
                adapter = training_dir / "adapter"
                files, binding = adapter_binding(cfg, args.condition, args.seed, adapter, config_path(cfg, "model_root"))
                artifacts[str(train_result)] = binding["sha256"]
                for name, digest in files.items():
                    artifacts[str(adapter / name)] = digest
                evaluate("object", adapter)
                evaluate("text", adapter)
        elif args.operation == "object_initial":
            evaluate("object", config_path(cfg, "initial_adapter"))
        elif args.operation == "pope":
            adapter = root / "runs" / args.condition / f"seed{args.seed}" / "train/adapter"
            evaluate("pope", adapter)
    except EngineeringRejected as error:
        outcome, rejection, rejection_evidence = "engineering_rejected", error.reason, error.evidence
    except BaseException as error:
        write_json(directory / "PROGRESS.json", {"status": "failed", "classification": "execution_or_contract_failure",
            "updated_at": now(), "role": role, "error_type": type(error).__name__, "error": str(error),
            "original_logs": [str(path) for path in logs], "existing_tasks_untouched": True})
        raise
    result = {"schema_version": "rarevlm.object_presence.worker.v1", "status": "completed",
        "outcome": outcome, "engineering_reason": rejection,
        "engineering_evidence_files_sha256": rejection_evidence, "operation": args.operation,
        "condition": args.condition, "seed": args.seed, "role": role, "completed_at": now(),
        "gpu_uuid": hardware["uuid"], "study_input_lock_sha256": sha_file(lock_path),
        "worker_input_lock_sha256": sha_file(directory / "INPUT_LOCK.json"),
        "artifact_files_sha256": artifacts, "completed_calls": history,
        "automatic_microbatch_fallback": False, "existing_tasks_untouched": True}
    write_json(complete, result)
    write_json(directory / "PROGRESS.json", {"status": "completed", "outcome": outcome,
        "engineering_reason": rejection, "updated_at": now(), "role": role})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "output-root", "study-lock"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--operation", choices=("preflight", "train_eval", "object_initial", "pope"), required=True)
    parser.add_argument("--condition", choices=("initial", "object_only", "object_replay"), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 42, 200), required=True)
    parser.add_argument("--screen-analysis", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
