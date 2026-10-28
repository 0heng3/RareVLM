"""One assigned UUID role; no source mutations or automatic runtime fallback."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import time

from vlm_joint.evaluate import visible_gpu_binding
from vlm_object_kl_eval.contracts import adapter_binding, path
from vlm_object_kl_eval.read import read_cell
from vlm_probe.evaluate import lock_file, now, read_json, sha_file, write_json
from .common import (call_reason, eval_arguments, read_receipt, role, train_arguments, verify_study, worker_dir)


class EngineeringRejected(RuntimeError):
    def __init__(self, reason_value, evidence):
        super().__init__(reason_value)
        self.reason, self.evidence = reason_value, evidence


def snapshot(file):
    if file is None or not file.exists():
        return None
    return {"sha256": sha_file(file), "mtime_ns": file.stat().st_mtime_ns}


def cache_artifacts(directory):
    return {str(file): sha_file(file) for file in directory.glob("*")
        if file.is_file() and file.suffix in (".pt", ".json")}


def call(place, name, artifacts, phase, command, preflight=None, cache_failure=None):
    log = place / "logs" / f"{phase}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    old = log.stat().st_size if log.exists() else 0
    files = {"preflight": preflight, "cache_failure": cache_failure}
    before = {kind: snapshot(file) for kind, file in files.items()}
    write_json(place / "PROGRESS.json", {"status": phase, "updated_at": now(), "role": name, "command": command})
    with log.open("a") as stream:
        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
    data = log.read_bytes()
    fresh = {}
    if completed.returncode:
        for kind, file in files.items():
            after = snapshot(file)
            if after is not None and after != before[kind]:
                fresh[kind] = {"file": str(file), "before": before[kind], "after": after}
    record = {"schema_version": "rarevlm.object_kl.call.v1", "phase": phase, "command": command,
        "returncode": completed.returncode, "completed_at": now(), "log_file": str(log),
        "log_file_sha256": sha_file(log), "log_start_byte": old, "log_end_byte": len(data),
        "log_segment_sha256": hashlib.sha256(data[old:]).hexdigest(), "fresh_failure_receipts": fresh}
    record_file = place / "attempts" / f"{phase}-{old}-{time.time_ns()}.call.json"
    write_json(record_file, record)
    evidence = {str(log): record["log_file_sha256"], str(record_file): sha_file(record_file)}
    evidence.update({item["file"]: item["after"]["sha256"] for item in fresh.values()})
    failure = call_reason(record, evidence) if completed.returncode else None
    if failure:
        raise EngineeringRejected(failure, evidence)
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)
    artifacts.update(evidence)
    if preflight is not None and preflight.exists():
        artifacts[str(preflight)] = sha_file(preflight)


def task_admission(directory, config_file, condition, seed, task, hardware):
    file = directory / "admissions" / f"{task}.json"
    lock_file(file, {"schema_version": "rarevlm.object_kl.gpu_admission.v1", "status": "admitted",
        "condition": condition, "seed": seed, "task": task, "gpu_uuid": hardware["uuid"],
        "config_sha256": sha_file(config_file)})
    return file


def run(args):
    config_file = args.config.resolve()
    config = read_json(config_file)
    root = path(config, "output_root")
    if args.output_root.resolve() != root:
        raise ValueError("KL worker output differs")
    verify_study(args.study_lock, config_file)
    name = role(args.operation, args.condition, args.seed)
    place = worker_dir(root, name)
    hardware = visible_gpu_binding()
    identity = {"schema_version": "rarevlm.object_kl.worker_input.v1", "role": name,
        "operation": args.operation, "condition": args.condition, "seed": args.seed,
        "study_input_lock_sha256": sha_file(args.study_lock), "config_sha256": sha_file(config_file),
        "hardware": hardware, "prior_phase_sha256": sha_file(args.screen_analysis or args.external_analysis) if args.screen_analysis or args.external_analysis else None}
    lock_file(place / "INPUT_LOCK.json", identity)
    if (place / "COMPLETE.json").exists():
        return read_receipt(place / "COMPLETE.json", sha_file(args.study_lock))
    artifacts = {str(place / "INPUT_LOCK.json"): sha_file(place / "INPUT_LOCK.json")}
    def evaluate(task, adapter, prior=None):
        admission = task_admission(place, config_file, args.condition, args.seed, task, hardware)
        artifacts[str(admission)] = sha_file(admission)
        call(place, name, artifacts, task, eval_arguments(config_file, root, args.condition, args.seed, task, adapter, admission, prior),
            root / "preflight" / args.condition / f"seed{args.seed}" / task / "PREFLIGHT.json")
        cell = read_cell(root, args.condition, args.seed, task, config_file, config)
        if not cell["complete"]: raise ValueError("evaluator exited without full completed cell")
        for kind in ("predictions", "evaluation", "input_lock"):
            artifacts[cell["provenance"][f"{kind}_file"]] = cell["provenance"][f"{kind}_sha256"]
    outcome, failure, evidence = "passed", None, {}
    try:
        if args.operation in ("preflight", "train_eval"):
            training_root = root / "engineering_runs" if args.operation == "preflight" else root
            train_dir = training_root / "runs/object_kl" / f"seed{args.seed}/train"
            if args.operation == "preflight" or not (train_dir / "TRAIN_RESULT.json").exists():
                call(place, name, artifacts, "engineering_preflight" if args.operation == "preflight" else "training",
                    train_arguments(config_file, training_root, args.seed, args.operation == "preflight"),
                    train_dir / "PREFLIGHT.json", train_dir / "teacher_cache/BUILD_FAILURE.json")
            artifacts.update(cache_artifacts(train_dir / "teacher_cache"))
            if args.operation == "train_eval":
                adapter = train_dir / "adapter"
                files, binding = adapter_binding(config, args.condition, args.seed, adapter, path(config, "model_root"), config_file=config_file)
                artifacts[str(train_dir / "TRAIN_RESULT.json")] = binding["sha256"]
                for file, digest in files.items(): artifacts[str(adapter / file)] = digest
                evaluate("object", adapter)
                evaluate("text", adapter)
        elif args.operation == "external_initial":
            evaluate("external", path(config, "initial_adapter"))
        elif args.operation == "external":
            evaluate("external", root / "runs/object_kl" / f"seed{args.seed}/train/adapter", args.screen_analysis)
        elif args.operation == "pope":
            evaluate("pope", root / "runs/object_kl" / f"seed{args.seed}/train/adapter", args.external_analysis)
    except EngineeringRejected as error:
        outcome, failure, evidence = "engineering_rejected", error.reason, error.evidence
    except BaseException as error:
        write_json(place / "PROGRESS.json", {"status": "execution_or_contract_failure", "error_type": type(error).__name__,
            "error": str(error), "scientific_failure": False, "updated_at": now()})
        raise
    receipt = {"schema_version": "rarevlm.object_kl.worker.v1", "status": "completed", "outcome": outcome,
        "engineering_reason": failure, "engineering_evidence_files_sha256": evidence,
        "role": name, "operation": args.operation, "condition": args.condition, "seed": args.seed,
        "completed_at": now(), "gpu_uuid": hardware["uuid"], "study_input_lock_sha256": sha_file(args.study_lock),
        "artifact_files_sha256": artifacts, "automatic_runtime_fallback": False, "existing_tasks_untouched": True}
    write_json(place / "COMPLETE.json", receipt)
    write_json(place / "PROGRESS.json", {"status": "completed", "outcome": outcome, "updated_at": now()})
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "output-root", "study-lock"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--operation", choices=("preflight", "train_eval", "external_initial", "external", "pope"), required=True)
    parser.add_argument("--condition", choices=("initial", "object_kl"), required=True)
    parser.add_argument("--seed", choices=(0, 42, 200), type=int, required=True)
    parser.add_argument("--screen-analysis", type=Path)
    parser.add_argument("--external-analysis", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
