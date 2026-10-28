"""Stable study/job identities and evidence-backed engineering rejection."""
from __future__ import annotations

import math
from pathlib import Path
import re
import sys

from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT
from vlm_object_eval.evaluate import CANDIDATES, source_hashes as eval_sources
from vlm_probe.evaluate import content_sha, lock_file, package_versions, private_path, read_json, sha_file

MEMORY_CAP_GIB = 65.
OOM = re.compile(r"CUDA out of memory|CUDA error: out of memory|torch\.OutOfMemoryError|CUDAOutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED", re.IGNORECASE)
OPERATIONS = ("preflight", "train_eval", "object_initial", "pope")


def source_hashes():
    from vlm_object_train.train import source_hashes as training_sources
    evaluation = eval_sources()
    result = {str(SOURCE_ROOT / key): value for key, value in evaluation.items()
        if not key.startswith("tools/")}
    result.update({str(SOURCE_ROOT / key): value for key, value in training_sources().items()})
    for key, value in evaluation.items():
        if key.startswith("tools/"):
            result[str(PROJECT_ROOT / key)] = value
    for folder in ("vlm_object_train", "vlm_object_presence", "vlm_object_presence_v2", "vlm_object_study"):
        for path in sorted((SOURCE_ROOT / folder).glob("*.py")):
            result[str(path)] = sha_file(path)
    for name in ("research_runtime/queue.py", "vlm_joint/pipeline.py", "vlm_precision/pipeline.py"):
        result[str(SOURCE_ROOT / name)] = sha_file(SOURCE_ROOT / name)
    path = PROJECT_ROOT / "tools/run_object_adapt_study.py"
    result[str(path)] = sha_file(path)
    return dict(sorted(result.items()))


def verified_files(files):
    for name, digest in files.items():
        path = private_path(name)
        if not path.is_file() or sha_file(path) != digest:
            raise ValueError(f"locked study file changed: {path}")


def verify_study(lock_path, config_file):
    lock = read_json(private_path(lock_path))
    if (lock.get("schema_version") != "rarevlm.object_presence.study_input.v1"
        or lock.get("config_file_sha256") != sha_file(config_file)
        or lock.get("source_sha256") != source_hashes() or lock.get("packages") != package_versions()):
        raise ValueError("study source/config identity changed")
    for key in ("protocol_files_sha256", "data_files_sha256", "model_files_sha256", "initializer_files_sha256"):
        verified_files(lock[key])
    return lock


def role_key(operation, condition, seed):
    if operation not in OPERATIONS or condition not in ("initial", *CANDIDATES) or seed not in (0, 42, 200):
        raise ValueError("unknown operation/condition/order seed")
    if (operation == "object_initial") != (condition == "initial"):
        raise ValueError("only the initial object role may use initial condition")
    if operation in ("preflight", "object_initial") and seed != 42:
        raise ValueError("engineering/initial roles use fixed seed42")
    return f"{operation}_{condition}_seed{seed}"


def worker_directory(root, role):
    if not re.fullmatch(r"[a-z0-9_]+", role):
        raise ValueError("unsafe worker role key")
    return root / "workers" / role


def job(config_file, root, operation, condition, seed, screen_file=None):
    role = role_key(operation, condition, seed)
    command = [sys.executable, "-m", "vlm_object_study.worker", "--config", str(config_file),
        "--output-root", str(root), "--operation", operation, "--condition", condition,
        "--seed", str(seed), "--study-lock", str(root / "STUDY_INPUT_LOCK.json")]
    if screen_file is not None:
        command.extend(["--screen-analysis", str(screen_file)])
    return {"key": role, "command": command,
        "completion_marker": str(worker_directory(root, role) / "COMPLETE.json")}


def train_arguments(config_file, root, condition, seed, *, preflight=False):
    command = [sys.executable, "-m", "vlm_object_train.train", "--config", str(config_file),
        "--condition", condition, "--seed", str(seed), "--output-root", str(root)]
    if preflight:
        command.append("--preflight")
    elif (root / "runs" / condition / f"seed{seed}" / "train/LATEST.json").exists():
        command.append("--resume")
    return command


def eval_arguments(config_file, root, condition, seed, task, adapter, admission, screen_file=None):
    command = [sys.executable, "-m", "vlm_object_eval.evaluate", "--config", str(config_file),
        "--condition", condition, "--seed", str(seed), "--task", task,
        "--adapter-dir", str(adapter), "--output-root", str(root), "--admission", str(admission)]
    if task == "pope":
        if screen_file is None:
            raise ValueError("POPE worker requires frozen three-seed internal selection")
        command.extend(["--screen-analysis", str(screen_file)])
    return command


def memory_rejection(result):
    values = []
    for key in ("peak_allocated_gib", "peak_reserved_gib"):
        value = result.get(key)
        if value is not None:
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("invalid engineering memory observation")
            values.append(value)
    for batch in result.get("batches", []):
        for key in ("batch_peak_allocated_gib", "batch_peak_reserved_gib"):
            value = batch.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("invalid preflight batch memory observation")
            values.append(value)
    return "preflight_memory_cap_exceeded" if values and max(values) > MEMORY_CAP_GIB else None


def engineering_reason(log_text, preflight=None):
    if preflight:
        cap = memory_rejection(preflight)
        if cap:
            return cap
        if "preflight_memory_cap_exceeded" in str(preflight.get("error", "")):
            return "preflight_memory_cap_exceeded"
    return "cuda_out_of_memory" if OOM.search(log_text) else None


def read_receipt(path, identity_sha=None):
    value = read_json(private_path(path))
    if value.get("schema_version") != "rarevlm.object_presence.worker.v1" or value.get("status") != "completed":
        raise ValueError("worker completion schema/status differs")
    if identity_sha is not None and value.get("study_input_lock_sha256") != identity_sha:
        raise ValueError("worker study identity differs")
    verified_files(value.get("artifact_files_sha256", {}))
    if value.get("outcome") == "engineering_rejected":
        if value.get("engineering_reason") not in ("cuda_out_of_memory", "preflight_memory_cap_exceeded"):
            raise ValueError("engineering rejection requires a real CUDA OOM or 65GiB cap")
        evidence = value.get("engineering_evidence_files_sha256", {})
        if not evidence:
            raise ValueError("engineering rejection has no original evidence")
        verified_files(evidence)
        observed = set()
        logs = "\n".join(Path(name).read_text(errors="replace") for name in evidence if Path(name).suffix == ".log")
        reason = engineering_reason(logs)
        if reason:
            observed.add(reason)
        for name in evidence:
            if Path(name).name == "PREFLIGHT.json":
                reason = engineering_reason(logs, read_json(Path(name)))
                if reason:
                    observed.add(reason)
        if value["engineering_reason"] not in observed:
            raise ValueError("engineering rejection reason is not present in original OOM/cap evidence")
    elif value.get("outcome") != "passed":
        raise ValueError("unknown worker outcome")
    return value


def engineering_admission(root, phase, config_file, receipts):
    rejected = {}
    for path in receipts:
        value = read_receipt(path, sha_file(root / "STUDY_INPUT_LOCK.json"))
        if value["outcome"] != "engineering_rejected":
            continue
        condition = value["condition"]
        if condition == "initial":
            raise RuntimeError("initial object evaluation engineering failure; no scientific comparison available")
        rejected.setdefault(condition, []).append({"seed": value["seed"], "operation": value["operation"],
            "reason": value["engineering_reason"], "worker_receipt": str(path),
            "worker_receipt_sha256": sha_file(path),
            "evidence_files_sha256": value["engineering_evidence_files_sha256"]})
    result = {"schema_version": "rarevlm.object_presence.engineering_admission.v1",
        "config_sha256": sha_file(config_file), "phase": phase,
        "rejected_candidates": rejected, "engineering_only": True,
        "alpha_unchanged": .0125, "automatic_microbatch_fallback": False}
    path = root / "engineering_admissions" / f"{phase}.json"
    lock_file(path, result)
    return path, result
