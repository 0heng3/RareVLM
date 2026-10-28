"""KL-specific source/role locks and real engineering failure evidence."""
from pathlib import Path
import hashlib
import re
import sys

from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT
from vlm_object_eval import evaluate as generation
from vlm_object_kl_eval.contracts import CANDIDATE, path, source_hashes as evaluation_sources
from vlm_object_study.common import engineering_reason as legacy_reason, memory_rejection
from vlm_probe.evaluate import lock_file, package_versions, read_json, sha_file

OPERATIONS = ("preflight", "train_eval", "external_initial", "external", "pope")


def source_hashes(config):
    result = {str(PROJECT_ROOT / name if name.startswith("tools/") else SOURCE_ROOT / name): digest
        for name, digest in evaluation_sources().items()}
    external = path(config, "external_data_root") / "SOURCE_LOCK.json"
    if external.exists():
        data_source = read_json(external)
        for name, digest in data_source["preparation_input_lock"]["code_files_sha256"].items():
            result[str(Path(name).resolve())] = digest
    return dict(sorted(result.items()))


def verify_files(files):
    for name, digest in files.items():
        if not Path(name).is_file() or sha_file(Path(name)) != digest:
            raise ValueError(f"locked KL study file changed: {name}")


def verify_study(file, config_file):
    config, lock = read_json(config_file), read_json(file)
    if (lock.get("schema_version") != "rarevlm.object_kl.study_input.v1"
        or lock.get("config_file_sha256") != sha_file(config_file)
        or lock.get("source_sha256") != source_hashes(config) or lock.get("packages") != package_versions()):
        raise ValueError("KL study source/config/package identity changed")
    for name in ("protocol_files_sha256", "data_files_sha256", "model_files_sha256", "initializer_files_sha256", "historical_files_sha256"):
        verify_files(lock[name])
    return lock


def role(operation, condition, seed):
    if operation not in OPERATIONS or seed not in (0, 42, 200):
        raise ValueError("unknown KL role or order seed")
    if (operation == "external_initial") != (condition == "initial"):
        raise ValueError("only external_initial may use the shared initial checkpoint")
    if condition not in ("initial", CANDIDATE) or (operation in ("preflight", "external_initial") and seed != 42):
        raise ValueError("fixed KL condition/pilot role differs")
    return f"{operation}_{condition}_seed{seed}"


def worker_dir(root, name):
    if not re.fullmatch(r"[a-z0-9_]+", name):
        raise ValueError("unsafe role")
    return root / "workers" / name


def job(config_file, root, operation, condition, seed, prior=None):
    name = role(operation, condition, seed)
    command = [sys.executable, "-m", "vlm_object_kl_study.worker", "--config", str(config_file),
        "--output-root", str(root), "--study-lock", str(root / "STUDY_INPUT_LOCK.json"),
        "--operation", operation, "--condition", condition, "--seed", str(seed)]
    if prior is not None:
        command.extend(["--screen-analysis" if operation == "external" else "--external-analysis", str(prior)])
    return {"key": name, "command": command, "completion_marker": str(worker_dir(root, name) / "COMPLETE.json")}


def train_arguments(config_file, root, seed, preflight=False):
    args = [sys.executable, "-m", "vlm_object_kl.train", "--config", str(config_file),
        "--condition", CANDIDATE, "--seed", str(seed), "--output-root", str(root)]
    if preflight:
        args.append("--preflight")
    elif (root / "runs" / CANDIDATE / f"seed{seed}/train/LATEST.json").exists():
        args.append("--resume")
    return args


def eval_arguments(config_file, root, condition, seed, task, adapter, admission, prior=None):
    args = [sys.executable, "-m", "vlm_object_kl_eval.evaluate", "--config", str(config_file),
        "--condition", condition, "--seed", str(seed), "--task", task, "--adapter-dir", str(adapter),
        "--output-root", str(root), "--admission", str(admission)]
    if condition == CANDIDATE and task in ("external", "pope"):
        if prior is None:
            raise ValueError("candidate confirmation needs passing prior phase")
        args.extend(["--screen-analysis" if task == "external" else "--external-analysis", str(prior)])
    return args


def reason(log, preflight=None, cache_failure=None):
    if re.search(r"RuntimeError:\s*formal_training_memory_cap_exceeded(?:\b|_)", log):
        return "formal_training_memory_cap_exceeded"
    if cache_failure:
        if memory_rejection(cache_failure):
            phase = cache_failure.get("phase", "teacher_cache")
            return "training_memory_cap_exceeded" if phase == "training" else "teacher_cache_memory_cap_exceeded"
        found = legacy_reason(log, cache_failure)
        if found:
            return found
    value = legacy_reason(log, preflight)
    if value:
        return value
    return None


def call_reason(record, evidence):
    """Reclassify only this failed invocation, never earlier append-log attempts."""
    if record.get("schema_version") != "rarevlm.object_kl.call.v1" or not record.get("returncode"):
        raise ValueError("engineering evidence must identify a failed current invocation")
    log_file = record["log_file"]
    if evidence.get(log_file) != record["log_file_sha256"]:
        raise ValueError("current invocation log is not bound to evidence")
    data = Path(log_file).read_bytes()
    start, end = record["log_start_byte"], record["log_end_byte"]
    if not 0 <= start <= end == len(data):
        raise ValueError("current invocation log byte range differs")
    segment = data[start:end]
    if hashlib.sha256(segment).hexdigest() != record["log_segment_sha256"]:
        raise ValueError("current invocation log segment changed")
    receipts = {}
    for kind, item in record["fresh_failure_receipts"].items():
        file = item["file"]
        if (kind not in ("preflight", "cache_failure") or item["before"] == item["after"]
            or item["after"] is None or evidence.get(file) != item["after"]["sha256"]):
            raise ValueError("engineering failure receipt is not fresh and bound")
        receipts[kind] = read_json(Path(file))
    return reason(segment.decode(errors="replace"), receipts.get("preflight"), receipts.get("cache_failure"))


def read_receipt(file, identity_sha):
    value = read_json(file)
    if (value.get("schema_version") != "rarevlm.object_kl.worker.v1" or value.get("status") != "completed"
        or value.get("study_input_lock_sha256") != identity_sha):
        raise ValueError("KL worker completion identity differs")
    verify_files(value["artifact_files_sha256"])
    if value["outcome"] == "engineering_rejected":
        evidence = value["engineering_evidence_files_sha256"]
        verify_files(evidence)
        attempts = [read_json(Path(name)) for name in evidence if name.endswith(".call.json")]
        if len(attempts) != 1 or call_reason(attempts[0], evidence) != value["engineering_reason"]:
            raise ValueError("engineering rejection lacks original matching evidence")
    elif value["outcome"] != "passed":
        raise ValueError("unknown KL worker outcome")
    return value
