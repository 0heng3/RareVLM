"""Supplemental metadata-only lock for the versioned stage33 recovery.

This module never imports the exporter, model, processor, scorer, or torch.  The
original STUDY remains immutable; supplemental recovery sources have their own
namespace.  Weight declarations are retained as provenance, never rehashed here.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET


SCHEMA = "rarevlm.lora_coordinate_shrink.recovery_input.v1"
CPU_SCHEMA = "rarevlm.lora_coordinate_shrink.recovery_cpu.v1"
REPO = Path(__file__).resolve().parents[2]
PRIVATE = Path("/opt/data/private")
ORIGINAL_STUDY_SHA256 = "283b63a86a8aa0f60892512e57dd0f9e46533a0796d7972e41b2a118e6d800af"
ARCHIVE_INDEX_SHA256 = "6fa1fcb884b9e984f974d8d80e886d59381f52185506d295114977f63672bfba"
FROZEN_CHECK_SHA256 = "01ef2ec651997d4fa288ecfc33157c5f916e67cb7efa5e713495511b7a8aa9af"
ARCHIVE_NAME = "execution_failure_archive_20261016T023011Z"
ORIGINAL_GPU_UUID = "GPU-e4083b4d-8512-2e2d-499d-8b433dd7232c"
ORIGINAL_SOURCE_COUNT = 96
SNAPSHOT_COUNT = 102
ORIGINAL_CPU_TESTS = 59
FAILURE = "completed replay-coverage training/teacher/adapter bytes changed"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _path(value):
    p = Path(value).expanduser().resolve()
    _require(p.is_relative_to(PRIVATE), "path outside persistent private workspace")
    return p


def file_sha256(path):
    p = _path(path)
    _require(p.suffix.lower() not in {".safetensors", ".pt", ".pth", ".bin"}, "weight reads forbidden in recovery lock")
    _require(p.is_file() and p.stat().st_size <= 16 * 1024 * 1024, "missing or oversized metadata/source file: " + str(p))
    h = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate JSON object key: " + key)
        result[key] = value
    return result


def _read(path):
    return json.loads(_path(path).read_text(), object_pairs_hook=_unique_object)


def canonical_runtime_sources(runtime_sources, *, project_root=None):
    """Apply the frozen runtime tools/project vs other/source path convention.

    Only the representation of keys changes.  SHA values are kept byte-for-byte,
    and no key is dropped, merged, or allowed to escape its prescribed root.
    """
    _require(isinstance(runtime_sources, dict), "runtime source map must be an object")
    project = _path(REPO if project_root is None else project_root)
    source = project / "src"
    canonical = {}
    for name, value in runtime_sources.items():
        _require(isinstance(name, str) and bool(name), "invalid runtime source key")
        _require(not Path(name).is_absolute() and "\\" not in name and all(part not in {"", ".", ".."} for part in name.split("/")), "runtime source path escape or noncanonical relative key: " + name)
        _require(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value), "invalid runtime source SHA256")
        base = project if name.startswith("tools/") else source
        path = (base / name).resolve()
        _require(path.is_relative_to(base.resolve()), "runtime source path escape: " + name)
        key = str(path)
        _require(key not in canonical, "duplicate canonical runtime source key: " + key)
        canonical[key] = value
    return canonical


def verify_runtime_sources(runtime_sources, original_sources, *, project_root=None):
    canonical = canonical_runtime_sources(runtime_sources, project_root=project_root)
    _require(isinstance(original_sources, dict) and len(original_sources) == ORIGINAL_SOURCE_COUNT and canonical == original_sources, "Object source map differs from original STUDY")
    return canonical


def _protocol():
    return REPO / "docs/VLM_LORA_COORDINATE_SHRINK_RECOVERY_V1_PROTOCOL.md"


def sources():
    """Return only supplemental source/tool/test/protocol raw-byte hashes."""
    package = REPO / "src/vlm_lora_coordinate_shrink_recovery_v1"
    paths = [package / (name + ".py") for name in ("__init__", "compat", "contracts", "analysis", "study", "lock")]
    actual = set(package.glob("*.py"))
    _require(actual == set(paths), "unexpected or missing recovery source module")
    tests = sorted((REPO / "tests").glob("test_lora_coordinate_shrink_recovery*.py"))
    _require(len(tests) == 3, "all three recovery test modules are required")
    paths += [REPO / "tools/run_lora_coordinate_shrink_recovery_v1.py", *tests, _protocol()]
    return {str(p.resolve()): file_sha256(p) for p in sorted(paths)}


def _run_inputs(config_path, config, supplemental):
    tests = [p for p in supplemental if Path(p).name.startswith("test_lora_coordinate_shrink_recovery")]
    paths = [str(config_path), str(_path(config["study_protocol"])), str(_protocol().resolve()), *tests]
    return {p: file_sha256(p) for p in sorted(set(paths))}


def _xml_count(path):
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    _require(bool(suites), "CPU XML has no test suite")
    _require(all(int(x.attrib.get("failures", 0)) == 0 and int(x.attrib.get("errors", 0)) == 0 and int(x.attrib.get("skipped", 0)) == 0 for x in suites), "CPU XML contains unsuccessful tests")
    return sum(int(x.attrib["tests"]) for x in suites)


def _no_self_reference(value, prohibited):
    if isinstance(value, dict):
        for k, v in value.items():
            _require(k not in prohibited, "CPU receipt has a self/lock reference")
            _no_self_reference(v, prohibited)
    elif isinstance(value, list):
        for item in value:
            _no_self_reference(item, prohibited)
    elif isinstance(value, str):
        _require(value not in prohibited, "CPU receipt has a self/lock reference")


def _payload(original_config, cpu_receipt):
    config_path, cpu_path = _path(original_config), _path(cpu_receipt)
    config = _read(config_path)
    _require(config["schema_version"] == "rarevlm.lora_coordinate_shrink.v1", "wrong original configuration schema")
    root = _path(config["output_root"])
    recovery = root / "recovery_v1"
    lock_path = recovery / "RECOVERY_INPUT_LOCK.json"
    files, cache = {}, {}

    def bind(path, expected=None):
        p = _path(path)
        key = str(p)
        if key not in cache:
            cache[key] = file_sha256(p)
        observed = cache[key]
        _require(expected is None or observed == expected, "frozen file bytes changed: " + key)
        files[key] = observed
        return observed

    study_path = root / "STUDY_INPUT_LOCK.json"
    bind(study_path, ORIGINAL_STUDY_SHA256)
    study = _read(study_path)
    _require(study["schema_version"] == "rarevlm.lora_coordinate_shrink.study_input.v1", "wrong original STUDY schema")
    bind(config_path, study["config_file_sha256"])
    _require(_path(study["config_path"]) == config_path, "original config path mismatch")
    original_sources = study["source_sha256"]
    _require(len(original_sources) == ORIGINAL_SOURCE_COUNT, "original scientific source count mismatch")
    for p, expected in original_sources.items():
        bind(p, expected)
    archive_root = root / "independent_review" / ARCHIVE_NAME
    index_path = archive_root / "INDEX.json"
    bind(index_path, ARCHIVE_INDEX_SHA256)
    index = _read(index_path)
    _require(index["status"] == "archived_execution_or_contract_failure" and index["goal_complete"] is False, "failure archive scope mismatch")
    _require(_path(index["source_output_root"]) == root, "archive root mismatch")
    archive_entries = {str(_path(item["original_file"])): item for item in index["files"]}
    _require(len(archive_entries) == index["evidence_file_count"], "duplicate archive original path")
    for item in archive_entries.values():
        _require(item["original_sha256"] == item["copy_sha256"] and item["original_unchanged_after_copy"] is True, "archive copy was not exact")

    def archived(relative):
        p = root / relative
        entry = archive_entries.get(str(p))
        _require(entry is not None, "required file absent from failure archive: " + str(p))
        return bind(p, entry["original_sha256"])

    frozen_path = archive_root / "FROZEN_SOURCE_CHECK.json"
    bind(frozen_path, FROZEN_CHECK_SHA256)
    frozen = _read(frozen_path)
    _require(index["frozen_source_check"]["file_sha256"] == FROZEN_CHECK_SHA256, "archive frozen-check mismatch")
    _require(frozen["all_match_source_snapshot"] is True and frozen["mismatch_count"] == 0 and frozen["checked_file_count"] == SNAPSHOT_COUNT, "original 102-file check failed")
    snapshot_path = root / "preparation/revision1/SOURCE_SNAPSHOT.json"
    archived("preparation/revision1/SOURCE_SNAPSHOT.json")
    _require(bind(snapshot_path) == frozen["source_snapshot_file_sha256"], "snapshot check anchor mismatch")
    snapshot = _read(snapshot_path)
    _require(snapshot["snapshot_file_count"] == SNAPSHOT_COUNT and len(snapshot["files_sha256"]) == SNAPSHOT_COUNT, "snapshot count mismatch")
    _require(all(snapshot["files_sha256"].get(p) == h for p, h in original_sources.items()), "original sources absent from 102 snapshot")
    for p, expected in snapshot["files_sha256"].items():
        bind(p, expected)
    _require(len(frozen["checks"]) == SNAPSHOT_COUNT and {x["file"]: x["expected_sha256"] for x in frozen["checks"]} == snapshot["files_sha256"], "frozen 102-check entries mismatch")
    _require(all(x["matches_snapshot"] is True and x["actual_sha256"] == x["expected_sha256"] for x in frozen["checks"]), "frozen check contains a mismatch")

    original_required = ["STUDY.json", "CPU_CONTRACTS.json", "cpu_contracts.xml", "PREPARATION.json", "independent_review/ROOT_PREPARATION_ACCEPTANCE.json", "independent_review/EXPORT_REVIEW.json", "preparation/revision1/CPU_EXECUTION.json", "preparation/revision1/PREPARE_EXECUTION.json", "producer_runs/lora_half32/seed42/HISTORICAL_PRODUCER_REFERENCE.json", "runs/lora_half32/seed42/shrink/EXPORT_MANIFEST.json", "LAUNCH.json", "STATUS.json", "controller.log", "queues/pilot42/STATUS.json", "queues/pilot42/logs/eval_lora_half32_seed42.log", "workers/eval_lora_half32_seed42/INPUT_LOCK.json", "workers/eval_lora_half32_seed42/PROGRESS.json"]
    for relative in original_required:
        archived(relative)
    original_cpu = _read(root / "CPU_CONTRACTS.json")
    _require(original_cpu["status"] == "passed" and original_cpu["returncode"] == 0 and original_cpu["source_sha256_before"] == original_sources == original_cpu["source_sha256_after"], "original CPU receipt mismatch")
    _require(_xml_count(root / "cpu_contracts.xml") == ORIGINAL_CPU_TESTS, "original CPU test count mismatch")
    preparation = _read(root / "PREPARATION.json")
    acceptance = _read(root / "independent_review/ROOT_PREPARATION_ACCEPTANCE.json")
    export_review = _read(root / "independent_review/EXPORT_REVIEW.json")
    _require(preparation["status"] == "passed" and acceptance["status"] == "passed_preparation_and_saved_export_only" and export_review["status"] == "passed_saved_export_formula_only", "original prepare/export acceptance missing")
    for value in (preparation, acceptance, export_review):
        _require(value["study_input_lock_sha256"] == ORIGINAL_STUDY_SHA256, "original acceptance STUDY mismatch")
    for relative in ("preparation/revision1/CPU_EXECUTION.json", "preparation/revision1/PREPARE_EXECUTION.json"):
        value = _read(root / relative)
        _require(value["status"] == "passed" and value["returncode"] == 0 and value["source_config_protocol_tests_unchanged"] is True, "real original prepare/CPU execution failed")
    # Acceptance itself is fixed by the failure archive.  Its listed scientific
    # evidence is bound explicitly above; do not recursively freeze its mutable
    # non-scientific learning snapshot or other presentation output.
    for p, expected in acceptance["helper_files_sha256"].items():
        bind(p, expected)

    producer_path = root / "producer_runs/lora_half32/seed42/HISTORICAL_PRODUCER_REFERENCE.json"
    producer = _read(producer_path)
    _require(producer["artifact_role"] == "unshrunk_producer" and producer["seed"] == 42 and producer["producer_was_shrunk"] is False, "wrong producer identity")
    _require(producer["source_binding_files_sha256"] == config["producer_reference_files_sha256"] == study["producer_reference_files_sha256"], "producer metadata chain differs")
    _require(len(producer["source_binding_files_sha256"]) == 8, "producer requires eight direct metadata proofs")
    for p, expected in producer["source_binding_files_sha256"].items():
        bind(p, expected)
    bind(producer["producer_result_file"], producer["producer_result_file_sha256"])
    export_path = root / "runs/lora_half32/seed42/shrink/EXPORT_MANIFEST.json"
    export = _read(export_path)
    _require(bind(export_path) == preparation["seed42_export_manifest_file_sha256"] == export_review["export_manifest_sha256"], "export manifest anchor mismatch")
    _require(export["source_producer_result_file"] == producer["producer_result_file"] and export["source_producer_result_file_sha256"] == producer["producer_result_file_sha256"], "export producer mismatch")
    _require(export["source_final_fp32_tensor_sha256"] == producer["expected_final_master_sha256"] and export["initial_canonical_fp32_tensor_sha256"] == producer["initial_canonical_master_sha256"], "export parameter identity mismatch")
    _require(export["alpha"] == 0.5 and export["tensor_count"] == 144 and export["parameter_elements"] == 1843200, "export scope mismatch")
    protocol = _path(config["study_protocol"])
    bind(protocol, _read(root / "STUDY.json")["protocol_sha256"])

    object_files = {str(root / "runs/lora_half32/seed42/object" / name): archived("runs/lora_half32/seed42/object/" + name) for name in ("EVALUATION.json", "INPUT_LOCK.json", "predictions.jsonl")}
    preflight_files = {str(root / "preflight/lora_half32/seed42/object" / name): archived("preflight/lora_half32/seed42/object/" + name) for name in ("PREFLIGHT.json", "INPUT_LOCK.json")}
    evaluation = _read(root / "runs/lora_half32/seed42/object/EVALUATION.json")
    object_input = _read(root / "runs/lora_half32/seed42/object/INPUT_LOCK.json")
    preflight = _read(root / "preflight/lora_half32/seed42/object/PREFLIGHT.json")
    pref_input = _read(root / "preflight/lora_half32/seed42/object/INPUT_LOCK.json")
    _require(evaluation["status"] == "completed" and evaluation["questions"] == 342 and evaluation["unique_images"] == 143 and evaluation["task"] == "object", "original Object is incomplete")
    _require(evaluation["predictions_sha256"] == files[str(root / "runs/lora_half32/seed42/object/predictions.jsonl")], "Object predictions SHA mismatch")
    verify_runtime_sources(object_input["source_sha256"], original_sources)
    _require(object_input["study_config_sha256"] == files[str(config_path)], "Object config mismatch")
    _require(preflight["status"] == "passed" and len(preflight["records"]) == 8, "original Object preflight is incomplete")
    worker_input = _read(root / "workers/eval_lora_half32_seed42/INPUT_LOCK.json")
    queue = _read(root / "queues/pilot42/STATUS.json")
    job = queue["jobs"]["eval_lora_half32_seed42"]
    uuids = [value["hardware"]["uuid"] for value in (object_input, pref_input, worker_input)] + [job["gpu_uuid"]]
    _require(all(value == ORIGINAL_GPU_UUID for value in uuids), "original eval GPU UUID mismatch")
    _require(queue["status"] == "failed" and queue["controller_pid"] == 518996 and job["pid"] == 519085 and job["state"] == "failed" and job["exit_code"] == 1, "original worker exit evidence mismatch")
    _require(index["observed_processes"] == {"518996": {"proc_exists": False}, "519085": {"proc_exists": False}}, "archive did not observe both original processes exited")
    status = _read(root / "STATUS.json")
    progress = _read(root / "workers/eval_lora_half32_seed42/PROGRESS.json")
    _require(status["status"] == progress["status"] == "execution_or_contract_failure" and progress["error"] == FAILURE and status["goal_complete"] is False, "original failure evidence changed")

    supplemental = sources()
    _require(not set(supplemental).intersection(original_sources), "recovery sources must remain separate from original STUDY")
    for p, expected in supplemental.items():
        bind(p, expected)
    receipt = _read(cpu_path)
    _no_self_reference(receipt, {str(cpu_path), str(lock_path)})
    _require(receipt["schema_version"] == CPU_SCHEMA and receipt["status"] == "passed" and receipt["returncode"] == 0, "new CPU receipt failed")
    _require(receipt["source_sha256_before"] == supplemental == receipt["source_sha256_after"], "new recovery source snapshot mismatch")
    run_inputs = _run_inputs(config_path, config, supplemental)
    _require(receipt["run_input_files_sha256_before"] == run_inputs == receipt["run_input_files_sha256_after"], "new CPU input snapshot mismatch")
    _require(all(type(receipt[k]) is int and receipt[k] == 0 for k in ("new_model_forwards", "gpu_operations", "optimizer_updates")), "CPU receipt includes scientific execution")
    _require(isinstance(receipt["files_sha256"], dict) and bool(receipt["files_sha256"]), "CPU output file hashes required")
    for p, expected in receipt["files_sha256"].items():
        bind(p, expected)
    bind(cpu_path)
    return {"schema_version": SCHEMA, "status": "prepared_supplemental_recovery_lock", "recovery_root": str(recovery), "original_config_path": str(config_path), "config_file_sha256": files[str(config_path)], "original_study_input_lock_path": str(study_path), "original_study_input_lock_sha256": ORIGINAL_STUDY_SHA256, "original_source_sha256": original_sources, "recovery_source_sha256": supplemental, "files_sha256": dict(sorted(files.items())), "original_object_files_sha256": object_files, "original_object_preflight_files_sha256": preflight_files, "original_eval_gpu_uuid": ORIGINAL_GPU_UUID, "cpu_receipt_path": str(cpu_path), "cpu_receipt_sha256": files[str(cpu_path)], "failure_archive_index_path": str(index_path), "failure_archive_index_sha256": ARCHIVE_INDEX_SHA256, "frozen_source_check_path": str(frozen_path), "frozen_source_check_sha256": FROZEN_CHECK_SHA256, "provenance": {"original_scientific_source_count": ORIGINAL_SOURCE_COUNT, "original_snapshot_file_count": SNAPSHOT_COUNT, "original_CPU_passed_tests": ORIGINAL_CPU_TESTS, "original_object_questions": 342, "original_object_preflight_questions": 8, "original_controller_pid": 518996, "original_worker_pid": 519085, "original_processes_observed_exited_in_archive": True, "weight_files_rehashed": 0, "parent_teacher_bank_read": False, "original_full_training_inputs_reencoded": 0, "fresh_Text_is_dynamic_not_in_static_lock": True}, "original_STUDY_not_modified_or_extended": True, "new_model_forwards": 0, "gpu_operations": 0, "new_scoring_calls": 0, "goal_complete": False}


def prepare_lock(original_config, cpu_receipt):
    """Create once after the source-snapshotted CPU receipt; never run export."""
    payload = _payload(original_config, cpu_receipt)
    path = Path(payload["recovery_root"]) / "RECOVERY_INPUT_LOCK.json"
    if path.exists():
        _require(_read(path) == payload, "existing recovery lock conflicts")
        return payload
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=".recovery-lock-", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            _require(_read(path) == payload, "concurrent recovery lock conflicts")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return payload


def verify_recovery(lock, original_config):
    """Recheck direct metadata and supplemental source identities, not weights."""
    stored = dict(lock) if isinstance(lock, dict) else _read(lock)
    _require(stored.get("schema_version") == SCHEMA, "wrong recovery lock schema")
    expected = _payload(original_config, stored["cpu_receipt_path"])
    _require(stored == expected, "recovery lock fields or dependencies changed")
    if not isinstance(lock, dict):
        _require(_path(lock) == Path(expected["recovery_root"]) / "RECOVERY_INPUT_LOCK.json", "recovery lock path mismatch")
    return stored
