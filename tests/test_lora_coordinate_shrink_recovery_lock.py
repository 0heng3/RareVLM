"""Synthetic metadata-only contracts: no real prepare, weights, or scoring."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from vlm_lora_coordinate_shrink_recovery_v1 import lock


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RecoveryLockTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="synthetic-recovery-lock-", dir="/opt/data/private")
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / "outputs"
        self.repo = self.base / "repo"

        def write(relative, value):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value, indent=2) + "\n" if not isinstance(value, str) else value)
            return path

        self.write = write
        self.original_protocol = self.repo / "docs/original.md"
        self.new_protocol = self.repo / "docs/VLM_LORA_COORDINATE_SHRINK_RECOVERY_V1_PROTOCOL.md"
        self.new_protocol.parent.mkdir(parents=True)
        self.original_protocol.write_text("original fixed protocol\n")
        self.new_protocol.write_text("supplemental fixed protocol\n")
        original_source = {}
        for i in range(2):
            p = self.repo / "src" / ("original_%d.py" % i)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("VALUE = %d\n" % i)
            original_source[str(p)] = digest_file(p)
        self.supplemental_paths = [self.repo / "recovery.py", self.repo / "test_lora_coordinate_shrink_recovery_lock.py", self.new_protocol]
        self.supplemental_paths[0].write_text("VALUE = 'recovery'\n")
        self.supplemental_paths[1].write_text("# synthetic CPU test module\n")
        chain = {}
        for i in range(8):
            p = write("proofs/proof%d.json" % i, {"synthetic_proof": i})
            chain[str(p)] = digest_file(p)
        result = Path(next(iter(chain)))
        self.config = self.repo / "original_config.json"
        self.config.write_text(json.dumps({"schema_version": "rarevlm.lora_coordinate_shrink.v1", "output_root": str(self.root), "study_protocol": str(self.original_protocol), "producer_reference_files_sha256": chain}))
        original_study = write("STUDY_INPUT_LOCK.json", {"schema_version": "rarevlm.lora_coordinate_shrink.study_input.v1", "config_path": str(self.config), "config_file_sha256": digest_file(self.config), "source_sha256": original_source, "producer_reference_files_sha256": chain})
        study_hash = digest_file(original_study)
        write("STUDY.json", {"protocol_sha256": digest_file(self.original_protocol)})
        write("CPU_CONTRACTS.json", {"status": "passed", "returncode": 0, "source_sha256_before": original_source, "source_sha256_after": original_source})
        write("cpu_contracts.xml", '<testsuites><testsuite tests="59" failures="0" errors="0" skipped="0"/></testsuites>')
        for relative in ("preparation/revision1/CPU_EXECUTION.json", "preparation/revision1/PREPARE_EXECUTION.json"):
            write(relative, {"status": "passed", "returncode": 0, "source_config_protocol_tests_unchanged": True})
        producer = {"artifact_role": "unshrunk_producer", "seed": 42, "producer_was_shrunk": False, "source_binding_files_sha256": chain, "producer_result_file": str(result), "producer_result_file_sha256": digest_file(result), "expected_final_master_sha256": "final", "initial_canonical_master_sha256": "initial"}
        write("producer_runs/lora_half32/seed42/HISTORICAL_PRODUCER_REFERENCE.json", producer)
        export = write("runs/lora_half32/seed42/shrink/EXPORT_MANIFEST.json", {"source_producer_result_file": str(result), "source_producer_result_file_sha256": digest_file(result), "source_final_fp32_tensor_sha256": "final", "initial_canonical_fp32_tensor_sha256": "initial", "alpha": .5, "tensor_count": 144, "parameter_elements": 1843200})
        write("PREPARATION.json", {"status": "passed", "study_input_lock_sha256": study_hash, "seed42_export_manifest_file_sha256": digest_file(export)})
        write("independent_review/ROOT_PREPARATION_ACCEPTANCE.json", {"status": "passed_preparation_and_saved_export_only", "study_input_lock_sha256": study_hash, "files_sha256": {}, "helper_files_sha256": {}})
        write("independent_review/EXPORT_REVIEW.json", {"status": "passed_saved_export_formula_only", "study_input_lock_sha256": study_hash, "export_manifest_sha256": digest_file(export)})
        snapshot_files = dict(original_source)
        snapshot_files[str(self.config)] = digest_file(self.config)
        snapshot = write("preparation/revision1/SOURCE_SNAPSHOT.json", {"snapshot_file_count": 3, "files_sha256": snapshot_files})
        write("LAUNCH.json", {"pid": 518996, "start_ticks": 422758130})
        write("STATUS.json", {"status": "execution_or_contract_failure", "goal_complete": False})
        write("controller.log", "Traceback: original controller failed\n")
        job = {"pid": 519085, "state": "failed", "exit_code": 1, "gpu_uuid": lock.ORIGINAL_GPU_UUID}
        write("queues/pilot42/STATUS.json", {"status": "failed", "controller_pid": 518996, "jobs": {"eval_lora_half32_seed42": job}})
        write("queues/pilot42/logs/eval_lora_half32_seed42.log", "Traceback: original worker failed\n")
        hardware = {"uuid": lock.ORIGINAL_GPU_UUID}
        write("workers/eval_lora_half32_seed42/INPUT_LOCK.json", {"hardware": hardware})
        write("workers/eval_lora_half32_seed42/PROGRESS.json", {"status": "execution_or_contract_failure", "error": lock.FAILURE})
        self.predictions = write("runs/lora_half32/seed42/object/predictions.jsonl", '{"synthetic_only": true}\n')
        write("runs/lora_half32/seed42/object/EVALUATION.json", {"status": "completed", "questions": 342, "unique_images": 143, "task": "object", "predictions_sha256": digest_file(self.predictions)})
        runtime_source = {str(Path(p).relative_to(self.repo / "src")): h for p, h in original_source.items()}
        write("runs/lora_half32/seed42/object/INPUT_LOCK.json", {"source_sha256": runtime_source, "study_config_sha256": digest_file(self.config), "hardware": hardware})
        write("preflight/lora_half32/seed42/object/PREFLIGHT.json", {"status": "passed", "records": [{} for _ in range(8)]})
        write("preflight/lora_half32/seed42/object/INPUT_LOCK.json", {"hardware": hardware})
        archive = self.root / "independent_review" / lock.ARCHIVE_NAME
        frozen = {"all_match_source_snapshot": True, "mismatch_count": 0, "checked_file_count": 3, "source_snapshot_file_sha256": digest_file(snapshot), "checks": [{"file": p, "expected_sha256": h, "actual_sha256": h, "matches_snapshot": True} for p, h in snapshot_files.items()]}
        frozen_path = write(str(archive.relative_to(self.root) / "FROZEN_SOURCE_CHECK.json"), frozen)
        original_paths = [p for p in self.root.rglob("*") if p.is_file() and not p.is_relative_to(archive)]
        entries = [{"original_file": str(p), "archive_file": str(archive / "evidence" / p.relative_to(self.root)), "original_sha256": digest_file(p), "copy_sha256": digest_file(p), "original_unchanged_after_copy": True} for p in original_paths]
        index = {"status": "archived_execution_or_contract_failure", "goal_complete": False, "source_output_root": str(self.root), "files": entries, "evidence_file_count": len(entries), "frozen_source_check": {"file_sha256": digest_file(frozen_path)}, "observed_processes": {"518996": {"proc_exists": False}, "519085": {"proc_exists": False}}}
        index_path = write(str(archive.relative_to(self.root) / "INDEX.json"), index)
        self.cpu = self.root / "recovery_v1/CPU_CONTRACTS.json"
        patches = [patch.object(lock, "REPO", self.repo), patch.object(lock, "ORIGINAL_SOURCE_COUNT", 2), patch.object(lock, "SNAPSHOT_COUNT", 3), patch.object(lock, "ORIGINAL_STUDY_SHA256", study_hash), patch.object(lock, "ARCHIVE_INDEX_SHA256", digest_file(index_path)), patch.object(lock, "FROZEN_CHECK_SHA256", digest_file(frozen_path)), patch.object(lock, "sources", side_effect=self.supplemental)]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        log = write("recovery_v1/synthetic_cpu.log", "Synthetic CPU contracts only; no real receipt.\n")
        source = self.supplemental()
        inputs = lock._run_inputs(self.config, json.loads(self.config.read_text()), source)
        self.receipt = {"schema_version": lock.CPU_SCHEMA, "status": "passed", "returncode": 0, "source_sha256_before": source, "source_sha256_after": source, "run_input_files_sha256_before": inputs, "run_input_files_sha256_after": inputs, "files_sha256": {str(log): digest_file(log)}, "new_model_forwards": 0, "gpu_operations": 0, "optimizer_updates": 0}
        self.save_receipt()

    def supplemental(self):
        return {str(p): digest_file(p) for p in sorted(self.supplemental_paths)}

    def save_receipt(self):
        self.cpu.parent.mkdir(parents=True, exist_ok=True)
        self.cpu.write_text(json.dumps(self.receipt))

    def test_prepare_verify_idempotent_and_domains_separate(self):
        result = lock.prepare_lock(self.config, self.cpu)
        path = self.root / "recovery_v1/RECOVERY_INPUT_LOCK.json"
        original = path.read_bytes()
        self.assertEqual(lock.verify_recovery(path, self.config), result)
        self.assertEqual(lock.prepare_lock(self.config, self.cpu), result)
        self.assertEqual(path.read_bytes(), original)
        self.assertFalse(set(result["original_source_sha256"]) & set(result["recovery_source_sha256"]))
        self.assertEqual(len(result["original_object_files_sha256"]), 3)
        self.assertEqual(len(result["original_object_preflight_files_sha256"]), 2)
        self.assertFalse(result["goal_complete"])

    def test_changed_object_bytes_rejected_before_lock(self):
        self.predictions.write_text(self.predictions.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "frozen file bytes changed"):
            lock.prepare_lock(self.config, self.cpu)
        self.assertFalse((self.cpu.parent / "RECOVERY_INPUT_LOCK.json").exists())

    def test_missing_object_input_rejected(self):
        (self.root / "runs/lora_half32/seed42/object/INPUT_LOCK.json").unlink()
        with self.assertRaisesRegex(ValueError, "missing or oversized"):
            lock.prepare_lock(self.config, self.cpu)

    def test_original_source_mutation_rejected(self):
        (self.repo / "src/original_0.py").write_text("MUTATED = True\n")
        with self.assertRaisesRegex(ValueError, "frozen file bytes changed"):
            lock.prepare_lock(self.config, self.cpu)

    def test_recovery_source_mutation_rejected(self):
        stored = lock.prepare_lock(self.config, self.cpu)
        self.supplemental_paths[0].write_text("MUTATED = True\n")
        with self.assertRaisesRegex(ValueError, "source snapshot mismatch"):
            lock.verify_recovery(stored, self.config)

    def test_lock_field_mutation_rejected(self):
        stored = lock.prepare_lock(self.config, self.cpu)
        changed = copy.deepcopy(stored)
        changed["original_object_files_sha256"][str(self.predictions)] = "0" * 64
        with self.assertRaisesRegex(ValueError, "lock fields or dependencies changed"):
            lock.verify_recovery(changed, self.config)

    def test_cpu_source_snapshot_mismatch_rejected(self):
        self.receipt["source_sha256_after"] = {}
        self.save_receipt()
        with self.assertRaisesRegex(ValueError, "source snapshot mismatch"):
            lock.prepare_lock(self.config, self.cpu)

    def test_cpu_receipt_self_and_lock_reference_rejected(self):
        for forbidden in (str(self.cpu), str(self.cpu.parent / "RECOVERY_INPUT_LOCK.json")):
            with self.subTest(forbidden=forbidden):
                self.receipt["extra_reference"] = forbidden
                self.save_receipt()
                with self.assertRaisesRegex(ValueError, "self/lock reference"):
                    lock.prepare_lock(self.config, self.cpu)

    def test_bool_gpu_counter_is_not_zero_operations(self):
        self.receipt["gpu_operations"] = False
        self.save_receipt()
        with self.assertRaisesRegex(ValueError, "scientific execution"):
            lock.prepare_lock(self.config, self.cpu)

    def test_cpu_input_mismatch_rejected(self):
        self.receipt["run_input_files_sha256_after"] = {}
        self.save_receipt()
        with self.assertRaisesRegex(ValueError, "input snapshot mismatch"):
            lock.prepare_lock(self.config, self.cpu)

    def test_failed_cpu_receipt_rejected(self):
        self.receipt["status"] = "failed"
        self.save_receipt()
        with self.assertRaisesRegex(ValueError, "CPU receipt failed"):
            lock.prepare_lock(self.config, self.cpu)

    def test_weight_file_never_read(self):
        nonexistent = self.base / "do_not_read.safetensors"
        with self.assertRaisesRegex(ValueError, "weight reads forbidden"):
            lock.file_sha256(nonexistent)

    def real_structure_source_maps(self):
        names = ["module_%02d/common.py" % i for i in range(94)] + ["tools/prepare_posttrain_pope.py", "tools/run_lora_coordinate_shrink_study.py"]
        relative = {name: hashlib.sha256(name.encode()).hexdigest() for name in names}
        absolute = {str(self.repo / name if name.startswith("tools/") else self.repo / "src" / name): value for name, value in relative.items()}
        return relative, absolute

    def test_runtime_relative_96_exactly_matches_study_absolute_96(self):
        relative, absolute = self.real_structure_source_maps()
        before = copy.deepcopy(relative)
        with patch.object(lock, "ORIGINAL_SOURCE_COUNT", 96):
            self.assertEqual(lock.verify_runtime_sources(relative, absolute), absolute)
        self.assertEqual(relative, before)
        self.assertEqual(len(relative), 96)
        self.assertEqual(len(absolute), 96)
        self.assertIn(str(self.repo / "tools/prepare_posttrain_pope.py"), absolute)

    def test_runtime_96_extra_missing_or_hash_change_is_rejected(self):
        relative, absolute = self.real_structure_source_maps()
        extra = dict(relative, **{"extra.py": "0" * 64})
        missing = dict(relative)
        missing.pop(next(iter(missing)))
        changed = dict(relative)
        changed[next(iter(changed))] = "0" * 64
        with patch.object(lock, "ORIGINAL_SOURCE_COUNT", 96):
            for altered in (extra, missing, changed):
                with self.subTest(altered_count=len(altered)):
                    with self.assertRaisesRegex(ValueError, "source map differs"):
                        lock.verify_runtime_sources(altered, absolute)

    def test_runtime_source_path_escape_and_alias_keys_rejected(self):
        for name in ("../outside.py", "tools/../../outside.py", "tools/../src/a.py", "a/./b.py", "a//b.py", "/absolute.py", "a\\b.py"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "escape or noncanonical"):
                    lock.canonical_runtime_sources({name: "0" * 64})
        outside = self.base / "outside"
        outside.mkdir()
        (self.repo / "src/outside_link").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "source path escape"):
            lock.canonical_runtime_sources({"outside_link/a.py": "0" * 64})

    def test_runtime_duplicate_canonical_paths_rejected(self):
        (self.repo / "src/alias").symlink_to(self.repo / "src", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "duplicate canonical"):
            lock.canonical_runtime_sources({"original_0.py": "0" * 64, "alias/original_0.py": "0" * 64})

    def test_duplicate_json_source_keys_rejected(self):
        path = self.write("duplicate-runtime-source.json", '{"source_sha256":{"a.py":"first","a.py":"second"}}')
        with self.assertRaisesRegex(ValueError, "duplicate JSON object key"):
            lock._read(path)


if __name__ == "__main__":
    unittest.main()
