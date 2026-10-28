"""CPU-only orchestration contracts: no subprocess/model/GPU execution."""
from copy import deepcopy
import importlib
import json
from pathlib import Path

import pytest

from vlm_object_eval.analysis import engineering_exclusions
from vlm_object_study import common
from vlm_object_study.worker import task_admission
from vlm_probe.evaluate import read_json, sha_file, write_json

pipeline = importlib.import_module("vlm_object_study.run")


def test_source_lock_covers_frozen_training_helpers_queue_and_public_launcher():
    from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT
    sources = common.source_hashes()
    for path in (SOURCE_ROOT / "vlm_alignment_signal/loss.py", SOURCE_ROOT / "vlm_object_train/train.py",
        SOURCE_ROOT / "vlm_object_eval/analysis.py", SOURCE_ROOT / "research_runtime/queue.py",
        SOURCE_ROOT / "vlm_object_presence_v2/data.py",
        PROJECT_ROOT / "tools/run_object_adapt_study.py"):
        assert sources[str(path)] == sha_file(path)
    assert all(Path(path).is_absolute() for path in sources)


def test_failed_quality_gate_cannot_be_admitted_by_diagnostic_scope_or_alias():
    assert pipeline.validate_quality_admission({"status": "admitted", "quality_gate_passed": True})
    assert pipeline.validate_quality_admission({"status": "admitted", "factual_quality_gate_passed": True})
    for value in ({"status": "admitted"}, {"status": "admitted", "quality_gate_passed": False},
        {"status": "admitted", "quality_gate_passed": True, "factual_quality_gate_passed": False},
        {"status": "admitted", "quality_gate_passed": True, "scope": "noisy_data_diagnostic"},
        {"status": "admitted", "quality_gate_passed": True, "automatic_diagnostic_bypass_allowed": True}):
        with pytest.raises(ValueError, match="diagnostic bypass"):
            pipeline.validate_quality_admission(value)


def test_training_cli_uses_real_seed_flag_and_explicit_complete_checkpoint_resume(tmp_path):
    config, root = tmp_path / "config.json", tmp_path / "outputs"
    command = common.train_arguments(config, root, "object_only", 42)
    assert "--seed42" not in command
    assert command[command.index("--seed") + 1] == "42"
    assert "--resume" not in command
    latest = root / "runs/object_only/seed42/train/LATEST.json"
    latest.parent.mkdir(parents=True)
    latest.write_text("{}")
    assert "--resume" in common.train_arguments(config, root, "object_only", 42)
    preflight = common.train_arguments(config, root, "object_only", 42, preflight=True)
    assert "--preflight" in preflight and "--resume" not in preflight


def test_pope_commands_require_selected_screen_and_explicit_order_seed(tmp_path):
    with pytest.raises(ValueError, match="internal selection"):
        common.eval_arguments(tmp_path / "cfg", tmp_path, "object_only", 0, "pope", tmp_path / "adapter", tmp_path / "admit")
    job = common.job(tmp_path / "cfg", tmp_path, "pope", "object_only", 200, tmp_path / "screen")
    command = job["command"]
    assert command[command.index("--seed") + 1] == "200"
    assert "--seed42" not in command
    assert command[command.index("--screen-analysis") + 1] == str(tmp_path / "screen")
    assert job["completion_marker"].endswith("workers/pope_object_only_seed200/COMPLETE.json")


def test_initial_is_one_shared_object_reference_and_never_a_new_trained_arm():
    assert common.role_key("object_initial", "initial", 42) == "object_initial_initial_seed42"
    for operation, condition, seed in (("train_eval", "initial", 42), ("pope", "initial", 42),
        ("object_initial", "initial", 0), ("preflight", "object_only", 0)):
        with pytest.raises(ValueError):
            common.role_key(operation, condition, seed)


def test_gpu_admission_uses_assigned_uuid_and_is_stable_on_same_role_resume(tmp_path):
    config = tmp_path / "config.json"
    config.write_text("{}")
    path = task_admission(tmp_path, config, "object_only", 42, "object", {"uuid": "GPU-assigned"})
    before = path.read_bytes()
    assert read_json(path)["gpu_uuid"] == "GPU-assigned"
    assert read_json(path)["config_sha256"] == sha_file(config)
    task_admission(tmp_path, config, "object_only", 42, "object", {"uuid": "GPU-assigned"})
    assert path.read_bytes() == before
    with pytest.raises(ValueError):
        task_admission(tmp_path, config, "object_only", 42, "object", {"uuid": "GPU-other"})


@pytest.mark.parametrize("field", ["peak_allocated_gib", "peak_reserved_gib"])
def test_memory_cap_is_strictly_above65_and_does_not_change_microbatch(field):
    assert common.memory_rejection({field: 65.}) is None
    assert common.memory_rejection({field: 65.0001}) == "preflight_memory_cap_exceeded"
    assert common.engineering_reason("ValueError: tensor shapes differ") is None
    assert common.engineering_reason("torch.cuda.OutOfMemoryError: CUDA out of memory.") == "cuda_out_of_memory"
    assert common.memory_rejection({"batches": [{"batch_peak_allocated_gib": 64., "batch_peak_reserved_gib": 66.}]}) == "preflight_memory_cap_exceeded"


def rejection_receipt(tmp_path, *, reason="cuda_out_of_memory", text="CUDA out of memory"):
    log = tmp_path / "original.log"
    log.write_text(text)
    receipt = tmp_path / "COMPLETE.json"
    value = {"schema_version": "rarevlm.object_presence.worker.v1", "status": "completed",
        "outcome": "engineering_rejected", "engineering_reason": reason,
        "condition": "object_only", "seed": 42, "operation": "preflight",
        "artifact_files_sha256": {}, "engineering_evidence_files_sha256": {str(log): sha_file(log)}}
    write_json(receipt, value)
    return receipt, value


def test_engineering_rejection_cannot_relabel_contract_error_or_lost_evidence(tmp_path):
    path, value = rejection_receipt(tmp_path)
    assert common.read_receipt(path)["outcome"] == "engineering_rejected"
    log = tmp_path / "original.log"
    log.write_text("ValueError: labels changed")
    value["engineering_evidence_files_sha256"][str(log)] = sha_file(log)
    write_json(path, value)
    with pytest.raises(ValueError, match="not present"):
        common.read_receipt(path)
    log.write_text("CUDA out of memory")
    with pytest.raises(ValueError, match="locked study file changed"):
        common.read_receipt(path)


def test_only_evidenced_engineering_exclusions_keep_alpha_unchanged(tmp_path):
    config = tmp_path / "config.json"
    config.write_text("{}")
    receipt, value = rejection_receipt(tmp_path)
    admission = {"schema_version": "rarevlm.object_presence.engineering_admission.v1",
        "config_sha256": sha_file(config), "engineering_only": True, "alpha_unchanged": .0125,
        "automatic_microbatch_fallback": False, "rejected_candidates": {"object_only": [{
            "worker_receipt": str(receipt), "worker_receipt_sha256": sha_file(receipt),
            "reason": "cuda_out_of_memory", "evidence_files_sha256": value["engineering_evidence_files_sha256"]}]}}
    path = tmp_path / "admission.json"
    write_json(path, admission)
    assert engineering_exclusions(path, config)[0] == {"object_only"}
    admission["alpha_unchanged"] = .025
    write_json(path, admission)
    with pytest.raises(ValueError, match="scientific thresholds"):
        engineering_exclusions(path, config)


def test_study_rechecks_bytes_before_a_worker_can_use_its_lock(tmp_path, monkeypatch):
    config, data = tmp_path / "config.json", tmp_path / "data.json"
    config.write_text("{}")
    data.write_text("first bytes")
    monkeypatch.setattr(common, "source_hashes", lambda: {"source": "fixed"})
    monkeypatch.setattr(common, "package_versions", lambda: {"package": "fixed"})
    lock = {"schema_version": "rarevlm.object_presence.study_input.v1", "config_file_sha256": sha_file(config),
        "source_sha256": {"source": "fixed"}, "packages": {"package": "fixed"}, "protocol_files_sha256": {}, "data_files_sha256": {str(data): sha_file(data)},
        "model_files_sha256": {}, "initializer_files_sha256": {}}
    path = tmp_path / "STUDY_INPUT_LOCK.json"
    write_json(path, lock)
    common.verify_study(path, config)
    data.write_text("changed bytes")
    with pytest.raises(ValueError, match="locked study file changed"):
        common.verify_study(path, config)


def fake_study(tmp_path, monkeypatch, *, pilot_eligible=("object_only", "object_replay"),
               selected="object_replay", confirmed=True, reject_engineering=False):
    root = tmp_path / "study"
    root.mkdir()
    config = tmp_path / "config.json"
    write_json(config, {"output_root": str(root)})
    write_json(root / "STUDY_INPUT_LOCK.json", {"fixed": True})
    monkeypatch.setattr(pipeline, "prepare", lambda path: ({"output_root": str(root)}, root, {"fixed": True}))
    monkeypatch.setattr(pipeline, "verify_study", lambda *args: {})
    monkeypatch.setattr(pipeline, "read_cell", lambda *args: {"complete": True})
    calls, analyses = [], []
    def queue(root_value, phase, jobs, identity):
        calls.append((phase, deepcopy(jobs)))
        return []
    monkeypatch.setattr(pipeline, "queue_stage", queue)
    def engineering(root_value, phase, config_value, receipts):
        path = root / f"engineering_{phase}.json"
        value = {"rejected_candidates": {name: [] for name in common.CANDIDATES} if reject_engineering else {}}
        write_json(path, value)
        return path, value
    monkeypatch.setattr(pipeline, "engineering_admission", engineering)
    def analyze(config_value, root_value, seeds, phase, **kwargs):
        analyses.append((seeds, phase, kwargs))
        path = root / "analysis" / ("pilot" if seeds == [42] else "replicated") / phase / "ANALYSIS.json"
        value = {"status": "complete", "eligible_candidates": list(pilot_eligible) if seeds == [42] else ([selected] if selected else []),
            "selected_candidate": selected if seeds != [42] else None, "true_improvement_goal_achieved": confirmed}
        write_json(path, value)
        return value, path
    monkeypatch.setattr(pipeline, "analyze", analyze)
    return config, root, calls, analyses


def test_pipeline_replicates_only_pilot_passers_and_confirms_one_selected_arm(tmp_path, monkeypatch):
    config, root, calls, analyses = fake_study(tmp_path, monkeypatch, pilot_eligible=("object_replay",))
    result = pipeline.run(config)
    assert [phase for phase, _ in calls] == ["engineering_preflights", "pilot42", "replication", "pope_confirmation"]
    replication = calls[2][1]
    assert len(replication) == 2
    assert {job["key"] for job in replication} == {"train_eval_object_replay_seed0", "train_eval_object_replay_seed200"}
    pope = calls[3][1]
    assert len(pope) == 3
    assert {job["key"] for job in pope} == {f"pope_object_replay_seed{seed}" for seed in (0, 42, 200)}
    assert result["status"] == "confirmation_passed_pending_independent_review"
    assert result["goal_complete"] is False
    assert (root / "TRIAL_EVIDENCE.json").is_file()
    assert analyses[-1][1] == "confirm"


def test_failed_pilot_is_completed_negative_without_replication_or_pope(tmp_path, monkeypatch):
    config, root, calls, _ = fake_study(tmp_path, monkeypatch, pilot_eligible=())
    result = pipeline.run(config)
    assert [phase for phase, _ in calls] == ["engineering_preflights", "pilot42"]
    assert result["status"] == "completed_negative"
    assert result["scientific_failure"] is True
    assert result["goal_complete"] is False
    assert not (root / "TRIAL_EVIDENCE.json").exists()


def test_failed_aggregate_screen_does_not_try_pope_or_choose_using_pope(tmp_path, monkeypatch):
    config, root, calls, _ = fake_study(tmp_path, monkeypatch, selected=None)
    result = pipeline.run(config)
    assert [phase for phase, _ in calls] == ["engineering_preflights", "pilot42", "replication"]
    assert result["status"] == "completed_negative"
    assert result["last_phase"] == "three_seed_internal_selection"
    assert not (root / "TRIAL_EVIDENCE.json").exists()


def test_oom_is_engineering_rejection_without_imputing_accuracy_failure(tmp_path, monkeypatch):
    config, root, calls, analyses = fake_study(tmp_path, monkeypatch, reject_engineering=True)
    result = pipeline.run(config)
    assert [phase for phase, _ in calls] == ["engineering_preflights"]
    assert analyses == []
    assert result["status"] == "engineering_rejected"
    assert result["scientific_failure"] is False
    assert result["goal_complete"] is False
