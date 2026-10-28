"""Independent CPU contracts; no GPU/model/real study preparation."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from vlm_object_kl_eval import contracts as c
from vlm_object_kl_eval import evaluate as ev
from vlm_object_kl_eval.analysis import assert_pair, external_gate, screen_gate, SCIENCE_FIELDS, validate_settings
from vlm_object_kl_study import common
from vlm_object_kl_study.worker import task_admission
from vlm_object_kl_study import worker
from vlm_probe.evaluate import content_sha, read_json, sha_file, write_json

pipeline = importlib.import_module("vlm_object_kl_study.run")


@pytest.mark.parametrize("category", ["missing", None, ""])
def test_text_empty_aliases_are_explicit_and_scores_inputs_unchanged(category):
    row = {"question_id": "q", "score": .6, "input_ids": [1, 2], "pixel_values": {"sha256": "fixed"}}
    if category != "missing": row["category"] = category
    rows = [row]
    before = deepcopy(rows)
    result, proof = c.canonical_rows(rows, "text")
    assert rows == before and "category" not in result[0]
    assert proof["noncategory_fields_sha256_before"] == proof["noncategory_fields_sha256_after"]


@pytest.mark.parametrize("category", ["random", "dog", False, 0])
def test_nonempty_text_category_is_never_relaxed(category):
    with pytest.raises(ValueError, match="nonempty Text"):
        c.canonical_rows([{"score": 1, "category": category}], "text")


@pytest.mark.parametrize("task", ["object", "external", "pope"])
def test_binary_categories_remain_exact_original_objects(task):
    rows = [{"category": "bird", "label": "yes", "score": 1}]
    normalized, _ = c.canonical_rows(rows, task)
    assert normalized is rows


def test_historical_and_new_text_pair_without_mutating_old_categories():
    from test_posttrain_eval_contracts import cache_fixture
    _, _, _, records = cache_fixture()
    old = deepcopy(records)
    for row in old: row["category"] = ""
    left, _ = c.canonical_rows(old, "text")
    right, _ = c.canonical_rows(records, "text")
    lock = {key: "fixed" for key in SCIENCE_FIELDS}
    assert_pair({"lock": lock, "rows": left}, {"lock": lock, "rows": right}, "text")
    assert old[0]["category"] == ""
    right[0]["pixel_values"]["sha256"] = "different"
    with pytest.raises(ValueError, match="pixel_values"):
        assert_pair({"lock": lock, "rows": left}, {"lock": lock, "rows": right}, "text")


def test_binary_category_mismatch_is_rejected_even_with_identical_input():
    from test_posttrain_eval_contracts import cache_fixture
    _, _, _, records = cache_fixture()
    for row in records: row.update(category="dog", label="yes")
    changed = deepcopy(records)
    changed[0]["category"] = "cat"
    lock = {key: "fixed" for key in SCIENCE_FIELDS}
    with pytest.raises(ValueError, match="category"):
        assert_pair({"lock": lock, "rows": records}, {"lock": lock, "rows": changed}, "object")


def valid_budget():
    names = {}
    for layer in range(36):
        for projection in ("q_proj", "v_proj"):
            for matrix in ("A", "B"):
                names[f"model.layers.{layer}.self_attn.{projection}.lora_{matrix}.default.weight"] = 2048 if projection == "v_proj" and matrix == "B" else 16384
    return {"trainable_names": names, "dtypes": {"lora_elements_by_dtype": {"float32": 1843200},
        "lora_tensors_by_dtype": {"float32": 144}, "params4bit_tensors": 0}}


def test_only_fp32_language_qv_exact_budget_allowed():
    value = valid_budget()
    c.validate_parameter_budget(value)
    value["dtypes"]["lora_elements_by_dtype"] = {"bfloat16": 1843200}
    with pytest.raises(ValueError, match="FP32"):
        c.validate_parameter_budget(value)
    value = valid_budget()
    first = next(iter(value["trainable_names"]))
    value["trainable_names"]["model.visual.q_proj.lora_A.weight"] = value["trainable_names"].pop(first)
    with pytest.raises(ValueError, match="FP32"):
        c.validate_parameter_budget(value)


def cache_fixture(tmp_path, monkeypatch):
    import torch
    import vlm_object_kl.cache as cache
    train_dir = tmp_path / "train"
    cache_dir = train_dir / "teacher_cache"
    cache_dir.mkdir(parents=True)
    model = tmp_path / "model"
    model.mkdir()
    write_json(model / "config.json", {"vocab_size": 5})
    actor_contract = {"question_id": "q", "assistant_tokens": 3, "microbatch_input_sha256": "pixels_and_ids"}
    actual = {"object": [], "replay": [actor_contract]}
    write_json(train_dir / "TRAINED_INPUT_CONTRACTS.json", actual)
    binding = {"initial_adapter_tensor_sha256": "initial", "frozen_parameter_sha256_before": "base",
        "teacher_role": "fixed_initial_before_optimizer"}
    write_json(cache_dir / "INPUT_LOCK.json", binding)
    manifest = {"teacher_parameters_sha256": "initial", "frozen_parameters_sha256": "base",
        "teacher_forward_questions": 1, "initial_parameters_rng_buffers_rope_restored": True,
        "entries": {"entry": {"identity": {"update_index": 0, "replay_microbatch_index": 0,
            "question_ids": ["q"], "input_contracts": [deepcopy(actor_contract)]}, "shape": [3, 5]}}}
    write_json(cache_dir / "MANIFEST.json", manifest)
    train = {**binding, "teacher_cache_directory": str(cache_dir),
        "teacher_cache_manifest_sha256": sha_file(cache_dir / "MANIFEST.json"),
        "teacher_cache_manifest_sha256_after": sha_file(cache_dir / "MANIFEST.json"), "teacher_cache_unchanged": True,
        "teacher_cache_binding_sha256": content_sha(binding), "actual_replay_exposures": 1,
        "trained_input_contracts_file_sha256": sha_file(train_dir / "TRAINED_INPUT_CONTRACTS.json"),
        "object_input_contract_sha256": content_sha([]), "replay_input_contract_sha256": content_sha(actual["replay"]),
        "teacher_cache_summary": {key: value for key, value in manifest.items() if key != "entries"}}
    monkeypatch.setattr(cache, "verify_frozen_cache", lambda *args: manifest)
    monkeypatch.setattr(cache, "load_entry", lambda *args: {"logprobs": torch.zeros(3, 5)})
    return train, train_dir, {"model_root": str(model)}, manifest


def test_cache_binds_before_after_sha_and_actual_actor_replay_inputs(tmp_path, monkeypatch):
    train, directory, config, manifest = cache_fixture(tmp_path, monkeypatch)
    assert c.validate_teacher_cache(train, directory, config)["actual_padded_actor_inputs_matched"]
    train["teacher_cache_manifest_sha256_after"] = "changed"
    with pytest.raises(ValueError, match="before/after"):
        c.validate_teacher_cache(train, directory, config)
    train["teacher_cache_manifest_sha256_after"] = train["teacher_cache_manifest_sha256"]
    manifest["entries"]["entry"]["identity"]["input_contracts"][0]["microbatch_input_sha256"] = "other_pixels"
    with pytest.raises(ValueError, match="actual actor"):
        c.validate_teacher_cache(train, directory, config)


def gate_rows():
    binary = [{"slice": "overall", "metrics": {"accuracy": {"delta_pp": 1.1,
        "one_sided_98_75_lower_pp": .01, "seed_delta_pp": [.1, 1.1, 2.1]},
        "recall": {"delta_pp": 0.}, "f1": {"delta_pp": 0.}, "specificity": {"delta_pp": -.5}}}]
    text = [{"slice": name, "delta_pp": 0., "one_sided_98_75_lower_pp": -.49, "seed_delta_pp": [-.5, 0., .5]}
        for name in ("overall", "rare_1_5", "frequent_gt20")]
    balanced = {"contrasts": [{"slice": "overall", "delta_pp": .01}]}
    return binary, text, balanced


def test_four_endpoint_family_and_pilot_point_guards_stay_frozen():
    validate_settings({})
    with pytest.raises(ValueError): validate_settings({"analysis": {"one_sided_confidence": .975}})
    binary, text, balanced = gate_rows()
    assert screen_gate(binary, text, balanced, replicated=False)["passed"]
    balanced["contrasts"][0]["delta_pp"] = 0.
    assert not screen_gate(binary, text, balanced, replicated=False)["passed"]
    binary, text, balanced = gate_rows()
    text[1]["delta_pp"] = -.000001
    assert not screen_gate(binary, text, balanced, replicated=False)["passed"]


def test_three_seed_screen_requires_points_not_each_seed_significance():
    binary, text, balanced = gate_rows()
    assert screen_gate(binary, text, balanced, replicated=True)["passed"]
    binary[0]["metrics"]["accuracy"]["seed_delta_pp"][0] = 0.
    assert not screen_gate(binary, text, balanced, replicated=True)["passed"]
    binary, text, balanced = gate_rows()
    text[0]["seed_delta_pp"][0] = -.500001
    assert not screen_gate(binary, text, balanced, replicated=True)["passed"]


def test_external_requires_f1_and_strict_positive_balanced_point():
    binary, text, balanced = gate_rows()
    prior = {"gate": {"passed": True}}
    assert external_gate(binary, balanced, prior)["passed"]
    balanced["contrasts"][0]["delta_pp"] = 0.
    assert not external_gate(binary, balanced, prior)["passed"]
    binary, text, balanced = gate_rows()
    binary[0]["metrics"]["f1"]["delta_pp"] = -.000001
    assert not external_gate(binary, balanced, prior)["passed"]
    binary, text, balanced = gate_rows()
    binary[0]["metrics"]["accuracy"]["one_sided_98_75_lower_pp"] = 0.
    assert not external_gate(binary, balanced, prior)["passed"]


def test_external_initial_is_allowed_but_candidate_is_blocked_before_internal_pass(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(ev, "external_admission", lambda config: {"quality_admitted": True})
    args = SimpleNamespace(task="external", condition="initial", screen_analysis=None, seed=42)
    assert ev.progression(args, {}, {})["initial_external_baseline_only"]
    args.condition = "object_kl"
    with pytest.raises(ValueError, match="internal three-seed"):
        ev.progression(args, {}, {})


def test_seed_cli_resume_and_new_uuid_admission(tmp_path):
    cfg = tmp_path / "config.json"; cfg.write_text("{}")
    args = common.train_arguments(cfg, tmp_path, 42)
    assert "--seed42" not in args and args[args.index("--seed") + 1] == "42"
    file = tmp_path / "runs/object_kl/seed42/train/LATEST.json"
    write_json(file, {})
    assert "--resume" in common.train_arguments(cfg, tmp_path, 42)
    admission = task_admission(tmp_path / "role", cfg, "object_kl", 42, "text", {"uuid": "GPU-a"})
    assert read_json(admission)["schema_version"] == "rarevlm.object_kl.gpu_admission.v1"
    with pytest.raises(ValueError): task_admission(tmp_path / "role", cfg, "object_kl", 42, "text", {"uuid": "GPU-b"})


def test_only_real_oom_or65_cap_is_engineering_rejection():
    assert common.reason("CUDA out of memory") == "cuda_out_of_memory"
    assert common.reason("ValueError: bad inputs", {"status": "failed", "error": "bad inputs"}) is None
    assert common.reason("RuntimeError: teacher_cache_memory_cap_exceeded",
        cache_failure={"phase": "teacher_cache", "peak_allocated_gib": 66, "peak_reserved_gib": 66}) == "teacher_cache_memory_cap_exceeded"
    assert common.reason("RuntimeError: formal_training_memory_cap_exceeded") == "formal_training_memory_cap_exceeded"


@pytest.mark.parametrize("kind", ["preflight", "cache_failure"])
def test_stale_cap_receipts_and_previous_log_do_not_hide_current_contract_failure(tmp_path, monkeypatch, kind):
    file = tmp_path / ("PREFLIGHT.json" if kind == "preflight" else "teacher_cache/BUILD_FAILURE.json")
    write_json(file, {"status": "failed", "peak_allocated_gib": 66, "error": "CUDA out of memory"})
    log = tmp_path / "logs/training.log"
    log.parent.mkdir()
    log.write_text("RuntimeError: formal_training_memory_cap_exceeded\nCUDA out of memory\n")
    before = worker.snapshot(file)
    def fake_run(command, stdout, **kwargs):
        stdout.write("ValueError: current input contract mismatch\n")
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(worker.subprocess, "run", fake_run)
    with pytest.raises(subprocess.CalledProcessError):
        worker.call(tmp_path, "test", {}, "training", ["fake"], **{kind: file})
    assert worker.snapshot(file) == before
    attempt = read_json(next((tmp_path / "attempts").glob("*.call.json")))
    assert attempt["fresh_failure_receipts"] == {}


@pytest.mark.parametrize("kind,expected", [("preflight", "preflight_memory_cap_exceeded"),
    ("cache_failure", "teacher_cache_memory_cap_exceeded")])
def test_fresh_cap_evidence_is_bound_to_current_log_segment_and_revalidated(tmp_path, monkeypatch, kind, expected):
    file = tmp_path / ("PREFLIGHT.json" if kind == "preflight" else "teacher_cache/BUILD_FAILURE.json")
    write_json(file, {"status": "failed", "peak_allocated_gib": 66})
    log = tmp_path / "logs/training.log"
    log.parent.mkdir()
    log.write_text("RuntimeError: formal_training_memory_cap_exceeded\n")
    def fake_run(command, stdout, **kwargs):
        write_json(file, {"status": "failed", "peak_allocated_gib": 67})
        stdout.write("RuntimeError: memory_cap_exceeded\n")
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(worker.subprocess, "run", fake_run)
    with pytest.raises(worker.EngineeringRejected) as caught:
        worker.call(tmp_path, "test", {}, "training", ["fake"], **{kind: file})
    error = caught.value
    assert error.reason == expected
    receipt = tmp_path / "COMPLETE.json"
    write_json(receipt, {"schema_version": "rarevlm.object_kl.worker.v1", "status": "completed",
        "study_input_lock_sha256": "study", "outcome": "engineering_rejected",
        "artifact_files_sha256": {}, "engineering_evidence_files_sha256": error.evidence,
        "engineering_reason": expected})
    assert common.read_receipt(receipt, "study")["engineering_reason"] == expected


def test_success_does_not_consume_stale_failure_and_locks_all_engineering_cache_files(tmp_path, monkeypatch):
    cache = tmp_path / "teacher_cache"
    for name in ("INPUT_LOCK.json", "MANIFEST.json", "BUILD_FAILURE.json"):
        write_json(cache / name, {"status": "failed", "peak_allocated_gib": 66})
    (cache / "batch.pt").write_bytes(b"full vocabulary teacher tensor")
    (cache / "ignored.txt").write_text("not a cache artifact")
    monkeypatch.setattr(worker.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0))
    artifacts = {}
    worker.call(tmp_path, "test", artifacts, "engineering_preflight", ["fake"], cache_failure=cache / "BUILD_FAILURE.json")
    artifacts.update(worker.cache_artifacts(cache))
    assert all(str(cache / name) in artifacts for name in ("INPUT_LOCK.json", "MANIFEST.json", "BUILD_FAILURE.json", "batch.pt"))
    assert str(cache / "ignored.txt") not in artifacts
    common.verify_files(artifacts)


def fake_pipeline(tmp_path, monkeypatch, *, pilot=True, screen=True, external=True, pope=True):
    root = tmp_path / "study"; root.mkdir()
    cfg = tmp_path / "config.json"; write_json(cfg, {"output_root": str(root)})
    write_json(root / "STUDY_INPUT_LOCK.json", {"fixed": True})
    monkeypatch.setattr(pipeline, "prepare", lambda file: ({}, root, {}))
    monkeypatch.setattr(pipeline, "verify_study", lambda *args: {})
    calls, analysis_calls = [], []
    def queue(root, phase, jobs, identity):
        calls.append((phase, deepcopy(jobs)))
        return [{"outcome": "passed"} for _ in jobs]
    monkeypatch.setattr(pipeline, "queue_stage", queue)
    def analyze(cfg, root, seeds, phase, **kwargs):
        analysis_calls.append((list(seeds), phase))
        passed = pilot if phase == "screen" and list(seeds) == [42] else screen if phase == "screen" else external if phase == "external" else pope
        file = root / "analysis" / ("pilot" if list(seeds) == [42] else "replicated") / phase / "ANALYSIS.json"
        result = {"status": "complete", "gate": {"passed": passed}}
        write_json(file, result)
        return result, file
    monkeypatch.setattr(pipeline, "analyze", analyze)
    return cfg, root, calls, analysis_calls


def test_pipeline_parallel_initial_external_never_selects_with_its_score(tmp_path, monkeypatch):
    cfg, root, calls, analysis_calls = fake_pipeline(tmp_path, monkeypatch)
    result = pipeline.run(cfg)
    assert [name for name, _ in calls] == ["engineering_preflight", "pilot42", "replication", "external_confirmation", "pope_confirmation"]
    assert {item["key"] for item in calls[1][1]} == {"train_eval_object_kl_seed42", "external_initial_initial_seed42"}
    assert len(calls[3][1]) == 3 and len(calls[4][1]) == 3
    assert analysis_calls[0] == ([42], "screen")
    assert result["confirmation_passed"] and result["goal_complete"] is False


@pytest.mark.parametrize("failed,expected", [("pilot", ["engineering_preflight", "pilot42"]),
    ("screen", ["engineering_preflight", "pilot42", "replication"]),
    ("external", ["engineering_preflight", "pilot42", "replication", "external_confirmation"])])
def test_failed_stage_prevents_later_candidate_prediction_and_keeps_goal_active(tmp_path, monkeypatch, failed, expected):
    cfg, root, calls, _ = fake_pipeline(tmp_path, monkeypatch, **{failed: False})
    result = pipeline.run(cfg)
    assert [name for name, _ in calls] == expected
    assert result["status"] == "completed_negative" and result["goal_complete"] is False
    assert not (root / "TRIAL_EVIDENCE.json").exists()
