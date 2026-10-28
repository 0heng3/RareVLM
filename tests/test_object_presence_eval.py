"""CPU science contracts for independent object-presence evaluation."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from vlm_object_eval import evaluate as ev
from vlm_object_eval.analysis import (_interval, assert_common_locks, assert_same_inputs,
    binary_contrast, COMMON_FIELDS, confirm_gate, scalar_contrast, screen_gate, validate_analysis_config)
from vlm_object_eval.analysis import select_candidate
from vlm_object_eval.scoring import diagnostics, first_yesno, metrics, official_prediction


REFERENCE = {"precision": "fp32", "training_quantization": "nf4_language_only",
    "min_pixels": 50176, "max_pixels": 200704, "max_new_tokens": 32, "do_sample": False}


@pytest.mark.parametrize("task,suffix", [("text", ev.TEXT_SUFFIX), ("object", ev.PROMPT_SUFFIX), ("pope", ev.PROMPT_SUFFIX)])
def test_generation_uses_only_original_image_question_and_frozen_task_suffix(task, suffix):
    raw = {"question_id": "1", "image_id": "a", "question": "Is there a dog?",
        "target": "leak", "answers": ["leak"] * 10, "label": "no",
        "bbox_evidence": ["leak"], "prompt_suffix": "leak", "category_name": "dog"}
    original = deepcopy(raw)
    clean = ev.inference_row(raw, task)
    assert set(clean) == {"question_id", "image_id", "question", "prompt_suffix"}
    assert clean["prompt_suffix"] == suffix
    assert raw == original


def test_both_arms_and_fixed_precision_bootstrap_are_preregistered():
    ev.validate_config({}, REFERENCE)
    validate_analysis_config({})
    for config in ({"candidates": ["object_only"]}, {"evaluation": {"batch_size": 1}},
        {"evaluation": {"visual_token_budget": 2048}}, {"evaluation": {"precision": "fp32"}}):
        with pytest.raises(ValueError):
            ev.validate_config(config, REFERENCE)
    for key, value in (("one_sided_confidence", .975), ("bootstrap_replicates", 2000),
        ("bootstrap_seed", 42), ("object_min_delta_pp", .5)):
        with pytest.raises(ValueError, match=key):
            validate_analysis_config({"analysis": {key: value}})


def test_original_text_manifest_metadata_cannot_override_new_task_identity(tmp_path, monkeypatch):
    rows = [{"question_id": "1", "image_id": "a"}, {"question_id": "2", "image_id": "b"}]
    images = {"a": {}, "b": {}, "training_image": {}}
    metadata = {"task": "textvqa", "manifest_path": str(tmp_path / "manifest.json"),
        "manifest_sha256": "content", "manifest_file_sha256": "file"}
    monkeypatch.setattr(ev.frozen_text, "dev_data", lambda path: (rows, images, tmp_path, metadata))
    result = ev.load_data({"asset_root": str(tmp_path)}, "text")
    assert "task" not in result[3]
    assert result[3]["original_task"] == "textvqa"
    assert result[3]["unique_images"] == 2


@pytest.mark.parametrize("text,value", [("Yes, there is.", "yes"), (" NO!", "no"),
    ("No\nthere", "no"), ("not", None), ("yesterday", None), ("There is no dog", None),
    ("yes/no", None), ("", None), ("No。", None)])
def test_first_yesno_is_separate_and_never_searches_later_tokens(text, value):
    assert first_yesno(text) == value
    audit = diagnostics(text)
    assert audit["first_yesno_invalid"] == (value is None)
    assert audit["official_prediction"] == official_prediction(text)


def test_malformed_outputs_remain_in_official_and_strict_denominators():
    rows = [{"prediction": "", "label": "no", "category": "random"},
        {"prediction": "No!", "label": "no", "category": "random"},
        {"prediction": "Yes, there is.", "label": "yes", "category": "random"}]
    result = metrics(rows)["overall"]
    assert result["n"] == 3
    assert result["accuracy"] == pytest.approx(1 / 3)
    assert result["first_yesno_accuracy"] == pytest.approx(2 / 3)
    assert result["first_yesno_invalid_rate"] == pytest.approx(1 / 3)
    assert result["invalid_rate"] == pytest.approx(2 / 3)


def object_cache_fixture():
    from test_posttrain_eval_contracts import cache_fixture
    rows, images, lock, records = cache_fixture()
    lock.update(condition="object_only", task="object")
    for index, (row, item) in enumerate(zip(rows, records)):
        row.update(label="yes" if index % 2 else "no", category="dog")
        item.update(condition="object_only", task="object", input_lock_sha256=ev.content_sha(lock),
            question_prompt_sha256=ev.content_sha(row["question"] + ev.PROMPT_SUFFIX),
            **ev.scoring_fields(item["prediction"], row, "object"))
    return rows, images, lock, records


def test_complete_batch_resume_never_reuses_partial_batch(tmp_path):
    rows, images, lock, records = object_cache_fixture()
    path = tmp_path / "predictions.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records[:6]) + '{"question_id":')
    recovered, repairs = ev.complete_cache(path, rows, images, lock)
    assert len(recovered) == len(path.read_text().splitlines()) == 4
    assert len(repairs) == 2
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    complete, repairs = ev.complete_cache(path, rows, images, lock, repair=False)
    assert len(complete) == 9
    assert complete[-1]["batch_size"] == 1
    assert repairs is None


@pytest.mark.parametrize("change", ("score", "token", "label", "parser", "lock", "pixel", "truncation"))
def test_cached_science_actual_ids_and_pixels_are_locked(tmp_path, change):
    rows, images, lock, records = object_cache_fixture()
    if change == "score":
        records[0]["score"] = 1.
    elif change == "token":
        records[0]["input_ids"][1] = 99
    elif change == "label":
        records[0]["label"] = "yes"
    elif change == "parser":
        records[0]["official_prediction"] = "no"
    elif change == "pixel":
        records[0]["pixel_values"]["shape"] = [3, 1176]
    elif change == "truncation":
        records[0]["input_truncated"] = True
    else:
        lock["hardware"] = {"uuid": "another-GPU"}
    path = tmp_path / "predictions.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    with pytest.raises(ValueError):
        ev.complete_cache(path, rows, images, lock, repair=False)


def test_image_clusters_preserve_unequal_question_counts_and_fixed_seed_axes():
    initial = [{"question_id": str(i), "image_id": name, "score": 0.}
        for i, name in enumerate(("a", "a", "b"))]
    first = [{**r, "score": float(r["image_id"] == "a")} for r in initial]
    result = scalar_contrast(initial, [first])
    assert result["image_clusters"] == 2
    assert result["questions"] == 3
    assert result["delta_pp"] == pytest.approx(200 / 3)
    assert result["image_cluster_bootstrap_95_pp"] == [0., 100.]
    opposite = [{**r, "score": float(r["image_id"] == "b")} for r in initial]
    result = scalar_contrast(initial, [first, opposite])
    assert result["checkpoint_axes"] == 2
    assert result["image_cluster_bootstrap_95_pp"] == [50., 50.]
    assert result["seed_delta_sample_sd_pp"] is not None
    with pytest.raises(ValueError, match="6000"):
        scalar_contrast(initial, [first], replicates=20)


def test_one_sided_98_75_lower_uses_alpha_point0125():
    sample = np.arange(6000, dtype=float)
    result = _interval(sample, 1., [1.])
    assert result["one_sided_98_75_lower_pp"] == pytest.approx(np.quantile(sample, .0125))
    assert result["one_sided_98_75_lower_pp"] != result["image_cluster_bootstrap_95_pp"][0]


def paired_binary_fixture():
    initial = []
    for image in ("a", "b"):
        for category in ("random", "popular", "adversarial"):
            for label in ("yes", "no"):
                initial.append({"question_id": f"{image}_{category}_{label}", "image_id": image,
                    "category": category, "label": label, "prediction": "No", "score": float(label == "no")})
    after = deepcopy(initial)
    for row in after:
        if row["image_id"] == "a":
            row.update(prediction=row["label"], score=1.)
    return initial, after


def test_pope_categories_share_image_clusters_not_category_image_pairs():
    initial, after = paired_binary_fixture()
    contrast, summaries = binary_contrast(initial, [after])
    overall = contrast[0]
    assert overall["questions"] == 12
    assert overall["image_clusters"] == overall["resampled_image_clusters"] == 2
    assert overall["delta_pp"] == 25
    assert overall["image_cluster_bootstrap_95_pp"] == [0., 50.]
    assert len(contrast) == 4
    for row in contrast:
        assert row["resampled_image_clusters"] == 2
        assert "specificity" in row["metrics"]
        assert row["metrics"]["precision"]["undefined_bootstrap_replicates"] == 6000
    assert summaries["initial"]["overall"]["n"] == 12
    json.dumps(contrast, allow_nan=False)


def passing_rows():
    binary = [{"slice": "overall", "metrics": {"accuracy": {"delta_pp": 1., "one_sided_98_75_lower_pp": .01, "seed_delta_pp": [.1, 1., 1.9]},
        "recall": {"delta_pp": 0.}, "f1": {"delta_pp": 0.}, "specificity": {"delta_pp": -.5}}}]
    binary.extend({**deepcopy(binary[0]), "slice": category} for category in ("random", "popular", "adversarial"))
    text = [{"slice": label, "delta_pp": 0., "one_sided_98_75_lower_pp": -.49}
        for label in ("overall", "rare_1_5", "frequent_gt20")]
    return binary, text


def passing_balanced(point=.01):
    return {"status": "complete", "contrasts": [{"slice": "overall", "delta_pp": point}]}


def test_phase_gates_enforce_strict_bounds_and_point_guard_meanings():
    binary, text = passing_rows()
    assert screen_gate(binary, text, balanced=passing_balanced())["passed"] is True
    assert confirm_gate(binary, {"passed": True})["passed"] is True
    binary[0]["metrics"]["accuracy"]["one_sided_98_75_lower_pp"] = 0
    assert screen_gate(binary, text, balanced=passing_balanced())["passed"] is False
    assert confirm_gate(binary, {"passed": True})["passed"] is False
    binary, text = passing_rows()
    text[0]["one_sided_98_75_lower_pp"] = -.5
    assert screen_gate(binary, text, balanced=passing_balanced())["passed"] is False
    binary, text = passing_rows()
    text[1]["delta_pp"] = -.000001
    assert screen_gate(binary, text, balanced=passing_balanced())["passed"] is False
    binary, text = passing_rows()
    binary[0]["metrics"]["recall"]["delta_pp"] = -.000001
    assert screen_gate(binary, text, balanced=passing_balanced())["passed"] is False
    assert confirm_gate(binary, {"passed": True})["passed"] is False
    binary, text = passing_rows()
    assert confirm_gate(binary, {"passed": False})["passed"] is False


def test_pilot_requires_positive_class_balanced_point_before_replication():
    binary, text = passing_rows()
    assert screen_gate(binary, text, balanced=passing_balanced())["passed"] is True
    for balanced in (None, passing_balanced(0.), passing_balanced(-.01)):
        result = screen_gate(binary, text, balanced=balanced)
        assert all(result["primary_checks"].values())
        assert result["point_protection_checks"]["class_balanced_mean_accuracy_point_gt0"] is False
        assert result["passed"] is False


def test_manifest_versions_are_explicit_and_v2_is_bound_to_current_protocol(tmp_path):
    image = str(tmp_path / "image.jpg")
    row = {"question_id": "object_presence_v2:000000000001:category18", "image_id": "a",
        "question": "Is there a dog?", "target": "yes", "image_path": image,
        "image_sha256": "a" * 64, "category_name": "dog", "prompt_suffix": ev.PROMPT_SUFFIX}
    manifest = {"schema_version": "rarevlm.object_presence.manifest.v2", "split": "val",
        "protocol_sha256": "b" * 64, "prompt_suffix": ev.PROMPT_SUFFIX, "rows": [row],
        "images": {"a": {"path": image, "sha256": "a" * 64, "width": 1, "height": 1}}}
    path = tmp_path / "VAL_MANIFEST.json"
    manifest["manifest_sha256"] = ev.content_sha(manifest)
    path.write_text(json.dumps(manifest))
    result = ev.object_data(path, "rarevlm.object_presence.manifest.v2", "b" * 64)
    assert result[3]["data_manifest_schema_version"] == "rarevlm.object_presence.manifest.v2"
    with pytest.raises(ValueError, match="data source"):
        ev.object_data(path, "rarevlm.object_presence.manifest.v1", "b" * 64)
    with pytest.raises(ValueError, match="protocol"):
        ev.object_data(path, "rarevlm.object_presence.manifest.v2", "c" * 64)
    disguised = {**manifest, "schema_version": "rarevlm.object_presence.manifest.v1"}
    with pytest.raises(ValueError, match="masquerade"):
        ev.validate_object_schema(disguised)


def test_replicated_screen_uses_three_seed_points_and_aggregate_ci():
    binary, text = passing_rows()
    text[0]["seed_delta_pp"] = [-.5, 0., .1]
    balanced = {"status": "complete", "contrasts": [{"slice": "overall", "delta_pp": .01}]}
    assert screen_gate(binary, text, replicated=True, balanced=balanced)["passed"] is True
    binary[0]["metrics"]["accuracy"]["seed_delta_pp"][0] = 0.
    assert screen_gate(binary, text, replicated=True, balanced=balanced)["passed"] is False
    binary, text = passing_rows()
    text[0]["seed_delta_pp"] = [-.500001, 0., .1]
    assert screen_gate(binary, text, replicated=True, balanced=balanced)["passed"] is False
    text[0]["seed_delta_pp"] = [-.5, 0., .1]
    balanced["contrasts"][0]["delta_pp"] = 0.
    assert screen_gate(binary, text, replicated=True, balanced=balanced)["passed"] is False


def test_pope_confirmation_guards_seed_directions_and_all_three_categories():
    binary, _ = passing_rows()
    assert confirm_gate(binary, {"passed": True})["passed"] is True
    binary[1]["metrics"]["accuracy"]["delta_pp"] = -.000001
    assert confirm_gate(binary, {"passed": True})["passed"] is False
    binary, _ = passing_rows()
    binary[0]["metrics"]["accuracy"]["seed_delta_pp"] = [0., 1., 2.]
    assert confirm_gate(binary, {"passed": True})["passed"] is False


def test_candidate_selection_uses_only_internal_object_then_text_then_fixed_name():
    contrasts = {}
    for name in ev.CANDIDATES:
        contrasts[f"{name}_minus_initial_object"] = [{"slice": "overall", "after_score_percent": 85.}]
        contrasts[f"{name}_minus_initial_text"] = [{"slice": "overall", "after_score_percent": 87.}]
        contrasts[f"{name}_minus_initial_pope"] = [{"slice": "overall", "after_score_percent": 100. if name == "object_replay" else 0.}]
    assert select_candidate(list(ev.CANDIDATES), contrasts)[0] == "object_only"
    contrasts["object_replay_minus_initial_text"][0]["after_score_percent"] = 87.1
    assert select_candidate(list(ev.CANDIDATES), contrasts)[0] == "object_replay"
    contrasts["object_only_minus_initial_object"][0]["after_score_percent"] = 85.1
    assert select_candidate(list(ev.CANDIDATES), contrasts)[0] == "object_only"


def test_pilot_screen_never_admits_pope(tmp_path):
    config = tmp_path / "config.json"
    config.write_text("{}")
    path = tmp_path / "pilot.json"
    path.write_text(json.dumps({"schema_version": "rarevlm.object_presence.screen.v1", "status": "complete",
        "config_sha256": ev.sha_file(config), "stage": "pilot_order_seed42", "seeds": [42],
        "eligible_candidates": ["object_only"], "selected_candidate": None}))
    with pytest.raises(ValueError, match="all three"):
        ev.screen_admission(path, config, "object_only", 42, {"sha256": "fixed"})


def test_actual_input_pairing_requires_every_audit_field():
    rows, images, lock, records = object_cache_fixture()
    assert_same_inputs(records, deepcopy(records))
    changed = deepcopy(records)
    changed[0]["pixel_values"]["sha256"] = "b" * 64
    with pytest.raises(ValueError, match="pixel_values"):
        assert_same_inputs(records, changed)
    changed = deepcopy(records)
    del changed[0]["image_grid_thw_tensor"]
    with pytest.raises(ValueError, match="image_grid_thw_tensor"):
        assert_same_inputs(records, changed)


def test_historical_reference_retains_old_source_but_not_different_scientific_inputs():
    original = {key: "fixed" for key in COMMON_FIELDS}
    candidate = {"lock": deepcopy(original)}
    historical = {"lock": deepcopy(original), "historical_reference": {"new_generation_questions": 0},
        "comparison_metadata": {key: "fixed" for key in ("data_manifest_sha256", "data_manifest_file_sha256", "questions", "unique_images")}}
    historical["lock"].update(source_sha256="old source", study_config_sha256="old config",
        study_config_content_sha256="old config content", input="original recorded prose")
    assert_common_locks(historical, candidate)
    assert historical["lock"]["source_sha256"] == "old source"
    candidate["lock"]["prompt_suffix"] = "different prompt"
    with pytest.raises(ValueError, match="prompt_suffix"):
        assert_common_locks(historical, candidate)


def test_gpu_admission_binds_exact_config_condition_task_and_physical_uuid(tmp_path):
    config = tmp_path / "config.json"
    config.write_text("{}")
    admission = {"schema_version": "rarevlm.object_presence.gpu_admission.v1", "status": "admitted",
        "condition": "object_only", "seed": 42, "task": "object", "gpu_uuid": "GPU-fixed",
        "config_sha256": ev.sha_file(config)}
    path = tmp_path / "admission.json"
    path.write_text(json.dumps(admission))
    assert ev.gpu_admission(path, config, "object_only", 42, "object", {"uuid": "GPU-fixed"})["gpu_admission_verified"]
    with pytest.raises(ValueError, match="gpu_uuid"):
        ev.gpu_admission(path, config, "object_only", 42, "object", {"uuid": "GPU-another"})
    with pytest.raises(ValueError, match="task"):
        ev.gpu_admission(path, config, "object_only", 42, "pope", {"uuid": "GPU-fixed"})


def test_initializer_absolute_path_is_shared_across_order_seeds():
    path = Path("/opt/data/private/longtail_vlm/rarevlm_pilot/outputs/vlm_weighted_sft_v1/runs/uniform/seed42/train/adapter")
    assert ev.initial_adapter_path({"initial_adapter": str(path)}) == path


@pytest.mark.parametrize("microbatch,accumulation", [(1, 16), (4, 4)])
def test_new_adapter_binding_requires_new_completed_ce_trainer_schema(tmp_path, monkeypatch, microbatch, accumulation):
    initial, candidate = tmp_path / "initial" / "adapter", tmp_path / "candidate" / "adapter"
    initial.mkdir(parents=True)
    candidate.mkdir(parents=True)
    model_root = tmp_path / "base" / "model"
    model_root.mkdir(parents=True)
    (model_root.parent / "model_provenance.json").write_text("{}")
    manifest = tmp_path / "TRAIN_MANIFEST.json"
    manifest.write_text(json.dumps({"rows": [{"question_id": str(i)} for i in range(16)]}))
    config = {"initial_adapter": str(initial), "train_manifest": str(manifest),
        "training_microbatch": microbatch, "training_accumulation": accumulation, "object_effective_batch": 16}
    files = {"adapter_config.json": "a", "adapter_model.safetensors": "b"}
    initial_binding = {"initial_train_result_path": str(initial.parent / "TRAIN_RESULT.json"),
        "initial_train_result_sha256": "initial", "sha256": "initial"}
    monkeypatch.setattr(ev.frozen_text, "adapter_binding", lambda *args: (files, initial_binding))
    monkeypatch.setattr(ev, "adapter_files", lambda *args: files)
    train = {"schema_version": "rarevlm.vlm_object_train.v1", "condition": "object_only", "seed": 42,
        "status": "completed", "config_sha256": ev.content_sha(config), "adapter_files_sha256": files,
        "initial_adapter_files_sha256": files, "trainable_parameters": 1843200,
        "numerical_path": "original_bf16", "visual_token_budget": 1024, "frozen_parameters_unchanged": True,
        "initial_adapter_dir": str(initial), "train_manifest_sha256": ev.sha_file(manifest),
        "train_manifest_file_sha256": ev.sha_file(manifest), "lora_parameter_precision": "fp32",
        "training_accumulation": accumulation, "training_microbatch_questions": microbatch,
        "object_effective_batch": 16,
        "replay_questions_per_update": 0, "replay_lambda": 0.,
        "model_provenance_sha256": ev.sha_file(model_root.parent / "model_provenance.json"),
        "planned_updates": 1, "completed_updates": 1, "actual_questions_seen": 16,
        "actual_object_exposures": 16, "actual_replay_exposures": 0}
    path = candidate.parent / "TRAIN_RESULT.json"
    path.write_text(json.dumps(train))
    assert ev.adapter_binding(config, "object_only", 42, candidate, model_root)[0] == files
    train["schema_version"] = "rarevlm.vlm_posttrain.v1"
    path.write_text(json.dumps(train))
    with pytest.raises(ValueError, match="schema_version"):
        ev.adapter_binding(config, "object_only", 42, candidate, model_root)


@pytest.mark.parametrize("microbatch,accumulation,effective", [(4, 16, 16), (1, 4, 16),
    (0, 16, 16), (4, 4, 32), (True, 16, 16), (None, None, None)])
def test_effective_training_batch_cannot_silently_change_or_fall_back(microbatch, accumulation, effective):
    with pytest.raises(ValueError, match="object_effective_batch16"):
        ev.training_batch_contract({"training_microbatch": microbatch,
            "training_accumulation": accumulation, "object_effective_batch": effective})


def test_failed_screen_blocks_pope_before_any_gpu_generation(tmp_path):
    config = tmp_path / "config.json"
    config.write_text("{}")
    path = tmp_path / "screen.json"
    path.write_text(json.dumps({"schema_version": "rarevlm.object_presence.screen.v1", "status": "complete",
        "config_sha256": ev.sha_file(config), "seeds": [42], "eligible_candidates": []}))
    for condition in ("initial", "object_only", "object_replay"):
        with pytest.raises(ValueError, match="did not pass"):
            ev.screen_admission(path, config, condition, 42, {"sha256": "fixed"})
