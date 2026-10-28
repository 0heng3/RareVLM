"""Meaningful CPU contracts for the independent fixed posttraining evaluation."""
from copy import deepcopy
import json

import pytest

from vlm_deep.common import PROMPT_SUFFIX
from vlm_posttrain_eval import evaluate as ev
from vlm_posttrain_eval.analysis import (assert_matched_training, gate,
    paired_interval, validate_analysis_config)


REFERENCE = {"precision": "fp32", "training_quantization": "nf4_language_only",
    "min_pixels": 50176, "max_pixels": 200704, "max_new_tokens": 32, "do_sample": False}


def test_training_prompt_cannot_change_evaluation_or_leak_references():
    row = {"question_id": "1", "image_id": "a", "question": "Which word?",
        "target": "secret", "answers": ["secret"] * 10, "prompt_suffix": "bad"}
    clean = ev.inference_row(row)
    assert set(clean) == {"question_id", "image_id", "question", "prompt_suffix"}
    assert clean["prompt_suffix"] == PROMPT_SUFFIX
    config = {"prompt_suffix": "", "training": {"max_new_tokens": 256}}
    assert ev.validate_config(config, REFERENCE)["max_new_tokens"] == 32
    for key, value in (("prompt_suffix", ""), ("max_new_tokens", 128), ("visual_token_budget", 512),
        ("do_sample", True), ("input_truncation", True)):
        with pytest.raises(ValueError, match=key):
            ev.validate_config({"evaluation": {key: value}}, REFERENCE)


def test_tensor_hash_is_actual_bytes_and_supports_bfloat16():
    import torch
    a = torch.tensor([[1, 2]], dtype=torch.bfloat16)
    before = ev.tensor_binding(a)
    assert before["dtype"] == "bfloat16"
    assert before["shape"] == [1, 2]
    a[0, 0] = 3
    assert before["sha256"] != ev.tensor_binding(a)["sha256"]


def cache_fixture():
    import torch
    rows = [{"question_id": str(i), "image_id": "a", "question": f"Q{i}", "answers": ["yes"] * 10}
        for i in range(9)]
    images = {"a": {"sha256": "f" * 64}}
    lock = {"model_key": "dpo_seed42", "condition": "dpo", "seed": 42,
        "model_revision": "fixed", "adapter_files_sha256": {"a": "b"}, "source_sha256": {"x": "y"}}
    records = []
    for i, row in enumerate(rows):
        ids, mask, generated = [151655, 10], [1, 1], [20]
        item = {"question_id": row["question_id"], "image_id": "a", "prediction": "yes", "score": 1.,
            "question_sha256": ev.content_sha(row["question"]), "image_sha256": images["a"]["sha256"],
            "task": "textvqa", "model_key": "dpo_seed42", "condition": "dpo", "seed": 42,
            "input_lock_sha256": ev.content_sha(lock), "source_binding_sha256": ev.content_sha(lock["source_sha256"]),
            "question_prompt_sha256": ev.question_prompt_sha(row), "numerical_path": "original_bf16",
            "precision": "bf16", "quantization": "original", "visual_token_budget": 1024,
            "metric": ev.METRIC, "model_revision": "fixed", "adapter_files_sha256": {"a": "b"},
            "batch_start": i // 4 * 4, "batch_size": min(4, 9 - i // 4 * 4),
            "image_grid_thw": [1, 2, 2], "visual_tokens": 1, "counted_image_tokens": 1,
            "premerge_patch_tokens": 4, "input_tokens": 2, "input_truncated": False,
            "input_ids": ids, "attention_mask": mask, "generated_ids": generated,
            "input_ids_sha256": ev.content_sha(ids), "attention_mask_sha256": ev.content_sha(mask),
            "generated_ids_sha256": ev.content_sha(generated), "generated_tail_slots": 1,
            "generated_nonpad_tokens": 1, "pad_token_id": 0, "padded_input_tokens": 2,
            "padded_input_ids_sha256": ev.content_sha(ids), "padded_attention_mask_sha256": ev.content_sha(mask),
            "pixel_values": {"dtype": "float32", "shape": [4, 1176], "sha256": "a" * 64},
            "image_grid_thw_tensor": ev.tensor_binding(torch.tensor([[1, 2, 2]], dtype=torch.int64)),
            **{k: .1 for k in ("batch_generation_seconds", "batch_preprocessing_seconds", "batch_h2d_seconds",
                "batch_peak_allocated_gib", "batch_peak_reserved_gib")}}
        records.append(item)
    return rows, images, lock, records


def test_resume_discards_only_incomplete_final_batch(tmp_path):
    rows, images, lock, records = cache_fixture()
    path = tmp_path / "predictions.jsonl"
    # Six complete JSONL rows plus an interrupted seventh row represent one
    # complete batch and one interrupted batch, not six reusable predictions.
    path.write_text("".join(json.dumps(r) + "\n" for r in records[:6]) + '{"question_id":')
    cached, repairs = ev.complete_cache(path, rows, images, 4, lock)
    assert len(cached) == 4
    assert len(path.read_text().splitlines()) == 4
    assert len(repairs) == 2
    # The final real batch has one QA and is legitimately complete.
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    assert len(ev.complete_cache(path, rows, images, 4, lock)[0]) == 9


@pytest.mark.parametrize("change", ("score", "token", "lock"))
def test_cache_rejects_score_input_or_runtime_change(tmp_path, change):
    rows, images, lock, records = cache_fixture()
    if change == "score":
        records[0]["score"] = 0.
    elif change == "token":
        records[0]["input_ids"][1] = 99
    else:
        lock["model_revision"] = "changed"
    path = tmp_path / "predictions.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    with pytest.raises(ValueError):
        ev.complete_cache(path, rows, images, 4, lock)


def test_image_bootstrap_keeps_same_image_questions_together():
    features = [{"question_id": str(i), "image_id": image, "sft_target_answer_frequency": "rare_1_5"}
        for i, image in enumerate(("a", "a", "b"))]
    before = [{**f, "score": 0.} for f in features]
    after = [{**f, "score": float(f["image_id"] == "a")} for f in features]
    result = paired_interval([before], [after], features)
    assert result["image_clusters"] == 2
    assert result["questions"] == 3
    assert result["delta_pp"] == pytest.approx(200 / 3)
    assert result["image_cluster_bootstrap_95_pp"] == [0., 100.]


def test_checkpoint_axes_are_averaged_before_image_bootstrap():
    features = [{"question_id": "1", "image_id": "a", "sft_target_answer_frequency": "rare_1_5"}]
    zero = [{**features[0], "score": 0.}]
    one = [{**features[0], "score": 1.}]
    result = paired_interval([zero, one], [one, zero], features)
    assert result["image_clusters"] == 1
    assert result["checkpoint_axes"] == 2
    assert result["image_cluster_bootstrap_95_pp"] == [0., 0.]
    assert result["seed_delta_sample_sd_pp"] == pytest.approx(2 ** .5 * 100)


def test_gate_requires_both_primary_effects_and_strict_positive_lower():
    base = [{"slice": label, "delta_pp": .5, "one_sided_97_5_lower_pp": .01}
        for label in ("overall", "rare_1_5", "frequent_gt20")]
    comparisons = {name: deepcopy(base) for name in ("dpo_minus_sft", "dpo_minus_initial", "sft_minus_initial")}
    assert gate(comparisons)["quality_passed"] is True
    comparisons["dpo_minus_initial"][0]["one_sided_97_5_lower_pp"] = 0
    assert gate(comparisons)["quality_passed"] is False
    with pytest.raises(ValueError, match="bootstrap_replicates"):
        validate_analysis_config({"analysis": {"bootstrap_replicates": 2000}})


def test_matched_training_rejects_unequal_chosen_exposure():
    left = {"initial_adapter_files_sha256": {"a": "b"}, "initial_adapter_tensor_sha256": "c",
        "actual_chosen_exposures": 5000, "train_manifest_sha256": "d"}
    right = {**left, "actual_chosen_exposures": 4999}
    with pytest.raises(ValueError, match="actual_chosen_exposures"):
        assert_matched_training(left, right)
