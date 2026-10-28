"""CPU-only paired image-cluster analysis of fixed initial/SFT/DPO dev cells."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics

from rarevlm_paths import SOURCE_ROOT
from vlm_deep.common import NORMALIZER, training_target
from vlm_probe.evaluate import (content_sha, lock_file, now, private_path,
    processor_for, read_json, read_manifest, sha_file, write_json)
from .evaluate import (CONDITIONS, FIXED, adapter_binding, complete_cache,
    config_path, dev_data, model_key, source_hashes)

LABELS = ("unseen_0", "rare_1_5", "medium_6_20", "frequent_gt20")
EXPECTED_COUNTS = dict(zip(LABELS, (687, 342, 171, 417)))
CONTRASTS = (("dpo_minus_sft", "sft", "dpo"),
    ("dpo_minus_initial", "initial", "dpo"), ("sft_minus_initial", "initial", "sft"))
COMMON_LOCK_FIELDS = ("posttrain_config_sha256", "posttrain_config_content_sha256",
    "reference_config_sha256", "source_sha256", "packages", "model_revision",
    "model_provenance_sha256", "model_weight_provenance_sha256", "processor_and_metadata_sha256",
    "actual_parameter_dtypes", "manifest_sha256", "manifest_file_sha256",
    "batch_size", "visual_token_budget", "question_order_sha256", "question_prompt_order_sha256",
    "prompt_suffix", "max_new_tokens", "do_sample", "numerical_path", "precision",
    "vision_precision", "quantization", "tf32_matmul", "tf32_cudnn")
PAIRED_INPUT_FIELDS = ("question_id", "image_id", "question_sha256", "question_prompt_sha256",
    "image_sha256", "image_grid_thw", "image_grid_thw_tensor", "visual_tokens",
    "input_tokens", "input_ids", "attention_mask", "input_ids_sha256", "attention_mask_sha256",
    "pixel_values", "padded_input_tokens", "padded_input_ids_sha256", "padded_attention_mask_sha256",
    "batch_start", "batch_size", "input_truncated")


def bucket(frequency):
    return LABELS[0] if frequency == 0 else LABELS[1] if frequency <= 5 else LABELS[2] if frequency <= 20 else LABELS[3]


def prepare_slices(asset_root, output_root):
    path = asset_root / "prepared/manifest.json"
    manifest = read_manifest(path)
    frequency = Counter()
    for row in manifest["splits"]["train_all"]:
        if row["target"] != training_target(row["answers"]):
            raise ValueError("actual train_all target differs from frozen raw-modal rule")
        frequency[NORMALIZER(row["target"])] += 1
    if len(manifest["splits"]["train_all"]) != 32985 or sum(frequency.values()) != 32985:
        raise ValueError("full actual train_all exposure differs")
    features = []
    for row in manifest["splits"]["dev"]:
        if row["target"] != training_target(row["answers"]):
            raise ValueError("dev raw-modal target differs")
        target = NORMALIZER(row["target"])
        features.append({"question_id": row["question_id"], "image_id": row["image_id"],
            "sft_target_normalized": target, "train_all_sft_target_qa_frequency": frequency[target],
            "sft_target_answer_frequency": bucket(frequency[target])})
    counts = Counter(f["sft_target_answer_frequency"] for f in features)
    if dict(counts) != EXPECTED_COUNTS:
        raise ValueError(f"fixed dev target-frequency slices differ: {dict(counts)}")
    lock = {"schema_version": "rarevlm.posttrain.dev_slices.v1", "manifest_path": str(path),
        "manifest_file_sha256": sha_file(path), "manifest_sha256": manifest["manifest_sha256"],
        "normalizer_source_sha256": sha_file(SOURCE_ROOT / "vlm_deep/third_party/vqa_eval_metric.py"),
        "training_target_source_sha256": sha_file(SOURCE_ROOT / "vlm_deep/common.py"),
        "definition": "normalize actual train_all raw-stripped modal target (lexical ties), one exposure per QA; dev uses its original modal target",
        "train_all_questions": 32985, "dev_questions": 1617, "dev_images": 1024,
        "slice_counts": dict(counts), "features": features,
        "target_frequency_sha256": content_sha(dict(sorted(frequency.items()))),
        "limitations": ["supervised answer-text exposure proxy; not semantic-class/pretraining frequency",
            "slices differ in ambiguity,length,image difficulty; fixed-checkpoint conditional CIs"]}
    lock["slice_lock_sha256"] = content_sha(lock)
    lock_file(output_root / "SLICE_LOCK.json", lock)
    return lock


def read_cell(root, condition, seed, references, images, config_path_value, config, processor=None):
    directory = root / "runs" / condition / f"seed{seed}" / "dev"
    evaluation_path = directory / "EVALUATION.json"
    if not evaluation_path.exists():
        return {"complete": False, "condition": condition, "seed": seed, "path": str(directory)}
    lock = read_json(directory / "INPUT_LOCK.json")
    evaluation = read_json(evaluation_path)
    expected = {"condition": condition, "seed": seed, "split": "dev",
        "model_key": model_key(condition, seed), "posttrain_config_sha256": sha_file(config_path_value),
        "posttrain_config_content_sha256": content_sha(config), "source_sha256": source_hashes(),
        **FIXED, "tf32_matmul": False, "tf32_cudnn": False, "n": 1617,
        "current_original_weight_bytes_verified": True}
    for key, value in expected.items():
        if lock.get(key) != value:
            raise ValueError(f"completed dev fixed binding differs: {key}")
    files, binding = adapter_binding(config, condition, seed, Path(lock["adapter_dir"]), config_path(config, "model_root"))
    if files != lock["adapter_files_sha256"] or binding != lock["training_binding"]:
        raise ValueError("completed training/adapter binding changed")
    prediction_path = directory / "predictions.jsonl"
    data = prediction_path.read_bytes()
    if not data.endswith(b"\n"):
        raise ValueError("completed predictions have interrupted tail")
    if len([json.loads(line) for line in data.splitlines()]) != len(references):
        raise ValueError("completed prediction length differs; analysis never repairs completed files")
    records, repairs = complete_cache(prediction_path, references, images, 4, lock, processor)
    if len(records) != len(references) or repairs:
        raise ValueError("completed predictions are incomplete")
    score = 100 * statistics.mean(r["score"] for r in records)
    if (evaluation.get("status") != "completed"
        or evaluation.get("predictions_sha256") != sha_file(prediction_path)
        or evaluation.get("input_lock_sha256") != content_sha(lock)
        or not math.isclose(score, evaluation["score_percent"], abs_tol=1e-10, rel_tol=0)):
        raise ValueError("completed prediction score/hash differs")
    return {"complete": True, "condition": condition, "seed": seed, "rows": records,
        "lock": lock, "evaluation": evaluation, "score_percent": score,
        "training": read_json(Path(binding["path"])),
        "provenance": {"predictions_file": str(prediction_path), "predictions_sha256": sha_file(prediction_path),
            "evaluation_file": str(evaluation_path), "evaluation_sha256": sha_file(evaluation_path),
            "input_lock_sha256": content_sha(lock), "training_result_sha256": binding["sha256"]}}


def assert_same_inputs(left, right):
    for key in COMMON_LOCK_FIELDS:
        if left["lock"].get(key) != right["lock"].get(key):
            raise ValueError(f"paired common evaluation lock differs: {key}")
    if len(left["rows"]) != len(right["rows"]):
        raise ValueError("paired dev lengths differ")
    for before, after in zip(left["rows"], right["rows"]):
        for key in PAIRED_INPUT_FIELDS:
            if before[key] != after[key]:
                raise ValueError(f"paired actual input differs: {key}")


def assert_matched_training(sft, dpo):
    """Hold chosen exposure, initializer and public data fixed, not optimizer loss."""
    required = ("initial_adapter_files_sha256", "initial_adapter_tensor_sha256",
        "actual_chosen_exposures", "train_manifest_sha256")
    for key in required:
        if key not in sft or key not in dpo or sft[key] != dpo[key]:
            raise ValueError(f"matched posttraining exposure/initializer differs: {key}")
    # Additional trainer evidence must agree when emitted by either condition.
    for key in ("global_pair_order_sha256", "chosen_input_contract_sha256", "trainable_parameters",
        "frozen_parameter_sha256_before", "completed_updates", "planned_updates"):
        if key in sft or key in dpo:
            if key not in sft or key not in dpo or sft[key] != dpo[key]:
                raise ValueError(f"matched posttraining contract differs: {key}")


def paired_interval(before_by_seed, after_by_seed, features, label="overall", replicates=6000, seed=20261015):
    """Average checkpoint axes first; resample image clusters, never seed x image."""
    import numpy as np
    if (not before_by_seed or len(before_by_seed) != len(after_by_seed)
        or any(len(r) != len(features) for r in [*before_by_seed, *after_by_seed])):
        raise ValueError("paired checkpoint/feature sizes differ")
    by_image = defaultdict(lambda: [0.0, 0])
    points, seed_differences = [], [[] for _ in before_by_seed]
    for index, feature in enumerate(features):
        if label != "overall" and feature["sft_target_answer_frequency"] != label:
            continue
        before_values, after_values = [], []
        for axis, (before, after) in enumerate(zip(before_by_seed, after_by_seed)):
            left, right = before[index], after[index]
            for key in ("question_id", "image_id"):
                if left[key] != right[key] or left[key] != feature[key]:
                    raise ValueError("unpaired question/image/feature")
            before_values.append(float(left["score"]))
            after_values.append(float(right["score"]))
            seed_differences[axis].append(float(right["score"]) - float(left["score"]))
        left_mean, right_mean = statistics.mean(before_values), statistics.mean(after_values)
        delta = right_mean - left_mean
        by_image[feature["image_id"]][0] += delta
        by_image[feature["image_id"]][1] += 1
        points.append((left_mean, right_mean))
    if not points:
        raise ValueError("empty predeclared slice")
    if replicates != 6000:
        raise ValueError("predeclared bootstrap requires 6000 replicates")
    matrix = np.asarray(list(by_image.values()), dtype=float)
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sampled = matrix[rng.integers(0, len(matrix), size=len(matrix))].sum(axis=0)
        estimates[index] = 100 * sampled[0] / sampled[1]
    seed_delta_pp = [100 * statistics.mean(values) for values in seed_differences]
    return {"slice": label, "before_score_percent": 100 * statistics.mean(p[0] for p in points),
        "after_score_percent": 100 * statistics.mean(p[1] for p in points),
        "delta_pp": 100 * statistics.mean(p[1] - p[0] for p in points),
        "questions": len(points), "image_clusters": len(matrix), "checkpoint_axes": len(before_by_seed),
        "image_cluster_bootstrap_95_pp": np.quantile(estimates, [.025, .975]).tolist(),
        "one_sided_97_5_lower_pp": float(np.quantile(estimates, .025)),
        "seed_delta_pp": seed_delta_pp,
        "seed_delta_sample_sd_pp": statistics.stdev(seed_delta_pp) if len(seed_delta_pp) > 1 else None,
        "bootstrap_replicates": replicates, "bootstrap_seed": seed,
        "uncertainty": "checkpoint-axis-mean paired image-cluster percentile bootstrap; fixed checkpoints; no seed x image independence; slice CIs pointwise exploratory"}


def validate_analysis_config(config):
    declared = config.get("analysis", {})
    expected = {"bootstrap_replicates": 6000, "bootstrap_seed": 20261015,
        "primary_min_delta_pp": .5, "one_sided_confidence": .975,
        "protection_margin_pp": .5}
    for key, value in expected.items():
        if declared.get(key, value) != value:
            raise ValueError(f"predeclared analysis differs: {key}")
    return {**expected, **declared}


def gate(contrasts, margin=.5):
    lookup = {name: {r["slice"]: r for r in rows} for name, rows in contrasts.items()}
    main_checks = {}
    for name in ("dpo_minus_sft", "dpo_minus_initial"):
        row = lookup[name]["overall"]
        main_checks[f"{name}_at_least_0_5pp"] = row["delta_pp"] >= .5
        main_checks[f"{name}_one_sided_97_5_lower_gt0"] = row["one_sided_97_5_lower_pp"] > 0
    # A stricter, separately labelled internal protection diagnostic. It cannot
    # establish transfer/general multimodal capability and does not trigger formal evaluation.
    protection_checks = {}
    for name in ("dpo_minus_initial", "sft_minus_initial"):
        protection_checks[f"{name}_overall_lower_gt_minus_margin"] = (
            lookup[name]["overall"]["one_sided_97_5_lower_pp"] > -margin)
    for name in ("dpo_minus_sft", "dpo_minus_initial"):
        for label in ("rare_1_5", "frequent_gt20"):
            protection_checks[f"{name}_{label}_lower_gt_minus_margin"] = (
                lookup[name][label]["one_sided_97_5_lower_pp"] > -margin)
    quality = all(main_checks.values())
    protection = all(protection_checks.values())
    return {"quality_passed": quality, "quality_checks": main_checks,
        "quality_family": "two primary overall contrasts, one-sided alpha=.05/2=.025 each",
        "internal_slice_protection_diagnostic_passed": protection,
        "internal_slice_protection_diagnostic_checks": protection_checks,
        "internal_slice_protection_margin_pp": margin,
        "protection_status": "additional conservative diagnostic; pointwise CIs; not an authorized automatic expansion gate or external capability evidence",
        "automatic_new_training": False, "automatic_official_evaluation": False,
        "interpretation": "single-checkpoint exploration until all predeclared seeds exist; dev already explored; no formal-set tuning or claim of general capability retention"}


def analyze(config_path_value, output_root, seeds=(42,), initial_seed=42):
    seeds = tuple(seeds)
    if seeds not in ((42,), (0, 42, 200)):
        raise ValueError("analysis axes must be pilot seed42 or complete ordered seeds0,42,200")
    config = read_json(config_path_value)
    settings = validate_analysis_config(config)
    asset_root = config_path(config, "asset_root")
    slices = prepare_slices(asset_root, output_root)
    references, images, _, _ = dev_data(asset_root)
    reference = read_json(config_path(config, "reference_config", "configs/vlm_deep_v2.json"))
    processor = processor_for(config_path(config, "model_root"), reference, 1024)
    initial = read_cell(output_root, "initial", initial_seed, references, images, config_path_value, config, processor)
    cells = {("initial", initial_seed): initial}
    for seed in seeds:
        for condition in ("sft", "dpo"):
            cells[(condition, seed)] = read_cell(output_root, condition, seed, references, images, config_path_value, config, processor)
    result = {"schema_version": "rarevlm.posttrain.dev_analysis.v1", "updated_at": now(),
        "status": "waiting_dev_predictions", "complete": all(c["complete"] for c in cells.values()),
        "seeds": list(seeds), "initial_seed": initial_seed,
        "initial_reused_across_posttrain_axes": len(seeds) > 1,
        "completed_cells": sum(c["complete"] for c in cells.values()), "expected_cells": len(cells),
        "config_sha256": sha_file(config_path_value), "analysis_source_sha256": sha_file(Path(__file__)),
        "slice_lock_sha256": slices["slice_lock_sha256"], "expected_predictions": 1617 * len(cells),
        "scope": "seed42 single-checkpoint exploration" if len(seeds) == 1 else "three posttraining seed axes; shared fixed initial adapter; image-cluster CIs conditional on checkpoints"}
    destination = output_root / "analysis" / ("pilot_seed42" if len(seeds) == 1 else "replication_seeds0_42_200")
    if not result["complete"]:
        result["pending"] = [c for c in cells.values() if not c["complete"]]
        write_json(destination / "ANALYSIS.json", result)
        return result
    for seed in seeds:
        sft, dpo = cells[("sft", seed)], cells[("dpo", seed)]
        assert_same_inputs(initial, sft)
        assert_same_inputs(sft, dpo)
        assert_matched_training(sft["training"], dpo["training"])
    by_condition = {"initial": [initial["rows"] for _ in seeds],
        **{condition: [cells[(condition, seed)]["rows"] for seed in seeds] for condition in ("sft", "dpo")}}
    contrasts = {}
    for contrast_index, (name, before, after) in enumerate(CONTRASTS):
        contrasts[name] = [paired_interval(by_condition[before], by_condition[after], slices["features"],
            label, settings["bootstrap_replicates"], settings["bootstrap_seed"] + contrast_index * 10 + index)
            for index, label in enumerate(("overall", *LABELS))]
    summaries = {}
    for condition in CONDITIONS:
        scores = [100 * statistics.mean(r["score"] for r in records) for records in by_condition[condition]]
        summaries[condition] = {"mean_score_percent": statistics.mean(scores),
            "seed_scores_percent": scores if condition != "initial" else [initial["score_percent"]],
            "seed_sample_sd_percent": statistics.stdev(scores) if len(scores) > 1 and condition != "initial" else None,
            "independent_trained_adapters": len(seeds) if condition != "initial" else 1}
    result.update(status="complete", validated_predictions=1617 * len(cells), contrasts=contrasts,
        condition_summary=summaries, gate=gate(contrasts, settings["protection_margin_pp"]),
        source_data={f"{c}_{s}": cell["provenance"] for (c, s), cell in cells.items()},
        prompt_limitations="public preference training may use long answers/no suffix; fixed eval original short phrase+greedy32; answer-format transfer is part of the measured outcome",
        capability_protection_recommendation="before any claim of general retention, preregister independent held-out preference chosen/rejected sequence-logprob margin and separate task-specific NI endpoints; do not tune on formal TextVQA/DocVQA/MMStar",
        heldout_preference_margin="not computed by this fixed VQA runner; separate label-free-forward diagnostic required",
        limitations=["public AI/human preference provenance must be stated; preference labels are not VQA truth",
            "internal dev historically exposed; not a new blind benchmark",
            "rare/frequent CIs are pointwise conditional diagnostics, not simultaneous long-tail claims",
            "quality gate does not prove general multimodal capability retention or seed stability"])
    write_json(destination / "ANALYSIS.json", result)
    lines = ["# 公开偏好后训练：固定内部 dev 配对分析", "",
        result["scope"], "", "| 比较 | 切片 | QA / 图像簇 | 差值 pp | 图像簇 95% CI |", "|---|---|---:|---:|---|"]
    for name, rows in contrasts.items():
        for row in rows:
            ci = row["image_cluster_bootstrap_95_pp"]
            lines.append(f"| {name} | {row['slice']} | {row['questions']} / {row['image_clusters']} | {row['delta_pp']:+.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] |")
    lines += ["", f"质量门控：{'通过' if result['gate']['quality_passed'] else '未通过'}。两个主整体比较各需 ≥0.5 pp，且单侧97.5%下界>0。",
        "", "按图像整簇重采样6000次；三种子先平均同图预测差值，样本SD另报，不将seed×image当独立样本。",
        "频次依据原train_all实际SFT target，属于答案文本监督曝光代理；切片区间是探索性点式区间。",
        "内部切片保护诊断不能证明跨任务能力保持；未据正式TextVQA/DocVQA/MMStar调整prompt、预算或checkpoint。",
        "公开偏好训练可能无suffix且答案较长，评测固定原短词prompt与greedy32；未自动扩种子或启动正式集。"]
    (destination / "findings.md").write_text("\n".join(lines) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--initial-seed", type=int, default=42)
    args = parser.parse_args()
    result = analyze(private_path(args.config), private_path(args.output_root), args.seeds, args.initial_seed)
    print(result["status"], "completed cells:", result["completed_cells"], "/", result["expected_cells"])


if __name__ == "__main__":
    main()
