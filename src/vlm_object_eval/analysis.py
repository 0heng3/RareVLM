"""CPU-only preregistered object/Text screen and exposed-POPE confirmation."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import math
from pathlib import Path
import statistics

import numpy as np

from vlm_posttrain_eval.analysis import LABELS, prepare_slices
from vlm_posttrain_pope.analysis import AUDIT_FIELDS
from vlm_probe.evaluate import content_sha, now, private_path, read_json, sha_file, write_json
from .evaluate import (CANDIDATES, FIXED, config_path, read_cell, screen_admission,
    source_hashes, validate_object_schema, verify_source_data)
from .scoring import metrics, official_prediction

REPLICATES = 6000
BOOTSTRAP_SEED = 20261015
CONFIDENCE = .9875
METRICS = ("accuracy", "f1", "recall", "specificity", "precision", "yes_rate")
COMMON_FIELDS = (*FIXED, "study_config_sha256", "study_config_content_sha256", "reference_config_sha256",
    "source_sha256", "packages", "model_revision", "model_config_sha256", "model_provenance_sha256",
    "model_weight_provenance_sha256", "processor_and_metadata_sha256", "actual_parameter_dtypes",
    "data_manifest_sha256", "data_manifest_file_sha256", "questions", "unique_images",
    "question_order_sha256", "question_prompt_order_sha256", "prompt_suffix", "input",
    "head_last_position_only", "explicit_lora_bf16_conversion", "current_original_weight_bytes_verified")


def validate_analysis_config(config):
    settings = {"bootstrap_replicates": REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED,
        "one_sided_confidence": CONFIDENCE, "object_min_delta_pp": 1.,
        "pope_min_delta_pp": .5, "text_noninferiority_margin_pp": .5,
        "specificity_point_margin_pp": .5}
    for key, value in settings.items():
        if config.get("analysis", {}).get(key, value) != value:
            raise ValueError(f"predeclared object analysis differs: {key}")
    return settings


def assert_same_inputs(before, after):
    if len(before) != len(after):
        raise ValueError("paired question lengths differ")
    for left, right in zip(before, after):
        for key in ("question_id", "image_id", *AUDIT_FIELDS):
            if key not in left or key not in right or left[key] != right[key]:
                raise ValueError(f"paired actual input audit differs or missing: {key}")
        for key in ("label", "category"):
            if key in left or key in right:
                if left.get(key) != right.get(key):
                    raise ValueError(f"paired scoring identity differs: {key}")


def assert_common_locks(initial, candidate):
    left, right = initial["lock"], candidate["lock"]
    historical = "historical_reference" in initial
    excluded = {"study_config_sha256", "study_config_content_sha256", "source_sha256", "input"} if historical else set()
    aliases = initial.get("comparison_metadata", {})
    for key in COMMON_FIELDS:
        if key in excluded:
            continue
        value = aliases[key] if key in ("data_manifest_sha256", "data_manifest_file_sha256", "questions", "unique_images") and historical else left.get(key)
        if value is None or key not in right or value != right[key]:
            raise ValueError(f"paired common evaluation lock differs: {key}")


def _interval(delta, point, seed_values):
    undefined = int(np.count_nonzero(~np.isfinite(delta)))
    return {"delta_pp": float(point) if point is not None else None,
        "image_cluster_bootstrap_95_pp": np.quantile(delta, [.025, .975]).tolist() if not undefined and point is not None else None,
        "one_sided_98_75_lower_pp": float(np.quantile(delta, .0125)) if not undefined and point is not None else None,
        "undefined_bootstrap_replicates": undefined,
        "seed_delta_pp": seed_values,
        "seed_delta_sample_sd_pp": statistics.stdev(seed_values) if len(seed_values) > 1 and all(v is not None for v in seed_values) else None}


def scalar_contrast(initial, candidates, *, replicates=REPLICATES, seed=BOOTSTRAP_SEED):
    """Average fixed checkpoint axes first; keep unequal questions per image."""
    if replicates != REPLICATES or seed != BOOTSTRAP_SEED:
        raise ValueError("fixed bootstrap requires 6000 replicates and seed20261015")
    if not initial or not candidates or any(len(rows) != len(initial) for rows in candidates):
        raise ValueError("paired nonempty checkpoint sizes required")
    groups = defaultdict(lambda: [0., 0.])
    seed_values = []
    for rows in candidates:
        for before, after in zip(initial, rows):
            if (before["question_id"], before["image_id"]) != (after["question_id"], after["image_id"]):
                raise ValueError("unpaired question/image")
        seed_values.append(100 * statistics.mean(after["score"] - before["score"] for before, after in zip(initial, rows)))
    for index, before in enumerate(initial):
        after = statistics.mean(rows[index]["score"] for rows in candidates)
        groups[before["image_id"]][0] += after - before["score"]
        groups[before["image_id"]][1] += 1
    matrix = np.asarray(list(groups.values()))
    weights = np.random.default_rng(seed).multinomial(len(matrix), np.full(len(matrix), 1 / len(matrix)), size=replicates)
    sampled = weights @ matrix
    delta = 100 * sampled[:, 0] / sampled[:, 1]
    return {**_interval(delta, statistics.mean(seed_values), seed_values),
        "before_score_percent": 100 * statistics.mean(r["score"] for r in initial),
        "after_score_percent": 100 * statistics.mean(statistics.mean(r["score"] for r in rows) for rows in candidates),
        "questions": len(initial), "image_clusters": len(matrix), "resampled_image_clusters": len(matrix),
        "checkpoint_axes": len(candidates), "bootstrap_replicates": replicates, "bootstrap_seed": seed,
        "uncertainty": "paired image-cluster percentile bootstrap conditional on these checkpoints; fixed seed axes averaged before cluster resampling; seed×image never independent"}


def _rates(counts):
    tp, fp, tn, fn = np.moveaxis(counts, -1, 0)
    numerators = {"accuracy": tp + tn, "f1": 2 * tp, "recall": tp,
        "specificity": tn, "precision": tp, "yes_rate": tp + fp}
    denominators = {"accuracy": tp + fp + tn + fn, "f1": 2 * tp + fp + fn,
        "recall": tp + fn, "specificity": tn + fp, "precision": tp + fp,
        "yes_rate": tp + fp + tn + fn}
    return {key: np.divide(numerators[key], value, out=np.full(value.shape, np.nan, dtype=float), where=value != 0)
        for key, value in denominators.items()}


def binary_contrast(initial, candidates, *, replicates=REPLICATES, seed=BOOTSTRAP_SEED):
    """Shared image draws across every category and checkpoint axis, including POPE."""
    if replicates != REPLICATES or seed != BOOTSTRAP_SEED:
        raise ValueError("fixed bootstrap requires 6000 replicates and seed20261015")
    if not initial or not candidates:
        raise ValueError("nonempty binary checkpoints required")
    groups = [initial, *candidates]
    for rows in groups:
        if len(rows) != len(initial) or len({r["question_id"] for r in rows}) != len(rows):
            raise ValueError("binary question coverage differs")
        for left, right in zip(initial, rows):
            for key in ("question_id", "image_id", "label", "category"):
                if left[key] != right[key]:
                    raise ValueError(f"unpaired binary identity: {key}")
            if right["score"] != float(official_prediction(right["prediction"]) == right["label"]):
                raise ValueError("saved official binary score differs")
    summaries = [metrics(rows) for rows in groups]
    image_ids = sorted({r["image_id"] for r in initial})
    image_index = {name: index for index, name in enumerate(image_ids)}
    slices = ["overall", *sorted({r["category"] for r in initial})]
    slice_index = {name: index for index, name in enumerate(slices)}
    counts = np.zeros((len(image_ids), len(slices), len(groups), 4), dtype=np.int64)
    for axis, rows in enumerate(groups):
        for row in rows:
            pred, label = official_prediction(row["prediction"]), row["label"]
            slot = 0 if pred == label == "yes" else 1 if pred == "yes" else 2 if label == "no" else 3
            for part in (0, slice_index[row["category"]]):
                counts[image_index[row["image_id"]], part, axis, slot] += 1
    weights = np.random.default_rng(seed).multinomial(len(image_ids), np.full(len(image_ids), 1 / len(image_ids)), size=replicates)
    sampled = (weights @ counts.reshape(len(image_ids), -1)).reshape(replicates, len(slices), len(groups), 4)
    sampled_rates = _rates(sampled)
    output = []
    for index, name in enumerate(slices):
        point_summaries = [group["overall"] if name == "overall" else group["by_category"][name] for group in summaries]
        values = {}
        for metric in METRICS:
            left = point_summaries[0][metric]
            right_values = [group[metric] for group in point_summaries[1:]]
            right = statistics.mean(right_values) if all(v is not None for v in right_values) else None
            seed_values = [100 * (value - left) if value is not None and left is not None else None for value in right_values]
            point = 100 * (right - left) if right is not None and left is not None else None
            delta = 100 * (sampled_rates[metric][:, index, 1:].mean(axis=-1) - sampled_rates[metric][:, index, 0])
            values[metric] = {"before": left, "after": right, **_interval(delta, point, seed_values)}
        output.append({"slice": name, "metrics": values, **values["accuracy"],
            "before_score_percent": 100 * values["accuracy"]["before"],
            "after_score_percent": 100 * values["accuracy"]["after"],
            "questions": point_summaries[0]["n"],
            "image_clusters": int(np.count_nonzero(counts[:, index, 0].sum(axis=-1))),
            "resampled_image_clusters": len(image_ids), "checkpoint_axes": len(candidates),
            "bootstrap_replicates": replicates, "bootstrap_seed": seed,
            "uncertainty": "same image-cluster draws across categories and fixed checkpoint axes; categories/secondary metrics have exploratory pointwise intervals"})
    return output, {"initial": summaries[0], "candidate_axes": summaries[1:]}


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def screen_gate(object_rows, text_rows, *, replicated=False, balanced=None):
    obj = next(row for row in object_rows if row["slice"] == "overall")["metrics"]
    text = {row["slice"]: row for row in text_rows}
    acc = obj["accuracy"]
    checks = {"object_accuracy_at_least_1pp": _finite(acc["delta_pp"]) and acc["delta_pp"] >= 1.,
        "object_accuracy_one_sided_98_75_lower_gt0": _finite(acc["one_sided_98_75_lower_pp"]) and acc["one_sided_98_75_lower_pp"] > 0,
        "text_one_sided_98_75_lower_gt_minus_0_5pp": _finite(text["overall"]["one_sided_98_75_lower_pp"]) and text["overall"]["one_sided_98_75_lower_pp"] > -.5}
    guards = {"object_yes_recall_point_nondecrease": _finite(obj["recall"]["delta_pp"]) and obj["recall"]["delta_pp"] >= 0,
        "object_no_specificity_point_at_least_minus_0_5pp": _finite(obj["specificity"]["delta_pp"]) and obj["specificity"]["delta_pp"] >= -.5}
    for label in ("rare_1_5", "frequent_gt20"):
        guards[f"text_{label}_point_nondecrease"] = label in text and _finite(text[label]["delta_pp"]) and text[label]["delta_pp"] >= 0
    if replicated:
        object_deltas = acc.get("seed_delta_pp", [])
        text_deltas = text["overall"].get("seed_delta_pp", [])
        guards["all_three_object_seed_points_gt0"] = len(object_deltas) == 3 and all(_finite(value) and value > 0 for value in object_deltas)
        guards["all_three_text_seed_points_at_least_minus_0_5pp"] = len(text_deltas) == 3 and all(_finite(value) and value >= -.5 for value in text_deltas)
    balanced_point = next((row["delta_pp"] for row in (balanced or {}).get("contrasts", []) if row["slice"] == "overall"), None)
    guards["class_balanced_mean_accuracy_point_gt0"] = (balanced or {}).get("status") == "complete" and _finite(balanced_point) and balanced_point > 0
    return {"passed": all(checks.values()) and all(guards.values()), "primary_checks": checks,
        "point_protection_checks": guards, "family": "two declared arms × object improvement and Text NI endpoints; one-sided alpha=.05/4=.0125",
        "point_protection_interpretation": "point guards only; not slice noninferiority, equivalence, or proof of long-tail improvement"}


def confirm_gate(pope_rows, screen):
    values = next(row for row in pope_rows if row["slice"] == "overall")["metrics"]
    acc = values["accuracy"]
    checks = {"screen_passed": screen["passed"],
        "pope_accuracy_at_least_0_5pp": _finite(acc["delta_pp"]) and acc["delta_pp"] >= .5,
        "pope_accuracy_one_sided_98_75_lower_gt0": _finite(acc["one_sided_98_75_lower_pp"]) and acc["one_sided_98_75_lower_pp"] > 0}
    guards = {f"pope_{name}_point_nondecrease": _finite(values[name]["delta_pp"]) and values[name]["delta_pp"] >= 0 for name in ("f1", "recall")}
    guards["pope_specificity_point_at_least_minus_0_5pp"] = _finite(values["specificity"]["delta_pp"]) and values["specificity"]["delta_pp"] >= -.5
    seed_deltas = acc.get("seed_delta_pp", [])
    guards["all_three_pope_seed_accuracy_points_gt0"] = len(seed_deltas) == 3 and all(_finite(value) and value > 0 for value in seed_deltas)
    by_category = {row["slice"]: row for row in pope_rows if row["slice"] != "overall"}
    for category in ("random", "popular", "adversarial"):
        point = by_category.get(category, {}).get("metrics", {}).get("accuracy", {}).get("delta_pp")
        guards[f"pope_{category}_mean_accuracy_point_nondecrease"] = _finite(point) and point >= 0
    return {"passed": all(checks.values()) and all(guards.values()), "primary_checks": checks,
        "point_protection_checks": guards, "confidence": CONFIDENCE,
        "point_protection_interpretation": "F1/recall/specificity point guards, not noninferiority"}


def _text_contrasts(initial, candidates, features):
    output = []
    for label in ("overall", *LABELS):
        indices = [index for index, feature in enumerate(features) if label == "overall" or feature["sft_target_answer_frequency"] == label]
        row = scalar_contrast([initial[i] for i in indices], [[rows[i] for i in indices] for rows in candidates])
        output.append({"slice": label, **row,
            "slice_interpretation": "actual SFT target-answer exposure; not semantic entity or pretraining frequency; slice CIs exploratory"})
    return output


def _balanced_diagnostic(config, object_initial, object_candidates):
    if "val_class_balanced_manifest" not in config:
        return {"status": "not_declared"}
    path = config_path(config, "val_class_balanced_manifest")
    manifest = read_json(path)
    expected_protocol = sha_file(config_path(config, "dataset_protocol")) if config.get("dataset_protocol") else None
    schema = validate_object_schema(manifest, config.get("object_manifest_schema_version"), expected_protocol)
    if (manifest.get("split") != "val_class_balanced"
        or content_sha({k: v for k, v in manifest.items() if k != "manifest_sha256"}) != manifest.get("manifest_sha256")):
        raise ValueError("class-balanced diagnostic manifest differs")
    full = read_json(config_path(config, "val_manifest", config.get("object_data_root")))
    if validate_object_schema(full, config.get("object_manifest_schema_version"), expected_protocol) != schema:
        raise ValueError("class-balanced and full validation manifest versions differ")
    raw_index = {row["question_id"]: row for row in full["rows"]}
    qids = {row["question_id"] for row in manifest["rows"]}
    if len(qids) != len(manifest["rows"]) or any(raw_index.get(row["question_id"]) != row for row in manifest["rows"]):
        raise ValueError("class-balanced rows must be exact full-validation subset")
    label_counts = Counter((row["category_name"], row.get("label", row.get("target"))) for row in manifest["rows"])
    if not label_counts or any(label_counts[category, "yes"] != label_counts[category, "no"]
        or label_counts[category, "yes"] == 0 for category, _ in label_counts):
        raise ValueError("class-balanced diagnostic must balance yes/no within every included category")
    selected = [[row for row in rows if row["question_id"] in qids] for rows in [object_initial, *object_candidates]]
    if any(len(rows) != len(qids) for rows in selected):
        raise ValueError("class-balanced predictions incomplete")
    contrast, summaries = binary_contrast(selected[0], selected[1:])
    return {"status": "complete", "manifest_path": str(path), "manifest_sha256": sha_file(path),
        "manifest_schema_version": schema,
        "question_ids_sha256": content_sha(sorted(qids)), "questions": len(qids),
        "contrasts": contrast, "summaries": summaries,
        "scope": "grounding diagnostic from exact full-val prediction subset; zero new generation; cannot replace full-val/POPE gates"}


def select_candidate(eligible, contrasts):
    """Only internal full-object accuracy, then Text, then fixed condition name."""
    scores = {}
    for condition in eligible:
        obj = next(row for row in contrasts[f"{condition}_minus_initial_object"] if row["slice"] == "overall")
        text = next(row for row in contrasts[f"{condition}_minus_initial_text"] if row["slice"] == "overall")
        scores[condition] = {"object_accuracy_percent": obj["after_score_percent"],
            "text_consensus_percent": text["after_score_percent"]}
    ranked = sorted(eligible, key=lambda name: (-scores[name]["object_accuracy_percent"],
        -scores[name]["text_consensus_percent"], name))
    return (ranked[0] if ranked else None), scores


def pilot_admission(path, config_file):
    path = private_path(path)
    value = read_json(path)
    if (value.get("schema_version") != "rarevlm.object_presence.screen.v1" or value.get("status") != "complete"
        or value.get("stage") != "pilot_order_seed42" or value.get("seeds") != [42]
        or value.get("config_sha256") != sha_file(config_file) or value.get("source_sha256") != source_hashes()):
        raise ValueError("replication requires the completed unchanged same-config pilot42 screen")
    eligible = value.get("eligible_candidates", [])
    if any(name not in CANDIDATES or not value.get("gates", {}).get(name, {}).get("passed") for name in eligible):
        raise ValueError("pilot eligible candidate/gate differs")
    verify_source_data(value.get("source_data", {}))
    return value


def engineering_exclusions(path, config_file):
    """Only byte-verified CUDA OOM/cap receipts may remove an unscored arm."""
    if path is None:
        return set(), None
    from vlm_object_study.common import read_receipt, verified_files
    path = private_path(path)
    value = read_json(path)
    if (value.get("schema_version") != "rarevlm.object_presence.engineering_admission.v1"
        or value.get("config_sha256") != sha_file(config_file) or value.get("engineering_only") is not True
        or value.get("alpha_unchanged") != .0125 or value.get("automatic_microbatch_fallback") is not False):
        raise ValueError("engineering exclusions cannot alter scientific thresholds")
    rejected = value.get("rejected_candidates", {})
    for condition, failures in rejected.items():
        if condition not in CANDIDATES or not failures:
            raise ValueError("invalid engineering-excluded candidate")
        for failure in failures:
            receipt_path = private_path(failure["worker_receipt"])
            if sha_file(receipt_path) != failure["worker_receipt_sha256"]:
                raise ValueError("engineering failure receipt bytes changed")
            receipt = read_receipt(receipt_path)
            if (receipt["condition"] != condition or receipt["outcome"] != "engineering_rejected"
                or failure["reason"] != receipt["engineering_reason"]):
                raise ValueError("engineering failure identity/reason differs")
            verified_files(failure["evidence_files_sha256"])
    return set(rejected), {"path": str(path), "sha256": sha_file(path), "rejected_candidates": rejected,
        "interpretation": "engineering rejection only; no negative accuracy imputed; alpha=.05/4 retained"}


def analyze(config_file, output_root, seeds=(42,), phase="screen", screen_file=None, pilot_file=None, engineering_file=None):
    config_file, output_root = private_path(config_file), private_path(output_root)
    config = read_json(config_file)
    settings = validate_analysis_config(config)
    if tuple(config.get("candidates", CANDIDATES)) != CANDIDATES:
        raise ValueError("both arms must be predeclared")
    seeds = sorted(set(seeds))
    if seeds not in ([42], [0, 42, 200]):
        raise ValueError("analysis uses pilot42 or the complete three order seeds0,42,200")
    if phase not in ("screen", "confirm"):
        raise ValueError("unknown analysis phase")
    excluded, engineering = engineering_exclusions(engineering_file, config_file)
    screen = read_json(private_path(screen_file)) if phase == "confirm" and screen_file else None
    if phase == "confirm":
        if (screen is None or screen.get("schema_version") != "rarevlm.object_presence.screen.v1"
            or screen.get("status") != "complete" or screen.get("config_sha256") != sha_file(config_file)
            or screen.get("seeds") != seeds or seeds != [0, 42, 200]
            or screen.get("stage") != "replicated_internal_selection"):
            raise ValueError("confirmation requires completed same-config/same-seeds screen")
        selected = screen.get("selected_candidate")
        if selected not in screen.get("eligible_candidates", []) or selected not in CANDIDATES:
            raise ValueError("no unique candidate was selected using internal object/Text results")
        candidates = [selected]
        tasks = ("pope",)
    else:
        if seeds == [42]:
            candidates = CANDIDATES
            pilot = None
        else:
            pilot_file = private_path(pilot_file or output_root / "analysis/pilot_seed42/screen/ANALYSIS.json")
            pilot = pilot_admission(pilot_file, config_file)
            candidates = pilot["eligible_candidates"]
        tasks = ("object", "text")
        candidates = [name for name in candidates if name not in excluded]
    cells = {}
    for task in tasks:
        cells[f"initial_42_{task}"] = read_cell(output_root, "initial", 42, task, config_file, config)
        for condition in candidates:
            for seed in seeds:
                key = f"{condition}_{seed}_{task}"
                cells[key] = read_cell(output_root, condition, seed, task, config_file, config)
    destination = output_root / "analysis" / ("pilot_seed42" if seeds == [42] else "seeds_" + "_".join(map(str, seeds))) / phase / "ANALYSIS.json"
    result = {"schema_version": f"rarevlm.object_presence.{phase}.v1", "updated_at": now(),
        "status": "waiting_predictions", "phase": phase, "seeds": seeds,
        "stage": "pilot_order_seed42" if seeds == [42] else "replicated_internal_selection" if phase == "screen" else "selected_candidate_pope_confirmation",
        "config_sha256": sha_file(config_file), "source_sha256": source_hashes(), "settings": settings,
        "initial_checkpoint": "one shared original full-data SFT42 checkpoint across order seeds",
        "scope": "single order-seed exploration" if seeds == [42] else "fixed order-seed replication; image uncertainty conditional on checkpoints",
        "pending_cells": [name for name, cell in cells.items() if not cell["complete"]],
        "limitations": ["POPE was exposed in prior study; not blind and pretraining overlap unknown",
            "class-balanced subset is supplementary grounding evidence only", "point protection does not establish slice NI or long-tail improvement"],
        "selection_rule": "Among internally eligible arms maximize aggregate full-object validation accuracy, then Text consensus, then ascending fixed condition name; POPE never enters selection"}
    if engineering is not None:
        result["engineering_exclusions"] = engineering
    if phase == "screen" and pilot is not None:
        result.update(pilot_screen_path=str(pilot_file), pilot_screen_sha256=sha_file(pilot_file),
            replication_candidates=list(candidates))
    if result["pending_cells"]:
        write_json(destination, result)
        return result, destination
    result.update(status="complete", source_data={name: cell["provenance"] for name, cell in cells.items()},
        contrasts={}, summaries={}, gates={}, by_seed={})
    if phase == "screen":
        slices = prepare_slices(config_path(config, "asset_root"), output_root / "analysis" / "text_slices")
        features = slices["features"]
        result["text_slice_lock_sha256"] = slices["slice_lock_sha256"]
    for condition in candidates:
        per_task = {}
        for task in tasks:
            initial = cells[f"initial_42_{task}"]
            matched = [cells[f"{condition}_{seed}_{task}"] for seed in seeds]
            for candidate in matched:
                assert_common_locks(initial, candidate)
                assert_same_inputs(initial["rows"], candidate["rows"])
            rows = [cell["rows"] for cell in matched]
            if task == "text":
                per_task[task] = _text_contrasts(initial["rows"], rows, features)
            else:
                per_task[task], summaries = binary_contrast(initial["rows"], rows)
                result["summaries"][f"{condition}_{task}"] = summaries
            result["contrasts"][f"{condition}_minus_initial_{task}"] = per_task[task]
        # Preserve individual order-seed evidence; never turn seeds×images into an n.
        for seed in seeds:
            single = {}
            for task in tasks:
                initial = cells[f"initial_42_{task}"]["rows"]
                rows = cells[f"{condition}_{seed}_{task}"]["rows"]
                single[task] = _text_contrasts(initial, [rows], features) if task == "text" else binary_contrast(initial, [rows])[0]
            point_checks = {"object_accuracy_point_gt0": single["object"][0]["delta_pp"] > 0,
                "text_point_at_least_minus_0_5pp": single["text"][0]["delta_pp"] >= -.5} if phase == "screen" else {
                "pope_accuracy_point_gt0": single["pope"][0]["delta_pp"] > 0}
            result["by_seed"][f"{condition}_{seed}"] = {"contrasts": single, "point_checks": point_checks,
                "scope": "per-seed intervals reported for exploration; individual CI significance is not an acceptance requirement"}
        if phase == "screen":
            balanced = _balanced_diagnostic(config,
                cells["initial_42_object"]["rows"], [cells[f"{condition}_{seed}_object"]["rows"] for seed in seeds])
            result.setdefault("class_balanced_diagnostics", {})[condition] = balanced
            result["gates"][condition] = screen_gate(per_task["object"], per_task["text"], replicated=seeds == [0, 42, 200], balanced=balanced)
        else:
            result["gates"][condition] = confirm_gate(per_task["pope"], screen["gates"][condition])
            for seed in seeds:
                binding = cells[f"{condition}_{seed}_pope"]["lock"]["training_binding"]
                screen_admission(screen_file, config_file, condition, seed, binding)
    passed = [condition for condition in candidates if result["gates"][condition]["passed"]]
    if phase == "screen":
        result["eligible_candidates"] = passed
        result["selected_candidate"], result["selection_scores"] = select_candidate(passed, result["contrasts"]) if seeds == [0, 42, 200] else (None, {})
        result["pope_confirmation_state"] = ("admitted_for_one_internally_selected_candidate" if passed else "not_run_screen_gate_failed") if seeds == [0, 42, 200] else ("not_run_pending_three_seed_internal_replication" if passed else "not_run_pilot_gate_failed")
    else:
        result["screen_analysis_path"] = str(private_path(screen_file))
        result["screen_analysis_sha256"] = sha_file(private_path(screen_file))
        result["confirmed_candidates"] = passed
        result["selected_candidate"] = screen["selected_candidate"]
        result["stable_gain_candidates"] = passed
        result["true_improvement_goal_achieved"] = bool(result["stable_gain_candidates"])
        result["screen_source_data"] = screen["source_data"]
    write_json(destination, result)
    return result, destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("screen", "confirm"), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", choices=(0, 42, 200), default=[42])
    parser.add_argument("--screen-analysis", type=Path)
    parser.add_argument("--pilot-screen", type=Path)
    parser.add_argument("--engineering-admission", type=Path)
    args = parser.parse_args()
    result, path = analyze(args.config, args.output_root, args.seeds, args.phase, args.screen_analysis, args.pilot_screen, args.engineering_admission)
    print(result["status"], str(path), flush=True)


if __name__ == "__main__":
    main()
