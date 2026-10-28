"""One KL candidate, four preregistered endpoints; paired image-cluster analysis."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from vlm_object_eval.analysis import binary_contrast, scalar_contrast, confirm_gate as original_pope_gate
from vlm_posttrain_eval.analysis import LABELS, prepare_slices
from vlm_posttrain_pope.analysis import AUDIT_FIELDS
from vlm_probe.evaluate import content_sha, now, read_json, sha_file, write_json
from .contracts import CANDIDATE, FIXED, external_data, load_data, path, source_hashes
from .read import phase_admission, read_cell, verify_source_data

SCIENCE_FIELDS = (*FIXED, "reference_config_sha256", "model_revision", "model_config_sha256",
    "model_provenance_sha256", "model_weight_provenance_sha256", "processor_and_metadata_sha256",
    "actual_parameter_dtypes", "packages", "prompt_suffix", "question_order_sha256",
    "question_prompt_order_sha256", "head_last_position_only", "explicit_lora_bf16_conversion",
    "current_original_weight_bytes_verified", "data_manifest_sha256", "data_manifest_file_sha256",
    "questions", "unique_images")


def assert_pair(before, after, task):
    aliases = before.get("comparison_metadata", {})
    for key in SCIENCE_FIELDS:
        value = aliases.get(key) if key in ("data_manifest_sha256", "data_manifest_file_sha256", "questions", "unique_images") and aliases else before["lock"].get(key)
        if value is None or value != after["lock"].get(key):
            raise ValueError(f"paired scientific data/model/processor contract differs: {key}")
    if len(before["rows"]) != len(after["rows"]):
        raise ValueError("paired full split lengths differ")
    for left, right in zip(before["rows"], after["rows"]):
        for key in ("question_id", "image_id", *AUDIT_FIELDS):
            if key not in left or key not in right or left[key] != right[key]:
                raise ValueError(f"paired actual inputs differ or missing: {key}")
        if task != "text":
            for key in ("label", "category"):
                if left[key] != right[key]:
                    raise ValueError(f"paired binary scoring identity differs: {key}")
        elif "category" in left or "category" in right:
            raise ValueError("Text reader must explicitly canonicalize only empty category aliases")


def text_contrasts(initial, candidates, features):
    result = []
    for label in ("overall", *LABELS):
        indices = [index for index, feature in enumerate(features)
            if label == "overall" or feature["sft_target_answer_frequency"] == label]
        result.append({"slice": label, **scalar_contrast([initial[i] for i in indices],
            [[rows[i] for i in indices] for rows in candidates])})
    return result


def balanced_contrast(config, task, initial, candidates):
    full, _, _, _ = load_data(config, task)
    if task == "external":
        subset, _, _, metadata = external_data(config, balanced=True)
    else:
        from vlm_object_eval.evaluate import validate_object_schema
        file = path(config, "val_class_balanced_manifest")
        document = read_json(file)
        validate_object_schema(document, config.get("object_manifest_schema_version"), sha_file(path(config, "dataset_protocol")))
        if (document.get("split") != "val_class_balanced" or content_sha({key: value for key, value in document.items() if key != "manifest_sha256"}) != document.get("manifest_sha256")):
            raise ValueError("fixed internal balanced manifest changed")
        subset = [{**row, "label": row["target"], "category": row["category_name"]} for row in document["rows"]]
        metadata = {"data_manifest_path": str(file), "data_manifest_file_sha256": sha_file(file)}
    full_index = {row["question_id"]: row for row in full}
    qids = {row["question_id"] for row in subset}
    if len(qids) != len(subset) or any(full_index.get(row["question_id"]) != row for row in subset):
        raise ValueError("balanced confirmation must be an exact full-manifest row subset")
    counts = Counter((row["category"], row["label"]) for row in subset)
    if not counts or any(counts[category, "yes"] == 0 or counts[category, "yes"] != counts[category, "no"] for category, _ in counts):
        raise ValueError("per-category diagnostic must balance yes/no exactly")
    selected = [[row for row in rows if row["question_id"] in qids] for rows in [initial, *candidates]]
    if any(len(rows) != len(qids) for rows in selected):
        raise ValueError("balanced full-prediction subset is incomplete")
    contrasts, summary = binary_contrast(selected[0], selected[1:])
    return {"status": "complete", "metadata": metadata, "questions": len(qids),
        "unique_images": len({row["image_id"] for row in selected[0]}), "contrasts": contrasts,
        "summary": summary, "new_generation_questions": 0,
        "scope": "fixed balanced subset of full predictions only; not a replacement for full-set confirmation"}


def overall(rows):
    return next(row for row in rows if row["slice"] == "overall")


def screen_gate(obj, text, balanced, *, replicated):
    values, txt = overall(obj)["metrics"], {row["slice"]: row for row in text}
    accuracy = values["accuracy"]
    checks = {"object_gain_at_least1pp": accuracy["delta_pp"] is not None and accuracy["delta_pp"] >= 1.,
        "object_lower98_75_gt0": accuracy["one_sided_98_75_lower_pp"] is not None and accuracy["one_sided_98_75_lower_pp"] > 0,
        "text_NI_lower98_75_gt_minus0_5": txt["overall"]["one_sided_98_75_lower_pp"] > -.5,
        "object_recall_point_nondecrease": values["recall"]["delta_pp"] is not None and values["recall"]["delta_pp"] >= 0,
        "object_specificity_point_at_least_minus0_5": values["specificity"]["delta_pp"] is not None and values["specificity"]["delta_pp"] >= -.5,
        "balanced_object_point_gt0": overall(balanced["contrasts"])["delta_pp"] > 0,
        "text_rare_point_nondecrease": txt["rare_1_5"]["delta_pp"] >= 0,
        "text_frequent_point_nondecrease": txt["frequent_gt20"]["delta_pp"] >= 0}
    if replicated:
        checks["all_three_object_seed_points_gt0"] = len(accuracy["seed_delta_pp"]) == 3 and all(value > 0 for value in accuracy["seed_delta_pp"])
        checks["all_three_text_seed_points_at_least_minus0_5"] = len(txt["overall"]["seed_delta_pp"]) == 3 and all(value >= -.5 for value in txt["overall"]["seed_delta_pp"])
    return {"passed": all(checks.values()), "checks": checks,
        "family": "one preregistered candidate × internal object/TextNI/external object/POPE accuracy; alpha=.05/4=.0125 each",
        "point_guard_interpretation": "point guards are not slice/recall/specificity NI or equivalence"}


def external_gate(contrasts, balanced, internal):
    values = overall(contrasts)["metrics"]
    acc = values["accuracy"]
    checks = {"internal_three_seed_screen_passed": internal["gate"]["passed"],
        "external_gain_at_least0_5pp": acc["delta_pp"] >= .5,
        "external_lower98_75_gt0": acc["one_sided_98_75_lower_pp"] > 0,
        "all_three_external_seed_points_gt0": len(acc["seed_delta_pp"]) == 3 and all(value > 0 for value in acc["seed_delta_pp"]),
        "external_recall_point_nondecrease": values["recall"]["delta_pp"] is not None and values["recall"]["delta_pp"] >= 0,
        "external_f1_point_nondecrease": values["f1"]["delta_pp"] is not None and values["f1"]["delta_pp"] >= 0,
        "external_specificity_point_at_least_minus0_5": values["specificity"]["delta_pp"] is not None and values["specificity"]["delta_pp"] >= -.5,
        "balanced_external_point_gt0": overall(balanced["contrasts"])["delta_pp"] > 0}
    return {"passed": all(checks.values()), "checks": checks, "Text_protection_retained_from_unchanged_screen": True,
        "scope": "external confirmation only; no hyperparameter/checkpoint selection or tuning"}


def validate_settings(config):
    expected = {"bootstrap_replicates": 6000, "bootstrap_seed": 20261015, "one_sided_confidence": .9875,
        "object_min_delta_pp": 1., "text_noninferiority_margin_pp": .5,
        "external_min_delta_pp": .5, "pope_min_delta_pp": .5, "specificity_point_margin_pp": .5}
    for key, value in expected.items():
        if config.get("analysis", {}).get(key, value) != value:
            raise ValueError(f"fixed four-endpoint analysis setting differs: {key}")
    return expected


def analyze(config_file, root, seeds=(42,), phase="screen", *, pilot_file=None, screen_file=None, external_file=None):
    config_file, root = Path(config_file).resolve(), Path(root).resolve()
    config = read_json(config_file)
    settings = validate_settings(config)
    seeds = sorted(set(seeds))
    if seeds not in ([42], [0, 42, 200]) or phase not in ("screen", "external", "confirm"):
        raise ValueError("only pilot42 or full three-order-seed analysis is allowed")
    prior = None
    if phase == "screen" and seeds == [0, 42, 200]:
        pilot_file = Path(pilot_file or root / "analysis/pilot_seed42/screen/ANALYSIS.json")
        prior = read_json(pilot_file)
        if (prior.get("schema_version") != "rarevlm.object_kl.screen.v1" or prior.get("status") != "complete"
            or prior.get("seeds") != [42] or prior.get("gate", {}).get("passed") is not True
            or prior.get("config_sha256") != sha_file(config_file) or prior.get("source_sha256") != source_hashes()):
            raise ValueError("three-seed expansion requires the unchanged passing pilot42")
        verify_source_data(prior["source_data"])
    elif phase in ("external", "confirm"):
        if seeds != [0, 42, 200]:
            raise ValueError("confirmation requires all three order seeds")
        prior = phase_admission(screen_file if phase == "external" else external_file, config_file,
            "screen" if phase == "external" else "external")
    tasks = ("object", "text") if phase == "screen" else ("external",) if phase == "external" else ("pope",)
    cells = {}
    for task in tasks:
        cells[f"initial_42_{task}"] = read_cell(root, "initial", 42, task, config_file, config)
        for seed in seeds:
            cells[f"{CANDIDATE}_{seed}_{task}"] = read_cell(root, CANDIDATE, seed, task, config_file, config)
    destination = root / "analysis" / ("pilot_seed42" if seeds == [42] else "seeds_0_42_200") / phase / "ANALYSIS.json"
    result = {"schema_version": f"rarevlm.object_kl.{phase}.v1", "status": "waiting_predictions",
        "stage": "pilot_order_seed42" if seeds == [42] else "replicated_internal_screen" if phase == "screen" else phase + "_confirmation",
        "updated_at": now(), "config_sha256": sha_file(config_file), "source_sha256": source_hashes(),
        "seeds": seeds, "candidate": CANDIDATE, "settings": settings,
        "pending_cells": [key for key, value in cells.items() if not value["complete"]],
        "initial_checkpoint": "one shared fixed SFT42 across order-seed axes",
        "limitations": ["fixed-checkpoint image-cluster uncertainty; seed sample SD separate; no seed×image independence",
            "old342 internal validation and new external are separate data lineages; public/pretraining overlap unknown",
            "four primary endpoint alpha=.05/4; point guards are not NI tests; no confirmation-based tuning"]}
    if result["pending_cells"]:
        write_json(destination, result)
        return result, destination
    result.update(status="complete", source_data={key: value["provenance"] for key, value in cells.items()},
        canonicalization={key: value["canonicalization"] for key, value in cells.items()}, contrasts={}, summaries={})
    for task in tasks:
        before = cells[f"initial_42_{task}"]
        after = [cells[f"{CANDIDATE}_{seed}_{task}"] for seed in seeds]
        for cell in after:
            assert_pair(before, cell, task)
        if task == "text":
            slices = prepare_slices(path(config, "asset_root"), root / "analysis/text_slices")
            values = text_contrasts(before["rows"], [cell["rows"] for cell in after], slices["features"])
            result["text_slice_lock_sha256"] = slices["slice_lock_sha256"]
        else:
            values, summary = binary_contrast(before["rows"], [cell["rows"] for cell in after])
            result["summaries"][task] = summary
        result["contrasts"][task] = values
        if task in ("object", "external"):
            result["class_balanced"] = balanced_contrast(config, task, before["rows"], [cell["rows"] for cell in after])
    if phase == "screen":
        result["gate"] = screen_gate(result["contrasts"]["object"], result["contrasts"]["text"], result["class_balanced"], replicated=len(seeds) == 3)
        result["eligible_candidates"] = [CANDIDATE] if result["gate"]["passed"] else []
        if prior:
            result.update(pilot_screen_path=str(pilot_file), pilot_screen_sha256=sha_file(pilot_file))
    elif phase == "external":
        result["gate"] = external_gate(result["contrasts"]["external"], result["class_balanced"], prior)
        result.update(screen_analysis_path=str(screen_file), screen_analysis_sha256=sha_file(screen_file))
    else:
        result["gate"] = original_pope_gate(result["contrasts"]["pope"], {"passed": prior["gate"]["passed"]})
        result.update(external_analysis_path=str(external_file), external_analysis_sha256=sha_file(external_file),
            confirmation_passed=result["gate"]["passed"], goal_complete=False, root_independent_prediction_review_required=True)
    write_json(destination, result)
    return result, destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("screen", "external", "confirm"), required=True)
    parser.add_argument("--seeds", type=int, choices=(0, 42, 200), nargs="+", default=[42])
    for name in ("pilot-screen", "screen-analysis", "external-analysis"):
        parser.add_argument(f"--{name}", type=Path)
    args = parser.parse_args()
    result, file = analyze(args.config, args.output_root, args.seeds, args.phase,
        pilot_file=args.pilot_screen, screen_file=args.screen_analysis, external_file=args.external_analysis)
    print(result["status"], str(file), flush=True)


if __name__ == "__main__":
    main()
