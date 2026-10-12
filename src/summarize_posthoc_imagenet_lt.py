"""Audit and summarize validation-tuned fixed-CE prior correction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from imagenet_lt_manifest import content_sha256
from summarize_imagenet_lt import metric, paired_class_image_bootstrap, sha256_file


SEEDS = (0, 42, 200)
GROUPS = ("oa", "many", "medium", "few")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--main-results", type=Path, required=True)
    parser.add_argument("--posthoc-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    manifest_hash = manifest["manifest_sha256"]
    if content_sha256({k: v for k, v in manifest.items() if k != "manifest_sha256"}) != manifest_hash:
        raise ValueError("manifest hash mismatch")
    labels = np.asarray([row["label"] for row in manifest["splits"]["test"]], dtype=np.uint16)
    zero = json.loads((args.main_results / "zero_shot.json").read_text())
    selections = json.loads((args.posthoc_dir / "validation_selections.json").read_text())
    if selections["manifest_sha256"] != manifest_hash:
        raise ValueError("validation selections use a different manifest")
    all_results = {"zs": [zero] * 3, "ce": [], "la": [], "posthoc": []}
    all_predictions = {"zs": [], "ce": [], "la": [], "posthoc": []}
    for seed in SEEDS:
        for method in ("ce", "la", "posthoc"):
            if method == "posthoc":
                result_path = args.posthoc_dir / f"imagenet_lt_seed{seed}_ce_posthoc.json"
                pred_path = args.posthoc_dir / f"imagenet_lt_seed{seed}_ce_posthoc_pred.npz"
            else:
                result_path = args.main_results / f"imagenet_lt_seed{seed}_{method}.json"
                pred_path = args.main_results / f"imagenet_lt_seed{seed}_{method}_pred.npz"
            result = json.loads(result_path.read_text())
            if (result["manifest_sha256"] != manifest_hash
                    or result["prediction_sha256"] != sha256_file(pred_path)):
                raise ValueError(f"result provenance mismatch: {result_path}")
            with np.load(pred_path) as payload:
                prediction = payload["predictions"]
                targets = payload["targets"]
            if (not np.array_equal(targets, labels)
                    or abs(float(np.mean(prediction == labels)) - result["test"]["oa"]) > 1e-6):
                raise ValueError(f"prediction contents disagree with result: {pred_path}")
            all_results[method].append(result)
            all_predictions[method].append(prediction)
        post = all_results["posthoc"][-1]
        ce = all_results["ce"][-1]
        if (post["ce_checkpoint_sha256"] != ce["checkpoint_sha256"]
                or post["selected_alpha"] != selections["seeds"][str(seed)]["selected_alpha"]
                or post["feature_artifact_sha256"] != ce["bank_metadata"]["feature_artifact_sha256"]):
            raise ValueError(f"post-hoc result is not tied to CE run: seed {seed}")
        for key in ("class_drift", "class_alignment", "class_centroid_alignment"):
            if not np.allclose(post["test"][key], ce["test"][key], rtol=0, atol=1e-6):
                raise ValueError(f"fixed-CE feature diagnostic changed: seed {seed}, {key}")
    zero_path = args.main_results / "zero_shot_pred.npz"
    if zero["prediction_sha256"] != sha256_file(zero_path):
        raise ValueError("zero-shot prediction hash mismatch")
    with np.load(zero_path) as payload:
        if not np.array_equal(payload["targets"], labels):
            raise ValueError("zero-shot labels differ")
        all_predictions["zs"] = [payload["predictions"]] * 3

    metrics = {}
    for method, results in all_results.items():
        metrics[method] = {
            group: {"mean": float(np.mean([metric(result, group) for result in results])),
                    "seed_values": [float(metric(result, group)) for result in results]}
            for group in GROUPS
        }
        for name in ("macro_f1", "nll", "ece15"):
            metrics[method][name] = float(np.mean([result["test"][name] for result in results]))
    contrasts = {}
    for baseline in ("ce", "la", "zs"):
        name = f"posthoc-{baseline}"
        contrasts[name] = {}
        for group in GROUPS:
            ids = (np.arange(1000) if group == "oa" else
                   np.asarray(manifest["groups"][group], dtype=int))
            deltas = [metric(a, group) - metric(b, group) for a, b in
                      zip(all_results["posthoc"], all_results[baseline])]
            interval = paired_class_image_bootstrap(
                all_predictions["posthoc"], all_predictions[baseline], labels, ids
            )
            contrasts[name][group] = {
                "mean_pp": 100 * float(np.mean(deltas)),
                "seed_values_pp": [100 * float(delta) for delta in deltas],
                "paired_class_image_bootstrap_95_pp": [100 * float(x) for x in interval],
            }
    few_mask = np.isin(labels, manifest["groups"]["few"])
    recoveries = {}
    for seed_idx, seed in enumerate(SEEDS):
        zs_correct = all_predictions["zs"][seed_idx] == labels
        ce_correct = all_predictions["ce"][seed_idx] == labels
        lost_by_ce = few_mask & zs_correct & ~ce_correct
        lost_count = int(lost_by_ce.sum())
        recoveries[str(seed)] = {"zs_correct_to_ce_wrong_few_images": lost_count}
        for method in ("la", "posthoc"):
            recovered = int((lost_by_ce & (all_predictions[method][seed_idx] == labels)).sum())
            recoveries[str(seed)][f"recovered_by_{method}"] = recovered
            recoveries[str(seed)][f"recovered_fraction_{method}"] = recovered / lost_count
    output = {
        "manifest_sha256": manifest_hash,
        "selection_rule": selections["selection_rule"],
        "selected_alpha": {seed: selections["seeds"][seed]["selected_alpha"] for seed in map(str, SEEDS)},
        "methods": metrics,
        "contrasts": contrasts,
        "few_lost_image_recovery": recoveries,
        "fixed_ce_features_verified": True,
        "inference_note": "Post-hoc alpha was selected per seed on validation OA; this has more validation tuning than fixed-tau LA and is a mechanism control, not a fair method leaderboard. Paired bootstrap is conditional on this test set.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
