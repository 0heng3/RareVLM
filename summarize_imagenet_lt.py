"""Summarize the fixed ImageNet-LT zero-shot/CE/LA replication."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import stats
from imagenet_lt_manifest import content_sha256


GROUPS = ("oa", "many", "medium", "few")
CONTRASTS = (("ce", "zs"), ("la", "ce"), ("la", "zs"))


def metric(result, group):
    return result["test"]["oa"] if group == "oa" else result["test"]["groups"][group]["accuracy"]


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paired_class_image_bootstrap(pred_a, pred_b, labels, ids, *, draws=1500, seed=20260930):
    """Resample classes and their paired test images, shared across seeds."""
    differences = np.mean([
        (a == labels).astype(np.float32) - (b == labels).astype(np.float32)
        for a, b in zip(pred_a, pred_b)
    ], axis=0)
    class_differences = [differences[labels == class_id] for class_id in ids]
    supports = {len(values) for values in class_differences}
    if supports != {50}:
        raise ValueError(f"expected 50 test images per class, got {supports}")
    class_differences = np.stack(class_differences, axis=0)
    rng = np.random.default_rng(seed)
    values = []
    for start in range(0, draws, 50):
        chunk = min(50, draws - start)
        class_draws = rng.integers(0, len(ids), size=(chunk, len(ids)))
        image_draws = rng.integers(0, 50, size=(chunk, len(ids), 50))
        sampled = np.take_along_axis(class_differences[class_draws], image_draws, axis=2)
        values.extend(sampled.mean(axis=(1, 2)).tolist())
    return [float(q) for q in np.quantile(values, [0.025, 0.975])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    if content_sha256({k: v for k, v in manifest.items() if k != "manifest_sha256"}) != manifest.get("manifest_sha256"):
        raise ValueError("manifest content hash mismatch")
    zero = json.loads((args.input_dir / "zero_shot.json").read_text())
    runs = {}
    for seed in (0, 42, 200):
        for method in ("ce", "la"):
            path = args.input_dir / f"imagenet_lt_seed{seed}_{method}.json"
            runs[(seed, method)] = json.loads(path.read_text())
    if any(r["manifest_sha256"] != manifest["manifest_sha256"] for r in [zero] + list(runs.values())):
        raise ValueError("results use a different manifest")
    classes = len(manifest["class_counts"])
    if classes != 1000:
        raise ValueError("expected 1000 classes")
    expected_labels = np.asarray([record["label"] for record in manifest["splits"]["test"]], dtype=np.uint16)
    expected_optimizer = {"name": "AdamW", "lr": 0.001, "weight_decay": 0.01,
                          "batch_size": 256, "epochs": 20}
    if zero.get("method") != "zero_shot":
        raise ValueError("zero-shot result method mismatch")
    reference_metadata = zero["bank_metadata"]
    for (seed, method), run in runs.items():
        checkpoint_path = args.input_dir / f"imagenet_lt_seed{seed}_{method}_adapter.pt"
        if (run.get("method") != method or run.get("train_seed") != seed
                or run.get("optimizer") != expected_optimizer
                or run.get("bank_metadata") != reference_metadata
                or run.get("checkpoint_sha256") != sha256_file(checkpoint_path)):
            raise ValueError(f"run provenance mismatch: seed {seed}, {method}")
    predictions = {}
    for key, result, path in [
        ("zs", zero, args.input_dir / "zero_shot_pred.npz"),
        *[((seed, method), runs[(seed, method)],
           args.input_dir / f"imagenet_lt_seed{seed}_{method}_pred.npz")
          for seed in (0, 42, 200) for method in ("ce", "la")],
    ]:
        if result.get("prediction_sha256") != sha256_file(path):
            raise ValueError(f"prediction artifact hash mismatch: {path}")
        with np.load(path) as payload:
            pred, targets = payload["predictions"], payload["targets"]
        if (pred.dtype != np.uint16 or not np.array_equal(targets, expected_labels)
                or pred.shape != expected_labels.shape
                or abs(float(np.mean(pred == targets)) - result["test"]["oa"]) > 1e-6):
            raise ValueError(f"prediction contents disagree with metrics or manifest: {path}")
        predictions[key] = pred
    log_frequency = np.log(np.asarray(manifest["class_counts"], dtype=float))
    zs_class = np.asarray(zero["test"]["class_accuracy"], dtype=float)
    by_method = {"zs": [zero], **{
        method: [runs[(seed, method)] for seed in (0, 42, 200)] for method in ("ce", "la")
    }}
    method_summary = {}
    for method, values in by_method.items():
        method_summary[method] = {
            name: {"mean": float(np.mean([metric(v, name) for v in values])),
                   "sd_across_seeds": (float(np.std([metric(v, name) for v in values], ddof=1))
                                       if len(values) > 1 else None)}
            for name in GROUPS
        }
        for name in ("macro_f1", "nll", "ece15"):
            method_summary[method][name] = float(np.mean([v["test"][name] for v in values]))

    paired_rows = []
    contrasts = {}
    for a, b in CONTRASTS:
        name = f"{a}-{b}"
        contrasts[name] = {}
        for group in GROUPS:
            ids = (np.arange(classes, dtype=int) if group == "oa" else
                   np.asarray(manifest["groups"][group], dtype=int))
            pred_a = [predictions[(seed, a)] for seed in (0, 42, 200)]
            pred_b = ([predictions["zs"]] * 3 if b == "zs" else
                      [predictions[(seed, b)] for seed in (0, 42, 200)])
            values = []
            for seed in (0, 42, 200):
                lhs = runs[(seed, a)]
                rhs = zero if b == "zs" else runs[(seed, b)]
                delta = metric(lhs, group) - metric(rhs, group)
                values.append(delta)
                predictions_a = predictions[(seed, a)]
                predictions_b = predictions["zs"] if b == "zs" else predictions[(seed, b)]
                mask = (np.ones(len(expected_labels), dtype=bool) if group == "oa" else
                        np.isin(expected_labels, ids))
                a_only = int(np.count_nonzero((predictions_a == expected_labels) &
                                              (predictions_b != expected_labels) & mask))
                b_only = int(np.count_nonzero((predictions_a != expected_labels) &
                                              (predictions_b == expected_labels) & mask))
                paired_rows.append({"contrast": name, "group": group, "seed": seed,
                                    "delta_pp": 100 * delta, "a_only_correct": a_only,
                                    "b_only_correct": b_only, "net_correct": a_only - b_only})
            interval = paired_class_image_bootstrap(
                pred_a, pred_b, expected_labels, ids
            )
            contrasts[name][group] = {
                "mean_pp": 100 * float(np.mean(values)),
                "sd_pp": 100 * float(np.std(values, ddof=1)),
                "seed_values_pp": [100 * float(v) for v in values],
                "paired_class_image_bootstrap_95_pp": [100 * v for v in interval],
            }

    class_rows, correlations = [], {}
    for seed in (0, 42, 200):
        ce = np.asarray(runs[(seed, "ce")]["test"]["class_accuracy"])
        la = np.asarray(runs[(seed, "la")]["test"]["class_accuracy"])
        deltas = {"ce_minus_zs": ce - zs_class, "la_minus_ce": la - ce}
        correlations[str(seed)] = {}
        for label, y in deltas.items():
            rho, pvalue = stats.spearmanr(y, log_frequency)
            # Rank-residual correlation adjusts descriptively for baseline ZS
            # difficulty/ceiling; neither coefficient is causal evidence.
            rank_y = stats.rankdata(y)
            rank_x = stats.rankdata(log_frequency)
            rank_z = stats.rankdata(zs_class)
            X = np.stack((rank_z, np.ones(classes)), axis=1)
            resid_y = rank_y - X @ np.linalg.lstsq(X, rank_y, rcond=None)[0]
            resid_x = rank_x - X @ np.linalg.lstsq(X, rank_x, rcond=None)[0]
            partial = float(np.corrcoef(resid_y, resid_x)[0, 1])
            correlations[str(seed)][label] = {
                "spearman_rho": float(rho), "descriptive_p": float(pvalue),
                "rank_partial_rho_controlling_zs_accuracy": partial,
            }
        for c in range(classes):
            class_rows.append({
                "seed": seed, "class_id": c, "wnid": manifest["label_to_wnid"][c],
                "train_count": manifest["class_counts"][c],
                "log_train_count": float(log_frequency[c]),
                "zero_shot_acc": float(zs_class[c]),
                "ce_acc": float(ce[c]), "la_acc": float(la[c]),
                "ce_minus_zs": float(ce[c] - zs_class[c]),
                "la_minus_ce": float(la[c] - ce[c]),
                "la_minus_zs": float(la[c] - zs_class[c]),
            })
    gate_a = all(metric(runs[(seed, "ce")], "few") < metric(zero, "few") and
                 metric(runs[(seed, "ce")], "many") > metric(zero, "many")
                 for seed in (0, 42, 200))
    gate_b = all(metric(runs[(seed, "la")], "few") > metric(runs[(seed, "ce")], "few") and
                 metric(runs[(seed, "la")], "oa") >= metric(runs[(seed, "ce")], "oa")
                 for seed in (0, 42, 200))
    gate_c_imagenet_condition = (
        all(metric(runs[(seed, "la")], "few") > metric(zero, "few") for seed in (0, 42, 200))
        and contrasts["la-zs"]["few"]["paired_class_image_bootstrap_95_pp"][0] > 0
    )
    summary = {
        "manifest_sha256": manifest["manifest_sha256"],
        "protocol": "ImageNet-LT frozen CLIP ViT-B/32; ZS plus CE/LA seeds 0/42/200; 20 epochs; validation-selected adapter epoch; 1000 classes",
        "methods": method_summary,
        "contrasts": contrasts,
        "frequency_correlations": correlations,
        "gates": {"A_ce_collapse_all_seeds": gate_a,
                  "B_la_mitigates_all_seeds": gate_b,
                  "C_imagenet_internal_condition": gate_c_imagenet_condition,
                  "C_independent_second_dataset_support": False,
                  "C_project_claim": False},
        "inference_note": "Three seeds share one test set. The paired bootstrap resamples classes and the same within-class test images across methods and seeds; it is conditional on ImageNet-LT. The independent CIFAR LA-vs-ZS Few interval crossed zero, so the project-level Gate C claim remains unpassed. Frequency and class semantics are fixed together in ImageNet-LT.",
    }
    write_csv(args.output_dir / "paired_deltas.csv", paired_rows)
    write_csv(args.output_dir / "class_results.csv", class_rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
