"""Summarize paired adapter results without tuning on the official test set."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METHODS = ("ce", "la", "ts")
GROUPS = ("oa", "many", "medium", "few")
COMPARISONS = (("la", "ce"), ("ts", "ce"), ("ts", "la"),
               ("ce", "zs"), ("la", "zs"), ("ts", "zs"))


def score(result, group):
    data = result["test"]
    return data["oa"] if group == "oa" else data["groups"][group]["accuracy"]


def rows_to_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_interval(run_by_key, manifests, a, b, group, *, repeats=3000, seed=20260930):
    """Exploratory hierarchical bootstrap over mappings and classes.

    Training seeds stay paired within a mapping. Only three independent
    frequency mappings exist, so this interval is descriptive, not a main
    dataset or new-class generalization guarantee.
    """
    rng = np.random.default_rng(seed)
    permutations = sorted(manifests)
    seeds = sorted({key[1] for key in run_by_key if key[1] >= 0})
    per_perm = {}
    for perm in permutations:
        ids = (np.arange(100, dtype=int) if group == "oa" else
               np.array(manifests[perm]["groups"][group], dtype=int))
        deltas = []
        for train_seed in seeds:
            lhs = run_by_key[(perm, train_seed, a)]["test"]["class_accuracy"]
            rhs = (run_by_key[(perm, train_seed, b)]["test"]["class_accuracy"]
                   if b != "zs" else run_by_key[(perm, -1, "zs")]["test"]["class_accuracy"])
            deltas.append((np.asarray(lhs) - np.asarray(rhs))[ids])
        per_perm[perm] = np.stack(deltas, axis=0)
    sampled = np.empty(repeats, dtype=float)
    for i in range(repeats):
        chosen_perms = rng.choice(permutations, size=len(permutations), replace=True)
        values = []
        for perm in chosen_perms:
            matrix = per_perm[int(perm)]
            chosen_classes = rng.integers(0, matrix.shape[1], size=matrix.shape[1])
            values.append(float(matrix[:, chosen_classes].mean()))
        sampled[i] = np.mean(values)
    return [float(x) for x in np.quantile(sampled, [0.025, 0.975])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = [json.loads(path.read_text()) for path in args.input_dir.glob("perm*_seed*.json")]
    run_by_key = {(int(r["permutation_seed"]), int(r["train_seed"]), r["method"]): r for r in runs}
    manifests = {
        int(m["permutation_seed"]): m
        for path in (args.input_dir / "manifests").glob("*.json")
        for m in [json.loads(path.read_text())]
    }
    zero = json.loads((args.input_dir / "zero_shot.json").read_text())
    for perm, value in zero["permutations"].items():
        run_by_key[(int(perm), -1, "zs")] = value
    seeds = sorted({r["train_seed"] for r in runs})
    if len(manifests) != 3 or len(seeds) != 3 or len(runs) != 27:
        raise ValueError("expected complete 3 permutations x 3 seeds x 3 losses")
    for perm in manifests:
        for seed in seeds:
            for method in METHODS:
                if (perm, seed, method) not in run_by_key:
                    raise ValueError(f"missing {(perm, seed, method)}")

    method_summary = {}
    for method in ("zs",) + METHODS:
        cells = ([run_by_key[(perm, -1, "zs")] for perm in sorted(manifests)]
                 if method == "zs" else
                 [run_by_key[(perm, seed, method)] for perm in sorted(manifests) for seed in seeds])
        method_summary[method] = {
            group: {"mean": float(np.mean([score(r, group) for r in cells])),
                    "sd": float(np.std([score(r, group) for r in cells], ddof=1))}
            for group in GROUPS
        }
        method_summary[method]["nll"] = float(np.mean([r["test"]["nll"] for r in cells]))
        method_summary[method]["ece15"] = float(np.mean([r["test"]["ece15"] for r in cells]))
        method_summary[method]["n_runs"] = len(cells)

    pair_rows, class_rows, contrast_summary = [], [], {}
    for perm in sorted(manifests):
        manifest = manifests[perm]
        class_group = {c: group for group in ("many", "medium", "few")
                       for c in manifest["groups"][group]}
        for seed in seeds:
            for method in METHODS:
                run = run_by_key[(perm, seed, method)]
                for c in range(100):
                    class_rows.append({
                        "permutation_seed": perm, "train_seed": seed, "method": method,
                        "class_id": c, "count": manifest["class_counts"][c],
                        "group": class_group[c],
                        "accuracy": run["test"]["class_accuracy"][c],
                        "zero_shot_accuracy": run_by_key[(perm, -1, "zs")]["test"]["class_accuracy"][c],
                        "delta_vs_zero_shot": (
                            run["test"]["class_accuracy"][c] -
                            run_by_key[(perm, -1, "zs")]["test"]["class_accuracy"][c]
                        ),
                    })
            for a, b in COMPARISONS:
                run_a = run_by_key[(perm, seed, a)]
                run_b = (run_by_key[(perm, seed, b)] if b != "zs" else
                         run_by_key[(perm, -1, "zs")])
                pa = np.load(args.input_dir / f"perm{perm}_seed{seed}_{a}_pred.npz")["predictions"]
                pb_path = (args.input_dir / "zero_shot_pred.npz" if b == "zs" else
                           args.input_dir / f"perm{perm}_seed{seed}_{b}_pred.npz")
                pb = np.load(pb_path)["predictions"]
                y = np.load(args.input_dir / "zero_shot_pred.npz")["targets"]
                for group in GROUPS:
                    mask = (np.ones(len(y), dtype=bool) if group == "oa" else
                            np.isin(y, manifest["groups"][group]))
                    a_only = int(np.sum((pa == y) & (pb != y) & mask))
                    b_only = int(np.sum((pa != y) & (pb == y) & mask))
                    pair_rows.append({
                        "permutation_seed": perm, "train_seed": seed,
                        "contrast": f"{a}-{b}", "group": group,
                        "delta_pp": 100 * (score(run_a, group) - score(run_b, group)),
                        "a_only_correct": a_only, "b_only_correct": b_only,
                        "net_correct": a_only - b_only,
                    })
    for a, b in COMPARISONS:
        key = f"{a}-{b}"
        contrast_summary[key] = {}
        for group in GROUPS:
            selected = [row["delta_pp"] for row in pair_rows
                        if row["contrast"] == key and row["group"] == group]
            interval = bootstrap_interval(run_by_key, manifests, a, b, group)
            contrast_summary[key][group] = {
                "mean_pp": float(np.mean(selected)),
                "sd_pp": float(np.std(selected, ddof=1)),
                "min_pp": float(np.min(selected)),
                "max_pp": float(np.max(selected)),
                "positive_pairs": int(np.sum(np.array(selected) > 0)),
                "pairs": len(selected),
                "exploratory_hierarchical_bootstrap_95_pp": [100*x for x in interval],
            }
    summary = {
        "protocol": "frozen CLIP ViT-B/32; CIFAR-100-LT 400/4 IR=100; 50/class validation; three random frequency-to-class mappings; three training seeds; 20 epochs; validation-selected adapter epoch",
        "interval_note": "Bootstrap resamples only 3 frequency mappings and classes within mappings; exploratory and conditional on CIFAR-100 and this one prompt/model.",
        "methods": method_summary,
        "contrasts": contrast_summary,
    }
    rows_to_csv(args.output_dir / "paired_deltas.csv", pair_rows)
    rows_to_csv(args.output_dir / "class_results.csv", class_rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
