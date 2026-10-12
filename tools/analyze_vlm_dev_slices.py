"""CPU-only exploratory slices of completed stage1 internal-dev predictions.

No generation, training, final-validation predictions, or selection rules are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Import existing metric contracts without creating files in frozen source trees.
sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from vlm_deep.common import NORMALIZER, read_manifest, training_target
from vlm_sft.common import sha_file, write_json

METHODS = ("qv_r8", "qkvo_r4", "qv_r4_merger")
SEEDS = (0, 42, 200)
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261010
SMALL_QUESTIONS = 50
SMALL_IMAGES = 30
NUMERIC = re.compile(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:\s*%)?")
SLICES = (
    ("all", "all"),
    ("answer_frequency", "unseen_0"),
    ("answer_frequency", "rare_1_5"),
    ("answer_frequency", "medium_6_20"),
    ("answer_frequency", "frequent_gt20"),
    ("answer_char_length", "short_le4"),
    ("answer_char_length", "medium_5_10"),
    ("answer_char_length", "long_gt10"),
    ("answer_text_proxy", "numeric_like"),
    ("answer_text_proxy", "other_text"),
)


def read_bound_json(path):
    data = path.read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def text_features(row, frequencies):
    raw = training_target(row["answers"])
    if raw is None or row.get("target") != raw:
        raise ValueError(f"training-target contract mismatch: {row['question_id']}")
    target = NORMALIZER(raw)
    frequency = frequencies[target]
    bucket = ("unseen_0" if frequency == 0 else "rare_1_5" if frequency <= 5
              else "medium_6_20" if frequency <= 20 else "frequent_gt20")
    length = len(target)
    return {
        "raw_target": raw,
        "normalized_target": target,
        "training_target_qa_frequency": frequency,
        "answer_frequency": bucket,
        "answer_char_length": "short_le4" if length <= 4 else "medium_5_10" if length <= 10 else "long_gt10",
        "normalized_target_characters": length,
        "answer_text_proxy": "numeric_like" if NUMERIC.fullmatch(target) else "other_text",
    }


def is_member(features, dimension, label):
    return dimension == "all" or features[dimension] == label


def summary(rows):
    n = len(rows)
    images = len({r["image_id"] for r in rows})
    return {
        "n": n,
        "unique_images": images,
        "score_percent": 100 * sum(float(r["score"]) for r in rows) / n if n else None,
        "small_slice": n < SMALL_QUESTIONS or images < SMALL_IMAGES,
        "note": "Exploratory; sparse slice" if n < SMALL_QUESTIONS or images < SMALL_IMAGES else "Exploratory",
    }


def paired_cluster_interval(base, alternative):
    import numpy as np

    if set(base) != set(alternative):
        raise ValueError("paired question IDs differ")
    grouped = defaultdict(lambda: [0.0, 0])
    for qid in sorted(base):
        b, a = base[qid], alternative[qid]
        if a["image_id"] != b["image_id"]:
            raise ValueError("paired image IDs differ")
        grouped[a["image_id"]][0] += float(a["score"]) - float(b["score"])
        grouped[a["image_id"]][1] += 1
    if not grouped:
        return {"delta_pp": None, "image_cluster_bootstrap_95_pp": None,
                "questions": 0, "unique_images": 0, "small_slice": True}
    values = [grouped[k] for k in sorted(grouped)]
    sums = np.asarray([x[0] for x in values], dtype=float)
    counts = np.asarray([x[1] for x in values], dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    estimates = []
    # Bound CPU memory; every selected image retains all its slice questions.
    for first in range(0, BOOTSTRAP_REPLICATES, 100):
        n = min(100, BOOTSTRAP_REPLICATES - first)
        indices = rng.integers(0, len(values), size=(n, len(values)))
        estimates.extend(100 * sums[indices].sum(axis=1) / counts[indices].sum(axis=1))
    questions = int(counts.sum())
    return {
        "delta_pp": float(100 * sums.sum() / counts.sum()),
        "image_cluster_bootstrap_95_pp": np.quantile(estimates, [0.025, 0.975]).tolist(),
        "questions": questions,
        "unique_images": len(values),
        "small_slice": questions < SMALL_QUESTIONS or len(values) < SMALL_IMAGES,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "scope": "Exploratory conditional image-cluster uncertainty; excludes training-seed uncertainty and multiplicity correction",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.asset_root / "prepared/manifest.json"
    manifest = read_manifest(manifest_path, verify_images=False)
    frequencies = Counter()
    for row in manifest["splits"]["train_10k"]:
        raw = training_target(row["answers"])
        if raw is None or row.get("target") != raw:
            raise ValueError("actual train_10k target differs from deterministic target")
        frequencies[NORMALIZER(raw)] += 1
    dev = {r["question_id"]: r for r in manifest["splits"]["dev"]}
    if len(dev) != len(manifest["splits"]["dev"]):
        raise ValueError("duplicate dev question IDs")
    features = {qid: text_features(row, frequencies) for qid, row in dev.items()}
    runs, skipped, slices, examples = {}, [], [], []
    configs = set()
    for method in METHODS:
        for seed in SEEDS:
            directory = args.run_root / f"stage1/{method}/seed{seed}/eval_dev"
            evaluation_path, predictions_path = directory / "EVALUATION.json", directory / "predictions.json"
            if not evaluation_path.exists() or not predictions_path.exists():
                skipped.append({"method": method, "seed": seed, "reason": "No complete internal-dev EVALUATION.json plus predictions.json"})
                continue
            evaluation, evaluation_sha = read_bound_json(evaluation_path)
            predictions, prediction_sha = read_bound_json(predictions_path)
            if evaluation["task"] != "textvqa" or evaluation["split"] != "dev":
                raise ValueError(f"not internal dev: {directory}")
            if evaluation["manifest_sha256"] != manifest["manifest_sha256"]:
                raise ValueError("dev manifest binding differs")
            if evaluation["predictions_sha256"] != prediction_sha:
                raise ValueError("dev prediction hash differs from evaluation binding")
            if evaluation["n"] != len(dev) or len(predictions) != len(dev):
                raise ValueError("incomplete dev predictions")
            lookup = {r["question_id"]: r for r in predictions}
            if len(lookup) != len(predictions) or set(lookup) != set(dev):
                raise ValueError("dev question IDs do not match exactly")
            for qid, prediction in lookup.items():
                if prediction["image_id"] != dev[qid]["image_id"]:
                    raise ValueError("dev image ID mismatch")
                score = float(prediction["score"])
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError("invalid saved score")
            if not math.isclose(summary(predictions)["score_percent"], evaluation["question_mean_score_percent"], abs_tol=1e-9):
                raise ValueError("saved dev scores differ from evaluation")
            configs.add(evaluation["config_sha256"])
            runs[(method, seed)] = {
                "lookup": lookup,
                "provenance": {
                    "method": method, "seed": seed,
                    "evaluation_path": str(evaluation_path.resolve()),
                    "evaluation_sha256": evaluation_sha,
                    "predictions_path": str(predictions_path.resolve()),
                    "predictions_sha256": prediction_sha,
                    "config_sha256": evaluation["config_sha256"],
                    "evaluation_source_sha256": evaluation["source_sha256"],
                    "completed_dev_score_percent": evaluation["score_percent"],
                },
            }
            for dimension, label in SLICES:
                members = [p for qid, p in lookup.items() if is_member(features[qid], dimension, label)]
                slices.append({"method": method, "seed": seed, "dimension": dimension,
                               "slice": label, **summary(members)})
            # At most three deterministic low-consensus examples per run, drawn
            # from different frequency buckets when possible; never representative.
            selected = []
            for bucket in ("unseen_0", "rare_1_5", "frequent_gt20", "medium_6_20"):
                candidates = [qid for qid, p in lookup.items() if float(p["score"]) < 1
                              and features[qid]["answer_frequency"] == bucket]
                if candidates:
                    selected.append(min(candidates, key=lambda qid: (float(lookup[qid]["score"]), qid)))
                if len(selected) == 3:
                    break
            for qid in selected:
                p = lookup[qid]
                examples.append({"method": method, "seed": seed, "question_id": qid,
                                 "image_id": p["image_id"], "raw_question": dev[qid]["question"],
                                 "raw_target": features[qid]["raw_target"], "raw_prediction": p["prediction"],
                                 "saved_consensus_score": p["score"], "text_features": features[qid],
                                 "note": "Deterministic learning example with consensus score below 1; no visual failure label or causal interpretation"})
    if not runs:
        raise ValueError("no complete internal-dev runs found")
    if len(configs) != 1:
        raise ValueError("completed dev runs use different frozen configs")
    pairs = []
    for alternative in ("qkvo_r4", "qv_r4_merger"):
        for seed in SEEDS:
            if ("qv_r8", seed) not in runs or (alternative, seed) not in runs:
                continue
            for dimension, label in SLICES:
                qids = [qid for qid in dev if is_member(features[qid], dimension, label)]
                base = {qid: runs[("qv_r8", seed)]["lookup"][qid] for qid in qids}
                adapted = {qid: runs[(alternative, seed)]["lookup"][qid] for qid in qids}
                pairs.append({"comparison": f"{alternative}-qv_r8", "seed": seed,
                              "dimension": dimension, "slice": label,
                              **paired_cluster_interval(base, adapted)})
    source_paths = [Path(__file__).resolve(), REPO / "vlm_deep/common.py",
                    REPO / "vlm_deep/third_party/vqa_eval_metric.py", REPO / "vlm_sft/common.py"]
    source_lock = args.run_root / "SOURCE_LOCK.json"
    result = {
        "schema_version": "rarevlm.learning_dev_slices.v1",
        "scope": "Exploratory CPU-only internal-dev diagnostics; frozen method selection remains unchanged",
        "source_sha256": {str(p.relative_to(REPO)): sha_file(p) for p in source_paths},
        "run_source_lock_sha256": sha_file(source_lock) if source_lock.exists() else None,
        "manifest": {"path": str(manifest_path.resolve()), "file_sha256": sha_file(manifest_path),
                     "manifest_sha256": manifest["manifest_sha256"], "frequency_split": "train_10k",
                     "training_questions": sum(frequencies.values()), "unique_normalized_training_targets": len(frequencies),
                     "empty_normalized_training_targets": frequencies[""],
                     "dev_questions": len(dev), "dev_unique_images": len({r["image_id"] for r in dev.values()})},
        "definitions": {
            "answer_frequency": "QA occurrence count of the actual deterministic raw train_10k target, then normalized with the existing standard NORMALIZER; not entity or visual-class frequency",
            "frequency_bins": {"unseen_0": "0", "rare_1_5": "1-5", "medium_6_20": "6-20", "frequent_gt20": ">20"},
            "answer_char_length": "Python character count of normalized deterministic dev target, including spaces: <=4, 5-10, >10",
            "numeric_like": "Full-string normalized integer or decimal, optional sign and percent suffix; text proxy only, not a visual error type",
            "score": "Question mean of saved standard TextVQA consensus scores; no new generation or rescoring",
            "small_slice": f"n < {SMALL_QUESTIONS} or unique_images < {SMALL_IMAGES}",
            "pairing": "Same seed and matched question/image IDs; images resampled with all their slice questions; question-weighted delta",
        },
        "included_runs": [r["provenance"] for r in runs.values()],
        "seed_coverage": [
            {"method": method, "completed_seeds": [seed for seed in SEEDS if (method, seed) in runs],
             "n_completed_seeds": sum((method, seed) in runs for seed in SEEDS),
             "note": "Incomplete seed coverage; do not infer a method winner" if sum((method, seed) in runs for seed in SEEDS) < len(SEEDS)
             else "Three completed seeds; exploratory slices still exclude seed uncertainty"}
            for method in METHODS
        ],
        "skipped_incomplete_runs": skipped,
        "slice_metrics": slices,
        "exploratory_paired_differences": pairs,
        "learning_examples": examples,
        "exploratory_counterfactual_hypotheses": [
            {"hypothesis": "An unseen/frequent dev score gap may reflect training-target frequency, answer complexity, or question composition.",
             "counterfactual_test": "In a separately specified future CPU analysis, hold saved predictions fixed and standardize frequency-group comparisons within answer-length and numeric-text strata. Persistence of the gap would retain an association, not prove a training-frequency cause.",
             "status": "Proposed only; not executed and not used for current selection"},
            {"hypothesis": "Differences on numeric_like may reflect answer formatting or question composition rather than visual reading or numerical reasoning.",
             "counterfactual_test": "Inspect dev-only raw references and saved predictions under a predeclared formatting audit; a visual-mechanism claim would additionally require independently annotated visual evidence and a separately frozen controlled experiment.",
             "status": "Proposed only; no visual failure type assigned and no training setting changed"},
        ],
        "limitations": [
            "Internal dev only; no official final/validation predictions are read or used to design training.",
            "Text proxies do not establish OCR, visual grounding, entity frequency, or visual failure causes.",
            "Deterministic modal target defines slices, while consensus scores retain all human references; target mismatch alone is not a correctness label.",
            "Frequency is restricted to actual train_10k QA targets; unseen does not mean unknown to the pretrained model.",
            "Some nonempty raw targets normalize to empty strings; their QA occurrences remain counted rather than being silently removed.",
            "Paired intervals are conditional on the completed training runs; seed uncertainty is excluded and many exploratory slices have no multiplicity correction.",
            "Small slices and hand-inspected learning examples do not support general performance claims.",
            "Method selection stays frozen to the existing overall dev mean, then dev token NLL and method-name tie breakers; these slices do not change it.",
            "Only complete run pairs available at this snapshot are compared; unequal numbers of completed seeds must not be interpreted as a fair method ranking.",
        ],
    }
    write_json(args.output, result)
    print(json.dumps({"output": str(args.output.resolve()), "complete_dev_runs": len(runs),
                      "slice_records": len(slices), "paired_slice_records": len(pairs),
                      "learning_examples": len(examples), "skipped_incomplete_runs": len(skipped)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
