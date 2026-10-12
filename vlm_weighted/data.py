"""Training-only answer-frequency weights and annotation audit for paired SFT.

The actual supervised answer is unchanged: raw-stripped modal, lexical tie break.
EvalAI normalization groups answers for frequency counting only. Dev references
are used solely to attach declared evaluation groups, never to compute weights.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from vlm_deep.common import NORMALIZER, content_sha, read_manifest, sha_file, training_target, write_json

SCHEMA = "rarevlm.vlm_weight_lock.v1"
QUESTION_COUNT = 32985
FORMULA = {
    "raw_weight": "clip((20 / training_normalized_target_frequency) ** 0.25, 1, 2)",
    "normalization": "weight = raw_weight / mean(raw_weight over all 32985 train_all QA)",
    "frequency_threshold": 20,
    "exponent": 0.25,
    "minimum_raw_weight": 1.0,
    "maximum_raw_weight": 2.0,
    "uniform_condition_weight": 1.0,
    "no_batch_or_subset_renormalization": True,
}
BIN_DEFINITIONS = {"unseen": "0", "rare": "1-5", "mid": "6-20", "frequent": ">20"}


def frequency_bin(frequency: int) -> str:
    if frequency < 0:
        raise ValueError("negative target frequency")
    if frequency == 0:
        return "unseen"
    if frequency <= 5:
        return "rare"
    if frequency <= 20:
        return "mid"
    return "frequent"


def _source_sha() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    files = (Path(__file__).resolve(), root / "vlm_deep/common.py",
             root / "vlm_deep/third_party/vqa_eval_metric.py", root / "vlm_sft/common.py")
    return {str(p.relative_to(root)): sha_file(p) for p in files}


def _mode(values: list[str]) -> tuple[str | None, int, int]:
    counts = Counter(values)
    if not counts:
        return None, 0, 0
    maximum = max(counts.values())
    tied = sorted(k for k, v in counts.items() if v == maximum)
    return tied[0], maximum, len(tied)


def _summary(values: list[float | int]) -> dict:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    quantiles = {}
    for percentile in (0, 25, 50, 75, 90, 95, 99, 100):
        position = (len(ordered) - 1) * percentile / 100
        low, high = math.floor(position), math.ceil(position)
        quantiles[str(percentile)] = ordered[low] + (ordered[high] - ordered[low]) * (position - low)
    return {"count": len(values), "mean": statistics.fmean(values), "quantiles": quantiles}


def _annotation(row: dict) -> dict:
    answers = row["answers"]
    if len(answers) != 10:
        raise ValueError(f"requires ten human answers: {row['question_id']}")
    target = training_target(answers)
    if target is None or row.get("target") != target:
        raise ValueError(f"actual raw-stripped training target changed: {row['question_id']}")
    normalized = [NORMALIZER(str(a)) for a in answers]
    normalized_target = NORMALIZER(target)
    raw_values = [str(a).strip() for a in answers if str(a).strip()]
    raw_mode, raw_count, raw_ties = _mode(raw_values)
    norm_mode, norm_count, norm_ties = _mode(normalized)
    norm_nonempty, nonempty_count, nonempty_ties = _mode([a for a in normalized if a])
    matching = sum(a == normalized_target for a in normalized)
    # Exact ten-reference leave-one-annotator-out score from match counts.
    score = (matching * min(max(matching - 1, 0) / 3, 1)
             + (10 - matching) * min(matching / 3, 1)) / 10
    return {
        "question_id": str(row["question_id"]), "image_id": str(row["image_id"]),
        "target": target, "normalized_target": normalized_target,
        "raw_target_match_count": raw_count, "normalized_target_match_count": matching,
        "target_loo_consensus_score": score, "raw_modal_tie_count": raw_ties,
        "normalized_modal_target": norm_mode, "normalized_modal_match_count": norm_count,
        "normalized_modal_tie_count": norm_ties,
        "normalized_nonempty_modal_target": norm_nonempty,
        "normalized_nonempty_modal_match_count": nonempty_count,
        "normalized_nonempty_modal_tie_count": nonempty_ties,
        "normalized_target_differs_from_normalized_modal": normalized_target != norm_mode,
        "normalized_target_differs_from_nonempty_normalized_modal": normalized_target != norm_nonempty,
        "empty_raw_reference_count": 10 - len(raw_values),
        "empty_normalized_reference_count": sum(not a for a in normalized),
        "target_character_count": len(target), "target_whitespace_word_count": len(target.split()),
        "normalized_target_character_count": len(normalized_target),
        "distinct_raw_reference_count": len(set(raw_values)),
        "distinct_normalized_reference_count": len(set(normalized)),
    }


def _leakage_checks(manifest: dict) -> dict:
    split_ids, split_qids, split_sha = {}, {}, {}
    for split in ("train_all", "dev", "final"):
        rows = manifest["splits"][split]
        split_ids[split] = {str(r["image_id"]) for r in rows}
        split_qids[split] = {str(r["question_id"]) for r in rows}
        if len(split_qids[split]) != len(rows):
            raise ValueError(f"duplicate QA in {split}")
        split_sha[split] = {manifest["images"][image_id]["sha256"] for image_id in split_ids[split]}
    pairs = {}
    for first, second in (("train_all", "dev"), ("train_all", "final"), ("dev", "final")):
        values = {"shared_image_ids": len(split_ids[first] & split_ids[second]),
                  "shared_question_ids": len(split_qids[first] & split_qids[second]),
                  "shared_manifest_image_byte_sha256": len(split_sha[first] & split_sha[second])}
        if any(values.values()):
            raise ValueError(f"split leakage {first}/{second}: {values}")
        pairs[f"{first}__{second}"] = values
    return {
        "pairs": pairs,
        "image_identity_scope": "declared image IDs and recorded image-file bytes SHA256; no perceptual-duplicate audit",
        "final_annotation_usage": "none; final image IDs, question IDs and recorded image hashes used only for overlap checks",
        "pretraining_overlap": "unknown, not assessed or claimed absent",
    }


def _row_summary(rows: list[dict]) -> dict:
    return {
        "questions": len(rows), "unique_images": len({r["image_id"] for r in rows}),
        "empty_normalized_targets": sum(not r["normalized_target"] for r in rows),
        "raw_modal_ties": sum(r["raw_modal_tie_count"] > 1 for r in rows),
        "normalized_modal_ties": sum(r["normalized_modal_tie_count"] > 1 for r in rows),
        "raw_normalized_modal_disagreements": sum(r["normalized_target_differs_from_normalized_modal"] for r in rows),
        "raw_nonempty_normalized_modal_disagreements": sum(r["normalized_target_differs_from_nonempty_normalized_modal"] for r in rows),
        "normalized_target_match_count_histogram": dict(sorted(Counter(r["normalized_target_match_count"] for r in rows).items())),
        "target_loo_consensus_score": _summary([r["target_loo_consensus_score"] for r in rows]),
        "target_character_count": _summary([r["target_character_count"] for r in rows]),
        "target_whitespace_word_count": _summary([r["target_whitespace_word_count"] for r in rows]),
        "empty_normalized_reference_count": sum(r["empty_normalized_reference_count"] for r in rows),
    }


def _csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot write empty audit CSV")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def build_weight_lock(manifest_path: str | Path, output_dir: str | Path) -> dict:
    """Build immutable weights and CPU diagnostics from the full training set."""
    manifest_path, output_dir = Path(manifest_path), Path(output_dir)
    manifest = read_manifest(manifest_path)
    training = manifest["splits"]["train_all"]
    if len(training) != QUESTION_COUNT:
        raise ValueError("this protocol requires all 32985 training QA")
    leakage = _leakage_checks(manifest)
    rows = [_annotation(row) for row in training]
    frequencies = Counter(row["normalized_target"] for row in rows)
    target_images = defaultdict(set)
    reference_coverage = Counter()
    for source_row, row in zip(training, rows):
        target_images[row["normalized_target"]].add(row["image_id"])
        reference_coverage.update({NORMALIZER(str(answer)) for answer in source_row["answers"]})
    raw_weights = [min(2.0, max(1.0, (20 / frequencies[r["normalized_target"]]) ** 0.25)) for r in rows]
    normalizer = math.fsum(raw_weights) / QUESTION_COUNT
    for row, raw_weight in zip(rows, raw_weights):
        frequency = frequencies[row["normalized_target"]]
        row.update(target_frequency=frequency, frequency_bin=frequency_bin(frequency),
                   training_target_unique_images=len(target_images[row["normalized_target"]]),
                   training_reference_coverage=reference_coverage[row["normalized_target"]],
                   raw_weight=raw_weight, weight=raw_weight / normalizer)
    dev_rows = [_annotation(row) for row in manifest["splits"]["dev"]]
    for row in dev_rows:
        frequency = frequencies[row["normalized_target"]]
        row.update(target_frequency=frequency, frequency_bin=frequency_bin(frequency),
                   training_target_unique_images=len(target_images[row["normalized_target"]]),
                   training_reference_coverage=reference_coverage[row["normalized_target"]])
    lock = {
        "schema_version": SCHEMA, "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": manifest["manifest_sha256"], "manifest_file_sha256": sha_file(manifest_path),
        "source_sha256": _source_sha(), "train_split": "train_all", "question_count": len(rows),
        "unique_training_images": len({r["image_id"] for r in rows}),
        "question_order_sha256": content_sha([r["question_id"] for r in rows]),
        "training_target_policy": "nonempty raw-stripped modal human answer, lexical tie break; target unchanged",
        "frequency_policy": "one actual normalized supervised target occurrence per train_all QA; includes empty normalized targets",
        "normalizer": "frozen vlm_deep.third_party.vqa_eval_metric.EvalAIAnswerProcessor",
        "formula": FORMULA, "normalization_constant": normalizer,
        "frequency_bins": BIN_DEFINITIONS,
        "dev_policy": "attach fixed train-only frequency groups; not used in f, Z or weight selection",
        "final_policy": "annotations not used; metadata used for leakage checks only",
        "rows": rows, "dev_rows": dev_rows,
    }
    lock["lock_sha256"] = content_sha(lock)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "WEIGHT_LOCK.json"
    if destination.exists():
        existing = read_weight_lock(destination, manifest_path)
        if existing != lock:
            raise ValueError("existing weight lock differs; use a new output protocol")
    else:
        write_json(destination, lock)
    target_rows = [{"normalized_target": target, "target_frequency": frequency,
                    "frequency_bin": frequency_bin(frequency), "unique_training_images": len(target_images[target]),
                    "training_reference_coverage": reference_coverage[target]}
                   for target, frequency in sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))]
    bins = {}
    for name in BIN_DEFINITIONS:
        group = [r for r in rows if r["frequency_bin"] == name]
        bins[name] = {**_row_summary(group), "sum_weight": math.fsum(r["weight"] for r in group),
                      "weight_share": math.fsum(r["weight"] for r in group) / QUESTION_COUNT,
                      "qa_share": len(group) / QUESTION_COUNT,
                      "weight": _summary([r["weight"] for r in group])}
    audit = {
        "schema_version": "rarevlm.vlm_annotation_audit.v1", "complete": True,
        "lock_sha256": lock["lock_sha256"], "weight_lock_file_sha256": sha_file(destination),
        "manifest_sha256": manifest["manifest_sha256"], "source_sha256": lock["source_sha256"],
        "training": _row_summary(rows), "dev": _row_summary(dev_rows),
        "training_frequency_bins": bins,
        "dev_frequency_bins": {name: _row_summary([r for r in dev_rows if r["frequency_bin"] == name]) for name in BIN_DEFINITIONS},
        "unique_normalized_training_targets": len(frequencies),
        "normalized_target_unique_image_coverage": _summary([len(target_images[t]) for t in frequencies]),
        "unique_target_count_by_frequency_bin": dict(Counter(frequency_bin(f) for f in frequencies.values())),
        "weights": {"normalization_constant": normalizer, "raw": _summary(raw_weights),
                    "normalized": _summary([r["weight"] for r in rows]),
                    "sum_normalized_weight": math.fsum(r["weight"] for r in rows),
                    "mean_normalized_weight": math.fsum(r["weight"] for r in rows) / QUESTION_COUNT,
                    "formula": FORMULA},
        "checks": {"training_target_exactly_matches_frozen_policy": True,
                   "one_record_per_real_training_qa": True, "training_qa_count": len(rows),
                   "dev_and_final_not_used_to_compute_weights": True, "leakage": leakage},
        "length_units": "characters and whitespace words, not model tokenizer tokens",
        "consensus_scope": "human-reference agreement, not a correctness label or calibrated uncertainty",
        "frequency_scope": "answer-text supervision exposure proxy, not semantic class or pretraining knowledge",
    }
    write_json(output_dir / "AUDIT.json", audit)
    _csv(output_dir / "training_rows.csv", rows)
    _csv(output_dir / "dev_groups.csv", dev_rows)
    _csv(output_dir / "target_frequency.csv", target_rows)
    return lock


def read_weight_lock(path: str | Path, manifest_path: str | Path | None = None) -> dict:
    """Verify the lock, full-population frequencies/weights, and optionally targets."""
    import json
    lock = json.loads(Path(path).read_text())
    if lock.get("schema_version") != SCHEMA:
        raise ValueError("weight lock schema changed")
    if lock.get("lock_sha256") != content_sha({k: v for k, v in lock.items() if k != "lock_sha256"}):
        raise ValueError("weight lock content changed")
    if lock["source_sha256"] != _source_sha():
        raise ValueError("weight lock source changed")
    if lock["formula"] != FORMULA:
        raise ValueError("weight formula changed")
    rows = lock["rows"]
    if len(rows) != QUESTION_COUNT or lock["question_count"] != QUESTION_COUNT:
        raise ValueError("incomplete training weights")
    if len({r["question_id"] for r in rows}) != QUESTION_COUNT:
        raise ValueError("duplicate training qid in weight lock")
    if lock["question_order_sha256"] != content_sha([r["question_id"] for r in rows]):
        raise ValueError("weight lock question order changed")
    frequencies = Counter(r["normalized_target"] for r in rows)
    normalizer = math.fsum(r["raw_weight"] for r in rows) / QUESTION_COUNT
    if not math.isclose(normalizer, lock["normalization_constant"], rel_tol=1e-14, abs_tol=1e-14):
        raise ValueError("training population normalization constant changed")
    for row in rows:
        f = frequencies[row["normalized_target"]]
        raw_weight = min(2.0, max(1.0, (20 / f) ** 0.25))
        if (row["target_frequency"] != f or row["normalized_target"] != NORMALIZER(row["target"])
                or row["frequency_bin"] != frequency_bin(f)
                or not math.isclose(row["raw_weight"], raw_weight, rel_tol=1e-14, abs_tol=1e-14)
                or not math.isclose(row["weight"], raw_weight / normalizer, rel_tol=1e-14, abs_tol=1e-14)):
            raise ValueError(f"training weight contract changed: {row['question_id']}")
    if not math.isclose(math.fsum(r["weight"] for r in rows), QUESTION_COUNT, rel_tol=1e-14):
        raise ValueError("training mean weight is not one")
    if manifest_path is not None:
        manifest_path = Path(manifest_path)
        manifest = read_manifest(manifest_path)
        if (lock["manifest_sha256"] != manifest["manifest_sha256"]
                or lock["manifest_file_sha256"] != sha_file(manifest_path)):
            raise ValueError("weight source manifest changed")
        for source, row in zip(manifest["splits"]["train_all"], rows):
            if (str(source["question_id"]) != row["question_id"]
                    or str(source["image_id"]) != row["image_id"]
                    or source["target"] != row["target"]
                    or training_target(source["answers"]) != row["target"]):
                raise ValueError("actual training QA order or raw target changed")
    return lock


def load_weights(path_or_lock: str | Path | dict, qids, condition: str = "frequency") -> list[float]:
    """Get per-QA weights in caller order; accepts a previously verified lock.

    Uniform and frequency conditions use the same immutable population lock.
    Cache the dictionary once in a runner rather than rereading it per batch.
    Sentinel computations still require a separate zero-valid mask in the loss.
    """
    if condition not in {"uniform", "frequency"}:
        raise ValueError("condition must be uniform or frequency")
    lock = path_or_lock if isinstance(path_or_lock, dict) else read_weight_lock(path_or_lock)
    by_qid = {str(row["question_id"]): row["weight"] for row in lock["rows"]}
    result = []
    for qid in qids:
        key = str(qid)
        if key not in by_qid:
            raise ValueError(f"qid outside complete training pool: {key}")
        result.append(1.0 if condition == "uniform" else by_qid[key])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    lock = build_weight_lock(args.manifest, args.output_dir)
    print({"weight_lock": str(args.output_dir / "WEIGHT_LOCK.json"),
           "questions": lock["question_count"], "normalization_constant": lock["normalization_constant"],
           "lock_sha256": lock["lock_sha256"]}, flush=True)


if __name__ == "__main__":
    main()
