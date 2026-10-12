"""CPU-only analysis of six fixed-weight, fixed-input numerical-path cells.

The bridge is an engineering sensitivity study on internal dev. It neither
trains a model nor proves mathematical/numerical equivalence between paths.
Live incomplete append-only predictions produce progress, never final effects.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
WORKSPACE = Path("/opt/data/private")
REPO = Path(__file__).resolve().parents[1]
DEFAULT_ASSET = WORKSPACE / "longtail_vlm/rarevlm_pilot/outputs/vlm_deep_v2"
DEFAULT_PROBE = WORKSPACE / "longtail_vlm/rarevlm_pilot/outputs/vlm_precision_bridge_v1"
NORMALIZER_PATH = REPO / "src" / "vlm_deep/third_party/vqa_eval_metric.py"
FONT_PATH = WORKSPACE / ".tools/fonts/NotoSansCJKsc-Regular.otf"
ROLES = ("base", "adapted")
PATHS = ("nf4_fp32", "original_fp32", "original_bf16")
PROTOCOLS = {"nf4_fp32": ("fp32", "nf4_language_only"),
             "original_fp32": ("fp32", "original"),
             "original_bf16": ("bf16", "original")}
CONTRASTS = [(role, first, second) for role in ROLES
             for first, second in (("nf4_fp32", "original_fp32"), ("original_fp32", "original_bf16"))]
REPEATED_BATCH_FIELDS = ("batch_size", "padded_input_tokens", "batch_generation_seconds",
                        "batch_preprocessing_seconds", "batch_h2d_seconds", "batch_peak_allocated_gib",
                        "batch_peak_reserved_gib", "timing_warmup")
INPUT_FIELDS = ("question_id", "image_id", "question_sha256", "image_sha256", "image_grid_thw",
                "visual_tokens", "counted_image_tokens", "resized_width", "resized_height",
                "input_tokens", "padded_input_tokens", "input_truncated", "batch_start", "batch_size")


def sha_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def content_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def private_path(path):
    resolved = path.resolve()
    if not resolved.is_relative_to(WORKSPACE):
        raise ValueError(f"artifact outside persistent workspace: {resolved}")
    return resolved


def write_json(path, value):
    private_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    private_path(path)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                         for k, v in row.items()} for row in rows)


def answer_normalizer():
    # Load only the pinned pure-Python answer processor, not torch/model modules.
    spec = importlib.util.spec_from_file_location("precision_bridge_normalizer", NORMALIZER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EvalAIAnswerProcessor()


def score_prediction(prediction, references, normalize):
    if len(references) != 10:
        raise ValueError("TextVQA consensus needs ten annotators")
    answer = normalize(prediction)
    refs = [normalize(str(ref)) for ref in references]
    return sum(min(sum(answer == ref for j, ref in enumerate(refs) if j != i) / 3, 1)
               for i in range(10)) / 10


def distribution(values):
    if not values:
        return None
    values = np.asarray(values, dtype=float)
    return {"n": len(values), "min": float(values.min()), "mean": float(values.mean()),
            "median": float(np.median(values)), "p05": float(np.quantile(values, .05)),
            "p95": float(np.quantile(values, .95)), "max": float(values.max())}


def manifest_data(asset_root):
    path = asset_root / "prepared/manifest.json"
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != "rarevlm.vlm_data.v2":
        raise ValueError("expected v2 data manifest")
    if content_sha({k: v for k, v in manifest.items() if k != "manifest_sha256"}) != manifest["manifest_sha256"]:
        raise ValueError("data manifest content hash mismatch")
    rows = manifest["splits"]["dev"]
    if len(rows) != 1617 or len({row["image_id"] for row in rows}) != 1024:
        raise ValueError("expected the complete frozen dev: 1617 QA / 1024 images")
    if len({str(row["question_id"]) for row in rows}) != len(rows):
        raise ValueError("duplicate dev qid")
    return path, manifest, rows


def resource_contention(probe_root):
    """Conservatively tag a whole role; sampled absence is not exclusivity."""
    path = probe_root / "RESOURCE_CONTENTION.json"
    raw = json.loads(path.read_text()) if path.exists() else None
    counts = raw.get("observations_by_role", {}) if raw else {}
    roles = {}
    for role in ROLES:
        count = counts.get(role)
        if count is not None and (not isinstance(count, int) or count < 0):
            raise ValueError("invalid contention observation count")
        roles[role] = {"resource_timing_status": "monitor_missing" if count is None else
                      "contended" if count > 0 else "no_observed_contention",
                      "sampled_contention_observations": count,
                      "exclusive_gpu_benchmark": False,
                      "scope": "Tag all three paths of a role if the monitor sampled any co-resident same-GPU compute process. Zero sampled events does not prove GPU exclusivity; no tasks were preempted."}
    return {"file": str(path), "file_sha256": sha_file(path) if raw else None,
            "monitor_snapshot": raw, "roles": roles,
            "events_file": str(probe_root / "RESOURCE_CONTENTION.jsonl"),
            "resource_snapshots_file": str(probe_root / "RESOURCE_SNAPSHOTS.jsonl")}


def check_lock(lock, role, pathname, manifest, manifest_path):
    precision, quantization = PROTOCOLS[pathname]
    expected = {"role": role, "numerical_path": pathname, "split": "dev", "task": "textvqa",
                "precision": precision, "quantization": quantization, "visual_token_budget": 256,
                "batch_size": 4, "max_new_tokens": 32, "do_sample": False,
                "tf32_matmul": False, "tf32_cudnn": False, "attention_implementation": "sdpa",
                "manifest_sha256": manifest["manifest_sha256"]}
    for key, value in expected.items():
        if lock.get(key) != value:
            raise ValueError(f"unexpected {role}/{pathname} input lock: {key}")
    if lock.get("manifest_file_sha256") != sha_file(manifest_path):
        raise ValueError("manifest file binding differs")
    if role == "base" and lock.get("adapter_files_sha256") is not None:
        raise ValueError("base role unexpectedly has an adapter")
    if role == "adapted":
        binding = lock.get("training_binding", {})
        if binding.get("method") != "qv_r8" or binding.get("seed") != 42 or not lock.get("adapter_files_sha256"):
            raise ValueError("adapted role is not the predeclared qv_r8 seed42 checkpoint")


def read_cells(probe_root, manifest_path, manifest, references):
    cells, progress, provenance = {}, [], {}
    normalize = answer_normalizer()
    for role in ROLES:
        for pathname in PATHS:
            directory = probe_root / "runs" / role / pathname
            prediction_path = directory / "predictions.jsonl"
            evaluation_path = directory / "EVALUATION.json"
            lock_path = directory / "INPUT_LOCK.json"
            lock = json.loads(lock_path.read_text()) if lock_path.exists() else None
            if lock is not None:
                check_lock(lock, role, pathname, manifest, manifest_path)
            data = prediction_path.read_bytes() if prediction_path.exists() else b""
            lines = data.splitlines(keepends=True)
            tail_ignored = bool(lines and not lines[-1].endswith(b"\n"))
            if tail_ignored:
                lines = lines[:-1]
            rows = [json.loads(line) for line in lines]
            if len(rows) > len(references) or (rows and lock is None):
                raise ValueError("invalid prediction count or absent input lock")
            for index, row in enumerate(rows):
                ref = references[index]
                image = manifest["images"][ref["image_id"]]
                expected = {"question_id": str(ref["question_id"]), "image_id": str(ref["image_id"]),
                            "question_sha256": content_sha(ref["question"]), "image_sha256": image["sha256"]}
                if any(str(row[key]) != value for key, value in expected.items()):
                    raise ValueError(f"dev identity/order/hash changed at {role}/{pathname}/{index}")
                precision, quantization = PROTOCOLS[pathname]
                if (row["numerical_path"], row["precision"], row["quantization"]) != (pathname, precision, quantization):
                    raise ValueError("prediction numerical-path identity differs")
                if row["input_lock_sha256"] != content_sha(lock):
                    raise ValueError("prediction input lock hash differs")
                if row["visual_token_budget"] != 256 or row["input_truncated"]:
                    raise ValueError("visual budget changed or input truncated")
                grid = row["image_grid_thw"]
                patches = math.prod(grid)
                visual = patches // 4
                if len(grid) != 3 or grid[0] != 1 or patches % 4 or not 0 < visual <= 256:
                    raise ValueError("unexpected image grid/visual budget")
                if row["visual_tokens"] != visual or row["counted_image_tokens"] != visual:
                    raise ValueError("visual token accounting differs")
                if row["resized_width"] != grid[2] * 14 or row["resized_height"] != grid[1] * 14:
                    raise ValueError("recorded image resize/grid differ")
                if row["padded_input_tokens"] < row["input_tokens"]:
                    raise ValueError("invalid input padding count")
                if row["batch_start"] != index // 4 * 4 or row["batch_size"] != min(4, len(references) - index // 4 * 4):
                    raise ValueError("fixed batch grouping changed")
                score = score_prediction(row["prediction"], ref["answers"], normalize)
                if not math.isfinite(row["score"]) or not math.isclose(row["score"], score, abs_tol=1e-12, rel_tol=0):
                    raise ValueError("prediction consensus score does not reproduce")
            evaluation = json.loads(evaluation_path.read_text()) if evaluation_path.exists() else None
            complete = evaluation is not None
            if complete:
                if lock is None or tail_ignored or len(rows) != len(references) or evaluation["n"] != len(references):
                    raise ValueError("evaluation claims completion with incomplete predictions")
                check_lock(evaluation, role, pathname, manifest, manifest_path)
                if evaluation["input_lock_sha256"] != content_sha(lock):
                    raise ValueError("evaluation input lock hash differs")
                if evaluation["predictions_sha256"] != hashlib.sha256(data).hexdigest():
                    raise ValueError("completed predictions file hash mismatch")
                if not math.isclose(evaluation["score_percent"], 100 * statistics.mean(row["score"] for row in rows), abs_tol=1e-10):
                    raise ValueError("completed evaluation metric differs")
            key = (role, pathname)
            cells[key] = {"rows": rows, "lock": lock, "complete": complete, "evaluation": evaluation}
            progress.append({"role": role, "numerical_path": pathname, "complete": complete,
                             "validated_saved_questions": len(rows), "expected_questions": len(references),
                             "incomplete_tail_ignored": tail_ignored})
            provenance[f"{role}:{pathname}"] = {"predictions_file": str(prediction_path),
                "predictions_snapshot_sha256": hashlib.sha256(data).hexdigest(),
                "input_lock_file": str(lock_path), "input_lock_sha256": content_sha(lock) if lock else None,
                "evaluation_file": str(evaluation_path), "evaluation_file_sha256": sha_file(evaluation_path) if complete else None}
    return cells, progress, provenance


def check_shared_inputs(cells):
    first = cells[("base", "nf4_fp32")]
    for key, cell in cells.items():
        for index, (ref, candidate) in enumerate(zip(first["rows"], cell["rows"])):
            if any(ref[field] != candidate[field] for field in INPUT_FIELDS):
                raise ValueError(f"encoded inputs/batch groups differ: {key}/{index}")
        for field in ("source_sha256", "packages", "model_revision", "model_config_sha256", "model_provenance_sha256",
                      "min_pixels", "max_new_tokens", "do_sample", "attention_implementation", "head_last_position_only"):
            if cell["lock"].get(field) != first["lock"].get(field):
                raise ValueError(f"six-cell common contract differs: {key}/{field}")
    for role in ROLES:
        original = cells[(role, "nf4_fp32")]["lock"]
        for pathname in PATHS[1:]:
            candidate = cells[(role, pathname)]["lock"]
            for field in ("adapter_files_sha256", "training_binding", "hardware"):
                if candidate.get(field) != original.get(field):
                    raise ValueError(f"within-role checkpoint/hardware differs: {role}/{pathname}/{field}")


def cell_summary(role, pathname, cell):
    rows = cell["rows"]
    groups = defaultdict(list)
    for row in rows:
        groups[(row["session_id"], row["batch_start"])].append(row)
    batches = []
    for (session, start), members in sorted(groups.items()):
        head = members[0]
        if len(members) != head["batch_size"]:
            raise ValueError("completed cell contains an incomplete batch")
        if any(any(member[field] != head[field] for field in REPEATED_BATCH_FIELDS) for member in members[1:]):
            raise ValueError("duplicated batch-level timing fields disagree")
        for field in REPEATED_BATCH_FIELDS:
            if field != "timing_warmup" and (not math.isfinite(float(head[field])) or float(head[field]) < 0):
                raise ValueError("negative/non-finite resource measurement")
        batches.append({"role": role, "numerical_path": pathname, "session_id": session, "batch_start": start,
                        **{field: head[field] for field in REPEATED_BATCH_FIELDS}})
    by_session = defaultdict(list)
    for batch in batches:
        by_session[batch["session_id"]].append(batch)
    for members in by_session.values():
        first = min(member["batch_start"] for member in members)
        if any(bool(member["timing_warmup"]) != (member["batch_start"] == first) for member in members):
            raise ValueError("session first-batch warmup rule differs")
    timed = [batch for batch in batches if not batch["timing_warmup"]]
    n = sum(batch["batch_size"] for batch in timed)
    generation = sum(batch["batch_generation_seconds"] for batch in timed)
    preprocessing = sum(batch["batch_preprocessing_seconds"] for batch in timed)
    transfer = sum(batch["batch_h2d_seconds"] for batch in timed)
    if n == 0 or generation <= 0:
        raise ValueError("no steady-state generation measurements")
    summary = {"role": role, "numerical_path": pathname, "precision": PROTOCOLS[pathname][0],
        "quantization": PROTOCOLS[pathname][1], "score_percent": 100 * statistics.mean(row["score"] for row in rows),
        "questions": len(rows), "unique_images": len({row["image_id"] for row in rows}),
        "visual_token_budget": 256, "visual_tokens": distribution([row["visual_tokens"] for row in rows]),
        "input_tokens": distribution([row["input_tokens"] for row in rows]),
        "generated_nonpad_tokens_including_eos": distribution([row["generated_nonpad_tokens"] for row in rows]),
        "all_batches": len(batches), "sessions": sorted(by_session),
        "excluded_session_warmup_batches": len(batches) - len(timed), "timed_batches": len(timed), "timed_questions": n,
        "generation_seconds_excluding_warmup": generation, "preprocessing_seconds_excluding_warmup": preprocessing,
        "h2d_seconds_excluding_warmup": transfer,
        "pure_generation_questions_per_second": n / generation,
        "generation_preprocessing_h2d_questions_per_second": n / (generation + preprocessing + transfer),
        "mean_amortized_generation_seconds_per_question": generation / n,
        "batch_generation_seconds_excluding_warmup": distribution([batch["batch_generation_seconds"] for batch in timed]),
        "all_preprocessing_seconds": sum(batch["batch_preprocessing_seconds"] for batch in batches),
        "all_h2d_seconds": sum(batch["batch_h2d_seconds"] for batch in batches),
        "peak_allocated_gib": max(batch["batch_peak_allocated_gib"] for batch in batches),
        "peak_reserved_gib": max(batch["batch_peak_reserved_gib"] for batch in batches),
        "actual_parameter_dtypes": cell["lock"].get("actual_parameter_dtypes"),
        "hardware": cell["lock"].get("hardware"),
        "timing_scope": "Count each (session_id,batch_start) once. Exclude the first batch of every session from all steady-state phase totals; include all complete batches for peaks/all preprocessing/H2D. Generation+preprocessing+H2D excludes loading, preflight, image checks, disk writes and other controller overhead. Batch-amortized latency is not independent single-request latency. Observed concurrency is not controlled to an idle-GPU benchmark."}
    reported = cell["evaluation"]["resources"]["pure_generation_questions_per_second"]
    if not math.isclose(summary["pure_generation_questions_per_second"], reported, rel_tol=1e-9):
        raise ValueError("worker and analysis generation throughput differ")
    return summary, batches


def bootstrap_difference(values, references, replicates, seed):
    clusters = defaultdict(lambda: [0.0, 0])
    for value, row in zip(values, references):
        clusters[row["image_id"]][0] += float(value)
        clusters[row["image_id"]][1] += 1
    grouped = np.asarray(list(clusters.values()), dtype=float)
    rng = np.random.default_rng(seed)
    estimates = []
    for start in range(0, replicates, 64):
        sampled = grouped[rng.integers(0, len(grouped), (min(64, replicates - start), len(grouped)))].sum(axis=1)
        estimates.extend((100 * sampled[:, 0] / sampled[:, 1]).tolist())
    return {"delta_pp": float(100 * grouped[:, 0].sum() / grouped[:, 1].sum()),
            "image_cluster_bootstrap_95_pp": np.quantile(estimates, [.025, .975]).tolist(),
            "questions": len(references), "unique_image_clusters": len(grouped), "bootstrap_replicates": replicates,
            "bootstrap_seed": seed,
            "uncertainty_scope": "Pointwise exploratory 95% percentile CI, resample entire images and retain every paired same-image QA. Conditional on fixed weights/input/dev sample; no training-seed uncertainty, equivalence margin or multiple-comparison correction."}


def paired_contrast(role, before, after, cells, references, replicates, seed):
    first, second = cells[(role, before)]["rows"], cells[(role, after)]["rows"]
    normalize = answer_normalizer()
    pairs = []
    transitions = defaultdict(int)
    for source, target in zip(first, second):
        a, b = float(source["score"]), float(target["score"])
        transitions[(round(a, 6), round(b, 6))] += 1
        pairs.append({"role": role, "before_path": before, "after_path": after,
                      "question_id": source["question_id"], "image_id": source["image_id"],
                      "before_prediction": source["prediction"], "after_prediction": target["prediction"],
                      "before_normalized_prediction": normalize(source["prediction"]),
                      "after_normalized_prediction": normalize(target["prediction"]),
                      "before_score": a, "after_score": b, "delta_score": b - a})
    n = len(pairs)
    raw = sum(pair["before_prediction"] == pair["after_prediction"] for pair in pairs)
    normalized = sum(pair["before_normalized_prediction"] == pair["after_normalized_prediction"] for pair in pairs)
    full_to_zero = sum(pair["before_score"] == 1 and pair["after_score"] == 0 for pair in pairs)
    zero_to_full = sum(pair["before_score"] == 0 and pair["after_score"] == 1 for pair in pairs)
    summary = {"role": role, "before_path": before, "after_path": after,
        "contrast": f"{role}:{after}-{before}",
        **bootstrap_difference([pair["delta_score"] for pair in pairs], references, replicates, seed),
        "raw_prediction_identical_questions": raw, "raw_prediction_agreement_percent": 100 * raw / n,
        "normalized_prediction_identical_questions": normalized, "normalized_prediction_agreement_percent": 100 * normalized / n,
        "full_credit_1_to_zero_0": full_to_zero, "zero_0_to_full_credit_1": zero_to_full,
        "full_credit_1_to_partial": sum(pair["before_score"] == 1 and 0 < pair["after_score"] < 1 for pair in pairs),
        "partial_to_full_credit_1": sum(0 < pair["before_score"] < 1 and pair["after_score"] == 1 for pair in pairs),
        "any_score_improvement_questions": sum(pair["delta_score"] > 1e-12 for pair in pairs),
        "any_score_regression_questions": sum(pair["delta_score"] < -1e-12 for pair in pairs),
        "score_unchanged_questions": sum(abs(pair["delta_score"]) <= 1e-12 for pair in pairs),
        "sum_positive_score_change": sum(max(pair["delta_score"], 0) for pair in pairs),
        "sum_negative_score_change": sum(min(pair["delta_score"], 0) for pair in pairs),
        "score_transition_matrix": [{"before_score": a, "after_score": b, "questions": count}
                                     for (a, b), count in sorted(transitions.items())],
        "agreement_scope": "Agreement concerns recorded decoded strings only; it is not parameter, logit, kernel, mathematical, or statistical equivalence. Full-credit and zero transitions are descriptive; preserve every partial score in matrix and paired CSV."}
    matrix = [{"role": role, "before_path": before, "after_path": after, **row}
              for row in summary["score_transition_matrix"]]
    return summary, pairs, matrix


def plot_results(directory, summaries, contrasts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties, fontManager
    if FONT_PATH.exists():
        fontManager.addfont(str(FONT_PATH))
        font = FontProperties(fname=str(FONT_PATH)).get_name()
    else:
        font = "DejaVu Sans"
    lookup = {(row["role"], row["numerical_path"]): row for row in summaries}
    colors = {"base": "#0072B2", "adapted": "#007A5E"}
    markers = {"base": "o", "adapted": "s"}
    role_labels = {"base": "固定底座", "adapted": "固定adapter · seed42"}
    timing_labels = {role: lookup[(role, "nf4_fp32")]["resource_timing_status"] for role in ROLES}
    monitor_caption = "；".join(f"{role}: {timing_labels[role]}" for role in ROLES)
    path_labels = ["NF4 / FP32", "原权重 / FP32", "原权重 / BF16"]
    files = []
    with plt.rc_context({"font.family": font, "font.size": 10, "svg.fonttype": "path", "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white", "savefig.facecolor": "white"}):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), layout="constrained")
        for role, shift in (("base", -.16), ("adapted", .16)):
            points = [lookup[(role, pathname)] for pathname in PATHS]
            x = np.arange(3) + shift
            axes[0, 0].scatter(x, [row["score_percent"] for row in points], marker=markers[role], s=65,
                               color=colors[role], label=role_labels[role])
            for coordinate, row in zip(x, points):
                axes[0, 0].annotate(f'{row["score_percent"]:.3f}', (coordinate, row["score_percent"]),
                                    xytext=(0, 8 if role == "adapted" else -15), textcoords="offset points", ha="center", fontsize=8)
            axes[0, 1].bar(x, [row["pure_generation_questions_per_second"] for row in points], width=.3,
                          color=colors[role], hatch="" if role == "base" else "//", label=role_labels[role])
            axes[1, 0].bar(x, [row["peak_allocated_gib"] for row in points], width=.3,
                          color=colors[role], hatch="" if role == "base" else "//", label=role_labels[role] + " allocated")
            axes[1, 0].scatter(x, [row["peak_reserved_gib"] for row in points], facecolors="none", edgecolors=colors[role],
                               marker=markers[role], s=75, label=role_labels[role] + " reserved")
        axes[0, 0].set_title("A 完整内部dev分数：每点一个固定权重", loc="left")
        axes[0, 0].set_ylabel("VQA一致得分（%）"); axes[0, 0].legend(fontsize=8)
        axes[0, 1].set_title("B 稳态纯生成吞吐", loc="left"); axes[0, 1].set_ylabel("QA / generation秒")
        axes[1, 0].set_title("C 完整单元峰值显存", loc="left"); axes[1, 0].set_ylabel("GiB"); axes[1, 0].legend(fontsize=7)
        for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
            ax.set_xticks(range(3), path_labels); ax.grid(axis="y", color="#DDDDDD", linewidth=.6)
        axes[0, 1].set_ylim(bottom=0); axes[1, 0].set_ylim(bottom=0)
        y = np.arange(len(contrasts))
        axes[1, 1].scatter([row["raw_prediction_agreement_percent"] for row in contrasts], y - .1,
                          marker="o", color="#0072B2", label="原始预测字符串一致")
        axes[1, 1].scatter([row["normalized_prediction_agreement_percent"] for row in contrasts], y + .1,
                          marker="s", color="#007A5E", label="评分规范化后字符串一致")
        axes[1, 1].set_yticks(y, [row["contrast"] for row in contrasts], fontsize=7)
        axes[1, 1].invert_yaxis(); axes[1, 1].set_xlim(0, 100)
        axes[1, 1].set_xlabel("一致率（%）；不构成等价证明")
        axes[1, 1].set_title("D 同题输出敏感性", loc="left"); axes[1, 1].legend(fontsize=7)
        fig.suptitle("数值路径工程桥接：固定权重来源、同输入、视觉预算256", fontsize=15, weight="bold")
        fig.supxlabel("每单元1617题/1024图；adapter为固定q/v rank8 seed42。不是多训练种子比较。\n"
                      "吞吐排除各session首批；原权重/BF16同时改变存储dtype与kernel路径。\n"
                      + monitor_caption + "；采样无竞争也不保证独占GPU。", fontsize=9)
        for ext in ("png", "svg"):
            path = directory / f"quality_resources.{ext}"; fig.savefig(path, dpi=180); files.append(path)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(12.5, 5), layout="constrained")
        for index, row in enumerate(contrasts):
            lower, upper = row["image_cluster_bootstrap_95_pp"]
            point = row["delta_pp"]
            # Small finite bootstrap samples may place the point outside the
            # percentile interval: draw interval and point independently.
            ax.hlines(index, lower, upper, color=colors[row["role"]], linewidth=2)
            ax.plot([lower, upper], [index, index], "|", color=colors[row["role"]], markersize=12)
            ax.plot(point, index, marker=markers[row["role"]], color=colors[row["role"]], markersize=7)
            ax.annotate(f"{point:+.3f} [{lower:+.3f}, {upper:+.3f}]", (upper, index), xytext=(8, 0),
                        textcoords="offset points", va="center", fontsize=9)
        ax.axvline(0, color="#555555", linestyle="--")
        ax.set_yticks(range(len(contrasts)), [row["contrast"] for row in contrasts]); ax.invert_yaxis()
        ax.set_xlabel("同图配对VQA差值（百分点）；95% image-cluster bootstrap区间")
        ax.set_title("四个预定工程比较：量化路径与BF16路径敏感性", loc="left", weight="bold")
        ax.grid(axis="x", color="#DDDDDD", linewidth=.6)
        lower, upper = ax.get_xlim(); ax.set_xlim(lower, upper + max((upper - lower) * .5, .5))
        fig.supxlabel("2000次整图配对重采样；区间条件于固定权重、内部dev，不含训练种子不确定性。\n"
                      "四比较未进行多重校正，探索性区间；未预定等价界限，跨零不证明等价。", fontsize=9)
        for ext in ("png", "svg"):
            path = directory / f"paired_numerical_deltas.{ext}"; fig.savefig(path, dpi=180); files.append(path)
        plt.close(fig)
    return [str(path) for path in files]


def analyze(asset_root, probe_root, replicates, seed):
    manifest_path, manifest, references = manifest_data(asset_root)
    output = probe_root / "analysis"; output.mkdir(parents=True, exist_ok=True)
    cells, progress, provenance = read_cells(probe_root, manifest_path, manifest, references)
    contention = resource_contention(probe_root)
    common = {"schema_version": "rarevlm.precision_bridge.analysis.v1", "updated_at": datetime.now(timezone.utc).isoformat(),
              "completed_cells": sum(cell["complete"] for cell in cells.values()), "expected_cells": 6,
              "manifest_sha256": manifest["manifest_sha256"], "cells": progress, "source_data": provenance,
              "resource_contention": contention}
    if not all(cell["complete"] for cell in cells.values()):
        result = {**common, "complete": False, "status": "partial_generation",
                  "scope": "Progress only; no partial final comparison or hypothesis selection."}
        write_json(output / "ANALYSIS.json", result)
        return result
    check_shared_inputs(cells)
    summaries, batches = [], []
    for role in ROLES:
        for pathname in PATHS:
            summary, batch_rows = cell_summary(role, pathname, cells[(role, pathname)])
            summary.update(contention["roles"][role])
            summaries.append(summary); batches.extend(batch_rows)
    contrasts, pairs, matrices = [], [], []
    for role, before, after in CONTRASTS:
        summary, pair_rows, matrix = paired_contrast(role, before, after, cells, references, replicates, seed)
        contrasts.append(summary); pairs.extend(pair_rows); matrices.extend(matrix)
    write_csv(output / "cell_metrics.csv", summaries)
    write_csv(output / "paired_contrasts.csv", contrasts)
    write_csv(output / "paired_questions.csv", pairs)
    write_csv(output / "score_transitions.csv", matrices)
    write_csv(output / "batch_resources.csv", batches)
    figures = plot_results(output, summaries, contrasts)
    result = {**common, "complete": True, "status": "completed", "summaries": summaries, "paired_contrasts": contrasts,
              "figures": figures, "checks": {"all_saved_predictions_rescored": len(references) * 6,
              "same_encoded_inputs_and_batch_groups": True, "same_within_role_checkpoint_hardware": True,
              "source_package_generation_contracts_match": True, "batch_timing_deduplicated_and_warmup_checked": True},
              "limitations": ["One fixed adapter seed42, one fixed internal dev set; this is not general multi-seed algorithm ranking.",
                  "Original-weight loading and BF16 storage/compute can change kernels, dtypes and numeric execution; no parameter/logit equivalence is assumed.",
                  "NF4-trained adapter on original weights is a deployment sensitivity bridge, not a new BF16 training experiment.",
                  "Paired image bootstrap retains all same-image questions; four pointwise intervals are exploratory and unadjusted.",
                  "String agreement or a CI crossing zero does not prove equivalence; no equivalence margin was predeclared.",
                  "Concurrent GPU jobs are preserved. Timing is observed under recorded concurrent load, not an isolated hardware benchmark.",
                  "Quality is full QA mean; resources count complete batches once and distinguish steady-state phase timing from end-to-end wall time."],
              "software_reference": {"title": "Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents",
                  "authors": "Kassis, T.; Agarwal, V.; He, Y.; Patel, D.; Brueckner, A. M.", "year": 2026,
                  "url": "https://doi.org/10.48550/arXiv.2609.00065", "scope": "Scientific visualization workflow"}}
    write_json(output / "ANALYSIS.json", result)
    figure_provenance = {"analysis_code": str(Path(__file__).resolve()), "analysis_code_sha256": sha_file(Path(__file__)),
        "manifest_file": str(manifest_path), "manifest_file_sha256": sha_file(manifest_path), "source_data": provenance,
        "resource_contention": contention,
        "source_tables": [str(output / name) for name in ("cell_metrics.csv", "paired_contrasts.csv", "batch_resources.csv", "score_transitions.csv")],
        "uncertainty": "95% pointwise percentile image-cluster paired bootstrap; conditional fixed weights; no multiplicity correction",
        "bootstrap_replicates": replicates, "bootstrap_seed": seed, "normalizer_source": str(NORMALIZER_PATH),
        "normalizer_source_sha256": sha_file(NORMALIZER_PATH), "score": "pinned TextVQA leave-one-annotator-out consensus; question mean",
        "transformations": ["No input filtering: all 1617 dev QA retained", "Resample complete image clusters, preserve question-weighted mean",
            "Deduplicate repeated batch resources by (session_id,batch_start)", "Exclude each session first batch from steady-state phase totals",
            "Exact decoded and EvalAI-normalized prediction agreement reported separately", "Full score transitions preserve partial credit"],
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "matplotlib")},
        "figure_exports": [{"file": path, "sha256": sha_file(Path(path)), "size_bytes": Path(path).stat().st_size} for path in figures],
        "dpi": 180, "formats": ["png", "svg"], "font_path": str(FONT_PATH), "svg_text": "glyph paths for portability",
        "destination": "Chinese educational research report, provisional general figures; no journal compliance claim",
        "alt_text": "Six fixed-weight internal-dev points compare NF4/FP32, original-weight/FP32, and original-weight/BF16 quality, generation throughput and memory for a base and one fixed adapter. Four paired image-cluster intervals quantify numerical-path sensitivity; separate raw and normalized string agreement is descriptive and does not establish equivalence."}
    write_json(output / "FIGURE_PROVENANCE.json", figure_provenance)
    lines = ["固定权重数值路径桥接：六单元完整；每单元1617题/1024图，视觉预算256。", "",
             "单个固定adapter seed42的部署数值敏感性研究，不是BF16训练或多种子算法排名。", ""]
    for row in summaries:
        lines.append(f"{row['role']} {row['numerical_path']}: VQA {row['score_percent']:.3f}%；纯生成 {row['pure_generation_questions_per_second']:.3f} QA/s；预处理+H2D+生成 {row['generation_preprocessing_h2d_questions_per_second']:.3f} QA/s；allocated峰值 {row['peak_allocated_gib']:.3f} GiB；计时标记 {row['resource_timing_status']}（并发观测 {row['sampled_contention_observations']}）。")
    lines.extend(["", "预定四比较（2000次整图配对条件95%区间，未多重校正）："])
    for row in contrasts:
        interval = row["image_cluster_bootstrap_95_pp"]
        lines.append(f"{row['contrast']}: {row['delta_pp']:+.3f} pp [{interval[0]:+.3f}, {interval[1]:+.3f}]；原字符串一致 {row['raw_prediction_agreement_percent']:.2f}%，规范化一致 {row['normalized_prediction_agreement_percent']:.2f}%；满分→零分 {row['full_credit_1_to_zero_0']}，零分→满分 {row['zero_0_to_full_credit_1']}；部分得分详见score_transitions.csv。")
    lines.extend(["", "输出一致率、跨零区间都不证明等价。存储dtype、kernel和并发任务可能影响速度，计时不含加载/预检/磁盘写入。",
                  "绘图软件参考：Kassis et al. (2026), https://doi.org/10.48550/arXiv.2609.00065。"])
    (output / "FINDINGS_ZH.txt").write_text("\n".join(lines) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET)
    parser.add_argument("--probe-root", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261011)
    args = parser.parse_args()
    args.probe_root = private_path(args.probe_root)
    if args.replicates != 2000:
        raise ValueError("declared bridge fixes 2000 bootstrap replicates")
    result = analyze(args.asset_root, args.probe_root, args.replicates, args.bootstrap_seed)
    print(json.dumps({"complete": result["complete"], "status": result["status"],
                      "completed_cells": result["completed_cells"], "analysis_file": str(args.probe_root / "analysis/ANALYSIS.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
