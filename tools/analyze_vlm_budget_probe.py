"""CPU-only slices and analysis for the separately declared visual-budget probe.

Slice membership is frozen before generation. Reference-derived frequency is
descriptive and never changes prompts, inference, or prediction-based selection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import re
import statistics
import sys
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
WORKSPACE = Path("/opt/data/private")
REPO = Path(__file__).resolve().parents[1]
DEFAULT_ASSET_ROOT = WORKSPACE / "longtail_vlm/rarevlm_pilot/outputs/vlm_deep_v2"
DEFAULT_OUTPUT_ROOT = WORKSPACE / "longtail_vlm/rarevlm_pilot/outputs/vlm_budget_probe_v1"
NORMALIZER_PATH = REPO / "src" / "vlm_deep/third_party/vqa_eval_metric.py"
TRAINING_TARGET_SOURCE = REPO / "src" / "vlm_deep/common.py"
NUMERIC_PROXY = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:\s*%)?"
BUDGETS = (256, 512, 1024)
ROLES = ("base", "adapted")
FONT_PATH = WORKSPACE / ".tools/fonts/NotoSansCJKsc-Regular.otf"
CASE_IDS = ("27482", "10476", "12257")
SLICE_DIMENSIONS = ("sft_target_answer_frequency", "reference_answer_frequency", "answer_char_length", "answer_text_proxy")
CONTRASTS = (
    ("base:512-256", {("base", 512): 1, ("base", 256): -1}),
    ("base:1024-256", {("base", 1024): 1, ("base", 256): -1}),
    ("adapted:512-256", {("adapted", 512): 1, ("adapted", 256): -1}),
    ("adapted:1024-256", {("adapted", 1024): 1, ("adapted", 256): -1}),
    ("(adapted:1024-256)-(base:1024-256)", {("adapted", 1024): 1, ("adapted", 256): -1, ("base", 1024): -1, ("base", 256): 1}),
)


def sha_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def content_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    if not path.resolve().is_relative_to(WORKSPACE):
        raise ValueError("output outside persistent workspace")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def answer_normalizer():
    # Import the pinned normalizer alone: no GPU, model loading or source pycache.
    spec = importlib.util.spec_from_file_location("budget_probe_pinned_vqa_normalizer", NORMALIZER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EvalAIAnswerProcessor()


def actual_training_target(row):
    # Same raw, stripped modal rule as frozen vlm_deep.common.training_target.
    values = [str(a).strip() for a in row["answers"] if str(a).strip()]
    counts = Counter(values)
    if not counts:
        raise ValueError("empty actual SFT target")
    target = min(counts, key=lambda a: (-counts[a], a))
    if row.get("target") != target:
        raise ValueError(f"manifest target differs from actual SFT rule: {row['question_id']}")
    return target


def frequency_bucket(count):
    return "unseen_0" if count == 0 else "rare_1_5" if count <= 5 else "medium_6_20" if count <= 20 else "frequent_gt20"


def prepare_slice_lock(asset_root, output_root):
    manifest_path = asset_root / "prepared/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["schema_version"] != "rarevlm.vlm_data.v2":
        raise ValueError("not the declared v2 manifest")
    computed = content_sha({k: v for k, v in manifest.items() if k != "manifest_sha256"})
    if computed != manifest["manifest_sha256"]:
        raise ValueError("manifest content hash mismatch")
    normalize = answer_normalizer()
    training = manifest["splits"]["train_all"]
    dev = manifest["splits"]["dev"]
    if {r["image_id"] for r in training} & {r["image_id"] for r in dev}:
        raise ValueError("training/dev image overlap")
    if len({str(r["question_id"]) for r in dev}) != len(dev):
        raise ValueError("duplicate internal dev questions")
    frequencies = Counter()
    target_frequencies = Counter()
    for row in training:
        # One QA may contribute several distinct normalized references, each once.
        unique_references = {normalize(str(a)) for a in row["answers"]}
        frequencies.update(a for a in unique_references if a)
        normalized_target = normalize(actual_training_target(row))
        # Normalization can remove articles/punctuation from a real nonempty
        # supervised string. Keep its empty normalized key instead of dropping QA.
        target_frequencies[normalized_target] += 1
    records = []
    grouped = defaultdict(list)
    for row in dev:
        normalized = [normalize(str(a)) for a in row["answers"]]
        reference_counts = Counter(a for a in normalized if a)
        representative = min(reference_counts, key=lambda a: (-reference_counts[a], a)) if reference_counts else ""
        count = frequencies[representative]
        normalized_actual_target = normalize(actual_training_target(row))
        target_count = target_frequencies[normalized_actual_target]
        length_bucket = "short_le4" if len(representative) <= 4 else "medium_5_10" if len(representative) <= 10 else "long_gt10"
        numeric_bucket = "numeric_like" if re.fullmatch(NUMERIC_PROXY, representative) else "other_text"
        feature = {"question_id": str(row["question_id"]), "image_id": row["image_id"],
            "question_sha256": content_sha(row["question"]), "image_sha256": manifest["images"][row["image_id"]]["sha256"],
            "modal_normalized_reference": representative, "modal_reference_annotator_count": reference_counts[representative],
            "all_normalized_references_empty": not bool(reference_counts),
            "normalized_reference_counts": dict(sorted(reference_counts.items())),
            "train_all_unique_reference_qa_frequency": count, "reference_answer_frequency": frequency_bucket(count),
            "sft_target_normalized": normalized_actual_target,
            "train_all_sft_target_qa_frequency": target_count, "sft_target_answer_frequency": frequency_bucket(target_count),
            "normalized_answer_characters": len(representative), "answer_char_length": length_bucket,
            "answer_text_proxy": numeric_bucket}
        records.append(feature)
        for dimension in ("reference_answer_frequency", "sft_target_answer_frequency", "answer_char_length", "answer_text_proxy"):
            grouped[(dimension, feature[dimension])].append(feature)
    slice_counts = [{"dimension": dimension, "slice": label, "questions": len(members),
                     "unique_images": len({r["image_id"] for r in members})}
                    for (dimension, label), members in sorted(grouped.items())]
    lock = {"schema_version": "rarevlm.budget_probe.slices.v1",
        "purpose": "Internal-dev exploratory grouping fixed before budget-probe prediction analysis; no official validation selection.",
        "manifest_path": str(manifest_path.resolve()), "manifest_file_sha256": sha_file(manifest_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "normalizer": {"class": "EvalAIAnswerProcessor", "source_path": str(NORMALIZER_PATH.resolve()), "source_sha256": sha_file(NORMALIZER_PATH)},
        "frequency_definition": {
            "training_split": "train_all", "interpretation": "auxiliary reference-coverage proxy; not actual supervised target exposure", "unit": "training QA containing the normalized reference",
            "rule": "Normalize every reference; discard empty strings; deduplicate within each training QA; increment each remaining normalized answer once per QA.",
            "not_sft_target_frequency": True, "not_pretraining_frequency": True,
            "training_questions": len(training), "training_unique_images": len({r["image_id"] for r in training}),
            "distinct_normalized_answers": len(frequencies), "total_unique_answer_qa_occurrences": sum(frequencies.values())},
        "sft_target_frequency_definition": {
            "training_split": "train_all", "interpretation": "primary actual one-epoch supervised target QA exposure after answer normalization",
            "rule": "Use each row.target verified against the frozen raw-stripped modal reference rule (ties lexical); normalize that one target; increment once per training QA.",
            "unit": "training QA whose actual supervised target normalizes to this answer", "target_source": str(TRAINING_TARGET_SOURCE.resolve()),
            "target_source_sha256": sha_file(TRAINING_TARGET_SOURCE), "training_questions": len(training),
            "distinct_normalized_targets": len(target_frequencies), "total_target_qa_occurrences": sum(target_frequencies.values()),
            "normalized_empty_target_rule": "Retain the empty normalized key for real nonempty supervised targets so every QA contributes exactly once.",
            "normalized_empty_target_qa_count": target_frequencies[""],
            "not_pretraining_frequency": True},
        "dev_representative_rule": "Normalize every dev reference; discard empty strings; choose highest annotator count; break ties by Unicode lexicographic order. If all normalize empty, representative is empty with explicit flag; auxiliary reference-coverage frequency is zero.",
        "frequency_bucket_rules": {"unseen_0": "frequency == 0", "rare_1_5": "1 <= frequency <= 5", "medium_6_20": "6 <= frequency <= 20", "frequent_gt20": "frequency > 20"},
        "length_bucket_rules": {"short_le4": "len(normalized modal reference) <= 4", "medium_5_10": "5 <= len <= 10", "long_gt10": "len > 10"},
        "numeric_proxy_rule": {"fullmatch_regex": NUMERIC_PROXY, "interpretation": "A text-form proxy after answer normalization, not a semantic task/OCR label. Time-like strings normalized to spaces may fall in other_text."},
        "dev_questions": len(dev), "dev_unique_images": len({r["image_id"] for r in dev}),
        "all_train_answer_frequencies": dict(sorted(frequencies.items())), "all_train_sft_target_frequencies": dict(sorted(target_frequencies.items())),
        "dev_target_rule": "Use dev row.target verified against the original raw-stripped modal target; normalize it and look up actual train_all target exposure separately from reference coverage.",
        "dev_features": records, "slice_counts": slice_counts,
        "limitations": ["Unseen means absent from train_all reference answers, not unseen in pretraining.",
            "Reference-derived slices describe saved evaluation outcomes; reference text is never an inference input.",
            "Question difficulty, string length and annotation ambiguity may confound answer-frequency associations.",
            "The frequency definition differs from earlier train_10k single-target slices; values must not be merged as the same statistic."]}
    lock["slice_lock_sha256"] = content_sha(lock)
    path = output_root / "SLICE_LOCK.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != lock:
            raise ValueError("pre-existing slice lock differs; refusing implicit replacement")
    else:
        write_json(path, lock)
    return lock


def distribution(values):
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return {"n": len(array), "min": float(array.min()), "mean": float(array.mean()),
            "median": float(np.median(array)), "p05": float(np.quantile(array, .05)),
            "p90": float(np.quantile(array, .90)), "p95": float(np.quantile(array, .95)), "max": float(array.max())}


def write_csv(path, rows):
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()} for row in rows)


def consensus_from_feature(prediction, feature, normalize):
    counts = feature["normalized_reference_counts"]
    normalized_prediction = normalize(prediction)
    matches = counts.get(normalized_prediction, 0) if normalized_prediction else 10 - sum(counts.values())
    return (matches * min(max(matches - 1, 0) / 3, 1) + (10 - matches) * min(matches / 3, 1)) / 10


def read_cells(probe_root, lock):
    """Snapshot append-only rows; incomplete cells never become final results."""
    features = lock["dev_features"]
    normalize = answer_normalizer()
    cells = {}; provenance = {}; progress = []
    for role in ROLES:
        for budget in BUDGETS:
            directory = probe_root / "runs" / role / f"budget_{budget}"
            prediction_file = directory / "predictions.jsonl"
            evaluation_file = directory / "EVALUATION.json"
            data = prediction_file.read_bytes() if prediction_file.exists() else b""
            # Reading during an append may expose a non-terminated tail: ignore it
            # without modifying the worker-owned file.
            pieces = data.splitlines(keepends=True)
            tail_ignored = bool(pieces and not pieces[-1].endswith(b"\n"))
            if tail_ignored:
                pieces = pieces[:-1]
            rows = [json.loads(line) for line in pieces]
            if len(rows) > len(features):
                raise ValueError("too many probe predictions")
            for i, row in enumerate(rows):
                feature = features[i]
                for key in ("question_id", "image_id", "question_sha256", "image_sha256"):
                    if str(row[key]) != str(feature[key]):
                        raise ValueError(f"probe does not follow locked dev identity/order: {role}/{budget}/{i}/{key}")
                if row["visual_token_budget"] != budget or row["input_truncated"]:
                    raise ValueError("budget mismatch or truncated input")
                grid = row["image_grid_thw"]
                visual_tokens = math.prod(grid) // 4
                if math.prod(grid) % 4 or row["visual_tokens"] != visual_tokens or row["counted_image_tokens"] != visual_tokens:
                    raise ValueError("visual token/grid accounting mismatch")
                if visual_tokens > budget or row["resized_width"] != grid[2] * 14 or row["resized_height"] != grid[1] * 14:
                    raise ValueError("visual budget or resized dimension mismatch")
                expected_score = consensus_from_feature(row["prediction"], feature, normalize)
                if not math.isfinite(row["score"]) or not math.isclose(row["score"], expected_score, abs_tol=1e-10):
                    raise ValueError("new probe consensus score mismatch")
                if row["input_exceeds_training_768"] != (row["input_tokens"] > 768):
                    raise ValueError("training-length exposure flag mismatch")
                if row["padded_input_tokens"] < row["input_tokens"]:
                    raise ValueError("invalid padded input accounting")
            evaluation = json.loads(evaluation_file.read_text()) if evaluation_file.exists() else None
            complete = evaluation is not None
            if complete:
                if tail_ignored or len(rows) != len(features) or evaluation["n"] != len(features):
                    raise ValueError("complete evaluation with incomplete predictions")
                if evaluation["predictions_sha256"] != hashlib.sha256(data).hexdigest():
                    raise ValueError("completed probe predictions hash mismatch")
                if evaluation["role"] != role or evaluation["split"] != "dev" or evaluation["visual_token_budget"] != budget:
                    raise ValueError("completed cell identity mismatch")
                score = 100 * statistics.mean(row["score"] for row in rows)
                if not math.isclose(score, evaluation["score_percent"], abs_tol=1e-10):
                    raise ValueError("completed cell metric mismatch")
                if evaluation["precision"] != "fp32" or evaluation["quantization"] != "nf4_language_only":
                    raise ValueError("unexpected numerical protocol")
                if evaluation["tf32_matmul"] or evaluation["tf32_cudnn"]:
                    raise ValueError("unexpected TF32 execution")
            cells[(role, budget)] = {"rows": rows, "complete": complete, "evaluation": evaluation}
            provenance[f"{role}:{budget}"] = {"prediction_file": str(prediction_file),
                "prediction_file_snapshot_sha256": hashlib.sha256(data).hexdigest(), "evaluation_file": str(evaluation_file),
                "evaluation_sha256": sha_file(evaluation_file) if complete else None}
            progress.append({"role": role, "visual_token_budget": budget, "complete": complete,
                "validated_saved_questions": len(rows), "expected_questions": len(features), "incomplete_tail_ignored": tail_ignored})
    return cells, progress, provenance


def cell_summary(role, budget, cell):
    rows = cell["rows"]
    by_batch = defaultdict(list)
    for row in rows:
        by_batch[(row["session_id"], row["batch_start"])].append(row)
    batches = []
    duplicated_fields = ("batch_size", "batch_generation_seconds", "batch_preprocessing_seconds", "batch_h2d_seconds",
                         "batch_peak_allocated_gib", "batch_peak_reserved_gib", "timing_warmup")
    for key, members in by_batch.items():
        head = members[0]
        if len(members) != head["batch_size"]:
            raise ValueError(f"completed cell contains partial batch: {role}/{budget}/{key}")
        for row in members[1:]:
            if any(row[name] != head[name] for name in duplicated_fields):
                raise ValueError("repeated batch timing fields disagree")
        batches.append(head)
    timed = [row for row in batches if not row["timing_warmup"]]
    timed_questions = sum(row["batch_size"] for row in timed)
    generation_seconds = sum(row["batch_generation_seconds"] for row in timed)
    if timed_questions == 0 or generation_seconds <= 0:
        raise ValueError("no post-warmup measurement")
    value = {"role": role, "visual_token_budget": budget, "score_percent": 100 * statistics.mean(r["score"] for r in rows),
        "questions": len(rows), "unique_images": len({r["image_id"] for r in rows}),
        "visual_tokens": distribution([r["visual_tokens"] for r in rows]), "input_tokens": distribution([r["input_tokens"] for r in rows]),
        "padded_input_tokens": distribution([r["padded_input_tokens"] for r in rows]),
        "generated_nonpad_tokens_including_eos": distribution([r["generated_nonpad_tokens"] for r in rows]),
        "input_exceeds_training_768_questions": sum(r["input_exceeds_training_768"] for r in rows),
        "input_exceeds_training_768_percent": 100 * sum(r["input_exceeds_training_768"] for r in rows) / len(rows),
        "sessions": sorted({r["session_id"] for r in rows}), "all_batches": len(batches),
        "excluded_session_warmup_batches": len(batches) - len(timed), "timed_batches": len(timed), "timed_questions": timed_questions,
        "model_generate_seconds_excluding_warmup": generation_seconds,
        "pure_generation_questions_per_second": timed_questions / generation_seconds,
        "mean_amortized_generation_seconds_per_question": generation_seconds / timed_questions,
        "batch_generation_seconds_excluding_warmup": distribution([r["batch_generation_seconds"] for r in timed]),
        "batch_amortized_generation_seconds_per_question": distribution([r["batch_generation_seconds"] / r["batch_size"] for r in timed]),
        "all_preprocessing_seconds": sum(r["batch_preprocessing_seconds"] for r in batches),
        "all_h2d_seconds": sum(r["batch_h2d_seconds"] for r in batches),
        "peak_allocated_gib": max(r["batch_peak_allocated_gib"] for r in batches),
        "peak_reserved_gib": max(r["batch_peak_reserved_gib"] for r in batches),
        "timing_scope": "Each (session_id,batch_start) counted once; every session first batch excluded from generation timing. Batch amortization is not independent single-request latency; no model-loading or historical-cache timing included."}
    resources = cell["evaluation"]["resources"]
    if not math.isclose(value["pure_generation_questions_per_second"], resources["pure_generation_questions_per_second"], rel_tol=1e-9):
        raise ValueError("analysis timing aggregate differs from worker")
    return value


def bootstrap_difference(values, selected_features, replicates, seed):
    images = defaultdict(lambda: [0.0, 0])
    for value, feature in zip(values, selected_features):
        images[feature["image_id"]][0] += float(value)
        images[feature["image_id"]][1] += 1
    grouped = np.asarray(list(images.values()), dtype=float)
    rng = np.random.default_rng(seed)
    estimates = []
    for start in range(0, replicates, 64):
        indices = rng.integers(0, len(grouped), (min(64, replicates - start), len(grouped)))
        sampled = grouped[indices].sum(axis=1)
        estimates.extend((100 * sampled[:, 0] / sampled[:, 1]).tolist())
    return {"delta_pp": float(100 * grouped[:, 0].sum() / grouped[:, 1].sum()),
        "image_cluster_bootstrap_95_pp": np.quantile(estimates, [.025, .975]).tolist(),
        "questions": int(grouped[:, 1].sum()), "unique_image_clusters": len(grouped), "bootstrap_replicates": replicates,
        "bootstrap_seed": seed, "small_slice": len(selected_features) < 50 or len(grouped) < 30,
        "scope": "Exploratory pointwise 95% percentile interval conditional on these fixed weights, including entire image question clusters. No training-seed uncertainty or multiple-comparison correction."}


def plot_results(analysis_root, summaries, cells, paired):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties, fontManager
    fontManager.addfont(str(FONT_PATH))
    font_name = FontProperties(fname=str(FONT_PATH)).get_name()
    colors = {"base": "#0072B2", "adapted": "#007A5E"}
    markers = {"base": "o", "adapted": "s"}
    labels = {"base": "原NF4底座", "adapted": "固定q/v rank8 · seed42"}
    lookup = {(r["role"], r["visual_token_budget"]): r for r in summaries}
    files = []
    with plt.rc_context({"font.family": font_name, "font.size": 10, "svg.fonttype": "path", "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white", "savefig.facecolor": "white"}):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
        for role in ROLES:
            points = [lookup[(role, b)] for b in BUDGETS]
            x = [r["visual_tokens"]["mean"] for r in points]
            axes[0, 0].plot(x, [r["score_percent"] for r in points], marker=markers[role], color=colors[role], label=labels[role])
            for r in points:
                axes[0, 0].annotate(f'≤{r["visual_token_budget"]}: {r["score_percent"]:.2f}',
                    (r["visual_tokens"]["mean"], r["score_percent"]), xytext=(0, 8 if role == "adapted" else -15),
                    textcoords="offset points", ha="center", fontsize=8)
            median = [r["batch_generation_seconds_excluding_warmup"]["median"] for r in points]
            low = [m - r["batch_generation_seconds_excluding_warmup"]["p05"] for m, r in zip(median, points)]
            high = [r["batch_generation_seconds_excluding_warmup"]["p95"] - m for m, r in zip(median, points)]
            axes[1, 0].errorbar(x, median, yerr=[low, high], marker=markers[role], color=colors[role], capsize=4, label=labels[role])
            axes[1, 1].plot(x, [r["peak_allocated_gib"] for r in points], marker=markers[role], color=colors[role], label=labels[role] + " allocated")
            axes[1, 1].plot(x, [r["peak_reserved_gib"] for r in points], marker=markers[role], color=colors[role], linestyle="--", alpha=.8, label=labels[role] + " reserved")
        axes[0, 0].set_title("A 质量：使用实际平均视觉 token", loc="left"); axes[0, 0].set_ylabel("内部 dev VQA得分（%）"); axes[0, 0].legend(fontsize=8)
        for ax in (axes[0, 0], axes[1, 0], axes[1, 1]):
            ax.set_xlabel("实际视觉 token / QA（均值）"); ax.grid(axis="y", color="#DDDDDD", linewidth=.6)
        token_arrays = [[r["visual_tokens"] for r in cells[("base", b)]["rows"]] for b in BUDGETS]
        axes[0, 1].boxplot(token_arrays, labels=[f"≤{b}" for b in BUDGETS], whis=(0, 100), showfliers=True,
                           medianprops={"color": "#222222"}, boxprops={"color": "#444444"})
        rng = np.random.default_rng(20261011)
        for i, values in enumerate(token_arrays, start=1):
            axes[0, 1].scatter(i + rng.uniform(-.08, .08, len(values)), values, s=3, color="#666666", alpha=.12)
        axes[0, 1].set_title("B 配置上限与实际 token 分布", loc="left"); axes[0, 1].set_xlabel("最大视觉 token 预算"); axes[0, 1].set_ylabel("实际视觉 token / QA")
        axes[0, 1].set_ylim(bottom=0); axes[0, 1].grid(axis="y", color="#DDDDDD", linewidth=.6)
        axes[1, 0].set_title("C 批生成时间：中位数与P05–P95", loc="left"); axes[1, 0].set_ylabel("generation秒 / batch（通常4题）"); axes[1, 0].legend(fontsize=8)
        axes[1, 1].set_title("D 每单元峰值显存", loc="left"); axes[1, 1].set_ylabel("GiB"); axes[1, 1].set_ylim(bottom=0); axes[1, 1].legend(fontsize=7)
        fig.suptitle("视觉预算干预：固定权重，检查质量与计算代价", fontsize=16, weight="bold")
        fig.supxlabel("单个固定adapter种子；实际token相同仍可能因答案长度产生不同生成耗时。计时排除每session首批。\n"
                      "质量曲线不表示跨训练种子置信区间；B箱须为全部范围，灰点为每QA，重复图像问题仍保留。", fontsize=9)
        for ext in ("png", "svg"):
            path = analysis_root / f"quality_cost.{ext}"; fig.savefig(path, dpi=180); files.append(path)
        plt.close(fig)
        overall = [r for r in paired if r["dimension"] == "all"]
        fig, ax = plt.subplots(figsize=(11, 4.6), layout="constrained")
        for i, row in enumerate(overall):
            low, high = row["image_cluster_bootstrap_95_pp"]
            ax.errorbar(row["delta_pp"], i, xerr=[[row["delta_pp"] - low], [high - row["delta_pp"]]], fmt="D", color="#0072B2", capsize=5)
            ax.annotate(f'{row["delta_pp"]:+.3f} [{low:+.3f}, {high:+.3f}]', (high, i), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9)
        ax.axvline(0, color="#555555", linestyle="--"); ax.set_yticks(range(len(overall)), [r["contrast"] for r in overall]); ax.invert_yaxis()
        ax.set_xlabel("同图配对VQA差值（百分点）；95% image-cluster bootstrap区间")
        ax.set_title("预先固定的五个比较：预算收益及适配×预算交互", loc="left", weight="bold"); ax.grid(axis="x", color="#DDDDDD", linewidth=.6)
        xmin, xmax = ax.get_xlim(); ax.set_xlim(xmin, xmax + max((xmax - xmin) * .5, .5))
        fig.supxlabel("每个区间固定底座/adapter，不含训练种子不确定性；探索性、多重比较未校正，不作确认性显著结论。", fontsize=9)
        for ext in ("png", "svg"):
            path = analysis_root / f"paired_budget_deltas.{ext}"; fig.savefig(path, dpi=180); files.append(path)
        plt.close(fig)
    return [str(p) for p in files]


def make_cases(asset_root, probe_root, cells, lock):
    """Illustrations from true image pixels and recorded resize dimensions only."""
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties, fontManager
    fontManager.addfont(str(FONT_PATH))
    manifest_path = asset_root / "prepared/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    rows = {str(r["question_id"]): r for r in manifest["splits"]["dev"]}
    features = {r["question_id"]: r for r in lock["dev_features"]}
    lookups = {key: {str(r["question_id"]): r for r in value["rows"]} for key, value in cells.items()}
    cases = []
    for qid in CASE_IDS:
        available = {key: value[qid] for key, value in lookups.items() if qid in value}
        if not available:
            continue
        row = rows[qid]; feature = features[qid]
        source = manifest_path.parent / manifest["images"][row["image_id"]]["path"]
        if sha_file(source) != manifest["images"][row["image_id"]]["sha256"]:
            raise ValueError("teaching source image hash mismatch")
        original = Image.open(source).convert("RGB")
        directory = probe_root / "cases" / qid; directory.mkdir(parents=True, exist_ok=True)
        case = {"question_id": qid, "question": row["question"], "image_id": row["image_id"], "source_image": str(source),
            "source_image_sha256": sha_file(source), "feature": feature, "predictions": [], "resize_illustrations": [],
            "scope": "Three declared teaching examples, not representative error prevalence. BICUBIC resize to recorded processor dimensions illustrates resolution only; not attention, OCR boxes, or an exact preprocessed model tensor."}
        with plt.rc_context({"font.family": FontProperties(fname=str(FONT_PATH)).get_name(), "font.size": 10,
                             "figure.facecolor": "white", "savefig.facecolor": "white"}):
            fig = plt.figure(figsize=(14, 7.8), layout="constrained"); grid = fig.add_gridspec(2, 4, height_ratios=[1.1, 1])
            ax = fig.add_subplot(grid[0, 0]); ax.imshow(original, interpolation="nearest"); ax.set_title(f"原始图片\n{original.width}×{original.height}"); ax.axis("off")
            for column, budget in enumerate(BUDGETS, start=1):
                ax = fig.add_subplot(grid[0, column]); ax.axis("off")
                sample = available.get(("base", budget), available.get(("adapted", budget)))
                if sample is None:
                    ax.text(.5, .5, "该预算尚无此题记录", ha="center", va="center"); ax.set_title(f"预算≤{budget}")
                    continue
                size = (sample["resized_width"], sample["resized_height"])
                resized = original.resize(size, Image.Resampling.BICUBIC)
                illustration = directory / f"resize_illustration_budget_{budget}.png"; resized.save(illustration)
                case["resize_illustrations"].append({"budget": budget, "recorded_size": list(size), "actual_visual_tokens": sample["visual_tokens"], "file": str(illustration), "sha256": sha_file(illustration)})
                ax.imshow(resized, interpolation="nearest"); ax.set_title(f"预算≤{budget} · 实际{sample['visual_tokens']} token\n记录尺寸{size[0]}×{size[1]}")
            for column, label in enumerate(("参考与切片", "预算≤256", "预算≤512", "预算≤1024")):
                ax = fig.add_subplot(grid[1, column]); ax.axis("off")
                if column == 0:
                    body = f"规范化参考：{feature['modal_normalized_reference']}\n\n实际监督目标频次：{feature['train_all_sft_target_qa_frequency']}\n标签：{feature['sft_target_answer_frequency']}\n\n辅助参考覆盖频次：{feature['train_all_unique_reference_qa_frequency']}\n标签：{feature['reference_answer_frequency']}"
                else:
                    budget = BUDGETS[column - 1]; parts = []
                    for role in ROLES:
                        prediction = available.get((role, budget))
                        if prediction is None:
                            parts.append(f"{role}: 尚未完成")
                            continue
                        parts.append(f"{role} 预测：\n{textwrap.fill(prediction['prediction'], width=30)}\nVQA得分：{prediction['score']:.2f}\n输入：{prediction['input_tokens']} token")
                        case["predictions"].append({"role": role, "budget": budget, "prediction": prediction["prediction"], "score": prediction["score"], "visual_tokens": prediction["visual_tokens"], "input_tokens": prediction["input_tokens"]})
                    body = "\n\n".join(parts)
                ax.text(.02, .98, label + "\n\n" + body, ha="left", va="top", transform=ax.transAxes, fontsize=10)
            fig.suptitle(f"真实案例 {qid}: {row['question']}", fontsize=15, weight="bold")
            fig.supxlabel("同图按记录尺寸BICUBIC缩放，统一展示面积只解释细节信息量；不是注意力图或processor像素张量重建。\n"
                          "答案标签用于事后评分和切片，模型生成仅接收图片与问题。单例不证明机制。", fontsize=9)
            board = directory / "teaching_board.png"; fig.savefig(board, dpi=180); plt.close(fig)
        case["teaching_board"] = str(board); case["teaching_board_sha256"] = sha_file(board)
        write_json(directory / "CASE.json", case); cases.append(case)
    return cases


def analyze_probe(asset_root, probe_root, lock, replicates, bootstrap_seed):
    analysis_root = probe_root / "analysis"; analysis_root.mkdir(parents=True, exist_ok=True)
    cells, progress, provenance = read_cells(probe_root, lock)
    complete = all(value["complete"] for value in cells.values())
    if not complete:
        cases = make_cases(asset_root, probe_root, cells, lock)
        value = {"schema_version": "rarevlm.budget_probe.analysis.v1", "complete": False,
                 "status": "partial_generation", "completed_cells": sum(value["complete"] for value in cells.values()),
                 "expected_cells": 6, "slice_lock_sha256": lock["slice_lock_sha256"], "cells": progress,
                 "cases": [{"question_id": c["question_id"], "teaching_board": c["teaching_board"]} for c in cases],
                 "scope": "Progress only. No pooled final comparison, hypothesis selection, or complete-results claim from incomplete cells."}
        write_json(probe_root / "ANALYSIS_PROGRESS.json", value); write_json(analysis_root / "ANALYSIS.json", value)
        return value
    for budget in BUDGETS:
        for base, adapted in zip(cells[("base", budget)]["rows"], cells[("adapted", budget)]["rows"]):
            for key in ("image_grid_thw", "visual_tokens", "input_tokens", "resized_width", "resized_height"):
                if base[key] != adapted[key]:
                    raise ValueError("base/adapter see different encoded inputs at same budget")
    summaries = [cell_summary(role, budget, cells[(role, budget)]) for role in ROLES for budget in BUDGETS]
    features = lock["dev_features"]
    slice_members = [("all", "all", list(range(len(features))))]
    for dimension in SLICE_DIMENSIONS:
        for label in sorted({r[dimension] for r in features}):
            slice_members.append((dimension, label, [i for i, row in enumerate(features) if row[dimension] == label]))
    scores = {key: np.asarray([r["score"] for r in value["rows"]], dtype=float) for key, value in cells.items()}
    slices = []; paired = []
    for dimension, label, indices in slice_members:
        chosen_features = [features[i] for i in indices]
        for role in ROLES:
            for budget in BUDGETS:
                slices.append({"role": role, "visual_token_budget": budget, "dimension": dimension, "slice": label,
                    "score_percent": float(100 * scores[(role, budget)][indices].mean()), "questions": len(indices),
                    "unique_images": len({row["image_id"] for row in chosen_features}),
                    "small_slice": len(indices) < 50 or len({row["image_id"] for row in chosen_features}) < 30,
                    "scope": "Exploratory descriptor. Actual SFT target exposure and auxiliary reference coverage are separate frequency definitions."})
        if dimension == "all":
            for contrast, coefficients in CONTRASTS:
                difference = sum(coefficient * scores[key][indices] for key, coefficient in coefficients.items())
                paired.append({"contrast": contrast, "dimension": dimension, "slice": label,
                    "prespecified_main_comparison": True,
                    **bootstrap_difference(difference, chosen_features, replicates, bootstrap_seed)})
    figures = plot_results(analysis_root, summaries, cells, paired)
    cases = make_cases(asset_root, probe_root, cells, lock)
    write_csv(analysis_root / "CELL_SUMMARY.csv", summaries); write_csv(analysis_root / "SLICE_SCORES.csv", slices)
    write_csv(analysis_root / "PAIRED_CONTRASTS.csv", paired)
    config_path = REPO / "configs/vlm_budget_probe_v1.json"
    value = {"schema_version": "rarevlm.budget_probe.analysis.v1", "complete": True, "status": "analyzed",
        "completed_cells": 6, "expected_cells": 6, "split": "internal_dev", "questions": len(features),
        "unique_images": len({r["image_id"] for r in features}), "fixed_adapter_seed": 42,
        "adapter_choice": json.loads(config_path.read_text())["adapter_selection"], "slice_lock_sha256": lock["slice_lock_sha256"],
        "probe_config_sha256": sha_file(config_path), "analysis_source_sha256": sha_file(Path(__file__)),
        "frequency_definition": lock["frequency_definition"], "sft_target_frequency_definition": lock["sft_target_frequency_definition"],
        "cell_summaries": summaries, "slice_scores": slices, "paired_contrasts": paired, "source_provenance": provenance,
        "all_saved_consensus_scores_independently_recomputed": True, "base_adapted_input_encoding_identical_at_same_budget": True,
        "figures": figures, "teaching_cases": [{"question_id": c["question_id"], "teaching_board": c["teaching_board"], "source_image": c["source_image"]} for c in cases],
        "limitations": ["Single fixed adapter seed; intervals do not measure training-seed uncertainty.",
            "Internal dev only; no official validation metrics are used to choose a budget or checkpoint.",
            "Budget increases can exceed original 256 visual tokens and 768 training input length; the latter proportion is explicitly reported.",
            "Five planned comparisons plus exploratory slices are pointwise, uncorrected; no confirmatory significance or general mechanism proof.",
            "Latency is batch generation and batch-amortized cost on concurrent independent A100 workers, not isolated single-request service latency.",
            "A resize illustration shows pixel detail, not learned attention or an exact processor tensor.",
            "Generation token counts include EOS when nonpad; the adapter may change output length and hence runtime."],
        "scope": "Fixed-weight visual-information intervention, image-cluster paired evaluation, quality/cost description; no retraining, official-val model selection, or backbone-ranking claim.",
        "scientific_visualization_reference": "Kassis et al. (2026), Scientific Agent Skills, https://arxiv.org/abs/2609.00065; current record checked 2026-10-11."}
    write_json(analysis_root / "ANALYSIS.json", value)
    manifest = {"audience": "project learning", "uncertainty": "paired 2000-replicate image-cluster percentile intervals on fixed predictions; latency bars are empirical P05/P95 batches, not CI",
        "data_tables": ["CELL_SUMMARY.csv", "SLICE_SCORES.csv", "PAIRED_CONTRASTS.csv"],
        "image_processing": "Only teaching images BICUBIC-resized to recorded dimensions; no denoising/contrast edits/crops/attention overlays. Original source files preserved.",
        "font": str(FONT_PATH), "source_provenance": provenance,
        "output_sha256": {str(Path(path).name): sha_file(Path(path)) for path in figures},
        "alt_text": "Quality and computation plots compare the unadapted NF4 base and a fixed q/v rank8 seed42 adapter across three visual budgets using actual token counts. Separate paired intervals describe budget gains and adaptation-by-budget interaction. Three real images illustrate recorded resize dimensions alongside authentic predictions."}
    write_json(analysis_root / "FIGURE_PROVENANCE.json", manifest)
    lines = ["视觉预算独立探索实验：六单元完整，固定adapter seed42，仅内部dev。", "",
             "实际SFT目标频次与辅助参考覆盖频次分别报告；所有缓存分数、token/grid及批计时聚合已核对。", ""]
    for row in summaries:
        lines.append(f"{row['role']} 预算≤{row['visual_token_budget']}: VQA {row['score_percent']:.3f}%；实际token均值 {row['visual_tokens']['mean']:.1f}；纯生成 {row['pure_generation_questions_per_second']:.3f}题/秒；超768输入 {row['input_exceeds_training_768_percent']:.2f}%")
    lines += ["", "预先固定的主比较（条件image-cluster95%区间，非训练种子CI；多重比较未校正）："]
    for row in paired:
        if row["dimension"] == "all":
            lines.append(f"{row['contrast']}: {row['delta_pp']:+.3f} pp {row['image_cluster_bootstrap_95_pp']}")
    lines += ["", "可用结果提出下一轮可验证的输入/数据研究假设，不能从单种子dev直接宣称普遍机制或更换底座收益。",
              "绘图参考：Kassis et al. (2026), https://arxiv.org/abs/2609.00065（已核验当前记录）。"]
    (analysis_root / "FINDINGS_ZH.txt").write_text("\n".join(lines) + "\n")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--probe-root", "--output-root", dest="output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261011)
    args = parser.parse_args()
    if not args.output_root.resolve().is_relative_to(WORKSPACE):
        raise ValueError("probe output outside persistent workspace")
    lock = prepare_slice_lock(args.asset_root, args.output_root)
    if args.prepare_only:
        print(json.dumps({"status": "slice_lock_prepared", "path": str(args.output_root / "SLICE_LOCK.json"),
            "slice_lock_sha256": lock["slice_lock_sha256"], "train_all_questions": lock["frequency_definition"]["training_questions"],
            "dev_questions": lock["dev_questions"], "slice_counts": lock["slice_counts"]}, ensure_ascii=False))
        return
    if args.replicates < 100:
        raise ValueError("insufficient bootstrap replicates")
    result = analyze_probe(args.asset_root, args.output_root, lock, args.replicates, args.bootstrap_seed)
    print(json.dumps({"complete": result["complete"], "status": result["status"], "completed_cells": result["completed_cells"],
                      "analysis_file": str(args.output_root / "analysis/ANALYSIS.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
