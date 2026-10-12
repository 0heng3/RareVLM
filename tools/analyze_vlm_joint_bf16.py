"""CPU-only analysis of original-weight BF16 at visual budgets 512 and 1024.

Four new cells retain two historical BF16/256 cells as labeled references.
Cross-run quality is paired by frozen dev questions. Historical speed is an
observed reference, never an isolated same-GPU speedup or a dtype effect.
"""
from __future__ import annotations

import argparse
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
REPO = Path(__file__).resolve().parents[1]
WORKSPACE = Path("/opt/data/private")
OUTPUTS = WORKSPACE / "longtail_vlm/rarevlm_pilot/outputs"
DEFAULT_ASSET = OUTPUTS / "vlm_deep_v2"
DEFAULT_PROBE = OUTPUTS / "vlm_joint_bf16_v1"
DEFAULT_REFERENCE = OUTPUTS / "vlm_precision_bridge_v1"
DEFAULT_NF4_BUDGET = OUTPUTS / "vlm_budget_probe_v1"
HELPER_PATH = REPO / "tools/analyze_vlm_precision_bridge.py"
_spec = importlib.util.spec_from_file_location("joint_bf16_cpu_helpers", HELPER_PATH)
helper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helper)
ROLES = ("base", "adapted")
BUDGETS = (256, 512, 1024)
NEW_BUDGETS = (512, 1024)
COMPARISONS = tuple((role, 256, after) for role in ROLES for after in NEW_BUDGETS)
COMMON_FIELDS = ("packages", "model_revision", "model_config_sha256", "model_provenance_sha256",
                 "min_pixels", "max_new_tokens", "do_sample", "attention_implementation",
                 "head_last_position_only", "autocast", "input", "input_truncation", "batch_size",
                 "tf32_matmul", "tf32_cudnn", "precision", "vision_precision", "quantization")
IDENTITY_FIELDS = ("question_id", "image_id", "question_sha256", "image_sha256")
ENCODING_FIELDS = helper.INPUT_FIELDS


def load_json(path):
    return json.loads(path.read_text())


def gpu_uuid(hardware):
    """New workers resolve physical UUID even when CUDA_VISIBLE_DEVICES is 0/1."""
    explicit = hardware.get("uuid")
    visible = hardware.get("visible_devices", "")
    return explicit or (visible if visible.startswith("GPU-") and "," not in visible else None)


def contention_monitor(root, historical=False):
    monitor = helper.resource_contention(root)
    for value in monitor["roles"].values():
        value["scope"] = ("Tag this role's " + ("historical BF16/256 reference" if historical else "two new BF16 high-budget cells")
            + " if any co-resident same-GPU compute process was sampled. Zero sampled events is not proof of exclusivity; no processes were preempted.")
    return monitor


def validate_lock(lock, role, budget, manifest_path, manifest):
    expected = {"role": role, "split": "dev", "task": "textvqa", "precision": "bf16",
                "vision_precision": "bf16", "quantization": "original", "visual_token_budget": budget,
                "numerical_path": "original_bf16", "batch_size": 4, "max_new_tokens": 32,
                "min_pixels": 64 * 28 * 28, "max_pixels": budget * 28 * 28,
                "do_sample": False, "tf32_matmul": False, "tf32_cudnn": False,
                "attention_implementation": "sdpa", "input_truncation": False,
                "manifest_sha256": manifest["manifest_sha256"],
                "manifest_file_sha256": helper.sha_file(manifest_path)}
    for key, value in expected.items():
        if lock.get(key) != value:
            raise ValueError(f"unexpected {role}/{budget} lock: {key}={lock.get(key)!r}")
    if role == "base" and lock.get("adapter_files_sha256") is not None:
        raise ValueError("base role unexpectedly contains an adapter")
    if role == "adapted":
        binding = lock.get("training_binding") or {}
        if binding.get("method") != "qv_r8" or binding.get("seed") != 42 or not lock.get("adapter_files_sha256"):
            raise ValueError("adapted checkpoint differs from predeclared full qv_r8 seed42")
    dtypes = lock.get("actual_parameter_dtypes") or {}
    if dtypes.get("params4bit_tensors") != 0 or set(dtypes.get("parameter_tensors_by_dtype", {})) != {"bfloat16"}:
        raise ValueError("actual loaded parameters are not unquantized BF16")
    if role == "adapted" and set(dtypes.get("lora_tensors_by_dtype", {})) != {"bfloat16"}:
        raise ValueError("adapted LoRA parameters are not BF16")
    gpu = gpu_uuid(lock.get("hardware") or {})
    if not gpu or not gpu.startswith("GPU-"):
        raise ValueError("expected an independently admitted single GPU UUID")
    if budget in NEW_BUDGETS:
        config_path = REPO / "configs/vlm_joint_bf16_v1.json"
        if lock.get("joint_config_sha256") != helper.sha_file(config_path):
            raise ValueError("new cell joint configuration binding differs")


def read_cell(directory, role, budget, manifest_path, manifest, references, origin):
    prediction_path = directory / "predictions.jsonl"
    evaluation_path = directory / "EVALUATION.json"
    lock_path = directory / "INPUT_LOCK.json"
    lock = load_json(lock_path) if lock_path.exists() else None
    if lock is not None:
        validate_lock(lock, role, budget, manifest_path, manifest)
    data = prediction_path.read_bytes() if prediction_path.exists() else b""
    lines = data.splitlines(keepends=True)
    tail_ignored = bool(lines and not lines[-1].endswith(b"\n"))
    if tail_ignored:
        lines = lines[:-1]
    rows = [json.loads(line) for line in lines]
    if len(rows) > len(references) or (rows and lock is None):
        raise ValueError(f"invalid prediction count or absent lock: {role}/{budget}")
    normalizer = helper.answer_normalizer()
    for index, row in enumerate(rows):
        ref = references[index]
        expected = {"question_id": str(ref["question_id"]), "image_id": str(ref["image_id"]),
                    "question_sha256": helper.content_sha(ref["question"]),
                    "image_sha256": manifest["images"][ref["image_id"]]["sha256"]}
        if any(str(row[key]) != value for key, value in expected.items()):
            raise ValueError(f"dev identity/order/hash differs at {role}/{budget}/{index}")
        if (row["numerical_path"], row["precision"], row["quantization"]) != ("original_bf16", "bf16", "original"):
            raise ValueError("prediction numerical-path identity differs")
        if row["input_lock_sha256"] != helper.content_sha(lock):
            raise ValueError("prediction input lock hash differs")
        if row["visual_token_budget"] != budget or row["input_truncated"]:
            raise ValueError("visual budget changed or input truncated")
        grid = row["image_grid_thw"]
        patches = math.prod(grid)
        actual_tokens = patches // 4
        if len(grid) != 3 or grid[0] != 1 or patches % 4 or not 0 < actual_tokens <= budget:
            raise ValueError("unexpected image grid/visual budget")
        if row["visual_tokens"] != actual_tokens or row["counted_image_tokens"] != actual_tokens:
            raise ValueError("actual visual token accounting differs")
        if row["resized_width"] != grid[2] * 14 or row["resized_height"] != grid[1] * 14:
            raise ValueError("recorded image resize/grid differ")
        if row["padded_input_tokens"] < row["input_tokens"] or bool(row["input_exceeds_training_768"]) != (row["input_tokens"] > 768):
            raise ValueError("input padding/training-limit accounting differs")
        if row["batch_start"] != index // 4 * 4 or row["batch_size"] != min(4, len(references) - index // 4 * 4):
            raise ValueError("fixed question batch groups differ")
        score = helper.score_prediction(row["prediction"], ref["answers"], normalizer)
        if not math.isfinite(row["score"]) or not math.isclose(row["score"], score, abs_tol=1e-12, rel_tol=0):
            raise ValueError("stored VQA consensus does not independently reproduce")
    evaluation = load_json(evaluation_path) if evaluation_path.exists() else None
    complete = evaluation is not None
    if complete:
        if lock is None or tail_ignored or len(rows) != len(references) or evaluation["n"] != len(references):
            raise ValueError("evaluation claims completion with incomplete predictions")
        validate_lock(evaluation, role, budget, manifest_path, manifest)
        if evaluation.get("status") != "completed" or evaluation["input_lock_sha256"] != helper.content_sha(lock):
            raise ValueError("completion status or input lock hash differs")
        if evaluation["predictions_sha256"] != hashlib.sha256(data).hexdigest():
            raise ValueError("completed predictions file hash differs")
        if not math.isclose(evaluation["score_percent"], 100 * statistics.mean(row["score"] for row in rows), abs_tol=1e-10):
            raise ValueError("completed evaluation mean differs")
    cell = {"rows": rows, "lock": lock, "complete": complete, "evaluation": evaluation, "origin": origin}
    progress = {"role": role, "visual_token_budget": budget, "origin": origin, "complete": complete,
                "validated_saved_questions": len(rows), "expected_questions": len(references),
                "incomplete_tail_ignored": tail_ignored}
    provenance = {"predictions_file": str(prediction_path), "predictions_snapshot_sha256": hashlib.sha256(data).hexdigest(),
                  "input_lock_file": str(lock_path), "input_lock_sha256": helper.content_sha(lock) if lock else None,
                  "evaluation_file": str(evaluation_path),
                  "evaluation_file_sha256": helper.sha_file(evaluation_path) if complete else None,
                  "origin": origin, "completed_at": evaluation.get("completed_at") if evaluation else None,
                  "hardware": lock.get("hardware") if lock else None}
    return cell, progress, provenance


def check_shared_contracts(cells):
    first = cells[("base", 256)]["lock"]
    for (role, budget), cell in cells.items():
        lock = cell["lock"]
        for field in COMMON_FIELDS:
            if lock.get(field) != first.get(field):
                raise ValueError(f"six-point scientific common contract differs: {role}/{budget}/{field}")
        # A new wrapper adds its own hashes. Common underlying implementations
        # must still match; do not require identical source dictionary keys.
        for path in set(first["source_sha256"]) & set(lock["source_sha256"]):
            if lock["source_sha256"][path] != first["source_sha256"][path]:
                raise ValueError(f"historical/current common source differs: {path}")
        within = cells[(role, 256)]["lock"]
        for field in ("adapter_files_sha256", "training_binding", "adapter_autocast_dtype", "explicit_lora_bf16_conversion", "actual_parameter_dtypes"):
            if lock.get(field) != within.get(field):
                raise ValueError(f"within-role checkpoint/numerical contract differs: {role}/{budget}/{field}")
    for role in ROLES:
        if cells[(role, 512)]["lock"]["hardware"] != cells[(role, 1024)]["lock"]["hardware"]:
            raise ValueError(f"new within-role budgets must use one admitted GPU: {role}")
        for field in ("source_sha256", "joint_config_sha256"):
            if cells[(role, 512)]["lock"].get(field) != cells[(role, 1024)]["lock"].get(field):
                raise ValueError(f"new within-role source/config differs: {role}/{field}")
    for budget in BUDGETS:
        base, adapted = cells[("base", budget)]["rows"], cells[("adapted", budget)]["rows"]
        for index, (before, after) in enumerate(zip(base, adapted)):
            if any(before[field] != after[field] for field in ENCODING_FIELDS):
                raise ValueError(f"same-budget base/adapted encoded input differs: {budget}/{index}")


def summarize(role, budget, cell, current_monitor, historical_monitor):
    summary, batches = helper.cell_summary(role, "original_bf16", cell)
    summary["visual_token_budget"] = budget
    summary["origin"] = cell["origin"]
    summary["completed_at"] = cell["evaluation"]["completed_at"]
    summary["gpu_uuid"] = gpu_uuid(summary["hardware"])
    summary["sessions"] = sorted({row["session_id"] for row in cell["rows"]})
    summary["input_exceeds_training_768_questions"] = sum(row["input_tokens"] > 768 for row in cell["rows"])
    summary["input_exceeds_training_768_percent"] = 100 * summary["input_exceeds_training_768_questions"] / len(cell["rows"])
    summary.update((historical_monitor if budget == 256 else current_monitor)["roles"][role])
    for row in batches:
        row.update(visual_token_budget=budget, origin=cell["origin"])
    return summary, batches


def paired_budget_contrast(role, before, after, cells, references, replicates, seed):
    aliases = {(role, str(before)): cells[(role, before)], (role, str(after)): cells[(role, after)]}
    summary, pairs, matrix = helper.paired_contrast(role, str(before), str(after), aliases, references, replicates, seed)
    summary["before_budget"] = before
    summary["after_budget"] = after
    summary["contrast"] = f"{role}:BF16 {after}-{before}"
    summary["comparison_scope"] = "Paired visual-budget sensitivity at original-weight BF16, with historical256 reference; fixed seed42 adapter only."
    for row in pairs + matrix:
        row.update(before_budget=before, after_budget=after)
    return summary, pairs, matrix


def historical_cost_tags(summaries):
    lookup = {(row["role"], row["visual_token_budget"]): row for row in summaries}
    tags = []
    for role in ROLES:
        history = lookup[(role, 256)]
        for budget in NEW_BUDGETS:
            target = lookup[(role, budget)]
            same_uuid = history["gpu_uuid"] == target["gpu_uuid"]
            tags.append({"role": role, "before_budget": 256, "after_budget": budget,
                "historical_gpu_uuid": history["gpu_uuid"], "new_gpu_uuid": target["gpu_uuid"],
                "same_gpu_uuid": same_uuid, "historical_completed_at": history["completed_at"], "new_completed_at": target["completed_at"],
                "historical_sessions": history["sessions"], "new_sessions": target["sessions"],
                "observed_generation_seconds_per_question_ratio": target["mean_amortized_generation_seconds_per_question"] / history["mean_amortized_generation_seconds_per_question"],
                "historical_timing_status": history["resource_timing_status"], "new_timing_status": target["resource_timing_status"],
                "causal_speedup_claim": False,
                "scope": "Different evaluation sessions/runs. GPU UUID and sampled contention are recorded; even same UUID does not control temporal load, kernels or output length. Descriptive observed cost ratio only."})
    return tags


def frequency_slices(nf4_root, cells, manifest, references):
    path = nf4_root / "SLICE_LOCK.json"
    if not path.exists():
        return [], {"available": False, "file": str(path)}
    lock = load_json(path)
    if helper.content_sha({k: v for k, v in lock.items() if k != "slice_lock_sha256"}) != lock["slice_lock_sha256"]:
        raise ValueError("historical slice lock content hash differs")
    if lock["manifest_sha256"] != manifest["manifest_sha256"] or len(lock["dev_features"]) != len(references):
        raise ValueError("historical SFT-frequency slice manifest differs")
    features = lock["dev_features"]
    for feature, ref in zip(features, references):
        if feature["question_id"] != str(ref["question_id"]) or feature["image_id"] != ref["image_id"]:
            raise ValueError("historical SFT-frequency question order differs")
    groups = defaultdict(list)
    for index, feature in enumerate(features):
        groups[feature["sft_target_answer_frequency"]].append(index)
    rows = []
    for role in ROLES:
        for budget in BUDGETS:
            for label, indices in sorted(groups.items()):
                images = {references[i]["image_id"] for i in indices}
                rows.append({"role": role, "visual_token_budget": budget, "dimension": "sft_target_answer_frequency",
                    "slice": label, "questions": len(indices), "unique_images": len(images),
                    "score_percent": 100 * statistics.mean(cells[(role, budget)]["rows"][i]["score"] for i in indices),
                    "small_slice": len(indices) < 50 or len(images) < 30,
                    "scope": "Exploratory descriptor of actual normalized supervised-target exposure in train_all. Unseen refers to this training split, not pretraining or semantic visual classes."})
    return rows, {"available": True, "file": str(path), "file_sha256": helper.sha_file(path),
                  "slice_lock_sha256": lock["slice_lock_sha256"],
                  "sft_target_frequency_definition": lock["sft_target_frequency_definition"]}


def cross_numeric_quality(nf4_root, summaries, manifest):
    path = nf4_root / "analysis/ANALYSIS.json"
    if not path.exists():
        return [], {"available": False, "file": str(path)}
    old = load_json(path)
    if not old.get("complete"):
        raise ValueError("historical NF4 budget analysis is not complete")
    rows = []
    lookup = {(row["role"], row["visual_token_budget"]): row for row in summaries}
    for point in old["cell_summaries"]:
        if point["visual_token_budget"] != 1024:
            continue
        role = point["role"]
        evaluation_path = nf4_root / "runs" / role / "budget_1024/EVALUATION.json"
        evaluation = load_json(evaluation_path)
        if evaluation["manifest_sha256"] != manifest["manifest_sha256"]:
            raise ValueError("historical NF4 quality manifest differs")
        bf16 = lookup[(role, 1024)]
        new = load_json(Path(bf16["evaluation_file"]))
        for field in ("model_revision", "model_config_sha256", "model_provenance_sha256", "adapter_files_sha256",
                      "training_binding", "batch_size", "min_pixels", "max_new_tokens", "do_sample",
                      "attention_implementation", "head_last_position_only", "input", "input_truncation"):
            if evaluation.get(field) != new.get(field):
                raise ValueError(f"historical NF4/BF16 shared source/input contract differs: {role}/{field}")
        rows.append({"role": role, "visual_token_budget": 1024,
                     "historical_nf4_fp32_score_percent": point["score_percent"],
                     "new_original_bf16_score_percent": bf16["score_percent"],
                     "observed_joint_path_delta_pp": bf16["score_percent"] - point["score_percent"],
                     "historical_evaluation_file": str(evaluation_path), "historical_evaluation_sha256": helper.sha_file(evaluation_path),
                     "historical_completed_at": evaluation.get("completed_at"),
                     "new_completed_at": bf16["completed_at"],
                     "scope": "Cross-numerical joint-path observation: storage, quantization and computation differ. High-budget original FP32 was not evaluated, so this does not isolate a dtype effect or establish numerical equivalence."})
    return rows, {"available": True, "file": str(path), "file_sha256": helper.sha_file(path),
                  "scope": "Previously completed NF4/FP32 quality-only reference; no historical speed attributed to BF16 alone."}


def fixed_cases(cells, references):
    lookup = {str(row["question_id"]): (index, row) for index, row in enumerate(references)}
    out = []
    for qid in ("27482", "10476", "12257"):
        if qid not in lookup:
            continue
        index, reference = lookup[qid]
        for role in ROLES:
            for budget in BUDGETS:
                row = cells[(role, budget)]["rows"][index]
                out.append({"question_id": qid, "image_id": reference["image_id"], "question": reference["question"],
                            "role": role, "visual_token_budget": budget, "prediction": row["prediction"], "score": row["score"],
                            "actual_visual_tokens": row["visual_tokens"], "resized_width": row["resized_width"], "resized_height": row["resized_height"],
                            "origin": cells[(role, budget)]["origin"],
                            "scope": "Three pre-existing teaching cases; descriptive display only, not representative accuracy or budget selection evidence."})
    return out


def plots(output, summaries, contrasts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties, fontManager
    font = "DejaVu Sans"
    if helper.FONT_PATH.exists():
        fontManager.addfont(str(helper.FONT_PATH))
        font = FontProperties(fname=str(helper.FONT_PATH)).get_name()
    lookup = {(row["role"], row["visual_token_budget"]): row for row in summaries}
    colors = {"base": "#0072B2", "adapted": "#007A5E"}
    markers = {"base": "o", "adapted": "s"}
    labels = {"base": "固定底座", "adapted": "固定adapter · seed42"}
    files = []
    with plt.rc_context({"font.family": font, "font.size": 10, "svg.fonttype": "path",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "savefig.facecolor": "white"}):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), layout="constrained")
        panels = [(axes[0, 0], "score_percent", "VQA一致得分（%）", "A 完整内部dev质量"),
                  (axes[0, 1], "pure_generation_questions_per_second", "QA / generation秒", "B 观测纯生成吞吐"),
                  (axes[1, 0], "peak_allocated_gib", "allocated GiB", "C 完整单元峰值显存")]
        for role in ROLES:
            points = [lookup[(role, budget)] for budget in BUDGETS]
            for ax, field, ylabel, title in panels:
                # Dashed segment explicitly crosses the historical/new boundary.
                ax.plot([0, 1], [points[0][field], points[1][field]], linestyle="--", color=colors[role], alpha=.65)
                ax.plot([1, 2], [points[1][field], points[2][field]], color=colors[role])
                ax.scatter([0], [points[0][field]], marker=markers[role], facecolors="white", edgecolors=colors[role], s=72)
                ax.scatter([1, 2], [points[1][field], points[2][field]], marker=markers[role], color=colors[role], s=62, label=labels[role])
                for x, point in enumerate(points):
                    ax.annotate(f"{point[field]:.3f}", (x, point[field]), xytext=(0, 9 if role == "adapted" else -15),
                                textcoords="offset points", ha="center", fontsize=8)
                ax.set_ylabel(ylabel); ax.set_title(title, loc="left")
                ax.set_xticks(range(3), ["256\n历史参考", "512\n本轮", "1024\n本轮"])
                ax.grid(axis="y", color="#DDDDDD", linewidth=.6)
            axes[1, 1].plot(range(3), [point["visual_tokens"]["mean"] for point in points],
                            marker=markers[role], color=colors[role], label=labels[role])
        axes[0, 0].legend(fontsize=8)
        axes[0, 1].set_ylim(bottom=0); axes[1, 0].set_ylim(bottom=0)
        axes[1, 1].set(title="D 实际视觉token均值", ylabel="merged visual tokens",
                       xticks=range(3), xticklabels=["256 历史", "512 本轮", "1024 本轮"])
        axes[1, 1].set_ylim(bottom=0); axes[1, 1].grid(axis="y", color="#DDDDDD", linewidth=.6)
        fig.suptitle("原权重BF16＋高视觉预算：四个新单元、两个历史参考", fontsize=15, weight="bold")
        fig.supxlabel("空心点／虚线＝历史256；吞吐受运行日期、GPU UUID、输出长度及并发影响，不能作为纯加速因果比较。", fontsize=8)
        for extension in ("png", "svg"):
            path = output / f"joint_bf16_quality_cost.{extension}"
            fig.savefig(path, dpi=180, facecolor="white"); files.append(str(path))
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(11, 5.2), layout="constrained")
        for index, row in enumerate(contrasts):
            low, high = row["image_cluster_bootstrap_95_pp"]
            color = colors.get(row.get("role"), "#555555")
            ax.plot([low, high], [index, index], color=color, linewidth=2)
            ax.scatter(row["delta_pp"], index, marker=markers.get(row.get("role"), "D"), color=color, s=60)
        ax.axvline(0, color="#777777", linestyle="--", linewidth=.8)
        ax.set_yticks(range(len(contrasts)), [row["contrast"] for row in contrasts])
        ax.invert_yaxis(); ax.set_xlabel("配对得分变化（百分点）；点与95%整图bootstrap区间")
        ax.set_title("四个预算比较＋一个交互：固定权重的内部dev敏感性", loc="left")
        ax.grid(axis="x", color="#DDDDDD", linewidth=.6)
        fig.supxlabel("2000次重采样；每次保留整图所有题；点状区间未做多重校正，不包含训练种子不确定性或精度等价检验。", fontsize=8)
        for extension in ("png", "svg"):
            path = output / f"joint_bf16_paired_effects.{extension}"
            fig.savefig(path, dpi=180, facecolor="white"); files.append(str(path))
        plt.close(fig)
    return files


def analyze(asset_root, probe_root, reference_root, nf4_root, replicates, seed):
    manifest_path, manifest, references = helper.manifest_data(asset_root)
    output = probe_root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    cells, progress, provenance = {}, [], {}
    for role in ROLES:
        for budget in BUDGETS:
            historical = budget == 256
            directory = (reference_root / "runs" / role / "original_bf16" if historical else
                         probe_root / "runs" / role / f"budget_{budget}")
            cell, status, source = read_cell(directory, role, budget, manifest_path, manifest, references,
                                             "historical_precision_bridge" if historical else "new_joint_bf16")
            cells[(role, budget)] = cell
            progress.append(status); provenance[f"{role}:{budget}"] = source
    current_monitor = contention_monitor(probe_root)
    historical_monitor = contention_monitor(reference_root, historical=True)
    common = {"schema_version": "rarevlm.joint_bf16.analysis.v1", "updated_at": datetime.now(timezone.utc).isoformat(),
              "completed_cells": sum(cells[(role, budget)]["complete"] for role in ROLES for budget in NEW_BUDGETS),
              "expected_cells": 4, "historical_reference_cells": 2,
              "historical_reference_cells_complete": sum(cells[(role, 256)]["complete"] for role in ROLES),
              "manifest_sha256": manifest["manifest_sha256"], "cells": progress, "source_data": provenance,
              "resource_contention": {"current": current_monitor, "historical": historical_monitor},
              "analysis_source_sha256": helper.sha_file(Path(__file__)), "cpu_helper_source_sha256": helper.sha_file(HELPER_PATH)}
    if not all(cell["complete"] for cell in cells.values()):
        result = {**common, "complete": False, "status": "partial_generation",
                  "pending": [row for row in progress if not row["complete"]],
                  "scope": "Progress only; no partial final comparison, throughput extrapolation or protocol selection."}
        helper.write_json(output / "ANALYSIS.json", result)
        return result
    check_shared_contracts(cells)
    summaries, batches = [], []
    for role in ROLES:
        for budget in BUDGETS:
            summary, batch_rows = summarize(role, budget, cells[(role, budget)], current_monitor, historical_monitor)
            summary["evaluation_file"] = provenance[f"{role}:{budget}"]["evaluation_file"]
            summaries.append(summary); batches.extend(batch_rows)
    contrasts, pairs, matrices = [], [], []
    for role, before, after in COMPARISONS:
        summary, pair_rows, matrix_rows = paired_budget_contrast(role, before, after, cells, references, replicates, seed)
        contrasts.append(summary); pairs.extend(pair_rows); matrices.extend(matrix_rows)
    interaction_values = [(a1024["score"] - a256["score"]) - (b1024["score"] - b256["score"])
                          for a1024, a256, b1024, b256 in zip(cells[("adapted", 1024)]["rows"],
                              cells[("adapted", 256)]["rows"], cells[("base", 1024)]["rows"], cells[("base", 256)]["rows"])]
    contrasts.append({"role": "interaction", "contrast": "(adapted1024−256)−(base1024−256)",
                      **helper.bootstrap_difference(interaction_values, references, replicates, seed),
                      "comparison_scope": "Interaction in budget benefit at original BF16; fixed NF4-trained seed42 adapter and frozen dev. Not a generalized training-method mechanism proof."})
    cost_tags = historical_cost_tags(summaries)
    slices, slice_source = frequency_slices(nf4_root, cells, manifest, references)
    cross_numeric, nf4_source = cross_numeric_quality(nf4_root, summaries, manifest)
    cases = fixed_cases(cells, references)
    for name, rows in (("CELL_SUMMARY.csv", summaries), ("PAIRED_CONTRASTS.csv", contrasts),
                       ("PAIRED_QUESTIONS.csv", pairs), ("SCORE_TRANSITIONS.csv", matrices),
                       ("BATCH_RESOURCES.csv", batches), ("HISTORICAL_COST_REFERENCES.csv", cost_tags),
                       ("SFT_FREQUENCY_SLICES.csv", slices), ("CROSS_NUMERICAL_QUALITY.csv", cross_numeric),
                       ("FIXED_TEACHING_CASES.csv", cases)):
        helper.write_csv(output / name, rows)
    figures = plots(output, summaries, contrasts)
    result = {**common, "complete": True, "status": "completed", "summaries": summaries, "cell_summaries": summaries,
              "paired_contrasts": contrasts, "historical_cost_references": cost_tags, "sft_frequency_slices": slices,
              "sft_frequency_slice_source": slice_source, "cross_numerical_quality": cross_numeric,
              "historical_nf4_quality_source": nf4_source, "fixed_teaching_cases": cases, "figures": figures,
              "checks": {"new_cells_verified": 4, "historical_reference_cells_verified": 2,
                         "all_saved_predictions_independently_rescored": len(references) * 6,
                         "same_budget_base_adapted_encoded_inputs": True,
                         "within_role_new_budgets_same_gpu_uuid": True,
                         "same_within_role_checkpoint_and_bf16_parameter_dtypes": True,
                         "common_source_package_generation_contracts_match": True,
                         "batch_resources_deduplicated_and_warmup_checked": True},
              "limitations": ["Four new evaluations, zero new training runs; one fixed NF4-trained adapter seed42 and one internal dev set.",
                  "Budget256 references are historical. Timing/GPU UUID/contention differences are explicitly labeled; observed speed ratios are not isolated causal speedups.",
                  "High-budget original FP32 has not been evaluated. Comparing high-budget BF16 against NF4/FP32 changes quantization and numerical path together and cannot isolate dtype effects.",
                  "No numerical or statistical equivalence margin was predeclared; CI crossing zero or output agreement cannot establish equivalence.",
                  "Five pointwise image-cluster bootstrap intervals are exploratory, unadjusted and conditional on fixed weights; no training-seed uncertainty.",
                  "Visual budget is an upper bound; actual merged token count and input lengths are measured per question. Higher-budget inputs may exceed the training limit without being truncated.",
                  "Existing GPU jobs were retained. Zero observed sampled contention is not an exclusivity guarantee.",
                  "Batch-amortized generation timing excludes loading, preflight, disk writes and other controller overhead; preprocessing+H2D+generation is not full deployment wall time.",
                  "SFT target frequency is a train_all answer-text exposure proxy, not pretraining knowledge or semantic visual-category long-tail evidence."],
              "scientific_visualization_reference": {"title": "Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents",
                  "authors": "Kassis, T.; Agarwal, V.; He, Y.; Patel, D.; Brueckner, A. M.", "year": 2026,
                  "url": "https://doi.org/10.48550/arXiv.2609.00065", "verified_on": "2026-10-11",
                  "current_arxiv_version_on_verification": "v2", "scope": "Previously applied scientific visualization workflow; current-version reference verified and retained"}}
    helper.write_json(output / "ANALYSIS.json", result)
    helper.write_json(output / "FIGURE_PROVENANCE.json", {
        "analysis_code": str(Path(__file__).resolve()), "analysis_code_sha256": helper.sha_file(Path(__file__)),
        "cpu_helper_source": str(HELPER_PATH), "cpu_helper_source_sha256": helper.sha_file(HELPER_PATH),
        "manifest_file": str(manifest_path), "manifest_file_sha256": helper.sha_file(manifest_path),
        "source_data": provenance, "resource_contention": result["resource_contention"],
        "source_tables": [str(path) for path in sorted(output.glob("*.csv"))],
        "slice_source": slice_source, "historical_nf4_quality_source": nf4_source,
        "uncertainty": "95% pointwise image-cluster percentile paired bootstrap; fixed weights, no multiplicity correction",
        "bootstrap_replicates": replicates, "bootstrap_seed": seed,
        "transformations": ["All frozen1617dev QA retained", "Resample all questions of each sampled image together",
            "Deduplicate repeated batch resources by session/batch_start", "Remove each session first batch from steady-state phase timing",
            "Show two historical256 references as hollow markers and historical/new connecting segments as dashed",
            "Do not treat cross-numerical comparisons as pure dtype effects or evidence of equivalence"],
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "matplotlib")},
        "figure_exports": [{"file": name, "sha256": helper.sha_file(Path(name)), "size_bytes": Path(name).stat().st_size} for name in figures],
        "dpi": 180, "font_path": str(helper.FONT_PATH), "svg_text": "glyph paths for portable Chinese rendering",
        "destination": "Chinese educational research report; no publisher-compliance claim",
        "alt_text": "Four new fixed-weight original-BF16 quality, generation throughput, memory and actual-token observations at512/1024 are compared with explicitly hollow historical256 references. Five paired image-cluster intervals characterize budget sensitivity and its adapter/base interaction; timing references do not prove an isolated speedup."})
    lines = ["# BF16与高视觉预算联合验证", "", "四个新评测单元完成；两项256预算为已完成的历史BF16参考。每格1617题/1024图。", "",
             "| 权重 | 预算 | 来源 | 得分% | 纯生成QA/s | 预处理+H2D+生成QA/s | 峰值allocated GiB |", "|---|---:|---|---:|---:|---:|---:|"]
    for row in summaries:
        lines.append(f"| {row['role']} | {row['visual_token_budget']} | {'历史' if row['visual_token_budget']==256 else '本轮'} | {row['score_percent']:.3f} | {row['pure_generation_questions_per_second']:.3f} | {row['generation_preprocessing_h2d_questions_per_second']:.3f} | {row['peak_allocated_gib']:.3f} |")
    lines.extend(["", "## 预定配对比较", "", "2000次整图bootstrap；95%点状条件区间，未多重校正。"])
    for row in contrasts:
        low, high = row["image_cluster_bootstrap_95_pp"]
        lines.append(f"- {row['contrast']}：{row['delta_pp']:+.3f} pp，区间 [{low:+.3f}, {high:+.3f}]。")
    lines.extend(["", "## 运行来源", ""])
    for row in summaries:
        lines.append(f"- {row['role']} budget{row['visual_token_budget']}：{row['completed_at']}，GPU `{row['gpu_uuid']}`，并发标记 `{row['resource_timing_status']}`。")
    lines.extend(["", "历史吞吐是跨运行观测参考，不能写成同卡纯加速效果。高预算原权重FP32尚未测量，所以不能分离BF16 dtype效应或证明精度等价。", "",
                  "## 限制", ""] + [f"- {item}" for item in result["limitations"]])
    lines.extend(["", "绘图软件参考：Kassis et al. (2026), Scientific Agent Skills, https://doi.org/10.48550/arXiv.2609.00065。", ""])
    (output / "FINDINGS.md").write_text("\n".join(lines))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET)
    parser.add_argument("--probe-root", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--nf4-budget-root", type=Path, default=DEFAULT_NF4_BUDGET)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261011)
    args = parser.parse_args()
    if args.replicates != 2000:
        raise ValueError("declared joint study fixes2000 image-cluster bootstrap replicates")
    roots = [helper.private_path(path) for path in (args.asset_root, args.probe_root, args.reference_root, args.nf4_budget_root)]
    result = analyze(*roots, args.replicates, args.bootstrap_seed)
    print(json.dumps({"complete": result["complete"], "status": result["status"],
                      "completed_cells": result["completed_cells"], "expected_cells": 4,
                      "historical_reference_cells_complete": result["historical_reference_cells_complete"],
                      "analysis_file": str(roots[1] / "analysis/ANALYSIS.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
