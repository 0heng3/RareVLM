#!/usr/bin/env python3
"""Build CPU-only teaching figures from saved VLM results, without evaluating models.

Run once against live outputs, then use ``--snapshot <figures>/snapshot`` to replay
the exact source bytes. PNG, SVG, raw CSV tables, and JSON provenance are emitted.
The existing environment needs only matplotlib, numpy, and Pillow; no GPU is used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PRIVATE = Path("/opt/data/private")
os.environ.setdefault("MPLCONFIGDIR", str(PRIVATE / ".tools/cache/matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from PIL import Image

METHODS = ("qv_r8", "qkvo_r4", "qv_r4_merger")
SEEDS = (0, 42, 200)
COLORS = {"qv_r8": "#0072B2", "qkvo_r4": "#B45F00", "qv_r4_merger": "#6F4C9B"}
LABELS = {"qv_r8": "q/v rank 8", "qkvo_r4": "q/k/v/o rank 4", "qv_r4_merger": "q/v rank 4\n+ merger"}
SEED_MARKERS = {0: "o", 42: "s", 200: "^"}
BUCKETS = ("unseen_0", "rare_1_5", "medium_6_20", "frequent_gt20")
BUCKET_LABELS = ("0次\n（训练目标未见）", "1–5次", "6–20次", ">20次")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, records: list[dict]) -> None:
    if not records:
        raise ValueError(f"No records for {path.name}; missing observations must not become zero")
    keys = list(dict.fromkeys(key for row in records for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def mean_sd(values: list[float]) -> tuple[float, float | None]:
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) >= 2 else None


def axes_style(ax, ylabel: str, ylim=(0, 100)) -> None:
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#d5dbe2", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0, pad=8)


def footer(fig, text: str) -> None:
    fig.text(0.06, 0.037, text, ha="left", va="bottom", fontsize=12, color="#374151", linespacing=1.5)


def seed_legend():
    return [Line2D([], [], linestyle="", marker=SEED_MARKERS[s], color="#374151",
                   markersize=8, label=f"seed {s}") for s in SEEDS]


def export(fig, stem: str, out: Path) -> dict:
    records = []
    for extension in ("png", "svg"):
        path = out / f"{stem}.{extension}"
        fig.savefig(path, dpi=100, facecolor="white", transparent=False)
        raw = path.read_bytes()
        item = {"path": path.name, "bytes": len(raw), "sha256": sha256(raw)}
        if extension == "png":
            with Image.open(path) as image:
                item.update(width_px=image.width, height_px=image.height, mode=image.mode,
                            dpi=image.info.get("dpi"))
        records.append(item)
    plt.close(fig)
    return {"outputs": records, "background": "opaque white", "svg_text": "glyph paths for portable appearance"}


def stage1_figure(summary: dict, out: Path) -> dict:
    rows = [r for r in summary["seed_records"] if r["stage"] == "stage1" and r["method"] in METHODS]
    write_csv(out / "stage1_seed_points.csv", rows)
    tasks = (("dev", "内部 dev", "TextVQA 一致性评分（%）"),
             ("textvqa", "TextVQA validation", "TextVQA 一致性评分（%）"),
             ("docvqa", "DocVQA validation", "ANLS × 100"),
             ("mmstar", "MMStar", "子类别宏平均准确率（%）"))
    fig, axes = plt.subplots(2, 2, figsize=(16, 10.8))
    fig.subplots_adjust(left=.08, right=.965, bottom=.20, top=.80, wspace=.20, hspace=.38)
    fig.suptitle("相同训练预算，LoRA 插入位置带来什么变化？", fontsize=26, x=.06, ha="left", y=.96)
    fig.text(.06, .902, "Stage 1：每次 10,000 QA、625 次更新；展示全部种子与均值 ± 种子样本标准差", fontsize=17)
    fig.legend(handles=seed_legend() + [Line2D([], [], color="#111827", marker="D", markersize=7,
               linestyle="", label="均值 ± 种子 SD（不是 CI）")], loc="upper left", bbox_to_anchor=(.052, .883),
               ncol=4, frameon=False, fontsize=13)
    aggregations = []
    missing = []
    for ax, (task, title, metric) in zip(axes.flat, tasks):
        axes_style(ax, metric)
        ax.set_yticks([0, 25, 50, 75, 100])
        n_examples = next((r["n"] for r in rows if r["task"] == task), None)
        ax.set_title(f"{title}  ·  {n_examples:,} 条问题", loc="left", fontsize=17, pad=12)
        ax.set_xticks(range(3), [LABELS[m] for m in METHODS], fontsize=14)
        ax.set_xlim(-.5, 2.65)
        for index, method in enumerate(METHODS):
            subset = sorted([r for r in rows if r["task"] == task and r["method"] == method], key=lambda r: r["seed"])
            completed = {r["seed"] for r in subset}
            absent = [s for s in SEEDS if s not in completed]
            missing.extend({"task": task, "method": method, "seed": s, "status": "not present in completed snapshot"} for s in absent)
            if not subset:
                ax.text(index, 50, "待完成", ha="center", color="#6b7280")
                continue
            values = [float(r["score_percent"]) for r in subset]
            mean, sd = mean_sd(values)
            for row in subset:
                jitter = {0: -.14, 42: -.03, 200: .08}[row["seed"]]
                ax.scatter(index + jitter, row["score_percent"], s=65, marker=SEED_MARKERS[row["seed"]],
                           color=COLORS[method], edgecolors="white", linewidth=.75, zorder=4)
            ax.errorbar(index + .25, mean, yerr=sd, fmt="D", color="#111827", markersize=6,
                        capsize=5, capthick=1.4, elinewidth=1.4, zorder=5)
            number = f"{mean:.3f}" + (f" ± {sd:.3f}" if sd is not None else "")
            note = f"n={len(subset)} 个种子" if not absent else f"n={len(subset)}；{', '.join(map(str, absent))} 待完成"
            ax.text(index + .07, 92, number, ha="center", fontsize=12, color="#111827")
            ax.text(index + .07, 83, note, ha="center", fontsize=10.2, color="#6b7280")
            aggregations.append({"method": method, "task": task, "metric": metric, "completed_seeds": sorted(completed),
                                 "n_seeds": len(subset), "mean_score_percent": mean, "sample_seed_sd": sd})
        baseline = summary["baseline_metrics"].get(task)
        if baseline:
            base = float(baseline["score_percent"])
            ax.axhline(base, color="#4b5563", linestyle=(0, (5, 4)), linewidth=1.4, zorder=1)
            ax.text(2.60, base - 5.5, f"NF4 底座 {base:.3f}", ha="right", fontsize=10.8,
                    color="#374151", bbox=dict(facecolor="white", edgecolor="none", pad=2))
        else:
            ax.text(.01, .045, "NF4 底座未进行 dev 评测", transform=ax.transAxes, fontsize=11, color="#6b7280")
    footer(fig, "各图区间统一为 0–100；任务指标不同，不能合并为一个总分。\n"
           "n=1 无法估计种子标准差。merger 策略的种子尚未齐全，不能据此公平地进行最终排名。\n"
           "种子 SD 描述训练重复之间的波动；它不是测试问题或图像的置信区间（CI）。")
    write_csv(out / "stage1_aggregates.csv", aggregations)
    details = export(fig, "stage1_results", out)
    details.update(raw_points="stage1_seed_points.csv", aggregates="stage1_aggregates.csv", missing=missing,
                   uncertainty="sample standard deviation across completed training seeds, ddof=1; null for n<2",
                   y_limits=[0, 100], transformations=["stage == stage1", "arithmetic mean over completed seeds"],
                   alt_en="Four common-scale panels show saved dev, TextVQA, DocVQA and MMStar metrics for three LoRA strategies. Individual seed markers and mean plus seed SD are displayed; the merger has one completed seed.",
                   description_zh="同一训练预算下比较三种LoRA插入策略。圆、方、三角分别是seed0/42/200，黑菱形与误差线为已完成种子的均值与样本标准差（不是置信区间）。第三策略仅一个种子，暂不能公平排名。TextVQA用一致性评分，DocVQA用ANLS，MMStar用子类别宏平均；不能把四项平均为一个总分。")
    return details


def numerical_figure(summary: dict, out: Path) -> dict:
    records = summary["numerical_path_matrix"]
    indexed = {r["condition"]: r for r in records}
    names = (("original_base", "original_pilot"), ("nf4_language_only_base", "nf4_language_only_pilot"))
    values = [[float(indexed[name]["score_percent"]) for name in row] for row in names]
    fig, ax = plt.subplots(figsize=(16, 9.4))
    fig.subplots_adjust(left=.06, right=.96, top=.90, bottom=.15)
    ax.set_axis_off()
    ax.set_xlim(-.2, 3.7)
    ax.set_ylim(-.65, 2.5)
    fig.suptitle("同一份历史 adapter，在不同数值路径下表现相同吗？", fontsize=25, x=.06, ha="left", y=.96)
    ax.text(1.22, 2.2, "无 adapter", ha="center", fontsize=20)
    ax.text(2.7, 2.2, "接入历史 pilot adapter", ha="center", fontsize=20)
    deltas = []
    for r, row in enumerate(names):
        y = 1.57 - r * 1.22
        label = "原始权重\nFP32 计算" if r == 0 else "语言权重 NF4\nFP32 计算"
        ax.text(.0, y, label, ha="left", va="center", fontsize=17, linespacing=1.4)
        for c, condition in enumerate(row):
            x = 1.22 + c * 1.48
            value = indexed[condition]
            rect = FancyBboxPatch((x-.55, y-.44), 1.10, .88, boxstyle="round,pad=.035,rounding_size=.04",
                                  facecolor="#eef3f7", edgecolor="#a0adba", linewidth=1.2)
            ax.add_patch(rect)
            ax.text(x, y+.07, f"{value['score_percent']:.3f}%", ha="center", va="center", fontsize=33, color="#111827")
            ax.text(x, y-.21, f"TextVQA  ·  {value['n']:,} 条问题", ha="center", fontsize=13, color="#4b5563")
        delta = values[r][1] - values[r][0]
        ax.annotate("", xy=(2.07, y), xytext=(1.85, y), arrowprops=dict(arrowstyle="->", color="#374151", lw=1.5))
        ax.text(1.97, y+.22, f"+{delta:.3f} pp", ha="center", fontsize=12.5, color="#374151")
        deltas.append({"path": indexed[row[0]]["quantization"], "pilot_minus_base_pp": delta})
    ax.text(.0, -.38, "横向比较：固定推理数值路径，观察接入 adapter 前后的差异。\n"
            "纵向比较：固定已保存的 adapter 状态，观察原始 / NF4 推理路径的差异。", fontsize=17, color="#374151", linespacing=1.6)
    footer(fig, "这里复用一份历史 pilot adapter 检查推理路径；没有分别训练四个模型。\n"
           "它无法分解“训练时量化”的因果作用。pp = 百分点；本图未添加未经估计的不确定性。")
    write_csv(out / "numerical_path_points.csv", records)
    write_csv(out / "numerical_path_deltas.csv", deltas)
    details = export(fig, "numerical_paths", out)
    details.update(raw_points="numerical_path_points.csv", derived_deltas="numerical_path_deltas.csv",
                   uncertainty="none: individual fixed-path evaluation scores", transformations=["adapter-minus-base subtraction within each numerical path"],
                   alt_en="A two-by-two matrix reports TextVQA scores for original and language-only NF4 weights, with no adapter or the same historical pilot adapter. This matrix compares inference paths, not retraining effects.",
                   description_zh="横着看是在同一数值路径下接入历史adapter前后的变化：原始权重+0.784个百分点，NF4+1.352个百分点。竖着看是保存的同一adapter状态在原始与NF4推理路径下的差异。这不是四种独立训练设置，不能作为训练量化的因果分解。")
    return details


def slices_figure(slices: dict, out: Path) -> dict:
    rows = [r for r in slices["slice_metrics"] if r["dimension"] == "answer_frequency"]
    write_csv(out / "answer_frequency_seed_points.csv", rows)
    population = []
    for bucket in BUCKETS:
        subset = [r for r in rows if r["slice"] == bucket]
        counts = {r["n"] for r in subset}
        images = {r["unique_images"] for r in subset}
        if len(counts) != 1 or len(images) != 1:
            raise ValueError(f"Inconsistent population across runs: {bucket}")
        population.append({"slice": bucket, "qa_count": counts.pop(), "unique_images": images.pop()})
    total = sum(r["qa_count"] for r in population)
    for row in population:
        row["fraction_percent"] = row["qa_count"] / total * 100
    if total != slices["manifest"]["dev_questions"]:
        raise ValueError("Frequency buckets do not cover the full dev population")
    write_csv(out / "answer_frequency_population.csv", population)
    fig, (left, right) = plt.subplots(1, 2, figsize=(16, 8.4), gridspec_kw={"width_ratios": [1, 1.5]})
    fig.subplots_adjust(left=.065, right=.967, bottom=.235, top=.76, wspace=.19)
    fig.suptitle("哪些问题更容易错？先用“训练答案频次”做诊断", fontsize=26, x=.055, ha="left", y=.95)
    fig.text(.055, .885, "仅内部 dev，共 1,617 条问题；频次来自实际训练的 10k QA 目标答案", fontsize=17)
    axes_style(left, "dev 问题数量", (0, max(r["qa_count"] for r in population) * 1.27))
    left.set_title("A. 每个频次桶包含多少问题", fontsize=19, loc="left", pad=15)
    bars = left.bar(range(4), [r["qa_count"] for r in population], width=.67,
                    color="#dde7ef", edgecolor="#29465b", linewidth=1.2,
                    hatch=["//", "..", "xx", "\\\\"])
    for bar, row in zip(bars, population):
        left.text(bar.get_x()+bar.get_width()/2, bar.get_height()+22,
                  f"{row['qa_count']}\n({row['fraction_percent']:.1f}%)", ha="center", fontsize=15, linespacing=1.4)
    left.set_xticks(range(4), BUCKET_LABELS, fontsize=13)
    left.set_xlabel("答案在训练 QA 目标中出现的次数", labelpad=14)
    axes_style(right, "TextVQA 一致性评分（%）", (0, 100))
    right.set_yticks([0, 25, 50, 75, 100])
    right.set_title("B. 各桶已保存的预测得分", fontsize=19, loc="left", pad=15)
    right.set_xlim(-.48, 3.48)
    right.set_xticks(range(4), BUCKET_LABELS, fontsize=13)
    right.set_xlabel("答案在训练 QA 目标中出现的次数", labelpad=14)
    aggregations = []
    coverage = []
    for mindex, method in enumerate(METHODS):
        completed = sorted({r["seed"] for r in rows if r["method"] == method})
        coverage.append({"method": method, "completed_seeds": completed, "n_seeds": len(completed),
                         "missing_seeds": [s for s in SEEDS if s not in completed]})
        for bindex, bucket in enumerate(BUCKETS):
            subset = [r for r in rows if r["method"] == method and r["slice"] == bucket]
            mean, sd = mean_sd([r["score_percent"] for r in subset])
            center = bindex + (mindex - 1) * .2
            for row in subset:
                jitter = {0: -.045, 42: 0., 200: .045}[row["seed"]]
                right.scatter(center+jitter, row["score_percent"], s=55, marker=SEED_MARKERS[row["seed"]],
                              edgecolors="white", linewidth=.75, color=COLORS[method], zorder=4)
            right.errorbar(center, mean, yerr=sd, fmt="_", markersize=17, color=COLORS[method], capsize=4,
                           elinewidth=1.6, capthick=1.5, zorder=5)
            aggregations.append({"method": method, "slice": bucket, "n_seeds": len(subset),
                                 "mean_score_percent": mean, "sample_seed_sd": sd})
    method_handles = [Line2D([], [], color=COLORS[m], marker="_", markersize=14, linewidth=2,
                       label=LABELS[m].replace("\n", " ") + f" (n={len([r for r in coverage if r['method']==m][0]['completed_seeds'])})") for m in METHODS]
    fig.legend(handles=method_handles, loc="upper left", bbox_to_anchor=(.052, .85), ncol=3, frameon=False, fontsize=13)
    right.legend(handles=seed_legend(), loc="lower right", ncol=3, frameon=False, fontsize=11)
    footer(fig, "评分轴范围为 0–100；散点是种子，短横线是均值，误差线是种子样本 SD（不是 CI）。\n"
           "答案频次是文本代理指标，不等于视觉类别或实体频次。训练目标未见，只代表未出现在这 10k QA 目标中。\n"
           "它不代表预训练未见；频次与得分的关联，也不能证明 OCR 或视觉长尾导致了错误。")
    write_csv(out / "answer_frequency_aggregates.csv", aggregations)
    details = export(fig, "answer_frequency_slices", out)
    details.update(raw_points="answer_frequency_seed_points.csv", population="answer_frequency_population.csv",
                   aggregates="answer_frequency_aggregates.csv", seed_coverage=coverage,
                   uncertainty="sample standard deviation over saved completed seeds; null for n<2",
                   y_limits_score=[0, 100], transformations=["answer_frequency dimension only", "pre-existing bins: 0, 1-5, 6-20, >20", "mean over completed seeds"],
                   alt_en="A zero-baseline bar chart shows 858, 328, 180 and 251 dev questions in four training-answer frequency buckets. Seed-score scatter and mean plus SD show lower scores for unseen targets, a text association that is not a visual causal claim.",
                   description_zh="四桶分别是训练目标中出现0次、1–5次、6–20次和20次以上。它们包含858/328/180/251条dev问题。右图保留每个seed与均值±种子标准差。未见答案桶分数较低只是文本频次关联，不能直接证明视觉长尾、OCR瓶颈或频次造成错误。")
    return details


def task_deltas_figure(summary: dict, out: Path) -> dict:
    tasks = (("textvqa", "TextVQA：一致性评分"), ("docvqa", "DocVQA：ANLS × 100"),
             ("mmstar", "MMStar：子类别宏平均"))
    rows = []
    for row in summary["seed_records"]:
        if row["stage"] == "stage1" and row["method"] in METHODS and row["task"] in dict(tasks):
            baseline = summary["baseline_metrics"][row["task"]]["score_percent"]
            rows.append({**row, "nf4_baseline_score_percent": baseline,
                         "delta_pp": row["score_percent"] - baseline})
    write_csv(out / "task_delta_seed_points.csv", rows)
    fig, axes = plt.subplots(1, 3, figsize=(16, 7.8), sharey=True)
    fig.subplots_adjust(left=.07, right=.975, bottom=.27, top=.68, wspace=.18)
    fig.suptitle("收益与代价：适配后的各任务变化", fontsize=28, x=.06, ha="left", y=.95)
    fig.text(.06, .875, "逐种子得分减去同任务 NF4 底座分数；公共纵轴为 −1.2 至 +3.5 个百分点（pp）", fontsize=17)
    fig.legend(handles=seed_legend() + [Line2D([], [], color="#111827", marker="D", markersize=7,
               linestyle="", label="均值 ± 种子 SD（不是 CI）")], loc="upper left", bbox_to_anchor=(.052, .84),
               ncol=4, frameon=False, fontsize=13)
    aggregates = []
    for ax, (task, label) in zip(axes, tasks):
        axes_style(ax, "相对 NF4 底座的变化（百分点）" if task == "textvqa" else "", (-1.2, 3.5))
        ax.set_yticks([-1, 0, 1, 2, 3])
        ax.axhline(0, color="#374151", linewidth=1.5, linestyle=(0, (5, 4)), zorder=1)
        baseline = summary["baseline_metrics"][task]["score_percent"]
        ax.set_title(f"{label}\nNF4 底座：{baseline:.3f}", fontsize=16, loc="left", pad=14)
        ax.set_xlim(-.45, 2.6)
        ax.set_xticks(range(3), ["q/v\nrank 8", "q/k/v/o\nrank 4", "q/v rank 4\n+ merger"], fontsize=13)
        for index, method in enumerate(METHODS):
            subset = [row for row in rows if row["task"] == task and row["method"] == method]
            if not subset:
                ax.text(index, 1.5, "待完成", ha="center", color="#6b7280")
                continue
            mean, sd = mean_sd([row["delta_pp"] for row in subset])
            for row in subset:
                jitter = {0: -.14, 42: -.03, 200: .08}[row["seed"]]
                ax.scatter(index+jitter, row["delta_pp"], s=65, marker=SEED_MARKERS[row["seed"]],
                           color=COLORS[method], edgecolors="white", linewidth=.75, zorder=4)
            ax.errorbar(index+.25, mean, yerr=sd, fmt="D", color="#111827", markersize=6,
                        capsize=5, capthick=1.4, elinewidth=1.4, zorder=5)
            number = f"{mean:+.3f}" + (f" ± {sd:.3f}" if sd is not None else "")
            ax.text(index+.05, 3.27, number, ha="center", fontsize=11.4)
            ax.text(index+.05, 2.98, f"n={len(subset)}" + ("；仅 seed0" if len(subset)==1 else " 个种子"),
                    ha="center", fontsize=10.6, color="#6b7280")
            aggregates.append({"task": task, "method": method, "baseline_score_percent": baseline,
                               "n_seeds": len(subset), "mean_delta_pp": mean, "sample_seed_sd": sd})
    footer(fig, "读零线：上方代表该任务指标提高，下方代表下降。三个任务指标各自解释，不能合成一个分数。\n"
           "SD 仅描述训练种子波动，不包含测试图像或底座评测的不确定性；n=1 无法估计种子 SD。\n"
           "merger 的 seed42/200 尚未出现在该冻结快照中，不能据此公平地进行最终策略排名。")
    write_csv(out / "task_delta_aggregates.csv", aggregates)
    details = export(fig, "task_deltas", out)
    details.update(raw_points="task_delta_seed_points.csv", aggregates="task_delta_aggregates.csv",
                   y_limits=[-1.2, 3.5], uncertainty="sample seed SD, not CI; no test-image or baseline uncertainty",
                   transformations=["stage1 saved result minus fixed same-task NF4 baseline", "arithmetic mean across completed seeds"],
                   alt_en="Three panels show per-seed changes from same-task NF4 baselines on a shared -1.2 to 3.5 percentage-point scale. TextVQA improves for all shown methods, DocVQA changes vary, and MMStar macro accuracy decreases. Merger has one completed seed.",
                   description_zh="各种子的同任务得分减去NF4底座得分。TextVQA约提升2.4–2.7个百分点；DocVQA提升幅度依策略而变；MMStar子类别宏平均略降。统一百分点轴与零基线使收益和代价易读，但不同任务的指标不能合并、种子不齐不能排名。")
    return details


def main() -> None:
    base = PRIVATE / "longtail_vlm/rarevlm_pilot"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=base / "outputs/vlm_deep_v2")
    parser.add_argument("--output", type=Path, default=base / "outputs/vlm_learning_v1/figures")
    parser.add_argument("--snapshot", type=Path, help="Replay from a previously emitted immutable source snapshot directory")
    parser.add_argument("--only", choices=["task_deltas"], help="Append only the task-delta figure; preserve existing figures")
    args = parser.parse_args()
    out = args.output.resolve()
    if PRIVATE not in out.parents:
        raise ValueError("Task-created outputs must stay under /opt/data/private")
    out.mkdir(parents=True, exist_ok=True)
    snapshot = out / "snapshot"
    snapshot.mkdir(exist_ok=True)
    paths = {"SUMMARY.json": args.source_root / "runs/SUMMARY.json",
             "seed_metrics.csv": args.source_root / "runs/seed_metrics.csv",
             "DEV_SLICES.json": args.source_root / "learning_dev_diagnostics/DEV_SLICES.json"}
    sources = []
    source_bytes = {}
    for name, original in paths.items():
        source = args.snapshot / name if args.snapshot else original
        raw = source.read_bytes()
        source_bytes[name] = raw
        destination = snapshot / name
        if destination.exists() and destination.read_bytes() != raw:
            raise FileExistsError(f"Refusing to overwrite a different source snapshot: {destination}; choose a new --output")
        if not destination.exists():
            destination.write_bytes(raw)
        sources.append({"name": name, "original_path": str(original), "read_path": str(source),
                        "snapshot": str(destination.relative_to(out)), "sha256": sha256(raw), "bytes": len(raw)})
    summary = json.loads(source_bytes["SUMMARY.json"])
    slices = json.loads(source_bytes["DEV_SLICES.json"])
    csv_rows = list(csv.DictReader(io.StringIO(source_bytes["seed_metrics.csv"].decode())))
    canonical = lambda row: (row["stage"], row["method"], int(row["seed"]), row["task"], float(row["score_percent"]))
    if sorted(map(canonical, csv_rows)) != sorted(map(canonical, summary["seed_records"])):
        raise ValueError("Live summary and CSV changed at different times; their seed records are inconsistent. Use a coherent snapshot.")
    cjk_font = PRIVATE / ".tools/fonts/NotoSansCJKsc-Regular.otf"
    if not cjk_font.exists():
        raise FileNotFoundError(f"Existing Chinese font required (no installation attempted): {cjk_font}")
    font_manager.fontManager.addfont(str(cjk_font))
    family = font_manager.FontProperties(fname=str(cjk_font)).get_name()
    style = {"font.family": family, "font.size": 15, "axes.labelsize": 14,
             "axes.titlesize": 19, "xtick.labelsize": 13, "ytick.labelsize": 13,
             "svg.fonttype": "path", "axes.unicode_minus": False, "figure.facecolor": "white"}
    with plt.rc_context(style):
        if args.only:
            figures = {"task_deltas": task_deltas_figure(summary, out)}
        else:
            figures = {"stage1_results": stage1_figure(summary, out),
                       "numerical_paths": numerical_figure(summary, out),
                       "answer_frequency_slices": slices_figure(slices, out),
                       "task_deltas": task_deltas_figure(summary, out)}
    manifest = {"schema_version": "rarevlm.learning_figures.v1", "created_at_asia_shanghai": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                "destination": "Chinese teaching webpage; Noto Sans CJK SC labels and SVG glyph paths",
                "submission_scope": "Teaching only; no journal-compliance claim", "cpu_only": True,
                "sources": sources, "source_summary_status": summary.get("status"), "figures": figures,
                "environment": {"python": platform.python_version(), "matplotlib": matplotlib.__version__, "numpy": np.__version__,
                                "font": str(cjk_font), "font_sha256": sha256(cjk_font.read_bytes())},
                "script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).read_bytes())},
                "software_reference": {"authors": ["Timothy Kassis", "Vinayak Agarwal", "Yuhuan He", "Darshil Patel", "Aubrey M. Brueckner"],
                     "year": 2026, "title": "Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents",
                     "doi": "10.48550/arXiv.2609.00065", "url": "https://arxiv.org/abs/2609.00065", "verified_version": "v2 (2026-09-02)", "verified_on": "2026-10-10"},
                "review": {"missing_values": "explicitly retained as pending; n=1 SD=null", "raw_seeds": "retained in CSV and snapshot",
                     "aggregation": "task means are never combined across metrics", "image_metadata": "Pillow inspected every exported PNG",
                     "manual_render_review": "pending", "accessibility": "color plus shapes/labels and CSV; no compliance certification"}}
    if args.only and (out / "FIGURES_PROVENANCE.json").exists():
        previous = json.loads((out / "FIGURES_PROVENANCE.json").read_text())
        if {x["name"]: x["sha256"] for x in previous["sources"]} != {x["name"]: x["sha256"] for x in sources}:
            raise ValueError("Append requires identical source snapshot hashes")
        previous["figures"].update(figures)
        previous["updated_at_asia_shanghai"] = manifest["created_at_asia_shanghai"]
        previous["append_script"] = manifest["script"]
        manifest = previous
    write_json(out / "FIGURES_PROVENANCE.json", manifest)
    print(json.dumps({"output": str(out), "figures": list(figures), "seed_records": len(summary["seed_records"]),
                      "source_summary_status": summary.get("status")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
