"""CPU-only post-hoc analysis of completed, immutable RareVLM v2 results.

Reads the logical run index rather than globbing legacy partial evaluations.
Does not train, generate, rescore labels, select checkpoints, or alter source runs.
Image clusters and training seeds remain distinct units of uncertainty.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

SEEDS = (0, 42, 200)
TASKS = ("dev", "textvqa", "docvqa", "mmstar")
METHODS = ("qv_r8", "qkvo_r4", "qv_r4_merger")
WORKSPACE = Path("/opt/data/private")
DEFAULT_RUN_ROOT = WORKSPACE / "longtail_vlm/rarevlm_pilot/outputs/vlm_deep_v2/runs"
METHOD_LABELS = {"qv_r8": "q/v rank8", "qkvo_r4": "q/k/v/o rank4", "qv_r4_merger": "q/v rank4 + merger"}
TASK_LABELS = {"dev": "内部 dev · TextVQA", "textvqa": "TextVQA · VQA score", "docvqa": "DocVQA · ANLS", "mmstar": "MMStar · 子类宏平均"}
SCOPE = ("Post-hoc exploratory analysis. Each image-cluster percentile interval is conditional on "
         "one fixed pair of trained checkpoints, not a training-seed interval. No multiplicity "
         "correction or confirmatory significance claim. Three-seed SD is reported separately.")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_bound(path, hashes):
    hashes[str(path.resolve())] = sha(path)
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_csv(path, rows):
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def metric(rows, task):
    if task == "mmstar":
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["subcategory"]].append(float(row["score"]))
        return 100 * statistics.mean(statistics.mean(v) for v in grouped.values())
    return 100 * statistics.mean(float(row["score"]) for row in rows)


def paired_clusters(base, adapted, task, replicates, seed):
    """Resample image IDs with replacement; keep all questions in each image.

    MMStar preserves its official subcategory macro mean on every replicate,
    rather than substituting a pooled question mean. Images spanning multiple
    subcategories retain their entire question cluster.
    """
    base_lookup = {r["question_id"]: r for r in base}
    adapted_lookup = {r["question_id"]: r for r in adapted}
    if len(base_lookup) != len(base) or len(adapted_lookup) != len(adapted) or set(base_lookup) != set(adapted_lookup):
        raise ValueError("duplicate/unpaired question IDs")
    categories = sorted({r.get("subcategory", "") for r in base}) if task == "mmstar" else [""]
    category_index = {name: i for i, name in enumerate(categories)}
    images = sorted({r["image_id"] for r in base})
    image_index = {name: i for i, name in enumerate(images)}
    sums = np.zeros((len(images), len(categories)), dtype=np.float64)
    counts = np.zeros_like(sums)
    for qid, a in adapted_lookup.items():
        b = base_lookup[qid]
        for key in ("image_id", "image_sha256", "question_sha256", "subcategory"):
            if a.get(key) != b.get(key):
                raise ValueError(f"unpaired {key}: {qid}")
        i = image_index[a["image_id"]]
        j = category_index[a.get("subcategory", "")] if task == "mmstar" else 0
        sums[i, j] += float(a["score"]) - float(b["score"])
        counts[i, j] += 1
    observed = float(100 * np.mean(sums.sum(axis=0) / counts.sum(axis=0)))
    expected = metric(adapted, task) - metric(base, task)
    if not math.isclose(observed, expected, abs_tol=1e-10):
        raise ValueError("paired estimator differs from benchmark metric")
    rng = np.random.default_rng(seed)
    estimates = []
    missing_category_replicates = 0
    while len(estimates) < replicates:
        batch = min(64, replicates - len(estimates))
        sampled = rng.integers(0, len(images), size=(batch, len(images)))
        sampled_counts = counts[sampled].sum(axis=1)
        valid = (sampled_counts > 0).all(axis=1)
        missing_category_replicates += int((~valid).sum())
        if valid.any():
            sampled_sums = sums[sampled].sum(axis=1)
            estimates.extend((100 * np.mean(sampled_sums[valid] / sampled_counts[valid], axis=1)).tolist())
    interval = np.quantile(estimates, [0.025, 0.975]).tolist()
    return {"delta_pp": observed, "image_cluster_bootstrap_95_pp": interval,
            "questions": len(base), "unique_image_clusters": len(images),
            "metric": "subcategory macro mean" if task == "mmstar" else "question mean",
            "bootstrap_replicates": replicates, "bootstrap_seed": seed,
            "empty_category_replicates_redrawn": missing_category_replicates,
            "interval_scope": SCOPE}


def make_figures(output, summaries, records, contrast_summaries, paired, font_path, hashes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties, fontManager

    fontManager.addfont(str(font_path))
    font_name = FontProperties(fname=str(font_path)).get_name()
    colors = {"qv_r8": "#0072B2", "qkvo_r4": "#8C4E00", "qv_r4_merger": "#007A5E"}
    seed_markers = {0: "o", 42: "s", 200: "^"}
    labels = ["q/v r8\n10k QA", "q/k/v/o r4\n10k QA", "q/v r4+merger\n10k QA", "q/v r8\n32,985 QA", "q/v r4+merger\n32,985 QA"]
    keys = [("stage1", "qv_r8"), ("stage1", "qkvo_r4"), ("stage1", "qv_r4_merger"), ("stage2", "qv_r8"), ("stage2", "qv_r4_merger")]
    lookup = {(r["stage"], r["method"], r["task"]): r for r in summaries}
    score_lookup = {(r["stage"], r["method"], r["seed"], r["task"]): r["score_percent"] for r in records}
    figure_files = []
    with plt.rc_context({"font.family": font_name, "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "path", "figure.facecolor": "white",
                         "savefig.facecolor": "white", "axes.labelcolor": "#222222", "text.color": "#222222"}):
        fig, axes = plt.subplots(2, 2, figsize=(13, 8.3), layout="constrained")
        limits = {"dev": (72.8, 75.8), "textvqa": (68, 74), "docvqa": (68, 71), "mmstar": (51.5, 56.5)}
        for ax, task in zip(axes.flat, TASKS):
            for i, (stage, method) in enumerate(keys):
                row = lookup[(stage, method, task)]
                ax.errorbar(i, row["mean_score_percent"], yerr=row["seed_sd_pp"], fmt="D", color="#222222",
                            markersize=5, capsize=5, linewidth=1.6, zorder=2)
                for offset, seed in zip((-.14, 0, .14), SEEDS):
                    ax.scatter(i + offset, score_lookup[(stage, method, seed, task)], marker=seed_markers[seed],
                               s=43, facecolors=colors[method], edgecolors="white", linewidths=.55, zorder=3)
                label_height = max(row["seed_scores_percent"] + [row["mean_score_percent"] + row["seed_sd_pp"]])
                ax.annotate(f'{row["mean_score_percent"]:.3f}', (i, label_height),
                            xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)
            baseline = lookup[("stage2", "qv_r8", task)]["baseline_score_percent"]
            if baseline is not None:
                ax.axhline(baseline, color="#555555", linestyle="--", linewidth=1.1)
                ax.text(4.45, baseline, f"  NF4底座 {baseline:.3f}", va="center", ha="right", fontsize=8,
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85})
            ax.set_xticks(range(len(keys)), labels)
            ax.set_ylim(*limits[task]); ax.set_xlim(-.5, 4.7)
            ax.set_title(TASK_LABELS[task], loc="left", weight="bold")
            ax.set_ylabel("得分（%）"); ax.grid(axis="y", color="#DDDDDD", linewidth=.6)
        for seed in SEEDS:
            axes[0, 0].scatter([], [], marker=seed_markers[seed], color="#555555", label=f"seed {seed}")
        axes[0, 0].errorbar([], [], yerr=[], fmt="D", color="#222222", label="均值 ± seed SD")
        axes[0, 0].legend(loc="lower left", ncol=2, fontsize=8, frameon=False)
        fig.suptitle("完整三种子结果：数据扩展有收益，跨任务表现不能只看 TextVQA", fontsize=16, weight="bold")
        fig.supxlabel("每点一个完整训练；误差为3个训练种子的样本SD，不是CI。各任务指标/坐标范围不同。\n"
                      "阶段2 q/v seed0 在V100训练，其余在A100；阶段1均为V100。公开验证集，预训练重叠未知。", fontsize=9)
        for ext in ("png", "svg"):
            path = output / f"full_results.{ext}"; fig.savefig(path, dpi=180); figure_files.append(path)
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), layout="constrained")
        delta_lookup = {(r["contrast"], r["task"]): r for r in contrast_summaries}
        for ax, task in zip(axes, ("textvqa", "docvqa", "mmstar")):
            for i, method in enumerate(("qv_r8", "qv_r4_merger")):
                contrast = f"{method}_stage2_minus_stage1"
                row = delta_lookup[(contrast, task)]
                ax.errorbar(i, row["mean_seed_delta_pp"], yerr=row["seed_delta_sd_pp"], fmt="D", color="#222222", capsize=5)
                for offset, seed in zip((-.12, 0, .12), SEEDS):
                    val = next(p["delta_pp"] for p in paired if p["contrast"] == contrast and p["task"] == task and p["seed"] == seed)
                    ax.scatter(i + offset, val, marker=seed_markers[seed], color=colors[method], s=45)
                label_height = max(row["seed_deltas_pp"] + [row["mean_seed_delta_pp"] + row["seed_delta_sd_pp"]])
                ax.annotate(f'{row["mean_seed_delta_pp"]:+.3f} pp', (i, label_height),
                            xytext=(0, 10), textcoords="offset points", ha="center", fontsize=10)
            ax.axhline(0, color="#555555", linestyle="--", linewidth=1)
            ax.set_xticks((0, 1), ("q/v rank8", "q/v rank4 + merger"))
            ax.set_ylim(-1.4, 2.5); ax.set_xlim(-.5, 1.5); ax.set_ylabel("阶段2 − 阶段1（百分点）")
            ax.set_title(TASK_LABELS[task], loc="left", weight="bold"); ax.grid(axis="y", color="#DDDDDD", linewidth=.6)
        fig.suptitle("10k → 32,985 QA：更多样本同时带来更多更新，不能单独归因于数据量", fontsize=15, weight="bold")
        fig.supxlabel("同方法、同名seed配对；三个原始种子点 + 均值±seed差值SD。硬件/布局随阶段变化，非纯数据量因果实验。", fontsize=9)
        for ext in ("png", "svg"):
            path = output / f"data_expansion_deltas.{ext}"; fig.savefig(path, dpi=180); figure_files.append(path)
        plt.close(fig)
    for path in figure_files:
        hashes[str(path.resolve())] = sha(path)
    manifest = {"audience": "project learning and review", "journal_submission": None,
                "raw_source": "immutable logical run index + completed EVALUATION/TRAIN_RESULT files",
                "data_table": "core_seed_metrics.csv and paired_contrast_seed_summary.csv",
                "uncertainty": "sample standard deviation across seeds {0,42,200}; not confidence intervals",
                "raw_points": "one marker per complete training seed; paired deltas never pool seed×question",
                "transformations": ["arithmetic mean across three seed-level metrics", "sample SD, ddof=1", "stage2-minus-stage1 seed-level differences"],
                "image_processing": "none; vector statistical plots only", "font": str(font_path),
                "software": {"numpy": np.__version__, "matplotlib": matplotlib.__version__},
                "svg_font_policy": "glyph paths for portable Chinese rendering",
                "output_files_sha256": {str(p.name): sha(p) for p in figure_files},
                "alt_text": "Four panels show internal dev, TextVQA, DocVQA and MMStar for five stage/method groups with three seed markers, means and seed SD. TextVQA improves with more data; DocVQA changes little; stage2 merger has higher but variable MMStar scores. The second figure shows the three task-specific paired seed changes from stage1 to stage2."}
    write_json(output / "FIGURE_PROVENANCE.json", manifest)
    return [str(p.resolve()) for p in figure_files]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_RUN_ROOT.parent / "final_analysis")
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261011)
    parser.add_argument("--font", type=Path, default=WORKSPACE / ".tools/fonts/NotoSansCJKsc-Regular.otf")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(WORKSPACE) or output.is_relative_to(args.run_root.resolve()):
        raise ValueError("analysis outputs must remain in persistent workspace outside original run tree")
    if args.replicates < 100:
        raise ValueError("insufficient bootstrap replicates")
    output.mkdir(parents=True, exist_ok=True)
    hashes = {}; root = args.run_root
    index = read_bound(root / "RUN_INDEX.json", hashes)
    summary = read_bound(root / "SUMMARY.json", hashes)
    verification = read_bound(root / "VERIFICATION.json", hashes)
    if not verification["passed"] or index["missing"] != {"trainings": [], "evaluations": []}:
        raise ValueError("original run not complete/verified")
    records = []; predictions = {}; trainings = {}; eval_index = {}
    for item in index["evaluations"]:
        if item["stage"] in ("stage1", "stage2"):
            eval_index[(item["stage"], item["method"], item["seed"], item["task"])] = item
    for item in index["trainings"]:
        path = Path(item["path"]); train = read_bound(path, hashes)
        key = (item["stage"], item["method"], item["seed"])
        trainings[key] = train
        for task in TASKS:
            logical = eval_index[(*key, task)]; evaluation_path = Path(logical["path"])
            evaluation = read_bound(evaluation_path, hashes)
            prediction_path = evaluation_path.parent / "predictions.json"
            rows = read_bound(prediction_path, hashes)
            if sha(prediction_path) != evaluation["predictions_sha256"]:
                raise ValueError("prediction content differs from completed evaluation binding")
            score = metric(rows, task)
            if len(rows) != evaluation["n"] or not math.isclose(score, evaluation["score_percent"], abs_tol=1e-9):
                raise ValueError("analysis aggregation differs from completed metric")
            predictions[(*key, task)] = rows
            records.append({"stage": key[0], "method": key[1], "seed": key[2], "task": task,
                "score_percent": score, "training_questions": train["question_count"],
                "unique_training_images": train["unique_training_images"], "updates": train["completed_updates"],
                "trainable_parameters": train["trainable_parameters"], "parameter_groups": train["parameter_groups"],
                "internal_dev_token_nll": train["internal_dev_token_nll"], "training_input_tokens": train["training_input_tokens"],
                "training_answer_tokens": train["training_answer_tokens"], "effective_batch": train["effective_batch"],
                "gpu_name": train["gpu_name"], "world_size": train["world_size"], "microbatch": train["microbatch"],
                "questions": len(rows), "unique_image_clusters": len({r["image_id"] for r in rows}),
                "training_path": str(path), "evaluation_path": str(evaluation_path), "predictions_path": str(prediction_path)})
    grouped = defaultdict(list)
    for row in records:
        grouped[(row["stage"], row["method"], row["task"])].append(row)
    summaries = []
    for (stage, method, task), members in sorted(grouped.items()):
        members = sorted(members, key=lambda r: r["seed"])
        if tuple(r["seed"] for r in members) != SEEDS:
            raise ValueError("expected exactly three training seeds")
        values = [r["score_percent"] for r in members]
        baseline = summary["baseline_metrics"].get(task, {}).get("score_percent")
        summaries.append({"stage": stage, "method": method, "task": task, "seeds": list(SEEDS),
            "seed_scores_percent": values, "mean_score_percent": statistics.mean(values), "seed_sd_pp": statistics.stdev(values),
            "baseline_score_percent": baseline, "mean_minus_nf4_baseline_pp": statistics.mean(values) - baseline if baseline is not None else None,
            "train_hardware_by_seed": {r["seed"]: r["gpu_name"] for r in members}})
    contrast_specs = [
        ("stage2_merger_minus_qv", ("stage2", "qv_r8"), ("stage2", "qv_r4_merger")),
        ("qv_r8_stage2_minus_stage1", ("stage1", "qv_r8"), ("stage2", "qv_r8")),
        ("qv_r4_merger_stage2_minus_stage1", ("stage1", "qv_r4_merger"), ("stage2", "qv_r4_merger")),
    ]
    paired = []
    for contrast, base_key, adapted_key in contrast_specs:
        for task in TASKS:
            for seed in SEEDS:
                base_train = trainings[(*base_key, seed)]; adapted_train = trainings[(*adapted_key, seed)]
                interval = paired_clusters(predictions[(*base_key, seed, task)], predictions[(*adapted_key, seed, task)],
                                            task, args.replicates, args.bootstrap_seed)
                paired.append({"contrast": contrast, "task": task, "seed": seed,
                    "base_stage": base_key[0], "base_method": base_key[1], "adapted_stage": adapted_key[0], "adapted_method": adapted_key[1],
                    "base_gpu": base_train["gpu_name"], "adapted_gpu": adapted_train["gpu_name"],
                    "training_hardware_changed": (base_train["gpu_name"], base_train["world_size"], base_train["microbatch"]) !=
                                                 (adapted_train["gpu_name"], adapted_train["world_size"], adapted_train["microbatch"]),
                    **interval})
    contrast_summaries = []
    for contrast, _, _ in contrast_specs:
        for task in TASKS:
            members = [r for r in paired if r["contrast"] == contrast and r["task"] == task]
            deltas = [r["delta_pp"] for r in members]
            contrast_summaries.append({"contrast": contrast, "task": task, "seeds": list(SEEDS), "seed_deltas_pp": deltas,
                "mean_seed_delta_pp": statistics.mean(deltas), "seed_delta_sd_pp": statistics.stdev(deltas),
                "positive_seed_count": sum(d > 0 for d in deltas), "negative_seed_count": sum(d < 0 for d in deltas),
                "scope": SCOPE})
    warnings = [
        "stage2 qv_r8 seed0 was trained on 4xV100; qv_r8 seeds42/200 and all merger runs were trained on 2xA100. Small differences cannot be attributed only to merger placement.",
        "Equal total trainable parameter counts compare budget allocation, not merger-only ablation: merger strategy also reduces language rank from8 to4.",
        "10k-to-32985 is a one-epoch scale change, with 625-to2062 updates and more token exposure. It does not isolate dataset size from compute, diversity or GPU layout.",
        "Stage1 selected two methods by generated internal dev mean only. Final public validation results are post-hoc evaluation; further hypotheses must be tested on an independently declared protocol.",
        "MMStar uses subcategory macro averaging, not overall pooled question accuracy; its bootstrap preserves this metric.",
        "Current experiments are TextVQA task adaptation, not yet a validated long-tail class benchmark or a mechanism proof for rare strings.",
        "Any answer-frequency slice of full training must recompute frequencies from train_all; earlier train_10k frequencies do not define full-training rarity.",
        "Public validation data may overlap pretraining, which is unknown. No claim of official held-out test/SOTA is justified.",
    ]
    write_csv(output / "core_seed_metrics.csv", records)
    write_csv(output / "core_task_summary.csv", summaries)
    write_csv(output / "paired_image_cluster_contrasts.csv", paired)
    write_csv(output / "paired_contrast_seed_summary.csv", contrast_summaries)
    figure_paths = make_figures(output, summaries, records, contrast_summaries, paired, args.font, hashes) if not args.no_figures else []
    analysis = {"schema_version": "rarevlm.final_analysis.v1", "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "completed_at": summary["status"]["updated_at"], "verification_reused": {k: verification[k] for k in ("passed", "completed_training_runs", "completed_evaluations", "stored_predictions_rescored")},
        "scope": SCOPE, "bootstrap": {"replicates": args.replicates, "seed": args.bootstrap_seed, "cluster_unit": "image", "seed_pooling": False},
        "numerical_path_scores_percent": {r["condition"]: r["score_percent"] for r in summary["numerical_path_matrix"]},
        "summaries": summaries, "paired_contrasts": paired, "paired_seed_summaries": contrast_summaries,
        "warnings": warnings, "backbone_recommendation": "Continue controlled depth on the current Qwen2.5-VL-3B backbone. Existing gains and task-specific transfer patterns do not establish a backbone capacity ceiling. Treat a newer/larger backbone as a separately budgeted transfer experiment after a reproducible visual-budget/data intervention.",
        "sources_sha256": hashes, "analysis_script_sha256": sha(Path(__file__)), "figures": figure_paths,
        "scientific_visualization_reference": {"authors": "Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel, Aubrey M. Brueckner", "year": 2026,
            "title": "Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents", "url": "https://arxiv.org/abs/2609.00065", "accessed": "2026-10-11", "current_record": "v2 revised2026-09-02"}}
    write_json(output / "FINAL_ANALYSIS.json", analysis)
    lines = ["本分析复用15次训练、67项评测的已通过核验记录，未重新训练/生成/评分标签。", "", "均值±SD为3个训练种子的描述统计；固定checkpoint的同图配对区间另见CSV/JSON。", ""]
    for row in summaries:
        if row["task"] != "dev":
            lines.append(f'{row["stage"]} {METHOD_LABELS[row["method"]]} {row["task"]}: {row["mean_score_percent"]:.3f} ± {row["seed_sd_pp"]:.3f}；对NF4底座 {row["mean_minus_nf4_baseline_pp"]:+.3f} pp')
    lines += ["", "核心配对（逐seed同图比较；阶段变化包含硬件/更新次数差异）："]
    for row in contrast_summaries:
        lines.append(f'{row["contrast"]} {row["task"]}: mean {row["mean_seed_delta_pp"]:+.3f} pp，seed差值SD {row["seed_delta_sd_pp"]:.3f}，逐seed {[round(v, 3) for v in row["seed_deltas_pp"]]}')
    lines += ["", "建议：继续深化当前底座；下一步以视觉预算/输入干预检验细粒度信息瓶颈。当前证据未显示必须换底座。",
              "必须限制结论：数据扩展同时增加更新数；merger策略同时改变语言rank；阶段2 q/v seed0为V100，其余为A100；公开验证集预训练重叠未知。",
              "此前按10k统计的答案频次不用于全量训练的长尾定义。MMStar的小幅变化仅作能力保持观察，不能宣布普遍遗忘或恢复。", "",
              "绘图规范参考：Kassis et al. (2026), Scientific Agent Skills, https://arxiv.org/abs/2609.00065（2026-10-11核验当前记录）。"]
    (output / "FINDINGS_ZH.txt").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(output), "seed_task_records": len(records), "paired_image_cluster_intervals": len(paired),
                      "bootstrap_replicates": args.replicates, "figures": figure_paths,
                      "all_analysis_metrics_match_completed_evaluations": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
