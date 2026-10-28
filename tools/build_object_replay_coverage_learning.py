#!/usr/bin/env python3
"""Render replay-coverage learning from JSON metadata only; never run a model."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUTPUTS = REPO.parents[1] / "outputs"
ROOT = OUTPUTS / "vlm_object_replay_coverage_v1"
CONFIG = REPO / "configs/vlm_object_replay_coverage_v1.json"
PROTOCOL = REPO / "docs/VLM_OBJECT_REPLAY_COVERAGE_V1_PROTOCOL.md"
ARMS = ("wide32", "repeat8x4")


def read(path):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {"read_status": "writing_or_unavailable", "path": str(path)}


def esc(value):
    return html.escape(str(value), quote=True)


def fmt(value):
    if value is None:
        return "待完成 / 未定义"
    if isinstance(value, float):
        return f"{value:.8g}"
    return esc(value)


def link(path, output, label=None):
    path = Path(path).resolve()
    if not path.exists():
        return "<span class='pending'>待保存：" + esc(path) + "</span>"
    return (f'<a href="{esc(os.path.relpath(path, output.parent))}">{esc(label or path.name)}</a>'
            f' <small>（相对链接 / <a href="{esc(path)}">绝对路径</a>）</small>')


def details(title, value):
    return "<details><summary>" + esc(title) + "</summary><pre>" + esc(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)) + "</pre></details>"


def table(headers, rows):
    return "<div class='scroll'><table><tr>" + "".join(f"<th>{esc(h)}</th>" for h in headers) + "</tr>" + "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows) + "</table></div>"


def fields(value, names):
    return {key: value[key] for key in names if key in value}


def source(path, output, needle):
    path = REPO / path
    if not path.exists():
        return link(path, output)
    lines = path.read_text().splitlines()
    line = next((i + 1 for i, text in enumerate(lines) if needle in text), 1)
    return link(path, output, f"{path.relative_to(REPO)}:{line}") + "<code>" + esc(f"{path}:{line}") + "</code>"


def first_update_panel(root, output):
    """Read the first actual formal update only; never derive probabilities."""
    value = "<h2>本轮首个正式更新：真实训练输入与QA分母</h2>"
    value += "<p>只读runs/{arm}/seed42/train/FIRST_UPDATE.json，不以工程预检替代正式更新。每条记录是一回训练曝光；同qid重复出现仍只有一个unique QA。文件表示该首步已保存，不证明181步完成或性能保持。</p>"
    summaries = {}
    for arm in ARMS:
        path = root / f"runs/{arm}/seed42/train/FIRST_UPDATE.json"
        first = read(path)
        value += "<h3>" + esc(arm) + " / seed42</h3>"
        if (not first or first.get("completed_updates") != 1
                or first.get("scope") != "actual first formal GPU update encoded inputs"
                or not isinstance(first.get("records"), list)):
            value += "<p class='pending'>准备中：本臂FIRST_UPDATE尚未完整保存；不补造实际题数、token或概率。</p>"
            summaries[arm] = {"status": "pending"}
            continue
        records = first["records"]
        rows, summary = [], {}
        for origin in ("object", "replay"):
            selected = [row for row in records if row.get("source") == origin]
            counts = Counter(str(row["question_id"]) for row in selected)
            multiplicity = dict(sorted(Counter(counts.values()).items()))
            actual_mask_tokens = [sum(label != -100 for label in row.get("labels", [])) for row in selected]
            summary[origin] = {"recorded_exposures": len(selected), "distinct_question_ids": len(counts),
                "distinct_image_ids": len({str(row["image_id"]) for row in selected}),
                "repetition_count_to_unique_QA_count": multiplicity,
                "actual_assistant_mask_tokens_across_exposures": sum(actual_mask_tokens),
                "all_saved_assistant_counts_match_labels": all(n == row.get("assistant_tokens") for n, row in zip(actual_mask_tokens, selected))}
            rows.append([origin, str(first.get(origin + "_count")), str(len(selected)), str(len(counts)),
                esc("；".join(f"{times}次×{n}题" for times, n in multiplicity.items())),
                str(sum(actual_mask_tokens)), esc(summary[origin]["all_saved_assistant_counts_match_labels"])])
        summaries[arm] = {"status": "saved_actual_first_formal_update", "total_records": len(records),
            "distinct_source_question_pairs": len({(row["source"], str(row["question_id"])) for row in records}),
            "sources": summary}
        value += table(["来源", "FIRST_UPDATE声明题数", "实际曝光记录", "distinct QA", "每题出现次数", "有效assistant token合计（含重复）", "与保存count匹配"], rows)
        value += details("首步实际范围", {"scope": first["scope"], "completed_updates": first["completed_updates"],
            "learning_rate": first.get("learning_rate"), "gradient_l2_before_clip": first.get("gradient_l2_before_clip"),
            "total_exposure_records": len(records), "distinct_source_question_pairs": summaries[arm]["distinct_source_question_pairs"]})
        value += "<div class='cases'>"
        for origin in ("object", "replay"):
            row = next((row for row in records if row.get("source") == origin), None)
            if row is None:
                continue
            labels, ids = row.get("labels", []), row.get("input_ids", [])
            positions = [i for i, label in enumerate(labels) if label != -100]
            start = max(0, positions[0] - 8) if positions else max(0, len(ids) - 16)
            stop = min(len(ids), (positions[0] + 8) if positions else len(ids))
            context = [{"position": i, "input_token_id": ids[i], "label": labels[i] if i < len(labels) else None,
                "assistant_supervised": i < len(labels) and labels[i] != -100,
                "causal_predicting_logit_position": i - 1 if i < len(labels) and labels[i] != -100 else None}
                for i in range(start, stop)]
            image_path = Path(row["image_path"])
            value += "<article><h4>本轮 " + esc(arm + "/" + origin) + " 首题</h4>"
            value += f"<img loading='lazy' src='{esc(os.path.relpath(image_path, output.parent))}' alt='本轮首正式更新实际训练原图'>"
            value += "<p>原问题：" + esc(row["question"]) + "</p><p>训练target：<strong>" + esc(row["target"]) + "</strong></p>"
            value += details("原提示后缀", row.get("prompt_suffix"))
            value += details("实际mask/token与输入片段（无重编码）", {
                **fields(row, ("question_id", "image_id", "input_tokens", "sequence_length", "visual_tokens", "assistant_tokens",
                    "truncated", "microbatch_input_sha256", "microbatch_question_position", "microbatch_size", "microbatch_sequence_length",
                    "image_grid_thw", "pixel_values_shape")),
                "actual_padded_input_ID_count": len(ids), "actual_label_count": len(labels),
                "actual_non_minus100_assistant_mask_count": len(positions),
                "actual_masked_prompt_or_padding_count": len(labels) - len(positions),
                "actual_supervised_label_positions": positions,
                "actual_supervised_label_token_IDs": [labels[i] for i in positions],
                "actual_input_ID_head16": ids[:16], "actual_input_ID_tail24": ids[-24:],
                "actual_assistant_boundary_context": context})
            value += link(image_path, output, "完整原图") + "</article>"
        value += "</div>" + link(path, output, "首个正式更新原始JSON")
    value += "<p>上述target进入训练teacher forcing/监督labels；它不是模型生成答案，也不进入生成评测的用户问题输入。label=−100的prompt/padding不计loss，label位置t由logits位置t−1预测。原图仅等比展示，无crop/伪注意力。本记录未保存学生逐token概率，页面不从target、平均loss或梯度范数伪造学生概率。</p>"
    return value, summaries


def completed_prediction_rows(prediction_path, evaluation_path):
    """Display reader only: no scoring, hashes, input encoding or cache repair."""
    evaluation = read(evaluation_path)
    if not evaluation or evaluation.get("status") != "completed":
        return None, "完整EVALUATION尚未保存"
    path = Path(prediction_path)
    if not path.is_file():
        return None, "保存预测文件缺失"
    try:
        with path.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        expected = evaluation.get("questions", evaluation.get("n"))
        if expected is not None and len(rows) != expected:
            return None, "完整EVALUATION与预测条数不一致，等待核对"
        if any(not all(key in row for key in ("question_id", "image_id", "prediction", "score")) for row in rows):
            return None, "预测展示必需字段缺失"
        index = {str(row["question_id"]): row for row in rows}
        if len(index) != len(rows):
            return None, "预测qid重复，等待核对"
        return index, None
    except (OSError, ValueError, TypeError):
        return None, "预测暂不可解析，不修复原文件"


def case_metadata(config, task):
    if task == "object":
        path = Path(config["val_manifest"])
    elif task == "text":
        path = Path(config["replay_manifest"])
    elif task == "external":
        path = Path(config["external_manifest"])
    else:
        path = Path(config["pope_data_root"]) / "manifest.json"
    document = read(path)
    if not document or "images" not in document:
        return None
    rows = document.get("splits", {}).get("dev") if task == "text" else document.get("rows", document.get("records"))
    if not isinstance(rows, list):
        return None
    return {"path": path, "rows": {str(row["question_id"]): row for row in rows}, "images": document["images"]}


def prediction_cases(config, root, output):
    value = "<h2>完整预测后的真实得分变化病例（seed42）</h2>"
    value += "<p>每个arm、每个任务独立选取：相对本轮initial历史引用，保存score不等的qid按str(question_id)字典序排序，取最先8条（不足取全部）。这是固定顺序的非随机展示，不代表错误比例或人工精选改进。只读取完整EVALUATION后的保存预测，绝不重评分、自动语义归因或改GT。</p>"
    summaries = {}
    for task in ("object", "text", "external", "pope"):
        arms = ARMS if task in ("object", "text") else ("wide32",)
        saved, reasons = {}, {}
        for arm in arms:
            directory = root / f"runs/{arm}/seed42/{task}"
            # Do not even read a partial predictions file before its completed marker.
            saved[arm], reasons[arm] = completed_prediction_rows(directory / "predictions.jsonl", directory / "EVALUATION.json")
        if not any(rows is not None for rows in saved.values()):
            for arm in arms:
                summaries[arm + "/" + task] = {"status": "pending"}
                value += "<p class='pending'>" + esc(arm + "/" + task) + "：准备中，完整结果未齐，不展示部分预测病例。</p>"
            if task in ("external", "pope"):
                value += "<p>repeat8x4 / " + esc(task) + "：未预注册为确认任务，不视作待补跑。</p>"
            continue
        reference_path = root / f"runs/initial/seed42/{task}/HISTORICAL_REFERENCE.json"
        declaration = read(reference_path) or {}
        provenance = declaration.get("source_data", {})
        initial, initial_reason = (completed_prediction_rows(provenance["predictions_file"], provenance["evaluation_file"])
            if declaration.get("schema_version") == "rarevlm.object_replay_coverage.historical_reference.v1"
            and provenance.get("predictions_file") and provenance.get("evaluation_file") else (None, "本轮initial历史引用尚未就绪"))
        metadata = case_metadata(config, task) if initial is not None else None
        for arm in arms:
            value += "<h3>" + esc(arm + "/" + task) + "：各自固定qid首8条score变化</h3>"
            rows = saved[arm]
            reason = reasons[arm] if rows is None else initial_reason if initial is None else "原图/参考清单尚未就绪" if metadata is None else None
            if reason:
                value += "<p class='pending'>准备中：" + esc(reason) + "。</p>"
                summaries[arm + "/" + task] = {"status": "pending"}
                continue
            if set(rows) != set(initial):
                value += "<p class='pending'>初始与本臂完整qid集合不一致，等待核对；不按交集静默选例。</p>"
                summaries[arm + "/" + task] = {"status": "qid_mismatch"}
                continue
            changed = sorted(qid for qid in rows if rows[qid]["score"] != initial[qid]["score"])
            selected = changed[:8]
            summaries[arm + "/" + task] = {"status": "complete_saved_predictions", "score_changed_QA": len(changed), "selected_qids": selected}
            value += "<p>本臂本任务共" + str(len(changed)) + "条保存score变化；本节展示" + str(len(selected)) + "条。未重新执行解析器或计分器。</p>"
            value += link(reference_path, output, "本轮initial历史引用") + " · " + link(root / f"runs/{arm}/seed42/{task}/predictions.jsonl", output, "本臂完整保存预测")
            value += "<div class='cases'>"
            for qid in selected:
                ref = metadata["rows"].get(qid)
                before, after = initial[qid], rows[qid]
                if not ref or str(ref["image_id"]) != str(before["image_id"]) or str(ref["image_id"]) != str(after["image_id"]):
                    value += "<article><p class='pending'>qid " + esc(qid) + "：问题/图像绑定缺失，待核对。</p></article>"
                    continue
                image = metadata["images"].get(ref["image_id"], {})
                raw_path = image.get("path") or ref.get("image_path")
                image_path = Path(raw_path) if raw_path else None
                if image_path and not image_path.is_absolute():
                    image_path = metadata["path"].parent / image_path
                value += "<article><h4>" + esc(arm + "/" + task + " qid=" + qid) + "</h4>"
                if image_path and image_path.is_file():
                    value += f"<img loading='lazy' src='{esc(os.path.relpath(image_path, output.parent))}' alt='保存预测对应真实原图'>"
                else:
                    value += "<p class='pending'>原图路径尚未就绪，不生成替代图片。</p>"
                value += "<p>原问题：" + esc(ref["question"]) + "</p>"
                answers = []
                for name in ("initial", "repeat8x4", "wide32"):
                    record = initial.get(qid) if name == "initial" else (saved.get(name) or {}).get(qid)
                    if record is None:
                        label = "未预注册此确认任务" if name == "repeat8x4" and task in ("external", "pope") else "完整保存记录待完成"
                        answers.append([name, label, "待完成", "待完成", "待完成"])
                    elif str(record["image_id"]) != str(ref["image_id"]):
                        answers.append([name, "图像绑定待核对", "待完成", "待完成", "待完成"])
                    else:
                        category = record.get("category")
                        answers.append([name, esc(record["prediction"]), fmt(record["score"]), fmt(record.get("visual_tokens")), esc(category) if category not in (None, "") else "Text无类别 / 未保存"])
                value += table(["真实保存模型", "原文回答", "原官方score", "实际visual tokens", "保存类别"], answers)
                reference = {"answers": ref.get("answers")} if task == "text" else {"label_or_target": ref.get("target", ref.get("label")), "category": ref.get("category_name", ref.get("category")), "category_id": ref.get("category_id")}
                value += details("原10参考共识来源（不改评分）" if task == "text" else "原二元GT/类别（非伪造人类参考）", reference)
                value += "<p>score变化=" + fmt(after["score"] - before["score"]) + "；仅保存分数差，不自动贴语义标签。</p>"
                if image_path:
                    value += link(image_path, output, "完整原图")
                value += "</article>"
            value += "</div>"
        if task in ("external", "pope"):
            value += "<p>repeat8x4确认任务未预注册；该列不借用旧CE/KL预测填充。</p>"
    value += "<p>表中的参考只用于生成后计分与展示，不进入生成用户输入。原图等比显示，无crop、重建输入或伪注意力；图片仅为本节保存qid对应原图，未作新视觉语义审查。此面板限seed42，三seed汇总及正式门槛仍看完整官方ANALYSIS。</p>"
    return value, summaries


def old_kl_prefix_panel(root, output):
    """Display the existing historical receipt, without rereading old predictions."""
    path = root / "independent_review/OLD_KL_PREFIX_CHANGE_REVIEW.json"
    receipt = read(path)
    value = "<h2>历史旧KL选定8题：保存token的分歧位置</h2>"
    if (not receipt or receipt.get("schema_version") != "rarevlm.object_replay_coverage.old_kl_prefix_change_review.v1"
            or receipt.get("status") != "completed_descriptive_review"):
        return value + "<p class='pending'>历史token分歧收据尚未就绪，不填假设IDs或位置。</p>"
    summary = receipt["summary"]
    first = summary["first_answer_token_changed"]
    endings = summary["initial_answer_ended"] + summary["old_kl_final_answer_ended"]
    later = summary["different_content_token"] - first
    value += "<p class='badge'>历史旧KL的选定8题，非当前wide32或repeat8x4结果；不能作因果归因或全量比例。这里只读已完成CPU收据，不重读旧预测、分词器、模型或概率，不重评分。</p>"
    value += "<p>收据汇总：<strong>" + str(first) + "题首token分歧、" + str(endings) + "题结束位置分歧、" + str(later) + "题后续内容分歧</strong>。首token不同发生在尚未输出答案内容时；结束位置分歧是共同内容前缀后有一方先结束；后续内容分歧发生在共同前缀之后。这些只描述分歧位置，不证明前缀漂移、覆盖不足或遗忘的原因。</p>"
    rows = []
    for case in receipt["cases"]:
        initial, old = case["initial"], case["old_kl_final"]
        difference = case["first_difference"]
        raw = case["raw_generated_suffix_first_difference"]
        if case["first_answer_token_changed"]:
            kind = "首个内容token不同"
        elif difference["kind"] == "initial_answer_ended":
            kind = "initial先结束，旧KL继续"
        elif difference["kind"] == "old_kl_final_answer_ended":
            kind = "旧KL先结束"
        else:
            kind = "后续内容token不同"
        def saved_answer(item):
            return ("<p>" + esc(item["prediction_raw"]) + "</p><p>原score=" + esc(item["saved_official_score"])
                + "</p><code>" + esc(item["generated_ids"]) + "</code>")
        rows.append(["<span data-old-kl-qid='" + esc(case["question_id"]) + "'>" + esc(case["question_id"]) + "</span>",
            kind, "<span data-prefix-position='" + str(difference["position_0_based"]) + "'>" + str(difference["position_0_based"]) + "</span>",
            str(case["longest_common_answer_prefix_length"]),
            esc(str(raw["initial_token_id"]) + " → " + str(raw["old_kl_final_token_id"])),
            saved_answer(initial), saved_answer(old),
            esc(str(initial["terminal"]["suffix_index"]) + " → " + str(old["terminal"]["suffix_index"]))])
    value += table(["历史qid", "收据分歧类型", "首分歧位置（0-based）", "共同内容前缀token数", "保存suffix分歧IDs：initial→旧KL", "initial实际回答 / 原分数 / 保存IDs", "旧KL实际回答 / 原分数 / 保存IDs", "EOS在suffix中的0-based位置：initial→旧KL"], rows)
    verification = receipt["generated_suffix_verification"]
    value += "<p>位置相对实际保存的generated suffix，不含prompt；第一个答案token位置为0。上表完整generated_ids保留EOS与其后padding，未由回答文本重新分词。内容前缀比较仅按收据声明排除首个EOS及其后padding；原生成后缀并未修改。EOS IDs=" + esc(verification["eos_token_ids"]) + "，pad ID=" + esc(verification["pad_token_id"]) + "。没有保存学生概率或margin，不能由token ID或分歧位置反推概率变化。</p>"
    value += link(path, output, "历史token分歧原收据") + " · " + link(OUTPUTS / "vlm_object_kl_v1/learning/学习页.html", output, "原KL八病例原图、参考与官方分数")
    return value + "<p>这8题为既定旧KL得分变化题，并非随机样本；其中位置类型的4/3/1计数不能推广到1617题、其他seed或本轮wide32。原图判断、原评分和旧KL负结果均保持。</p>"


def first25_timing_panel(root, output):
    """Show the already reviewed timing receipt without recalibrating costs."""
    path = root / "independent_review/FIRST25_TIMING_REVIEW.json"
    receipt = read(path)
    value = "<h2>首25步实测计时：仅锚点后的active训练估计</h2>"
    if (not receipt or receipt.get("schema_version") != "rarevlm.object_replay_coverage.first25_timing_review.v1"
            or receipt.get("status") != "measured25step_training_timing_only"):
        return value + "<p class='pending'>准备中：首25步计时审查收据尚未保存。</p>"
    value += "<p class='badge'>本面板仅读取已完成收据，不重读checkpoint、不重算验收成本。0.8–1.3倍工程规划带是人为假设的操作余量，不是置信区间或实测方差。</p>"
    value += "<p>进度锚点UTC：<strong>" + esc(receipt["estimate_anchor_progress_updated_at"]) + "</strong>；收据审查UTC：" + esc(receipt["reviewed_at"]) + "。这些是当时快照，刷新本页不会滚动倒计时或据此推断当前进度。</p>"
    value += table(["已验收记录项", "收据原值", "范围"], [
        ["计时条件", esc(receipt["calibrated_condition"]), "仅这一臂已校准，不外推wide速度"],
        ["已完成 / 计划正式更新", str(receipt["calibration_completed_optimizer_updates"]) + " / " + str(receipt["planned_optimizer_updates"]), "不是teacher缓存批次"],
        ["锚点实际Object / Replay曝光", str(receipt["actual_object_QA_at25"]) + " / " + str(receipt["actual_replay_QA_at25"]), "训练曝光，非生成评测"],
        ["累计active训练秒", esc(receipt["actual_training_active_seconds_at25"]), "含该实现的输入/引用读取/H2D/F/B/clip/Adam，不是纯kernel"],
        ["实测平均active秒 / update", esc(receipt["measured_mean_active_seconds_per_update"]), "首25步算术均值，非后续恒速承诺"],
        ["锚点剩余正式更新", str(receipt["remaining_updates_from25"]), "只针对本臂训练更新"],
        ["剩余active训练估计秒", esc(receipt["estimated_remaining_training_active_seconds_from25"]), "从25步锚点算起，不是从本页当前时间算起"],
        ["工程规划范围秒（非CI）", esc(receipt["engineering_planning_range_seconds_from25"]), "0.8×–1.3×，不宣称统计覆盖概率"]])
    wide = receipt.get("wide_teacher_cache", {})
    value += details("同一收据中的wide快照（非当前实时读数）", fields(wide, (
        "completed_atomic_microbatches", "total_microbatches", "completed_teacher_QA", "total_teacher_QA",
        "cache_counts_are_optimizer_updates", "latest_saved_formal_progress", "manifest_frozen_exists")))
    value += "<p><strong>不是总pilot或整目标ETA。</strong>估计不包括wide剩余teacher构建、checkpoint I/O、完整评测、复制种子、外部/POPE确认或其他等待。序列/像素长度、缓存I/O和竞争会改变速度；只有control到25步，wide缓存进度不能等同正式训练更新，也不产生准确率或保持结论。</p>"
    return value + link(path, output, "首25步实测计时原收据")


def status_panel(root, output):
    value = "<h2>当前保存阶段与正式结果</h2>"
    complete = read(root / "COMPLETION.json")
    value += details("COMPLETION / STATUS（缺失为null，文件存在不等于进程存活）", {"completion": complete, "status": read(root / "STATUS.json")})
    if not complete:
        value += "<p class='pending'>准备中 / 执行中：尚无本轮COMPLETION，不预填性能结论。</p>"
    for arm, seed in (("wide32", 42), ("repeat8x4", 42), ("wide32", 0), ("wide32", 200)):
        directory = root / f"runs/{arm}/seed{seed}/train"
        result = read(directory / "TRAIN_RESULT.json")
        progress = read(directory / "PROGRESS.json")
        if seed != 42 and not result and not progress:
            continue
        value += f"<h3>{esc(arm)} / order seed{seed}</h3>"
        value += details("实际训练记录" if result else "准备中：训练结果未保存", fields(result, (
            "status", "completed_at", "completed_updates", "actual_questions_seen", "actual_object_exposures",
            "actual_replay_exposures", "replay_coverage", "mean_object_ce", "mean_replay_kl", "mean_replay_ce",
            "mean_replay_ce_scope", "teacher_source", "teacher_cache_unchanged", "frozen_parameters_unchanged",
            "training_active_seconds", "model_load_seconds", "teacher_cache_seconds_current_session",
            "preflight_seconds_current_session", "session_wall_seconds", "peak_gpu_allocated_gib",
            "peak_gpu_reserved_gib", "compute_accounting")) if result else {"progress": progress})
        if result:
            value += link(directory / "TRAIN_RESULT.json", output)
        teacher = read(directory / "TEACHER_SOURCE.json")
        value += details("实际TEACHER_SOURCE（缺失为待完成）", teacher)
        for task in ("object", "text", "external", "pope"):
            path = root / f"runs/{arm}/seed{seed}/{task}/EVALUATION.json"
            evaluation = read(path)
            if evaluation:
                value += details("完整评测记录：" + task, fields(evaluation, ("status", "questions", "unique_images", "score_percent", "metric", "completed_at", "resources", "current_session_wall_seconds"))) + link(path, output)
    paths = sorted((root / "analysis").glob("**/ANALYSIS.json")) if (root / "analysis").is_dir() else []
    if not paths:
        value += "<p class='pending'>本轮完整ANALYSIS尚未出现；旧KL预测不作为本轮结果。</p>"
    for path in paths:
        result = read(path) or {}
        value += details("分析：" + str(path.relative_to(root)), fields(result, (
            "status", "stage", "seeds", "updated_at", "candidate", "exploratory_control", "control_can_be_selected",
            "pending_cells", "gate", "eligible_candidates", "confirmation_passed", "goal_complete",
            "equal_teacher_compute_claim", "limitations"))) + link(path, output, "本轮完整ANALYSIS")
        if result.get("status") != "complete":
            continue
        rows = []
        for task, contrasts in result.get("contrasts", {}).items():
            for row in contrasts:
                if row["slice"] in ("overall", "rare_1_5", "frequent_gt20"):
                    rows.append([esc(task + "/" + row["slice"])] + [fmt(row.get(key)) for key in (
                        "before_score_percent", "after_score_percent", "delta_pp", "one_sided_98_75_lower_pp", "seed_delta_sample_sd_pp")])
        value += table(["wide32相对固定初始", "初始%", "训练后%", "差值pp", "98.75%下界pp", "seed差值样本SD pp"], rows)
        balanced = result.get("class_balanced")
        if balanced and balanced.get("status") == "complete":
            row = next(row for row in balanced["contrasts"] if row["slice"] == "overall")
            value += "<p>官方class_balanced：" + fmt(balanced["questions"]) + "QA / " + fmt(balanced["unique_images"]) + "图，逐类yes/no相等、QA-pooled，非类别macro；由完整预测过滤，不新增生成。</p>" + details("均衡accuracy结果", fields(row, ("before_score_percent", "after_score_percent", "delta_pp", "one_sided_98_75_lower_pp")))
        control = result.get("exploratory_control_contrasts")
        if control:
            control_rows = []
            for comparison, tasks in control.get("contrasts", {}).items():
                for task, contrasts in tasks.items():
                    for row in contrasts:
                        if row["slice"] in ("overall", "rare_1_5", "frequent_gt20"):
                            control_rows.append([esc(comparison), esc(task + "/" + row["slice"]), fmt(row.get("delta_pp")), fmt(row.get("image_cluster_bootstrap_95_pp"))])
            value += "<h3>pilot42探索性控制比较，不用于选择候选</h3>" + table(["比较", "端点", "差值pp", "探索95%CI pp"], control_rows)
    return value


def engineering_costs(root, output):
    value = "<h2>额外工程与教师成本：按实际保存字段分列</h2>"
    found = False
    for arm in ARMS:
        for scope, base in (("独立工程", root / "engineering_runs"), ("正式内置工程", root)):
            directory = base / f"runs/{arm}/seed42/train"
            path = directory / "PREFLIGHT.json"
            preflight = read(path)
            if not preflight:
                continue
            found = True
            value += details(arm + " / " + scope, fields(preflight, (
                "status", "completed_at", "reference_scope", "actual_new_teacher_forward_questions",
                "historical_teacher_bank_questions", "discarded_engineering_optimizer_updates", "formal_optimizer_updates",
                "initial_parameters_restored_exactly", "teacher_cache_unchanged", "record", "peak_allocated_gib", "peak_reserved_gib"))) + link(path, output)
    for path in sorted((root / "preflight").glob("**/PREFLIGHT.json")) if (root / "preflight").is_dir() else []:
        data = read(path) or {}
        found = True
        value += details("评测预检：" + str(path.relative_to(root)), {"status": data.get("status"),
            "saved_generated_QA_records": len(data.get("records", [])), "saved_generation_batches": len(data.get("batches", [])),
            "scope": "工程生成另计，不把正式预测QA数当整轮全部生成"}) + link(path, output)
    if not found:
        value += "<p class='pending'>准备中：尚无本轮实际PREFLIGHT或成本收据。</p>"
    return value + "<p>planned teacher成本不等：wide32需要新固定初始教师软目标；repeat8x4按已绑定旧完整词表缓存的重复引用消费。实际新教师forward QA、历史bank QA、秒、缓存字节及当前会话分别以上方TEACHER_SOURCE/TRAIN_RESULT为准；不能把历史生产算作新生成，或把复用写成没有历史成本。同5792回放曝光不等于同教师信息、同token、同FLOPs或同壁钟。</p>"


def old_fixed_images(output):
    root = OUTPUTS / "vlm_object_kl_gradient_v1"
    value = "<h2>沿用既有固定六张训练原图</h2><p>来源为旧梯度诊断checkpoint25/100/175各自既定block的object首题与replay首题，仅教学引用；不是本轮coverage的实际FIRST_UPDATE，也不是本轮新预测。未用dev选图、未重新看图作语义审查。整图等比展示，无生成图、crop或伪注意力。</p><div class='cases'>"
    count = 0
    for step, group in ((25, "early_late"), (100, "middle"), (175, "early_late")):
        metadata_path = root / f"groups/{group}/points/step_{step:06d}_block_{step:06d}/ACTUAL_INPUTS.json"
        records = read(metadata_path) or {}
        for source in ("object", "replay"):
            if not records.get(source):
                continue
            row = records[source][0]
            if not all(key in row for key in ("image_path", "question", "target")):
                continue
            path = Path(row["image_path"])
            value += f"<article><h3>旧固定block{step} / {esc(source)}首题</h3><img loading='lazy' src='{esc(os.path.relpath(path, output.parent))}' alt='既有固定训练原图'>"
            value += "<p>问题：" + esc(row["question"]) + "</p><p>训练target：" + esc(row["target"]) + "</p>"
            value += details("原实际输入记录字段", fields(row, ("question_id", "image_id", "assistant_tokens", "input_tokens", "visual_tokens", "image_grid_thw")))
            value += link(path, output, "原图") + " · " + link(metadata_path, output, "原固定block ACTUAL_INPUTS") + "</article>"
            count += 1
    return value + "</div>", count


CALCULATOR = r"""
<h2>每题独立均值：先token、再QA，最后两路相加</h2>
<pre>L = (1/B_object) Σ_i [(1/T_i) Σ_answer CE_i]
  + 1 × (1/32) Σ_32次replay曝光_j [(1/T_j) Σ_answer Σ_vocab q_initial·(log q_initial − log p_actor)]
正常B_object=16，末步实际B_object=8；replay分母两臂都为32。
若8题各重复4遍且每次同题loss相同：(4Σ_8题 l_j)/32 = (Σ_8题 l_j)/8。</pre>
<p>32次回放不与16个object混成一个48分母。重复题计为曝光，不变成新unique；固定θ且每次同题梯度相同的理想算术中，重复8题四遍的均值梯度也等于原8题均值，λ不会被放大四倍。原dropout0等固定条件下的代数等式，不宣称不同浮点累加顺序逐位相同；实际累加与训练轨迹差异仍需实测。长答案先按本题有效assistant token均值，不因token数多自动变成更高题权重。完整词表先求和，不再除词表大小；prompt/padding的label=−100不计，原结束token计入。</p>
<p class="badge">下面是假设的两题归一算例，不是本轮训练loss或新实验。</p>
<label>假设第1题有效token数 / token loss总和：<input id="t1" type="number" min="1" value="3"><input id="s1" type="number" min="0" step="0.1" value="0.3"></label>
<label>假设第2题有效token数 / token loss总和：<input id="t2" type="number" min="1" value="6"><input id="s2" type="number" min="0" step="0.1" value="1.2"></label>
<button id="calculate">计算假设每题均值</button><pre id="calculation"></pre>
<script>function calculate(){let t1=Number(document.getElementById('t1').value),t2=Number(document.getElementById('t2').value),s1=Number(document.getElementById('s1').value),s2=Number(document.getElementById('s2').value);let output=document.getElementById('calculation');if(![t1,t2].every(x=>Number.isInteger(x)&&x>0)||![s1,s2].every(x=>Number.isFinite(x)&&x>=0)){output.textContent='请输入正整数token数和非负有限loss总和。';return}let a=s1/t1,b=s2/t2;output.textContent='假设每题loss：'+a.toFixed(6)+' / '+b.toFixed(6)+'\n假设QA平均：'+((a+b)/2).toFixed(6)+'\n各复制4遍，仍按8次曝光平均：'+((4*a+4*b)/8).toFixed(6)+'\n这里不改变实际训练或λ=1。'}document.getElementById('calculate').onclick=calculate;calculate();</script>
<h2>GT prefix与greedy为什么要分开验证</h2>
<p>训练KL比较同一图像、问题以及原训练GT前缀下的完整词表分布；greedy生成会按模型自己上一步输出继续。既定训练前缀的平均KL小，不直接保证heldout输入或自己生成路径的短答保持。小分布变化可能改变argmax，但本页没有逐题学生margin证据，不把这一数学可能性认定为实际掉分的唯一原因。</p>
"""


def history(output):
    path = OUTPUTS / "vlm_object_kl_v1/analysis/pilot_seed42/screen/ANALYSIS.json"
    data = read(path)
    value = "<h2>动机来自已完成负结果，覆盖因果仍是假设</h2>"
    if data and data.get("status") == "complete":
        rows = []
        for task in ("object", "text"):
            for row in data["contrasts"][task]:
                if row["slice"] in ("overall", "rare_1_5", "frequent_gt20"):
                    rows.append([task + "/" + row["slice"], fmt(row["delta_pp"]), fmt(row["one_sided_98_75_lower_pp"])])
        value += table(["旧KL seed42端点", "差值pp", "98.75%下界pp"], rows)
    value += "<p>旧KL物体收益通过、Text保持失败。既有梯度三点也没有证明覆盖不足的因果、普遍冲突或必须调λ。本轮检验更广unique回放的实际性能，保持λ=1；不能把准备/训练完成或低mean KL当作真实提升。</p>"
    return value + link(REPO / "docs/VLM_OBJECT_KL_V1_RESULTS.md", output, "旧KL结果") + " · " + link(REPO / "docs/VLM_OBJECT_KL_GRADIENT_V1_RESULTS.md", output, "旧梯度诊断结果")


def build(config_path):
    config = read(config_path)
    if config and config.get("experiment") != "vlm_object_replay_coverage_v1":
        raise ValueError("expected this coverage-v1 config; do not mix studies")
    root = Path(config["output_root"]).resolve() if config else ROOT
    if root != ROOT.resolve():
        raise ValueError("writes only outputs/vlm_object_replay_coverage_v1/learning/学习页.html")
    output = root / "learning/学习页.html"
    protocol = Path(config.get("study_protocol", PROTOCOL)) if config else PROTOCOL
    body = "<h1>回放覆盖预算：wide32与repeat8x4</h1><p>研究问题：在相同5792次replay曝光下，增加unique训练问题能否保留物体收益并通过Text保持及后续确认？这是待验证假设；旧失败和原门槛保留。</p>"
    body += "<p class='pending'>快照UTC：" + esc(datetime.now(timezone.utc).isoformat()) + "；只读JSON元数据，不读取.pt、不加载模型/processor、不运行forward/GPU。没有启动观察器。</p>"
    if not config or not protocol.is_file():
        body += "<p class='pending'>准备中：新配置或事前协议尚未齐；下表为根代理给定的计划，不冒充已冻结/已执行结果。</p>"
    body += link(config_path, output, "本轮配置") + " · " + link(protocol, output, "事前协议")
    body += "<h2>计划对照：unique与曝光是两个数</h2>" + table([
        "pilot42条件", "计划unique QA", "原32985池覆盖", "实际曝光目标", "每次更新replay曝光", "更新 / λ", "角色"], [
        ["wide32", "5792", "17.56%", "5792", "32个不同问题", "181 / 1", "唯一候选；依协议晋级"],
        ["repeat8x4", "1448", "4.39%", "5792", "原8题各重复4次", "181 / 1", "pilot42探索性控制；不可被选择"]])
    body += "<p>两臂同一初始SFT42、同完整2888物体QA及相同object顺序；unique范围是监督问题覆盖，不是视觉语义类别/预训练知识覆盖。wide32同时改变每步问题组成与梯度方差，不能把后续差异全部归为纯覆盖因果。下方actual字段出现前，上述5792等数值仅为设计目标。</p>"
    first_update_html, first_update_summaries = first_update_panel(root, output)
    case_html, prediction_case_summaries = prediction_cases(config or {}, root, output)
    body += first_update_html + first25_timing_panel(root, output) + status_panel(root, output) + case_html + CALCULATOR
    extended_reading = root / "independent_review/PREFIX_OBJECTIVE_READING.md"
    if extended_reading.is_file():
        body += "<p>延伸阅读：" + link(extended_reading, output, "GT前缀与保持目标的边界") + "</p>"
    body += old_kl_prefix_panel(root, output)
    images, count = old_fixed_images(output)
    body += images + engineering_costs(root, output) + history(output)
    body += "<h2>真实代码与验收</h2><p>只有wide32按预注册门槛相对共同初始接受筛选；repeat8x4仅pilot42描述性控制，不参与选候选。内部通过后才复制wide32的0/200，再做完整外部和9000题POPE；Text整体非劣、rare/frequent保护和均衡子集guard保持。实际gate、阶段和成本以本轮完整ANALYSIS/COMPLETION为准；单seed控制差值不能自动写成三seed因果胜出。</p><ul>"
    for path, needle in (("src/vlm_object_replay_coverage/train.py", "def backward_update"),
                         ("src/vlm_object_replay_coverage/reference.py", "class TeacherSource"),
                         ("src/vlm_object_replay_coverage/analysis.py", "def analyze"),
                         ("src/vlm_object_train/loss.py", "def normalized_qa_loss"),
                         ("src/vlm_object_kl/loss.py", "def per_question_kl"),
                         ("src/vlm_deep/common.py", "def consensus")):
        body += "<li>" + source(path, output, needle) + "</li>"
    body += "</ul><p>手动刷新：</p><pre>source /opt/data/private/.tools/activate.sh\nsource /opt/data/private/.tools/vlm-a100-env.sh\n&quot;$VLM_A100_PYTHON&quot; " + esc(Path(__file__).resolve()) + "</pre><p>重建后刷新浏览器；没有后台watcher。公开外部/POPE预训练重叠未知；旧完整initial预测复用单列，不称本轮新生成或独立重复。</p>"
    style = "body{font:16px/1.65 system-ui,sans-serif;max-width:1220px;margin:auto;padding:28px;color:#183247;background:#f5f8fa}h2{margin-top:38px;border-bottom:1px solid #cad7df}table{width:100%;border-collapse:collapse;background:white;font-size:14px}td,th{padding:9px;border:1px solid #cad7df;text-align:left;vertical-align:top}th{background:#e6eef4}.scroll{overflow:auto}pre,code{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.5 monospace}pre{padding:14px;background:#e8eff4;max-height:420px;overflow:auto}code{display:block}details{margin:12px 0;padding:10px;background:white}summary{cursor:pointer}.pending,.badge{background:#fff1d0;padding:12px}.cases{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:16px}.cases article{padding:14px;background:white}.cases img{width:100%;height:330px;object-fit:contain;background:#eee}a{color:#155e94}small{font-size:11px}input,button{font:inherit;padding:5px;margin:7px}input{width:100px}label{display:block}"
    page = "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>回放覆盖预算学习页</title><style>" + style + "</style><body>" + body + "</body></html>"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    print(json.dumps({"HTML": str(output), "bytes": output.stat().st_size,
        "config_present": bool(config), "protocol_present": protocol.is_file(), "old_fixed_training_images": count,
        "first_formal_update_metadata": first_update_summaries,
        "complete_prediction_case_panels": prediction_case_summaries,
        "model_loads": 0, "tensor_reads": 0, "model_forwards": 0, "GPU_operations": 0, "watchers_started": 0}, ensure_ascii=False))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    build(args.config.resolve())


if __name__ == "__main__":
    main()
