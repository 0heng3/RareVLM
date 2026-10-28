#!/usr/bin/env python3
"""Read saved evidence and render the KL learning page; never load a model.

One shot by default. --watch refreshes the same HTML from saved CPU artifacts.
Torch is imported only when a formal, frozen full-vocabulary .pt cache exists.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import time

REPO = Path(__file__).resolve().parents[1]
OUTPUTS = REPO.parents[1] / "outputs"
DEFAULT_CONFIG = REPO / "configs/vlm_object_kl_v1.json"


def read_json(path):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"read_status": "文件正在写入或不可解析；下次刷新重读", "path": str(path)}


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def escaped(value):
    return html.escape(str(value), quote=True)


def pretty(value):
    return "<pre>" + escaped(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)) + "</pre>"


def link(path, output, label=None):
    path = Path(path).resolve()
    relative = os.path.relpath(path, output.parent)
    return (f'<a href="{escaped(relative)}">{escaped(label or path.name)}</a>'
            f' <small>（相对链接；<a href="{escaped(str(path))}">绝对路径</a>）</small>')


def source(path, output, needle):
    path = REPO / path
    lines = path.read_text().splitlines()
    line = next((i + 1 for i, text in enumerate(lines) if needle in text), 1)
    return link(path, output, f"{path.relative_to(REPO)}:{line}") + f"<code>{escaped(path)}:{line}</code>"


def table(headers, rows):
    return ("<div class='scroll'><table><thead><tr>" + "".join(f"<th>{escaped(h)}</th>" for h in headers)
            + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row)
            + "</tr>" for row in rows) + "</tbody></table></div>")


def fmt(value, signed=False):
    if value is None:
        return "待完成"
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        return f"{value:+.6f}" if signed else f"{value:.6f}"
    return escaped(value)


def details(title, value):
    return "<details><summary>" + escaped(title) + "</summary>" + pretty(value) + "</details>"


def compact_result(value):
    keys = ("status", "completed_at", "completed_updates", "planned_updates", "updated_at",
            "actual_questions_seen", "actual_object_exposures", "actual_replay_exposures",
            "mean_object_ce", "mean_replay_kl", "mean_replay_ce", "mean_replay_ce_scope",
            "teacher_cache_unchanged", "lora_delta_l2", "lora_delta_max_abs", "changed_lora_tensors",
            "initial_parameters_restored_exactly", "rng_restored", "formal_optimizer_updates",
            "discarded_engineering_optimizer_updates", "memory_cap_passed", "record", "error",
            "training_active_seconds", "model_load_seconds", "preflight_seconds_current_session",
            "teacher_cache_seconds_current_session", "session_wall_seconds", "peak_gpu_allocated_gib",
            "peak_gpu_reserved_gib", "peak_allocated_gib", "peak_reserved_gib", "compute_accounting")
    return {key: value[key] for key in keys if key in value}


def historical(config, output):
    root = Path(config["historical_object_study_root"])
    path = root / "analysis/pilot_seed42/screen/ANALYSIS.json"
    analysis = read_json(path)
    if not analysis or analysis.get("status") != "complete":
        return "<p>旧 v2 完整分析当前不可用，不填补估计值。</p>"
    rows = []
    for arm, name in (("object_only", "Only"), ("object_replay", "硬标签 Replay")):
        for task in ("object", "text"):
            for row in analysis["contrasts"][f"{arm}_minus_initial_{task}"]:
                if row["slice"] not in ("overall", "rare_1_5", "frequent_gt20"):
                    continue
                rows.append([name, task + " / " + escaped(row["slice"]),
                             fmt(row.get("before_score_percent")), fmt(row.get("after_score_percent")),
                             fmt(row.get("delta_pp"), True), fmt(row.get("one_sided_98_75_lower_pp"), True)])
    result = ("<h2>旧 v2：回放硬标签 CE 未证明保持</h2><p class='badge'>历史 seed42；不是新 KL 结果</p>"
              + table(["条件", "任务 / 切片", "初始 %", "训练后 %", "差值 pp", "单侧 98.75% 下界 pp"], rows)
              + "<p>Only 的物体门槛通过，但 Text rare/frequent 点保护失败。硬标签 Replay 的物体下界为 0，"
              "Text 下界未严格大于 −0.5 pp，切片点保护也失败。两臂均无候选晋级。切片点差不等于已证明总体显著退化；"
              "整体非劣也不能证明切片非劣。以上为图像簇条件区间、单 checkpoint 顺序种子探索。</p>"
              + link(path, output, "旧 v2 完整官方分析") + details("旧 v2 原始联合门槛", analysis["gates"]))
    cost_rows = []
    for arm in ("object_only", "object_replay"):
        value = read_json(root / f"runs/{arm}/seed42/train/TRAIN_RESULT.json") or {}
        cost_rows.append([arm] + [fmt(value.get(key)) for key in (
            "actual_object_exposures", "actual_replay_exposures", "training_active_seconds",
            "session_wall_seconds", "peak_gpu_allocated_gib", "peak_gpu_reserved_gib")])
    result += table(["历史条件", "Object QA", "额外 Text QA", "训练 active 秒", "分支 session 秒", "allocated GiB", "reserved GiB"], cost_rows)
    result += "<p>Replay 增加 1448 次真实 Text 曝光；同初始化、更新数、参数量和视觉上限不代表等信息或等 FLOPs。分支 session 不相加冒充整轮壁钟。</p>"
    review_path = root / "independent_review/OBJECT_VAL_CASE_REVIEW.json"
    review = read_json(review_path) or {}
    wanted = {"object_presence_v2:000000456900:category16", "object_presence_v2:000000239881:category18"}
    cases = [case for case in review.get("cases", []) if case["question_id"] in wanted]
    predictions = {}
    for arm in ("initial", "object_only", "object_replay"):
        pred_path = root / f"runs/{arm}/seed42/object/predictions.jsonl"
        if pred_path.exists():
            with pred_path.open() as stream:
                predictions[arm] = {row["question_id"]: row for row in map(json.loads, stream)
                                    if row["question_id"] in wanted}
    result += "<h3>真实原图：鸟笼与厨房里的狗</h3><p>两图是旧 v2 的保存病例，未依据新 KL 预测挑选；下面显示完整原图，浏览器等比缩放，不是裁剪、模型注意力或新模型输入。</p><div class='cases'>"
    for case in cases:
        image = Path(case["image_path"])
        result += ("<article><img loading='lazy' src='" + escaped(os.path.relpath(image, output.parent))
                   + "' alt='旧 v2 真实原图'><h4>" + escaped(case["question"]) + "</h4><p>target="
                   + escaped(case["target"]) + "；原图 " + escaped(f"{case['image_width']}×{case['image_height']}")
                   + "；实际 visual tokens=" + escaped(case.get("visual_tokens")) + "。</p>"
                   + table(["旧模型", "已保存回答", "已保存得分"], [[arm, escaped(predictions.get(arm, {}).get(case["question_id"], {}).get("prediction", "缺失")),
                     fmt(predictions.get(arm, {}).get(case["question_id"], {}).get("score"))] for arm in predictions])
                   + "<p>" + escaped(case.get("visual_review", {}).get("visible_evidence", "未审查")) + "</p>"
                   + details("原标注框依据（仅展示，不作模型 crop）", case["bbox_evidence"])
                   + "<code>" + escaped(case["question_id"]) + "</code>" + link(image, output, "原图文件") + "</article>")
    return result + "</div><p>这两处具体漏检保留于三份旧预测；不能由两个病例推断整体失败的因果份额，也不能预告 KL 会修复它们。</p>" + link(review_path, output, "旧病例独立审查")


def token_names(config):
    path = Path(config["model_root"]) / "tokenizer.json"
    value = read_json(path) or {}
    names = {index: token for token, index in value.get("model", {}).get("vocab", {}).items()}
    names.update({item["id"]: item["content"] for item in value.get("added_tokens", [])})
    return names, path


def first_batch_review_panel(root, output):
    """Display the independent JSON receipt only; do not reopen cache tensors."""
    path = root / "independent_review/TEACHER_FIRST_BATCH_REVIEW.json"
    receipt = read_json(path)
    result = "<h2>单批工程缓存独立审查：真实概率先看一题</h2>"
    if not receipt:
        return result + "<p class='pending'>独立首批审查收据尚未出现。</p>"
    if (receipt.get("schema_version") != "rarevlm.object_kl.independent_teacher_first_batch_review.v1"
            or receipt.get("status") != "passed_for_one_saved_engineering_batch_only"
            or receipt.get("reviewed_atomic_batches") != 1
            or not receipt.get("checks") or not all(value is True for value in receipt["checks"].values())):
        return result + "<p class='error'>收据合同或审查状态不符，不展示实际概率。</p>" + link(path, output)
    question = receipt["questions"][0]
    result += "<p class='badge'>单批工程缓存审查，未全量冻结／无正式效果。下列状态是收据审查时的快照，不替代当前工程或正式训练状态。</p>"
    result += "<p>本面板只读独立 JSON 收据，不重新打开 .pt、不读取全缓存，也不加载模型或新增 forward。真实 top-5 是展示摘要；训练保存并使用完整 151936 词表。</p>"
    result += pretty({"reviewed_at": receipt["reviewed_at"], "receipt_sha256": sha_file(path),
        "tensor_shape": receipt["tensor_shape"], "tensor_dtype": receipt["tensor_dtype"],
        "per_question_assistant_token_counts": receipt["per_question_assistant_token_counts"],
        "question_id": question["question_id"],
        "actual_target_token_ids_including_end": question["actual_target_token_ids"],
        "actual_target_decoded_including_end_tokens": question["actual_target_decoded_including_end_tokens"],
        "source_cache_batch": receipt["source"]["cache_batch_file"],
        "source_cache_batch_file_sha256": receipt["source"]["cache_batch_file_sha256"],
        "max_absolute_row_logsumexp": receipt["max_absolute_row_logsumexp"]})
    rows = []
    for token in question["assistant_prediction_rows"][:2]:
        q_target = token["teacher_label_probability"]
        alternatives = [item for item in token["top5"] if item["token_id"] != token["label_token_id"]]
        alternative = alternatives[0] if alternatives else None
        rows.append([str(token["assistant_token_index"]),
            escaped(f"{token['label_token_id']} / {token['label_token_decoded']}"),
            escaped(f"label t={token['label_position_in_actual_padded_input']}；logits t−1={token['causal_prediction_input_position']}"),
            f"{q_target:.9f}", f"{q_target - 1:+.9f}",
            (escaped(f"ID {alternative['token_id']} ({alternative['decoded']})：+{alternative['probability']:.9f}") if alternative else "未记录"),
            "0（同初始 p=q 的公式值）", pretty(token["top5"])])
    result += table(["首题 token 序号", "真实 GT ID / 解码", "真实因果位置", "真实教师 q(GT)",
        "同初始时 CE 对 GT logit 梯度 q(GT)−1", "同初始时 CE 对一项非 GT logit 梯度 q(v)",
        "同初始时 KL 梯度 p−q", "收据真实 top-5"], rows)
    result += ("<p>CE/KL 梯度列是把收据真实教师 q 代入“同输入、同固定初始参数时 p=q”推导的单 token、未归一 logits 梯度；"
        "不是新增 actor 前向、实测梯度或更新后概率。首题 q(GT)&lt;1，CE 仍提高硬标签 logit、降低其他 logit；同初始 KL 梯度为0。"
        "实际回放 loss 还要按本题4个有效答案 token、每更新8题分别归一，λ=1；不能把这里的原始梯度直接当完整 update 梯度。"
        "KL 近0与临时更新后的响应，仍需等待 PREFLIGHT 的实际验证。</p>")
    return result + details("首批独立检查项与当时边界", {"checks": receipt["checks"],
        "cache_state_at_review_end": receipt["cache_state_at_review_end"], "limitations": receipt["limitations"]}) + link(path, output, "首批独立审查完整收据")


def external_initial_panel(root, output):
    """Summarize the completed independent initial-baseline receipt only."""
    path = root / "independent_review/EXTERNAL_INITIAL_REVIEW.json"
    receipt = read_json(path)
    result = "<h2>初始外部基线：非 KL 收益</h2>"
    if not receipt:
        return result + "<p class='pending'>外部初始基线独立审查收据尚未出现。</p>"
    if (receipt.get("schema_version") != "rarevlm.object_kl.external_initial_review.v1"
            or receipt.get("status") != "passed" or receipt.get("baseline_only") is not True):
        return result + "<p>收据尚未通过完整初始基线审查，不展示估计分数。</p>" + link(path, output)
    full = receipt["overall"]
    balanced = receipt["per_category_yes_no_balanced_subset"]["overall"]
    result += "<p class='badge'>原始 initial checkpoint 的新外部完整基线；不是新 KL 候选结果、提升或训练效果。</p>"
    rows = []
    for name, item in (("完整外部基线", full), ("逐类 yes/no 均衡子集（QA-pooled）", balanced)):
        rows.append([name, str(item["questions"]), str(item["unique_images"])]
            + [fmt(item["rates_percent"][key]) for key in ("accuracy", "precision", "f1", "recall", "specificity")])
    result += table(["基线范围", "QA", "图像", "Accuracy %", "Precision %", "F1 %", "Recall %", "Specificity %"], rows)
    result += table(["完整4372题：GT / 预测", "预测 yes", "预测 no"], [
        ["GT yes", f"TP = {full['TP']}", f"FN = {full['FN']}"],
        ["GT no", f"FP = {full['FP']}", f"TN = {full['TN']}"]])
    result += (f"<p>完整基线 {full['FP'] + full['FN']} 个错误中 {full['FN']} 个是漏检（FN），"
        "这是保存预测的描述，不能据此调整当前训练设置或预告 KL 会修复这些错误。</p>"
        "<p>3438题是完整预测的固定严格整行子集，18个类别内各自 yes/no 数相等，总计1719/1719；"
        "各类 QA 数仍可不同。因此上述均衡子集指标按全部 QA 合并计算（QA-pooled），不是18类宏平均，"
        "也不替代4372题完整确认或完整9000题 POPE。</p>")
    return result + "<p>审查时间 UTC：" + escaped(receipt["reviewed_at_utc"]) + "；" + link(path, output, "完整初始基线独立审查收据") + "</p>"


def negative_result_cases(output):
    """Render the portable negative-result export and its eight viewed cases."""
    path = REPO / "results/vlm_object_kl_v1_metrics.json"
    metrics = read_json(path)
    if not metrics or metrics.get("schema_version") != "rarevlm.object_kl.portable_results.v1":
        return ""
    result = "<h2>本轮负结果与8条真实得分变化病例</h2>"
    result += "<p class='error'>completed_negative：物体门槛通过；Text整体非劣、rare/frequent点保护失败。未执行0/200复制、candidate外部确认或POPE，用户目标仍active。</p>"
    result += "<p>训练回放mean KL虽低，仍未通过heldout Text保持门槛。它是不断变化训练批次的在线平均，不是最终全训练集或heldout分布距离，也不能预判greedy答案不变。</p>"
    result += link(REPO / "docs/VLM_OBJECT_KL_V1_RESULTS.md", output, "完整负结果文档") + " · " + link(path, output, "便携指标与病例JSON")
    accounting = metrics["operation_accounting"]
    result += table(["操作", "实际计数", "范围"], [
        ["新正式评分 model×QA", str(accounting["new_formal_scored_model_x_QA"]), "candidate Object342 + Text1617 + initial external4372"],
        ["评测工程预检生成", str(accounting["new_evaluation_preflight_generated_QA"]), "3份PREFLIGHT各8题，独立于正式评分"],
        ["已记录自回归生成QA合计", str(accounting["new_recorded_autoregressive_generated_QA_including_eval_preflights"]), "包含24题预检；不与teacher-forced前向混算"],
        ["历史初始Object/Text复用", "342 + 1617", "不是本轮新增生成"],
        ["正式 / 已丢弃工程 optimizer更新", "181 / 2", "丢弃工程状态，未混入正式Adam"],
        ["两份teacher cache额外前向QA", "工程1448 / 正式1448", "每份完整151936词表FP32缓存4,372,745,568 bytes"],
        ["候选外部 / POPE / 复制种子", "0 / 0 / 0", "筛选失败，无后续确认"]])
    result += "<p>正式训练active1751.599秒、trainer session3051.211秒；工程与正式教师纯前向分别611.506/619.327秒，完整controller session5599.568秒。阶段口径与额外响应检查见完整结果，不称等信息、等FLOPs或端到端加速。</p>"
    review = metrics["Text_visual_case_review"]
    result += "<p>复用已保存的全部8条非零得分变化：1升7降；原文变化总计16条。仅这8张原图各查看一次，属于助手局部观察，不是盲评人类真值或全量机制比例。下面原图等比显示，无裁剪、重建输入或伪注意力。</p><div class='cases'>"
    for case in review["cases"]:
        image_path = Path(case["image_path"])
        result += "<article id='" + escaped(case["case_id"]) + "'><img loading='lazy' src='" + escaped(os.path.relpath(image_path, output.parent)) + "' alt='Text真实原图'>"
        result += "<h3>qid " + escaped(case["question_id"]) + "：" + escaped(case["question"]) + "</h3>"
        result += "<p>" + escaped(case["slice"]) + "；实际训练目标频次=" + str(case["target_frequency"]) + "；原图 " + escaped(f"{case['image_width']}×{case['image_height']}") + "。</p>"
        result += table(["保存的模型", "真实原文回答", "官方共识分"], [[escaped(arm), escaped(value["prediction"]), fmt(value["score"])] for arm, value in case["predictions"].items()])
        result += "<p><strong>助手局部观察：</strong>" + escaped(case["visual_review"]["visible_evidence_and_scoring_boundary"]) + "</p>"
        result += details("原始10参考（GT/评分保持）", case["answers"]) + link(image_path, output, "完整原图") + "</article>"
    return result + "</div><p>没有因病例改GT、官方评分或门槛；也不由相同答案错误倒推新KL与历史CE的共同因果机制。下一梯度诊断需独立事前协议，准备或诊断完成都不等于真实提升。</p>"


def teacher_panel(config, root, output, seed, top_k, engineering=False, metadata_only=False):
    scope = "工程预检" if engineering else "正式训练"
    directory = (root / "engineering_runs" if engineering else root) / f"runs/object_kl/seed{seed}/train/teacher_cache"
    manifest_path = directory / "MANIFEST.json"
    manifest = read_json(manifest_path)
    result = "<h2>真实" + scope + " teacher 缓存：只读 1 题</h2>"
    result += ("<p class='badge'>独立工程 scope：临时优化步骤将丢弃，缓存不能冒充正式训练证据。</p>" if engineering else "<p>此面板仅接受正式 runs/ 下的缓存，不以工程缓存替代。</p>")
    result += "<p>这里只展示教师 q，未保存的当前学生 p 不可由平均 loss 反推；因此不展示伪造的实际 p−q。</p>"
    if not manifest or manifest.get("status") != "frozen":
        progress = read_json(directory / "PROGRESS.json")
        estimate = read_json(directory / "ESTIMATE.json")
        return result + "<p class='pending'>准备中：" + scope + "完整 teacher_cache/MANIFEST.json 尚未冻结，没有可展示的真实教师概率。</p>" + details("现存缓存进度与预算（缺失为 null）", {"progress": progress, "estimate": estimate})
    if metadata_only:
        return result + "<p>本次为只读 JSON 刷新，未打开 .pt；真实首批概率见独立审查收据面板。</p>" + details("已保存缓存元数据（未重审张量）", {
            key: manifest.get(key) for key in ("status", "vocabulary_size", "logprob_dtype", "full_vocabulary",
                "teacher_forward_questions", "teacher_forward_assistant_tokens", "teacher_forward_seconds_all_batches", "cache_file_bytes")}) + link(manifest_path, output)
    try:
        binding_path = directory / "INPUT_LOCK.json"
        binding = read_json(binding_path)
        if (manifest.get("schema_version") != "rarevlm.object_kl.teacher_cache.v1"
                or manifest.get("full_vocabulary") is not True or manifest.get("logprob_dtype") != "float32"
                or manifest.get("temperature") != 1.0 or manifest.get("direction") != "initial_to_actor"
                or not binding or content_sha(binding) != manifest.get("binding_sha256")
                or manifest.get("teacher_parameters_sha256") != binding.get("initial_adapter_tensor_sha256")
                or manifest.get("frozen_parameters_sha256") != binding.get("frozen_parameter_sha256_before")):
            raise ValueError("完整词表/FP32/T=1/固定初始教师/输入绑定不一致")
        entries = manifest["entries"]
        key = sorted(entries)[0]
        entry = entries[key]
        cache_path = directory / f"{key}.pt"
        if sha_file(cache_path) != entry["file_sha256"]:
            raise ValueError("所选缓存文件 SHA 不一致")
        import torch  # CPU tensor deserialization only; no model/processor/CUDA call.
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        values, counts = payload["logprobs"], payload["counts"]
        identity = entry["identity"]
        if payload["identity"] != identity or identity["binding_sha256"] != manifest["binding_sha256"]:
            raise ValueError("所选缓存实际输入 identity 不一致")
        if (values.dtype != torch.float32 or values.ndim != 2 or values.requires_grad
                or counts.dtype != torch.int64 or counts.ndim != 1 or not bool((counts > 0).all())
                or int(counts.sum()) != values.shape[0] or list(values.shape) != entry["shape"]
                or values.shape[1] != manifest["vocabulary_size"] or not bool(torch.isfinite(values).all())
                or not torch.allclose(torch.logsumexp(values, -1), torch.zeros(values.shape[0]), atol=1e-5, rtol=0)):
            raise ValueError("FP32 完整词表 shape/counts/归一化检查失败")
        digest = hashlib.sha256()
        for name, tensor in (("logprobs", values), ("counts", counts)):
            tensor = tensor.detach().cpu().contiguous()
            digest.update(name.encode()); digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode()); digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        if digest.hexdigest() != payload["tensor_sha256"] or payload["tensor_sha256"] != entry["tensor_sha256"]:
            raise ValueError("所选完整缓存 tensor SHA 不一致")
        names, tokenizer_path = token_names(config)
        labels = identity["actual_assistant_labels"][0]
        positions = [i for i, value in enumerate(labels) if value != -100]
        if len(positions) != int(counts[0]) or counts.tolist() != identity["assistant_tokens"]:
            raise ValueError("有效 assistant labels 与 counts 不一致")
        n = int(counts[0])
        q = values[:n].exp()
        rows = []
        for token_index, position in enumerate(positions):
            token_id = labels[position]
            probs, ids = q[token_index].topk(min(top_k, values.shape[1]))
            top = [{"token_id": int(index), "tokenizer原始项": names.get(int(index), "未找到"), "q": float(prob)}
                   for prob, index in zip(probs, ids)]
            rows.append([str(token_index), str(position), str(position - 1), str(token_id),
                         escaped(names.get(token_id, "未找到")), f"{float(q[token_index, token_id]):.9g}",
                         f"{float(probs.sum()):.9g}", pretty(top)])
        first = {"question_id": identity["question_ids"][0], "assistant_tokens": n,
                 "actual_input_ids": identity["actual_input_ids"][0],
                 "actual_attention_mask": identity["actual_attention_mask"][0],
                 "actual_assistant_labels": labels,
                 "input_contract": identity["input_contracts"][0]}
        result += "<p class='good'>读取成功：真实" + scope + " FP32 缓存；所选文件、完整张量 SHA 与固定初始参数/输入锁匹配。此处只核对所选条目；整库完整性由训练验证器负责。</p>"
        result += pretty({"scope": scope, "cache_file": str(cache_path), "file_sha256": entry["file_sha256"],
                          "tensor_shape": list(values.shape), "dtype": str(values.dtype),
                          "selected_question_id": first["question_id"], "answer_tokens_including_end": n,
                          "full_vocabulary_size": values.shape[1], "teacher_parameters_sha256": manifest["teacher_parameters_sha256"],
                          "total_teacher_forward_questions": manifest["teacher_forward_questions"],
                          "total_cache_file_bytes": manifest["cache_file_bytes"],
                          "total_teacher_forward_seconds": manifest["teacher_forward_seconds_all_batches"]})
        result += table(["答案 token 序号", "label 位置 t", "预测 logits 位置 t−1", "target ID", "tokenizer 原始项", "真实 q(target)", "展示 top-K 概率和", "真实 top-K"], rows)
        result += "<p>每个答案 token 的训练 KL 都遍历完整词表求和；表中 top-K 仅为阅读，未覆盖概率质量仍参与训练。tokenizer 原始项可能含 Ġ/Ċ 等编码符号，不冒充完整表面答案。换行/结束 token 也按实际 labels 纳入。</p>"
        return result + details("第 1 题完整实际 IDs / labels / mask / 输入合同", first) + link(manifest_path, output) + link(binding_path, output) + link(tokenizer_path, output)
    except Exception as error:
        return result + "<p class='error'>拒绝展示实际概率：" + escaped(f"{type(error).__name__}: {error}") + "。不以假设值替代；训练文件不作修改。</p>" + link(manifest_path, output)


def current_panel(config, root, output, seed):
    train = root / f"runs/object_kl/seed{seed}/train"
    engineering = root / f"engineering_runs/runs/object_kl/seed{seed}/train"
    completion = read_json(root / "COMPLETION.json")
    result = read_json(train / "TRAIN_RESULT.json")
    progress = read_json(train / "PROGRESS.json")
    analyses = sorted((root / "analysis").glob("**/ANALYSIS.json")) if (root / "analysis").exists() else []
    stage = "准备中：尚无正式训练结果；工程通过和缓存完成均不是质量提升。"
    if progress:
        stage = "已保存正式进度：" + str(progress.get("status", "未声明")) + "；以记录时间为准，不由文件存在推断进程存活。"
    if result and result.get("status") == "completed":
        stage = "正式训练已保存；等待/读取完整评测分析，训练完成不是目标成功。"
    if completion and completion.get("status") == "completed_negative":
        stage = "本轮completed_negative：物体门槛通过，但Text非劣及rare/frequent点保护失败；目标尚未完成。"
    elif completion:
        stage = "工作流有 COMPLETION 记录；最终判定以完整官方分析和根代理复核为准。"
    engineering_cache = read_json(engineering / "teacher_cache/PROGRESS.json")
    value = "<h2>新 KL 当前阶段与真实成本</h2><p class='pending'>" + escaped(stage) + "</p>"
    value += details("独立工程 teacher 缓存当前保存进度", engineering_cache or {"status": "pending"})
    for label, path in (("独立工程 PREFLIGHT（临时更新已丢弃）", engineering / "PREFLIGHT.json"),
                        ("正式训练内置 PREFLIGHT", train / "PREFLIGHT.json"),
                        ("正式训练 PROGRESS", train / "PROGRESS.json"), ("正式 TRAIN_RESULT", train / "TRAIN_RESULT.json")):
        data = read_json(path)
        value += details(label + ("：已保存" if data else "：待完成"), compact_result(data) if data else {"status": "pending", "expected_path": str(path)})
        if data:
            value += link(path, output)
    history_summary = {}
    for scope, directory in (("独立工程", engineering), ("正式训练内置工程预检", train)):
        history = directory / "PREFLIGHT_HISTORY.jsonl"
        records = []
        if history.is_file():
            for line in history.read_text().splitlines():
                try:
                    records.append(json.loads(line))
                except ValueError:
                    continue  # An incomplete concurrent append is re-read next time.
        history_summary[scope] = {"saved_complete_records": len(records),
            "recorded_discarded_optimizer_updates_sum": sum(row.get("discarded_engineering_optimizer_updates", 0) for row in records),
            "scope": "仅已完整保存的预检历史；不含未落盘失败，不将临时步骤计入正式更新"}
    value += details("预检额外步骤：读取全部已保存 HISTORY，分 scope 统计", history_summary)
    for path in analyses:
        data = read_json(path) or {}
        summary = {key: data[key] for key in ("schema_version", "status", "updated_at", "phase", "stage", "seeds", "pending_cells", "eligible_candidates", "selected_candidate", "gate", "gates", "screen_gates", "external_gate", "goal_gate", "true_improvement_goal_achieved", "limitations") if key in data}
        value += details("新 KL 完整阶段分析：" + str(path.relative_to(root)), summary) + link(path, output, "打开完整分析 JSON（不是局部日志）")
        if data.get("status") == "complete":
            rows = []
            for task, contrasts in data.get("contrasts", {}).items():
                if not isinstance(contrasts, list):
                    continue
                for contrast in contrasts:
                    if contrast.get("slice") in ("overall", "rare_1_5", "frequent_gt20"):
                        rows.append([escaped(task), escaped(contrast["slice"]), fmt(contrast.get("delta_pp"), True),
                                     fmt(contrast.get("one_sided_98_75_lower_pp"), True),
                                     fmt(contrast.get("seed_delta_sample_sd_pp")), str(contrast.get("checkpoint_axes", "待记录"))])
            value += table(["本轮已完成任务", "切片", "差值 pp", "98.75%下界 pp", "seed差值样本SD pp", "checkpoint axes"], rows)
            balanced = data.get("class_balanced")
            if balanced and balanced.get("status") == "complete":
                overall_row = next(row for row in balanced["contrasts"] if row["slice"] == "overall")
                value += "<p>官方class_balanced：" + str(balanced["questions"]) + "QA / " + str(balanced["unique_images"]) + "图，逐类yes/no相等，按QA合并（非类别macro）；原完整预测过滤，新增生成0。</p>"
                value += table(["内部均衡accuracy初始%", "训练后%", "差值pp", "保护规则"], [[fmt(overall_row["before_score_percent"]), fmt(overall_row["after_score_percent"]), fmt(overall_row["delta_pp"], True), "附加点保护；不替代完整val或POPE"]])
    if not analyses:
        value += "<p>新 KL 尚无完整 ANALYSIS；不把旧 Only/Replay 分数回填为本轮结果。</p>"
    if completion:
        value += details("工作流 COMPLETION 原始记录", completion) + link(root / "COMPLETION.json", output)
    value += "<p>成本分别读取教师预计算 QA/秒/字节、训练 object 与 replay QA/token/秒、预检秒、模型加载秒、session 壁钟、allocated/reserved。不存在的字段显示待完成；没有测量 FLOPs，不能由 QA 数或历史时间声称等算力或训练加速。</p>"
    return value


CALCULATOR = r"""
<h2>可调教学算例：CE 的 p−onehot 与 KL 的 p−q</h2>
<p class="badge">以下数字全部是假设；三词表教学模型，不是实际 Qwen 词表或本轮缓存。</p>
<p>在同一图像、问题及真实目标前缀下，q 是固定初始教师概率，p 是学生概率，y 是硬标签。
CE=−log p(y)，KL(q‖p)=Σᵥqᵥ(log qᵥ−log pᵥ)。相对于学生 logits 的梯度分别是 p−onehot(y) 与 p−q（T=1）。梯度下降会降低正梯度对应的 logit、提高负梯度对应的 logit。</p>
<label>假设学生未归一权重（A/B/C）：<input id="weights" value="0.5,0.3,0.2"></label>
<label>固定教师 q：<input id="teacher" value="0.7,0.2,0.1"></label>
<label>硬标签 y：<select id="target"><option value="0">A</option><option value="1">B</option><option value="2">C</option></select></label>
<button id="calc">计算</button><button id="equal">令 p=q，看初始状态</button>
<div id="calculation" aria-live="polite"></div>
<p>p=q 时 KL=0、梯度全零；CE 仍会把分布推向唯一硬目标。因此回放 CE 仍在“再学习训练标签”，不是直接约束原模型的完整输出分布。
KL 约束只覆盖既定回放输入与原目标前缀；未约束的输入、未生成路径、优化冲突和有限回放覆盖仍可变化。它是否保持 heldout 能力必须靠实际门槛验证。</p>
<script>
function calculate(){try{let p=document.getElementById('weights').value.split(',').map(Number),q=document.getElementById('teacher').value.split(',').map(Number),y=Number(document.getElementById('target').value);
if(p.length!==3||q.length!==3||![...p,...q].every(x=>Number.isFinite(x)&&x>0))throw Error('每组请输入三个大于0的数');let ps=p.reduce((a,b)=>a+b),qs=q.reduce((a,b)=>a+b);p=p.map(x=>x/ps);q=q.map(x=>x/qs);
let ce=-Math.log(p[y]),kl=q.reduce((s,x,i)=>s+x*(Math.log(x)-Math.log(p[i])),0),f=x=>x.toFixed(6);
document.getElementById('calculation').innerHTML='<p>假设 CE='+f(ce)+'；假设 KL='+f(kl)+'</p><table><tr><th>假设 token</th><th>学生 p</th><th>教师 q</th><th>CE 梯度</th><th>KL 梯度</th></tr>'+p.map((x,i)=>'<tr><td>'+['A','B','C'][i]+'</td><td>'+f(x)+'</td><td>'+f(q[i])+'</td><td>'+f(x-(i===y?1:0))+'</td><td>'+f(x-q[i])+'</td></tr>').join('')+'</table>';
}catch(e){document.getElementById('calculation').textContent=e.message;}}
document.getElementById('calc').onclick=calculate;document.getElementById('equal').onclick=()=>{document.getElementById('weights').value=document.getElementById('teacher').value;calculate()};calculate();
</script>
"""


def render(config_path, seed, top_k, metadata_only=False):
    config = read_json(config_path)
    if not config or config.get("experiment") != "vlm_object_kl_v1":
        raise ValueError("expected real vlm_object_kl_v1 config")
    root = Path(config["output_root"]).resolve()
    expected = (OUTPUTS / "vlm_object_kl_v1").resolve()
    if root != expected:
        raise ValueError("learning generator writes only outputs/vlm_object_kl_v1/learning/学习页.html")
    output = root / "learning/学习页.html"
    generated = datetime.now(timezone.utc).isoformat()
    command = f'source /opt/data/private/.tools/activate.sh\nsource /opt/data/private/.tools/vlm-a100-env.sh\n"$VLM_A100_PYTHON" {REPO / "tools/build_object_kl_learning.py"} --metadata-only'
    completed = read_json(root / "COMPLETION.json")
    intro = "本轮已完成负结果：物体收益未伴随Text保持验收通过，真实提升目标仍active。" if completed and completed.get("status") == "completed_negative" else "研究假设：用固定初始模型的分布约束替代硬标签回放 CE，可能保留物体收益并保持原 Text 能力。当前没有预设成功结论。"
    body = "<h1>物体适配 × 固定初始教师 KL：边做边学</h1><p>" + intro + "</p>"
    body += "<p>本页快照 UTC：<strong>" + escaped(generated) + "</strong>；formal seed=" + str(seed) + "。</p>"
    body += "<p>生成器只读保存记录，不加载模型、processor，不新增 forward/GPU，也不改标签、评分或门槛。重新运行即可读取新的缓存、预检、训练和完整分析。</p>"
    body += details("仅JSON的手动刷新命令（本轮已收尾）", {"command": command})
    body += "<p>浏览器刷新只重读已生成 HTML，不会触发实验或重建快照；本轮绑定controller的轻量刷新在COMPLETION后退出，不持续轮询旧study。</p>"
    body += negative_result_cases(output)
    body += current_panel(config, root, output, seed)
    body += external_initial_panel(root, output)
    body += historical(config, output) + CALCULATOR
    body += "<h2>实际 loss、QA/token 归一与静态 reference 防错</h2><pre>L = (1/B_obj) Σ_i [(1/T_i) Σ_t CE_object(i,t)]\n  + λ (1/B_replay) Σ_j [(1/T_j) Σ_t Σ_v q_initial(v|x_j,y_&lt;t) · (log q_initial − log p_actor)]\nλ=1，T（蒸馏温度）=1；T_i/T_j 是实际 assistant token 数，包含结束 token。</pre>"
    body += "<p>先对词表求和，再对每题有效答案 token 求均值，最后分别对 object/replay QA 求均值。不能再除词表大小，也不能让长答案因 token 多而得到更大题权重。micro4 的 loss 按本次真实 object/replay 总题数累计；末步 object8 题按8归一。prompt/padding 的 label=−100，不参与 loss；label t 由 logits t−1 预测。回放 hard CE 只 detach 记录，未反传。</p>"
    body += "<p>教师必须是同一个固定初始 SFT42（BF16 舍入后 FP32 master）的完整分布，而不是每步 current actor.detach；后者每次都会得到 p=q，约束失效。缓存绑定 initial/base/source/config/UUID 与实际 IDs、mask、labels、像素/grid。预计算之后恢复 LoRA、RNG、registered buffers、rope/cache 状态；训练前后校验缓存 SHA。使用原训练 GT 前缀的 teacher forcing，没有新采样，属于保持蒸馏，不称 on-policy GKD、RL 或模型压缩。</p>"
    body += first_batch_review_panel(root, output)
    body += teacher_panel(config, root, output, seed, top_k, metadata_only=metadata_only)
    body += teacher_panel(config, root, output, seed, top_k, engineering=True, metadata_only=metadata_only)
    body += "<h2>验收与可追到代码的入口</h2><p>pilot42 完整342物体题/143图与1617 Text题/1024图先过内部联合门槛；仅通过后复核0/200顺序种子，再完成4372外部物体题/1791图和完整9000 POPE题/共享500图。四主端点 alpha=.05/4，6000次图像簇 bootstrap，单侧98.75%下界；seed均值与样本SD另列。Text整体非劣、rare/frequent点保护和逐类均衡guard不能用整体收益代替。新独立外部集不回填挽救旧v2失败。</p>"
    body += "<ul>" + "".join("<li>" + source(path, output, needle) + "</li>" for path, needle in (
        ("src/vlm_object_kl/loss.py", "def assistant_logprobs"),
        ("src/vlm_object_kl/loss.py", "def per_question_kl"),
        ("src/vlm_object_kl/train.py", "def backward_update"),
        ("src/vlm_object_kl/cache.py", "def verify_frozen_cache"),
        ("src/vlm_object_kl/cache.py", "def build_teacher_cache"),
        ("src/vlm_object_kl/inputs.py", "def cache_key"),
        ("src/vlm_object_kl_eval/evaluate.py", "def "),
        ("src/vlm_object_kl_eval/analysis.py", "def "),
        ("tools/build_object_kl_learning.py", "def render"))) + "</ul>"
    body += "<p>" + link(config_path, output, "新 KL 配置") + " · " + link(config["study_protocol"], output, "事前 KL 协议") + " · " + link(root / "STUDY_INPUT_LOCK.json", output, "STUDY 输入锁（存在后读取）") + "</p>"
    body += "<p>生成输入仅用图像、问题与声明提示，target仅训练/生成后评分。近重复及预训练重叠未知；POPE已有历史暴露。局部病例、训练/工程完成、缓存冻结均不等于真实目标成功，最终结论需要完整预测、门槛、成本与独立复核。</p>"
    style = "body{font:16px/1.7 system-ui,sans-serif;max-width:1180px;margin:auto;padding:28px;color:#172b3a;background:#f5f7fa}h1,h2,h3{line-height:1.3}h2{margin-top:42px;border-bottom:2px solid #d8e2ea;padding-bottom:10px}table{border-collapse:collapse;width:100%;font-size:14px;background:white}td,th{padding:9px;border:1px solid #d2dde6;text-align:left;vertical-align:top}th{background:#e8eff5}pre,code{font:13px/1.5 ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere}pre{padding:14px;background:#e9eff4;max-height:430px;overflow:auto}code{display:block}a{color:#145e9b}small{font-size:11px}details{margin:12px 0;padding:9px;background:white}summary{cursor:pointer}.pending,.badge{background:#fff2d0;padding:12px}.good{background:#dff1e5;padding:12px}.error{background:#ffe2df;padding:12px}.cases{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:20px}.cases article{background:white;padding:15px}.cases img{width:100%;height:430px;object-fit:contain;background:#eee}.scroll{overflow:auto}input,select,button{font:inherit;margin:8px;padding:6px}label{display:block}#auto{display:inline}"
    page = "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>固定初始教师KL学习页</title><style>" + style + "</style><body>" + body + "</body></html>"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    print(json.dumps({"html": str(output), "generated_at_utc": generated, "bytes": output.stat().st_size,
                      "model_loads": 0, "model_forwards": 0, "gpu_operations": 0}, ensure_ascii=False), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, choices=(0, 42, 200), default=42)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--metadata-only", action="store_true", help="read JSON receipts/metadata only; never open .pt")
    parser.add_argument("--watch", action="store_true", help="repeat read-only snapshots until Ctrl-C")
    parser.add_argument("--interval", type=float, default=30)
    args = parser.parse_args()
    if not 1 <= args.top_k <= 32 or args.interval < 5:
        parser.error("top-k must be1..32 and refresh interval>=5seconds")
    try:
        while True:
            render(args.config.resolve(), args.seed, args.top_k, metadata_only=args.metadata_only)
            if not args.watch:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
