#!/usr/bin/env python3
"""Render a derived, local-only learning page from fixed-checkpoint diagnostics.

This presentation tool is outside the scientific input lock. It does not load
models, score predictions, train, select checkpoints, or read confirmation data.
Run after ANALYSIS.json is completed; incomplete sources produce a pending page.
"""
from __future__ import annotations

import argparse
import hashlib
from html import escape
import json
import math
import os
from pathlib import Path
from urllib.parse import quote


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT.parents[1] / "outputs/vlm_fixed_error_readout_v1"
DEFAULT_OUTPUT = PROJECT / "docs/learning/固定模型_错误分型诊断.html"
ROLES = ("initial", "ce", "ce_rank")
LABELS = {"initial": "初始 SFT42", "ce": "CE 控制", "ce_rank": "CE＋跨图排序"}
TASK_LABELS = {"train_selected": "固定训练诊断组", "screen": "完整筛选集"}
KINDS = ("canonical", "alias")
NEAR_ZERO = 0.125
SALT = "fixed_error_readout_v1_learning_examples_v1"


def read_json(path):
    return json.loads(Path(path).read_text())


def h(value):
    return escape(str(value), quote=True)


def fmt(value, digits=3):
    if value is None:
        return "—"
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        return f"{value:.{digits}f}" if digits else str(int(value))
    return str(value)


def table(headers, rows):
    return ('<div class="scroll"><table><thead><tr>'
            + "".join(f"<th>{h(x)}</th>" for x in headers)
            + "</tr></thead><tbody>"
            + "".join("<tr>" + "".join(f"<td>{h(x)}</td>" for x in row) + "</tr>"
                      for row in rows) + "</tbody></table></div>")


def local_url(path, output):
    path = Path(path).resolve(strict=True)
    return quote(os.path.relpath(path, output.parent).replace(os.sep, "/"), safe="/")


def local_link(path, output, label):
    path = Path(path)
    return f'<a href="{h(local_url(path, output))}">{h(label)}</a>' if path.is_file() else h(label)


def select_fixed(candidates, category):
    """Presentation-only selection, after classification, never training selection."""
    def key(pair):
        payload = json.dumps([SALT, category, pair["task"], pair["pair_id"]],
                             ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()
    return min(candidates, key=key) if candidates else None


def margin_svg(values, kind, scale):
    """A signed numerical bar chart; no image-space/attention interpretation."""
    left, zero, right, height = 150, 380, 610, 205
    factor = (right - zero) / scale
    parts = [f'<svg viewBox="0 0 650 {height}" role="img" '
             f'aria-label="{h(kind)} 三模型正负图 margin，零轴位于中央">',
             f'<text x="12" y="18" class="svgtitle">{h(kind)}：yes − no</text>',
             f'<line x1="{zero}" y1="30" x2="{zero}" y2="183" stroke="#26334a"/>',
             f'<text x="{left}" y="199" text-anchor="middle">−{scale:g}</text>',
             f'<text x="{zero}" y="199" text-anchor="middle">0</text>',
             f'<text x="{right}" y="199" text-anchor="middle">+{scale:g}</text>']
    for index, role in enumerate(ROLES):
        y = 46 + 47 * index
        value = values[role]["readouts"][kind]
        parts.append(f'<text x="12" y="{y + 12}">{h(LABELS[role])}</text>')
        for offset, name, color, mark in (
                (0, "positive_margin", "#216db8", "正图"),
                (18, "negative_margin", "#17947e", "负图")):
            margin = float(value[name])
            if not math.isfinite(margin):
                raise ValueError("Cannot render a nonfinite readout")
            end = zero + margin * factor
            near = abs(margin) <= NEAR_ZERO
            stroke = '#d57600' if near else color
            x, width = min(zero, end), abs(end - zero)
            parts += [f'<rect x="{x:.3f}" y="{y + offset}" width="{width:.3f}" '
                      f'height="10" fill="{color}"/>',
                      f'<circle cx="{end:.3f}" cy="{y + offset + 5}" r="3" '
                      f'fill="{stroke}"><title>{h(mark)} {margin:+.6f}'
                      f'{"；近零" if near else ""}</title></circle>',
                      f'<text x="{zero + 8 if margin <= 0 else zero - 8}" '
                      f'y="{y + offset + 9}" '
                      f'text-anchor="{"start" if margin <= 0 else "end"}">'
                      f'{h(mark)} {margin:+.3f}{" *" if near else ""}</text>']
    return "".join(parts) + "</svg>"


def sources(root, output):
    paths = [(root / "ANALYSIS.json", "完整分析 JSON"),
             (root / "PAIR_DIAGNOSES.jsonl", "逐对原始分型"),
             (root / "data/COHORT.json", "固定诊断队列"),
             (root / "CHANGED_CASES.json", "全部升降病例"),
             (PROJECT / "src/vlm_fixed_error_readout/readout.py", "提示词末位读出代码"),
             (PROJECT / "src/vlm_fixed_error_readout/analyze.py", "分型统计代码")]
    return "<p>" + " · ".join(local_link(path, output, label) for path, label in paths) + "</p>"


def summary(analysis):
    rows = []
    for task in TASK_LABELS:
        for role in ROLES:
            value = analysis["results"][task][role]["all"]
            gen = value["generation"]
            change = analysis["comparisons"][task].get(role)
            fixed = change["initial_wrong"]["generation"]["corrected_pairs"] if change else "—"
            broken = change["initial_correct"]["generation"]["broken_pairs"] if change else "—"
            rows.append([TASK_LABELS[task], LABELS[role],
                         f'{gen["correct_qa"]:g}/{value["qa"]}',
                         fmt(gen["qa_accuracy_percent"]),
                         f'{gen["both_correct_pairs"]}/{value["pairs"]}',
                         fixed, broken])
    parts = ["<h2>实际生成：哪些原错学会了，哪些原对丢失了</h2>",
             table(["题组", "模型", "QA 正确/总数", "QA 准确率 %", "两图全对/总对",
                    "相对初始修复原错对", "原全对新增错"], rows),
             '<p class="note">训练组只有固定 340 题、170 对：45 个初始错对、45 个初始全对保持锚点，'
             '另 80 对用于保留原四题批次。该组富集了错例，<b>不是全部 600 对训练数据的准确率</b>。'
             '完整筛选为 512 题、256 对，初始错对/全对分母为 24/232。</p>']
    rows = []
    for task in TASK_LABELS:
        for role in ROLES:
            result = analysis["results"][task][role]["all"]
            for kind in KINDS:
                value = result["readouts"][kind]
                counts = value["classification_counts_actual_generation_failed_pairs"]
                rows.append([TASK_LABELS[task], LABELS[role], kind,
                             result["generation"]["failed_pairs"],
                             counts["ordered_but_threshold_wrong"],
                             counts["order_reversed_or_tied"],
                             counts["readout_both_correct"],
                             value["near_boundary_pairs"]])
    parts += ["<h2>把真实生成失败拆成不同问题</h2>",
              table(["题组", "模型", "读出", "生成失败对", "已排序但阈值错", "逆序/并列",
                     "读出双对但生成错", "全部对中近零对"], rows)]
    return "".join(parts)


def manifest_maps(cohort):
    result = {}
    for task, reference in cohort["manifests"].items():
        path = Path(reference["path"]).resolve(strict=True)
        value = read_json(path)
        result[task] = (path, value, {row["question_id"]: row for row in value["splits"]["dev"]})
    return result


def case_block(title, candidates, category, index, manifests, output, image_paths):
    chosen = select_fixed(candidates, category)
    if chosen is None:
        return f'<section class="case"><h3>{h(title)}</h3><p>本次完整结果中此类为 0 例，无病例展示。</p></section>'
    task, pair_id = chosen["task"], chosen["pair_id"]
    values = {role: index[(task, role, pair_id)] for role in ROLES}
    manifest_path, manifest, rows = manifests[task]
    pictures = []
    for name, key in (("正图", "positive_question_id"), ("负图", "negative_question_id")):
        row = rows[chosen[key]]
        path = (manifest_path.parent / manifest["images"][row["image_id"]]["path"]).resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"Missing original case image: {path}")
        image_paths.add(path)
        pictures.append('<figure>'
                        f'<a href="{h(local_url(path, output))}"><img loading="lazy" '
                        f'src="{h(local_url(path, output))}" alt="{h(name)} 原图，{h(row["question"])}"></a>'
                        f'<figcaption>{h(name)} · 原 GT：<b>{h(row["target"])}</b><br>'
                        f'question_id：{h(row["question_id"])}</figcaption></figure>')
    generated = []
    numeric = []
    max_abs = 1.0
    for role in ROLES:
        generation = values[role]["generation"]
        generated.append([LABELS[role], generation["positive_prediction"],
                          generation["positive_primary_score"], generation["negative_prediction"],
                          generation["negative_primary_score"], "是" if generation["both_correct"] else "否"])
        for kind in KINDS:
            value = values[role]["readouts"][kind]
            max_abs = max(max_abs, abs(value["positive_margin"]), abs(value["negative_margin"]))
            numeric.append([LABELS[role], kind, fmt(value["positive_margin"], 6),
                            fmt(value["negative_margin"], 6), fmt(value["gap"], 6),
                            fmt(value["center"], 6), "是" if value["ordered"] else "否",
                            "是" if value["both_binary_correct"] else "否",
                            "是" if value["near_boundary"] else "否"])
    scale = math.ceil(max_abs)
    charts = '<div class="charts">' + "".join(margin_svg(values, kind, scale) for kind in KINDS) + '</div>'
    return ('<section class="case">' + f'<h3>{h(title)}</h3>'
            f'<p class="question">原题：{h(chosen["question"])}</p>'
            f'<p class="small">{h(TASK_LABELS[task])} · 候选 {len(candidates)} 对中固定哈希选 1 对 · '
            f'{h(pair_id)}</p><div class="pictures">' + "".join(pictures) + '</div>'
            + table(["模型", "正图真实生成", "正图分数", "负图真实生成", "负图分数", "实际双全对"], generated)
            + charts + '<p class="small">蓝色为正图、绿色为负图；每个条形从零轴出发。橙点/* 表示 '
            '|margin|≤0.125；只标记近边界，不调整零阈值。两图使用相同刻度，显示的是实际 logits '
            '数值读出，<b>不是空间注意力图</b>。</p>'
            + table(["模型", "读出", "m正图", "m负图", "gap", "center", "严格排对", "条件双对", "近零"], numeric)
            + '</section>')


def render(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    required = [root / "ANALYSIS.json", root / "PAIR_DIAGNOSES.jsonl", root / "data/COHORT.json"]
    missing = [path.name for path in required if not path.is_file()]
    analysis = read_json(required[0]) if required[0].is_file() else None
    completed = not missing and analysis.get("status") == "completed"
    intro = ('<header><p class="eyebrow">RareVLM · 边做边学</p><h1>固定模型：错误究竟出在哪一层？</h1>'
             '<p>读取同一初始模型与已保存的 CE / CE＋跨图排序 Final200；本轮 '
             '<b>0 次训练更新，不评确认集</b>。训练题只补必要生成，筛选沿用已保存的实际生成得分。</p></header>')
    teaching = ('<section><h2>先分清三个不同条件</h2>'
                '<ol><li><b>canonical margin</b>：m=logit(yes)−logit(no)，使用固定的两个原始词 token。</li>'
                '<li><b>alias margin</b>：m=logsumexp(YES 词形 logits)−logsumexp(NO 词形 logits)，'
                '汇总预先固定的大小写、前导空格等单 token 词形；不据结果临时改词形集合。</li>'
                '<li><b>实际生成</b>：保存的完整 greedy 回答及其原评分。上述两种条件读出都不能替代它。</li></ol>'
                '<p class="formula">gap=m正图−m负图；center=(m正图+m负图)/2</p>'
                '<p>gap&gt;0 只要求两图顺序正确。两图条件判定都正确还需 '
                '<b>m正图&gt;0 且 m负图&lt;0</b>；margin=0 是边界并列，不暗算为正确。'
                '共同向 yes 偏移能保持 gap，却让负图越过零轴，因此“排对”不等于“都答对”。'
                '若条件读出双对而实际生成失败，则单列生成不一致，不强行归为阈值错误。</p>'
                '<p class="note">近零标准固定为任一 |margin|≤0.125，仅作数值敏感性提示。'
                '新读出默认保留完整输出头，使用 prompt-only、无 labels、一次 no-cache 前向与 FP32 归约；'
                'canonical、alias、完整词表 top1 的差异仍需分别分析。</p></section>')
    image_paths = set()
    if completed:
        if analysis.get("training_updates") != 0 or analysis.get("confirm_evaluated") is not False:
            raise ValueError("This teaching page is only for the zero-update, no-confirm diagnostic")
        cohort = read_json(required[2])
        pairs = [json.loads(line) for line in required[1].read_text().splitlines() if line.strip()]
        index = {(row["task"], row["role"], row["pair_id"]): row for row in pairs}
        if len(index) != len(pairs):
            raise ValueError("Duplicate pair diagnoses")
        rank_train = [row for row in pairs if row["task"] == "train_selected" and row["role"] == "ce_rank"
                      and row.get("anchor_group") == "initial_wrong"]
        repaired = [row for row in rank_train if row["generation"]["both_correct"]]
        unrepaired = [row for row in rank_train if not row["generation"]["both_correct"]]
        failed = [row for row in pairs if row["task"] == "screen" and row["role"] == "ce_rank"
                  and not row["generation"]["both_correct"]]
        ordered = [row for row in failed if row["readouts"]["alias"]["classification"] == "ordered_but_threshold_wrong"]
        reversed_or_tied = [row for row in failed if row["readouts"]["alias"]["classification"] == "order_reversed_or_tied"]
        mismatch = [row for row in failed if row["readouts"]["alias"]["readout_both_correct_but_generation_failed"]]
        other = reversed_or_tied if reversed_or_tied else mismatch
        other_title = "筛选失败：alias 逆序或并列" if reversed_or_tied else "筛选失败：alias 条件双对而生成失败"
        definitions = [("训练原错：Final200 已修复", repaired, "train_repaired"),
                       ("训练原错：Final200 仍未修复", unrepaired, "train_unrepaired"),
                       ("筛选失败：alias 已排对但阈值错", ordered, "screen_ordered_threshold"),
                       (other_title, other, "screen_other")]
        maps = manifest_maps(cohort)
        body = summary(analysis) + teaching + '<h2>四类固定哈希代表对：对着原图和代码理解</h2>'
        body += '<p>病例在完整结果分型后，按固定盐与 pair_id 的 SHA256 最小值选择。'
        body += '它们用于讲解，不是新增评测样本、专家重标注或总体效果估计。</p>'
        for title, candidates, category in definitions:
            body += case_block(title, candidates, category, index, maps, output, image_paths)
        body += '<h2>这轮证据能支持什么</h2><ul>'
        body += "".join(f'<li>{h(finding)}</li>' for finding in analysis.get("findings", [])) + '</ul>'
        body += ('<p class="note">训练锚点修复/未修复、gap 或 center 变化是固定 checkpoint 的描述。'
                 '本轮没有隔离数据、目标函数、学习率或模型架构的因果贡献，也没有证明新损失、校准或延长训练必然有效。'
                 '本次不是新能力增益、全训练集准确率或三种子结论；历史负结果保持不变。</p>')
    else:
        reason = '缺少：' + '、'.join(missing) if missing else f'分析状态：{analysis.get("status", "未知")}'
        body = f'<section class="note"><h2>诊断结果待完成</h2><p>{h(reason)}。'
        body += '尚不展示总计、病例答案或 margin 数值，不推测未生成的结果。结果齐全后重新运行本工具。</p></section>' + teaching
    provenance = []
    for path in required:
        if path.is_file():
            provenance.append(f'<li>{h(path.name)}：<code>{hashlib.sha256(path.read_bytes()).hexdigest()}</code></li>')
    footer = ('<footer><h2>原始证据与源码</h2>' + sources(root, output)
              + f'<p>直接引用的 {len(image_paths)} 张不同原图均已检查存在。图片未复制、裁剪或重绘；'
              '浏览器只按比例显示，点击可打开原文件。</p><ul>' + "".join(provenance)
              + f'</ul><p class="small">病例选择盐：{h(SALT)}。本页为派生教学展示，不属于冻结科学输入。</p></footer>')
    css = '''body{margin:0;background:#f4f6fa;color:#203047;font:16px/1.7 system-ui,sans-serif}main{max-width:1180px;margin:auto;padding:32px 24px}h1{font-size:32px;line-height:1.3}h2{font-size:23px;margin-top:34px}h3{font-size:20px}header,section.case{background:white;padding:24px;border-radius:14px;margin:22px 0}header{border-top:6px solid #216db8}.eyebrow{color:#216db8;font-weight:600}.note{background:#e9f0fa;padding:16px;border-left:4px solid #3970aa}.small,figcaption{font-size:13px;color:#536173}.formula{font:18px ui-monospace,monospace;background:#edf2f7;padding:14px}.question{font-size:20px;font-weight:600}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px;margin:18px 0}th,td{padding:9px;border:1px solid #d6dfeb;text-align:left;vertical-align:top}th{background:#e8eef6;white-space:nowrap}tbody tr:nth-child(even){background:#f8fafc}.pictures,.charts{display:grid;grid-template-columns:1fr 1fr;gap:18px}figure{margin:8px 0}img{width:100%;height:340px;object-fit:contain;background:#eef1f5}svg{width:100%;background:#f8fafc;border:1px solid #dfe6ef}svg text{font:12px system-ui,sans-serif;fill:#344258}.svgtitle{font-weight:700}a{color:#185d9d}code{overflow-wrap:anywhere;font-size:12px}footer{border-top:1px solid #cbd4e0;margin-top:34px;padding-top:10px}@media(max-width:750px){main{padding:16px}.pictures,.charts{grid-template-columns:1fr}header,section.case{padding:16px}h1{font-size:26px}}'''
    document = '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
    document += '<meta name="viewport" content="width=device-width,initial-scale=1">'
    document += f'<title>固定模型：错误分型诊断 | RareVLM</title><style>{css}</style></head><body><main>'
    document += intro + body + footer + '</main></body></html>\n'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document)
    return {"status": "completed_page" if completed else "pending_page", "output": str(output),
            "original_images_checked": len(image_paths), "model_forwards": 0, "training_updates": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(render(args.root, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
