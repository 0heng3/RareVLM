#!/usr/bin/env python3
"""Render read-only learning evidence; never import a model or execute a forward.

This presentation tool intentionally lives outside the locked scientific source
list. It checks the real study source/config/protocol bytes before and after a
render. Missing runtime artifacts remain pending, and rerunning refreshes them.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime
import hashlib
from html import escape
from html.parser import HTMLParser
import json
import math
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

PRIVATE = Path("/opt/data/private")
PROJECT = Path(__file__).resolve().parents[1]
PILOT = PROJECT.parents[1]
OUTPUTS = PILOT / "outputs"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def persistent(value):
    path = Path(value).resolve()
    if path != PRIVATE and PRIVATE not in path.parents:
        raise ValueError(f"presentation artifact outside persistent workspace: {path}")
    return path


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def pretty(value):
    return escape(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def number(value, digits=3):
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) and not isinstance(value, bool) else "pending"


def details(title, value):
    return f"<details><summary>{escape(title)}</summary><pre>{pretty(value)}</pre></details>"


class Snapshot:
    def __init__(self):
        self.files = {}
        self.pending = []

    def read(self, path, required=False):
        path = persistent(path)
        if not path.is_file():
            if required:
                raise FileNotFoundError(path)
            return None
        raw = path.read_bytes()
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            if required:
                raise
            self.pending.append(str(path))
            return None
        self.files[str(path)] = hashlib.sha256(raw).hexdigest()
        return value

    def lines(self, path):
        path = persistent(path)
        if not path.is_file():
            return []
        raw = path.read_bytes()
        values = []
        for line in raw.splitlines():
            try:
                values.append(json.loads(line))
            except json.JSONDecodeError:
                self.pending.append(str(path))
        self.files[str(path)] = hashlib.sha256(raw).hexdigest()
        return values


def guard(root, config_file):
    lock_path = root / "STUDY_INPUT_LOCK.json"
    lock = json.loads(lock_path.read_text())
    sources = lock["source_sha256"]
    if str(Path(__file__).resolve()) in sources:
        raise ValueError("learning renderer unexpectedly belongs to scientific source lock")
    mismatches = [name for name, digest in sources.items() if sha(name) != digest]
    if mismatches:
        raise ValueError("locked scientific source changed: " + ", ".join(mismatches))
    if sha(config_file) != lock["config_file_sha256"]:
        raise ValueError("locked study config changed")
    for name, digest in lock["protocol_files_sha256"].items():
        if sha(name) != digest:
            raise ValueError("locked protocol changed: " + name)
    return {"scientific_sources_checked": len(sources), "scientific_sources_unchanged": True,
        "config_unchanged": True, "protocol_files_unchanged": True,
        "study_input_lock_sha256": sha(lock_path), "renderer_is_outside_scientific_lock": True}


def source_pages(directory):
    names = ("vlm_object_train/loss.py", "vlm_object_train/train.py", "vlm_deep/common.py",
        "vlm_sft/common.py", "vlm_alignment_signal/loss.py", "vlm_object_eval/analysis.py",
        "vlm_object_presence_v2/data.py")
    links = {}
    for name in names:
        path = PROJECT / "src" / name
        text = path.read_text()
        filename = name.replace("/", "_") + ".html"
        code = "\n".join(f'<span class="line" id="L{index}"><a href="#L{index}">{index:4}</a> {escape(line)}</span>'
            for index, line in enumerate(text.splitlines(), 1))
        page = f'<!doctype html><meta charset="utf-8"><title>{escape(name)}</title><style>body{{font:15px/1.6 monospace;margin:24px;background:#f7f8fa}}pre{{white-space:pre-wrap}}a{{color:#728397;text-decoration:none}}.line:target{{background:#fff0b0}}</style><h2>{escape(name)}</h2><p>只读源码副本；SHA256 {sha(path)}</p><pre>{code}</pre>'
        write(directory / "code" / filename, page)
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                links[f"{name}:{node.name}"] = (f"code/{filename}#L{node.lineno}", node.lineno)
    return links


def code_link(links, name, function):
    href, line = links[f"{name}:{function}"]
    return f'<a href="{href}">{escape(name)}:{line} · {escape(function)}</a>'


def token_decoder(model_root):
    try:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(model_root / "tokenizer.json"))
        return lambda token: tokenizer.decode([int(token)], skip_special_tokens=False)
    except (ImportError, OSError, ValueError):
        return lambda token: "token text unavailable"


def current_state(root, snapshot, cfg):
    state = snapshot.read(root / "STATUS.json") or {}
    preflights, updates, trains, evaluations, first_updates = [], [], [], [], []
    for path in sorted(root.glob("**/PREFLIGHT.json")):
        value = snapshot.read(path)
        if not value:
            continue
        kind = "独立工程预检" if "engineering_runs" in path.parts else "正式训练内置预检" if "train" in path.parts else "评估输入预检"
        if value.get("batches"):
            allocated = max(row.get("batch_peak_allocated_gib", 0) for row in value["batches"])
            reserved = max(row.get("batch_peak_reserved_gib", 0) for row in value["batches"])
        else:
            allocated, reserved = value.get("peak_allocated_gib"), value.get("peak_reserved_gib")
        preflights.append({"path": str(path), "kind": kind, "condition": value.get("condition", "/".join(path.parts[-4:-1])),
            "status": value.get("status"), "allocated": allocated, "reserved": reserved,
            "discarded_steps": value.get("discarded_engineering_optimizer_updates", 0), "record": value.get("record"),
            "restored_exactly": value.get("initial_parameters_restored_exactly"),
            "rng_restored": value.get("rng_restored"),
            "runtime_buffers_and_rope_cache_restored": value.get("model_runtime_buffers_and_rope_cache_restored"),
            "formal_optimizer_updates": value.get("formal_optimizer_updates"), "error": value.get("error")})
    for condition in ("object_only", "object_replay"):
        for seed in (42, 0, 200):
            directory = root / "runs" / condition / f"seed{seed}" / "train"
            result, progress = snapshot.read(directory / "TRAIN_RESULT.json"), snapshot.read(directory / "PROGRESS.json")
            first = snapshot.read(directory / "FIRST_UPDATE.json")
            if first:
                first_updates.append({"condition": condition, "seed": seed, "source_file": str(directory / "FIRST_UPDATE.json"), **first})
            if result:
                trains.append({"condition": condition, "seed": seed, "source_file": str(directory / "TRAIN_RESULT.json"), **result})
            value = result or progress
            updates.append({"condition": condition, "seed": seed, "completed": value.get("completed_updates") if value else None,
                "planned": value.get("planned_updates") if value else None, "status": value.get("status") if value else "pending"})
    for path in sorted(root.glob("runs/**/EVALUATION.json")):
        value = snapshot.read(path)
        if value:
            evaluations.append({"path": str(path), "condition": value.get("condition"), "seed": value.get("seed"),
                "task": value.get("task", path.parent.name), "score_percent": value.get("score_percent"),
                "questions": value.get("questions"), "status": value.get("status")})
    history = [row for path in root.glob("**/PREFLIGHT_HISTORY.jsonl") for row in snapshot.lines(path)]
    return {"state": state, "preflights": preflights, "updates": updates, "trains": trains,
        "evaluations": evaluations, "first_updates": first_updates,
        "actual_discarded_engineering_steps": sum(row.get("discarded_engineering_optimizer_updates", 0) for row in history)}


def audit_cards(cfg, snapshot, directory, decode):
    from PIL import Image
    data = persistent(cfg["object_data_root"])
    audit = snapshot.read(data / "semantic_audit/FROZEN_64.json", True)
    review = snapshot.read(OUTPUTS / "vlm_object_presence_review_v2/SEMANTIC_REVIEW.json", True)
    if sha(data / "semantic_audit/FROZEN_64.json") != review["source_audit_sha256"]:
        raise ValueError("semantic review differs from fixed64 source")
    raw = {row["question_id"]: row for row in audit["rows"]}
    chosen = [row for row in sorted(review["rows"], key=lambda row: row["audit_index"]) if row["judgment"] == "clear"][:4]
    qids = {row["question_id"] for row in chosen}
    encodings = {row["question_id"]: row for row in snapshot.lines(data / "ENCODING_CHECKS.jsonl") if row["question_id"] in qids}
    cards, bindings = [], []
    for item in chosen:
        row, enc = raw[item["question_id"]], encodings[item["question_id"]]
        source = persistent(row["image_path"])
        if sha(source) != row["image_sha256"]:
            raise ValueError("teaching original image bytes changed")
        with Image.open(source) as image:
            width, height, fmt = *image.size, image.format
        extension = ".png" if fmt == "PNG" else ".jpg" if fmt == "JPEG" else ".image"
        target = directory / "images" / f"audit_{item['audit_index']:02d}{extension}"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or sha(target) != sha(source):
            shutil.copyfile(source, target)
        labels, ids = enc["labels"], enc["input_ids"]
        valid = [(index, label) for index, label in enumerate(labels) if label != -100]
        if len(valid) != enc["assistant_tokens"] or any(ids[index] != label for index, label in valid):
            raise ValueError("saved assistant mask/labels differ from actual IDs")
        tail = []
        for index in range(max(0, len(ids) - 18), len(ids)):
            selected = labels[index] != -100
            token = decode(ids[index])
            tail.append(f'<span class="token {"supervised" if selected else "masked"}" title="position {index}; label {labels[index]}">{ids[index]}<small>{escape(json.dumps(str(token), ensure_ascii=False))}</small></span>')
        boxes = "".join(f'<rect x="{box["bbox_current_image_xywh"][0]}" y="{box["bbox_current_image_xywh"][1]}" width="{box["bbox_current_image_xywh"][2]}" height="{box["bbox_current_image_xywh"][3]}"/>' for box in row["bbox_evidence"])
        overlay = f'<svg class="box-overlay" viewBox="0 0 {width} {height}" aria-label="COCO审查框">{boxes}</svg>'
        assistant = [{"position": index, "id": value, "token": decode(value)} for index, value in valid]
        cards.append(f'<article class="sample"><div class="photo"><img src="images/{target.name}" alt="审查样本{item["audit_index"]}原图">{overlay}</div><div class="sample-content"><span class="pill">固定审查 #{item["audit_index"]} · target={row["target"]}</span><h3>{escape(row["question"])}</h3><p>{escape(item["reason_zh"])}</p><p class="muted">助手审查结论 clear；展示原图 {width}×{height}。框仅辅助审查，模型接收整图。</p><div class="stats"><span>输入 {enc["sequence_length"]} token</span><span>视觉 {enc["visual_tokens"]} token</span><span>监督 {enc["assistant_tokens"]} token</span></div><div class="tokens">{"".join(tail)}</div><p class="muted">灰色为不计loss的prefix，绿色为真实assistant监督；这里只展示序列末尾18个token。</p>{details("展开实际 input_ids / labels 与像素、grid锁（fresh CPU编码，不含forward）", {"question_id": row["question_id"], "prompt_suffix": row["prompt_suffix"], "assistant_targets": assistant, **enc})}</div></article>')
        bindings.append({"audit_index": item["audit_index"], "question_id": row["question_id"], "target": row["target"],
            "original_image": str(source), "copied_image": str(target), "image_sha256": sha(target),
            "sequence_length": enc["sequence_length"], "visual_tokens": enc["visual_tokens"], "assistant_tokens": len(valid)})
    return "".join(cards), bindings, review


def render(config_file):
    snapshot = Snapshot()
    config_file = persistent(config_file)
    cfg = snapshot.read(config_file, True)
    root = persistent(cfg["output_root"])
    initial_guard = guard(root, config_file)
    directory = root / "learning"
    directory.mkdir(parents=True, exist_ok=True)
    links = source_pages(directory)
    decode = token_decoder(persistent(cfg["model_root"]))
    cards, image_bindings, review2 = audit_cards(cfg, snapshot, directory, decode)
    review1 = snapshot.read(OUTPUTS / "vlm_object_presence_review_v1/SEMANTIC_REVIEW.json", True)
    encoding = snapshot.read(persistent(cfg["object_data_root"]) / "ENCODING_SUMMARY.json", True)
    manifest = snapshot.read(persistent(cfg["train_manifest"]), True)
    validation = snapshot.read(persistent(cfg["val_manifest"]), True)
    balanced = snapshot.read(persistent(cfg["val_class_balanced_manifest"]), True)
    runtime = current_state(root, snapshot, cfg)
    mechanism = snapshot.read(OUTPUTS / "vlm_public_posttrain_failure_review_v1/MECHANISM.json", True)
    previous = snapshot.read(OUTPUTS / "vlm_public_posttrain_failure_review_v1/SUMMARY.json", True)
    prediction = snapshot.read(OUTPUTS / "vlm_public_posttrain_failure_review_v1/PREDICTION_DIAGNOSTICS.json", True)
    count, updates = len(manifest["rows"]), math.ceil(len(manifest["rows"]) / cfg["object_effective_batch"])
    replay = updates * cfg["replay_per_update"]
    completed = sum(row["completed"] or 0 for row in runtime["updates"])
    preflight_rows = "".join(f'<tr><td>{escape(row["kind"])}</td><td>{escape(str(row["condition"]))}</td><td>{escape(str(row["status"]))}</td><td>{number(row["allocated"])} / {number(row["reserved"])}</td><td>{row["discarded_steps"]}</td></tr>' for row in runtime["preflights"])
    if not preflight_rows:
        preflight_rows = '<tr><td colspan="5">pending：尚未写出完整PREFLIGHT.json；不把模型加载或排队写成预检通过。</td></tr>'
    update_rows = "".join(f'<tr><td>{row["condition"]}</td><td>{row["seed"]}</td><td>{row["completed"] if row["completed"] is not None else "pending"} / {row["planned"] if row["planned"] is not None else updates}</td><td>{escape(str(row["status"]))}</td></tr>' for row in runtime["updates"])
    eval_rows = "".join(f'<tr><td>{escape(str(row["condition"]))}</td><td>{row["seed"]}</td><td>{escape(str(row["task"]))}</td><td>{row["questions"]}</td><td>{number(row["score_percent"], 6)}</td></tr>' for row in runtime["evaluations"])
    if not eval_rows:
        eval_rows = '<tr><td colspan="5">pending：新候选没有完整评估分数，当前不能宣称提分或能力保持。</td></tr>'
    first_blocks = []
    for value in runtime["first_updates"]:
        records = value.get("records", [])
        first_blocks.append(f'<article class="panel"><h3>{value["condition"]} / seed{value["seed"]} · 实际首更新</h3><p>真实object={value.get("object_count")} QA、replay={value.get("replay_count")} QA；lr={value.get("learning_rate")}；clip前梯度L2={number(value.get("gradient_l2_before_clip"), 6)}。</p><p class="muted">FIRST_UPDATE保存实际输入、labels与梯度；没有逐题logp/CE字段，逐题实测CE仍为pending，不作额外forward重算。</p>{details("展开实际首更新IDs/labels记录", value)}</article>')
    if not first_blocks:
        first_blocks = ['<div class="notice">实际首更新输入、梯度、logp/CE：pending。已有fresh CPU token合同不冒充实际GPU首更新。</div>']
    train_summaries = [{key: row.get(key) for key in ("condition", "seed", "completed_updates", "actual_object_exposures", "actual_replay_exposures", "mean_object_ce", "mean_replay_ce", "training_object_input_tokens", "training_replay_input_tokens", "training_object_microbatches", "training_replay_microbatches", "peak_gpu_allocated_gib", "peak_gpu_reserved_gib", "compute_accounting")} for row in runtime["trains"]]
    review_rows = "".join(f'<tr><td>{name}</td><td>{item["counts"]["clear"]}/{item["counts"]["total"]}</td><td>{number(item["clear_percent"], 4)}%</td><td>[{number(item["wilson95_percent"][0], 3)}, {number(item["wilson95_percent"][1], 3)}]%</td><td>{"通过" if item["gate"]["passed"] else "拒绝"}</td></tr>' for name, item in (("v1 原总体", review1), ("v2 新总体", review2)))
    initial_pope = prediction["pope"]["models"]["initial"]["overall"]
    dpo_pope = prediction["pope"]["models"]["dpo"]["overall"]
    pref = mechanism["heldout"]["summary"]["dpo"]
    data_train = mechanism["data_distribution"]["train"]
    ce_code = code_link(links, "vlm_object_train/loss.py", "per_question_ce")
    loss_code = code_link(links, "vlm_object_train/loss.py", "normalized_qa_loss")
    micro_code = code_link(links, "vlm_object_train/train.py", "backward_update")
    mask_code = code_link(links, "vlm_sft/common.py", "answer_labels")
    encode_code = code_link(links, "vlm_deep/common.py", "encode")
    scoring_code = code_link(links, "vlm_object_eval/analysis.py", "screen_gate")
    confirm_code = code_link(links, "vlm_object_eval/analysis.py", "confirm_gate")
    qualify_code = code_link(links, "vlm_object_presence_v2/data.py", "qualifies")
    now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    analysis_payload = []
    for path in sorted((root / "analysis").glob("**/ANALYSIS.json")):
        value = snapshot.read(path)
        if value:
            analysis_payload.append({"path": str(path), "status": value.get("status"), "stage": value.get("stage"),
                "eligible_candidates": value.get("eligible_candidates"), "selected_candidate": value.get("selected_candidate"),
                "gates": value.get("gates"), "true_improvement_goal_achieved": value.get("true_improvement_goal_achieved")})
    body = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>真实物体问答、逐题CE与Text回放</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f4f3ee;color:#142e35;font:16px/1.75 system-ui,sans-serif}}main{{max-width:1160px;margin:auto;padding:32px 24px 80px}}h1{{font-size:36px;line-height:1.3;max-width:850px;margin:10px 0 22px}}h2{{margin:44px 0 16px;font-size:25px;border-top:1px solid #ccd5d2;padding-top:26px}}h3{{font-size:19px;margin:12px 0}}p{{margin:10px 0}}a{{color:#087c82}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#edf1f1;padding:14px;font:12px/1.6 monospace;max-height:420px;overflow:auto}}code{{background:#e9eeeb;padding:2px 5px;border-radius:4px}}details{{border-top:1px solid #dce3df;margin-top:12px;padding-top:10px}}summary{{cursor:pointer;color:#087c82}}.muted,small{{color:#566d72;font-size:13px}}.eyebrow{{letter-spacing:.15em;font-size:12px;color:#42736a}}.hero,.panel{{background:#fff;border:1px solid #d4ded9;border-radius:15px;padding:22px;margin:16px 0}}.hero{{border-left:6px solid #11998e}}.notice{{background:#fff2cf;border-left:4px solid #cc9e2d;padding:14px 18px;margin:15px 0}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px}}.sample{{background:#fff;border:1px solid #d4ded9;border-radius:15px;overflow:hidden}}.photo{{position:relative;line-height:0;background:#142e35}}.photo img{{width:100%;height:auto;display:block}}.box-overlay{{position:absolute;inset:0;width:100%;height:100%;display:none;pointer-events:none}}.show-boxes .box-overlay{{display:block}}.box-overlay rect{{fill:none;stroke:#ffde59;stroke-width:4;vector-effect:non-scaling-stroke}}.sample-content{{padding:20px}}.pill{{display:inline-block;background:#e1f4ec;color:#145642;border-radius:20px;padding:3px 10px;font-size:12px}}.stats{{display:flex;flex-wrap:wrap;gap:8px;font-size:12px;margin:12px 0}}.stats span{{background:#edf1f0;padding:3px 8px;border-radius:5px}}.tokens{{display:flex;gap:4px;flex-wrap:wrap}}.token{{font:11px monospace;padding:4px;border-radius:4px;border:1px solid #cbd4d5;min-width:40px;text-align:center}}.token small{{display:block;font-size:9px;max-width:100px;overflow:hidden}}.masked{{color:#728188;background:#edf0f1}}.supervised{{background:#dcf8e7;border-color:#28a16a;color:#14502c}}table{{width:100%;border-collapse:collapse;font-size:14px;background:#fff}}td,th{{padding:10px 12px;border-bottom:1px solid #e0e6e2;text-align:left;vertical-align:top}}.table-wrap{{overflow:auto}}th{{background:#e8efeb}}.equation{{font:20px/2 Georgia,serif;background:#e8f4ef;border-radius:10px;padding:14px;overflow:auto}}.split{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}label{{display:block;margin:10px 0}}input[type=number]{{width:85px;padding:8px;border:1px solid #afc6c2;border-radius:5px}}input[type=range]{{width:100%}}.bar{{display:flex;height:32px;background:#e6eae6;border-radius:7px;overflow:hidden}}.bar span{{display:block;min-width:0}}#object-bar{{background:#159a8c}}#replay-bar{{background:#e6a54b}}.big{{font-size:30px;font-weight:650}}.footer{{font-size:12px;color:#58706f;margin-top:40px}}@media(max-width:760px){{.grid,.split{{grid-template-columns:1fr}}h1{{font-size:29px}}main{{padding:20px 14px}}}}
</style><main><div class="eyebrow">RAREVLM · 真实记录学习页 · V2</div><h1>从一张真实图片，走到逐题交叉熵与Text回放</h1><p>面向已有Python／PyTorch基础的逐步解释。页面生成时间 {now}；只读本地证据，0新forward、0新训练。</p>
<div class="hero"><p><strong>当前阶段：</strong>{escape(str(runtime["state"].get("status", "pending")))}</p><p>已保存正式更新合计 <strong>{completed}</strong>；工程临时step实际记录 <strong>{runtime["actual_discarded_engineering_steps"]}</strong>（已丢弃／恢复，不能计为正式更新）。未写出的字段均为pending。</p><p class="muted">源锁检查：{initial_guard["scientific_sources_checked"]}个科学源码、配置和协议字节一致。新候选是否真实提分，须等完整内部筛选和全9000题POPE确认。</p></div>
<h2>01 · 真实图片与固定监督</h2><p>取v2固定64样本中按审查序号最先出现的4个clear样本，不按模型结果挑图。图像保持原始字节，问题与yes/no来自固定COCO规则；助手审查可复核，不能当独立人工真值。</p><label><input id="boxes-toggle" type="checkbox"> 显示COCO框：仅审查展示，模型仍接收整图；负例无目标框。</label><div class="grid">{cards}</div>
<h2>02 · 哪些token参与loss？</h2><div class="panel"><p>完整训练序列包含图像占位、用户问题、固定yes/no提示和assistant答案。只有assistant区域的label保留真实token ID，其余设为<code>−100</code>。</p><p>本轮四例的assistant监督均含3个token：<code>no=2152</code>或<code>yes=9693</code>，再加<code>&lt;|im_end|&gt;=151645</code>与换行<code>198</code>。这三个token都计入Tᵢ，不能只训练yes/no而漏掉结束标记。</p><p>前向输出向右移一位对齐下一个token，prompt与padding的−100不计入logp。保存的CPU合同实际序列长度{encoding["sequence_length_min"]}–{encoding["sequence_length_max"]}，视觉token {encoding["visual_tokens_min"]}–{encoding["visual_tokens_max"]}；1024只是上限，3230次全新CPU编码均未截断。</p><p>代码：{encode_code}；{mask_code}；{code_link(links, "vlm_alignment_signal/loss.py", "sequence_logps")}。</p></div>
<h2>03 · 从一个token到一次optimizer.step</h2><div class="equation">ℓᵢ = − (1 / Tᵢ) ∑<sub>t∈assistantᵢ</sub> log p<sub>θ</sub>(yᵢ,ₜ | imageᵢ, questionᵢ, yᵢ,&lt;ₜ)<br>L<sub>object</sub> = (1 / N<sub>object</sub>) ∑ᵢ ℓᵢ<br>L<sub>A</sub> = L<sub>object</sub>； L<sub>B</sub> = L<sub>object</sub> + 1 × L<sub>Text replay</sub></div><p>先对每题有效assistant token取均值，再按真实问题数取均值。Text答案长度可不同；按题平均避免长答案仅因token更多而自动占更大权重。代码：{ce_code}；{loss_code}。</p>
<div class="split"><div class="panel"><h3>真实协议的分母</h3><p>每次更新object最多16题，由microbatch4×4累积；B额外8题Text，由microbatch4×2累积。每个object微批贡献<code>∑ℓ/16</code>，每个replay微批贡献<code>∑ℓ/8</code>，最后只做一次optimizer.step。</p><p>训练{count}题，共{updates}更新；最后object块实际{count % 16 or 16}题，分母随之改为{count % 16 or 16}，replay仍8题。两个来源独立归一，不能把它们合成24题统一均值。代码：{micro_code}。</p></div><div class="panel"><h3>交互教学：两项loss的比例</h3><p class="muted">以下CE数值是手填教学假设，默认1与0.5，绝非模型实测；滑块不修改协议、不重训、不改任何实验文件。实际训练λ固定1。</p><label>假设 object mean CE <input id="object-ce" type="number" min="0" max="20" step="0.1" value="1"></label><label>假设 replay mean CE <input id="replay-ce" type="number" min="0" max="20" step="0.1" value="0.5"></label><label>教学 λ = <output id="lambda-output">1.00</output><input id="lambda" type="range" min="0" max="2" step="0.01" value="1"></label><div class="bar"><span id="object-bar"></span><span id="replay-bar"></span></div><p><span style="color:#159a8c">object</span> + <span style="color:#a56816">λ × replay</span> = <strong id="total-loss"></strong></p><p id="fraction-note" class="muted"></p></div></div>
<div class="table-wrap"><table><thead><tr><th>条件</th><th>object监督</th><th>额外Text监督</th><th>正式更新</th></tr></thead><tbody><tr><td>A · object_only</td><td>{count} QA</td><td>0</td><td>{updates}（计划）</td></tr><tr><td>B · object_replay</td><td>同{count} QA／顺序</td><td>{replay} QA曝光</td><td>同{updates}（计划）</td></tr></tbody></table></div><p class="notice">B多了Text数据、token和前后向计算。相同object顺序、更新数、LoRA参数量与视觉上限不代表等数据或等算力；结果只能解释这套回放方案的整体作用。</p>
<h2>04 · 真实GPU预检、更新与输入记录</h2><p>独立工程预检与正式训练内置预检分别记录；临时optimizer step恢复原LoRA、RNG与运行时buffer，不能混入正式训练step。65GiB为allocated/reserved上限，超过即工程拒绝，不自动降低microbatch。</p><div class="table-wrap"><table><thead><tr><th>记录类型</th><th>条件／role</th><th>状态</th><th>allocated / reserved GiB</th><th>丢弃step</th></tr></thead><tbody>{preflight_rows}</tbody></table></div><div class="table-wrap"><table><thead><tr><th>条件</th><th>order seed</th><th>完成／计划更新</th><th>状态</th></tr></thead><tbody>{update_rows}</tbody></table></div>{"".join(first_blocks)}{details("已完成TRAIN_RESULT的真实CE／token／额外计算汇总（缺项pending）", train_summaries if train_summaries else {"status": "pending", "mean_object_ce": None, "mean_replay_ce": None})}{details("完整预检快照与真实输入检查", runtime["preflights"])}
<h2>05 · 为什么从上一轮DPO失败走到这一步？</h2><div class="panel"><p>旧DPO确实学到相对偏好：heldout {pref["reference_centered_positive_count"]}/439对margin改善，均值{number(pref["reference_centered_margin"]["mean"], 5)}。但公开chosen平均{number(data_train["chosen_tokens"]["mean"], 2)} token，{number(data_train["chosen_longer_than_32_tokens_percent"], 2)}%超过32 token，和短答／物体存在问答的目标有真实差异；因果份额尚未由消融隔离。</p><p>旧POPE初始FP={initial_pope["FP"]}、FN={initial_pope["FN"]}，漏检占错误{number(previous["pope_initial_missed_objects_share_of_errors_percent"], 2)}%。DPO后FP={dpo_pope["FP"]}、FN={dpo_pope["FN"]}，少3个误报同时多3个漏检，accuracy仍87.1%。新监督因此同时保留真实正例与不存在负例；是否改善仍需完整评估。</p><p class="muted">这不是已证明DPO无效，也没有证明只换数据或加回放就一定成功。新对照改变任务监督，B另加Text回放；不把多个变化归为单一原因。</p></div>
<h2>06 · v1拒绝与v2准入：不同总体，不能算配对提质</h2><div class="table-wrap"><table><thead><tr><th>固定审查</th><th>明确／总数</th><th>明确比例</th><th>Wilson95%区间</th><th>同一质量门槛</th></tr></thead><tbody>{review_rows}</tbody></table></div><p>v1保留质量拒绝，不启动建模；v2另限定18类动物／交通、正例短边≥24px且分割面积比例≥0.005，并排除全部旧64审查图。新64中的61例clear不能解释为旧64配对提升，也不是全量语义准确率或随机错标率。代码：{qualify_code}。</p>
<h2>07 · 怎样才算真实提升？</h2><p>完整内部object验证{len(validation["rows"])}题，另有平衡诊断{len(balanced["rows"])}题（从同一完整预测按固定qid取子集，0新增生成）。pilot与三seed都须平衡子集点差&gt;0；内部accuracy≥+1pp、单侧98.75%图像簇下界&gt;0，同时保护Text。通过内部三seed筛选后，按object→Text→固定名称选一个候选，才做三seed全POPE9000确认。</p><p>POPE确认要求均值至少+0.5pp及单侧98.75%下界&gt;0，三seed准确率点差全&gt;0、三category均值不降，并保留F1／recall／specificity和Text／rare／frequent保护。6000次bootstrap按图像整簇抽样；POPE三类共享500图，不按1500图或seed×image计独立样本。代码：{scoring_code}；{confirm_code}。</p><div class="table-wrap"><table><thead><tr><th>新条件</th><th>seed</th><th>完整任务</th><th>题数</th><th>保存分数%</th></tr></thead><tbody>{eval_rows}</tbody></table></div>{details("已保存的阶段分析（未写出时为pending）", analysis_payload if analysis_payload else {"status": "pending"})}<p class="muted">共同强初始Text87.111936%／POPE87.1%是显式复用的历史预测，不是这页新跑出的基线；新object验证初始仍须真实生成。POPE曾在旧研究暴露，不能称盲测或预训练未见。成功标准不等于用户已独立掌握，最终结论由完整证据与独立审查决定。</p>
<div class="footer">刷新命令：<code>PYTHONPATH=src "$VLM_A100_PYTHON" tools/build_object_adapt_learning.py --config {escape(str(config_file))}</code><br>本页是独立展示产物，不在scientific source lock列表；生成器只写learning目录。原图、CPU输入合同、GPU首更新与教学假设均已明确区分。</div></main><script>
const update=()=>{{const a=Math.max(0,Number(document.getElementById('object-ce').value)||0),b=Math.max(0,Number(document.getElementById('replay-ce').value)||0),l=Number(document.getElementById('lambda').value),total=a+l*b,ratio=total?a/total:0,replayRatio=total?l*b/total:0;document.getElementById('lambda-output').textContent=l.toFixed(2);document.getElementById('total-loss').textContent=total.toFixed(4);document.getElementById('object-bar').style.width=(ratio*100)+'%';document.getElementById('replay-bar').style.width=(replayRatio*100)+'%';document.getElementById('fraction-note').textContent=total?'教学计算贡献：object '+(ratio*100).toFixed(1)+'%，replay '+(replayRatio*100).toFixed(1)+'%。不是梯度贡献或真实训练曲线。':'教学总loss为0，比例无定义。不是梯度贡献或真实训练曲线。';}};['object-ce','replay-ce','lambda'].forEach(id=>document.getElementById(id).addEventListener('input',update));document.getElementById('boxes-toggle').addEventListener('change',event=>document.body.classList.toggle('show-boxes',event.target.checked));update();
</script></html>'''
    html = directory / "学习页.html"
    write(html, body)
    final_guard = guard(root, config_file)
    proof = {"schema_version": "rarevlm.object_adapt.learning.v1", "generated_at": now,
        "new_model_forwards": 0, "new_optimizer_updates": 0, "scientific_files_modified": False,
        "renderer_sha256": sha(Path(__file__)), "html_path": str(html), "html_sha256": sha(html),
        "source_guard_before": initial_guard, "source_guard_after": final_guard,
        "input_files_sha256": snapshot.files, "pending_partial_json_files": snapshot.pending,
        "original_image_bindings": image_bindings, "stage_snapshot": runtime["state"],
        "actual_completed_formal_updates_in_saved_records": completed,
        "actual_discarded_engineering_steps_in_saved_history": runtime["actual_discarded_engineering_steps"]}
    # Validate every local image/source anchor without launching a browser/model.
    class Links(HTMLParser):
        def __init__(self):
            super().__init__()
            self.links = []
        def handle_starttag(self, tag, attrs):
            self.links.extend(value for key, value in attrs if key in ("src", "href") and value)
    parser = Links()
    parser.feed(body)
    broken = []
    for link in parser.links:
        if link.startswith(("https:", "http:", "#", "data:")):
            continue
        target = directory / link.split("#", 1)[0]
        if not target.is_file():
            broken.append(link)
    if broken:
        raise ValueError("broken learning-page links: " + repr(broken))
    proof["local_links_checked"] = len(parser.links)
    proof["local_links_broken"] = []
    write(directory / "LEARNING_BUILD.json", json.dumps(proof, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"html": str(html), "source_guard": final_guard, "images": len(image_bindings),
        "local_links_checked": len(parser.links), "formal_updates_snapshot": completed,
        "preflights_snapshot": len(runtime["preflights"]), "new_forwards": 0}, ensure_ascii=False), flush=True)
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/vlm_object_adapt_v2.json")
    args = parser.parse_args()
    render(args.config)


if __name__ == "__main__":
    main()
