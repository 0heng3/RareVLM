#!/usr/bin/env python3
"""Render saved gradient JSON as a learning page; no tensors/models/forward."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
import math
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUTPUTS = REPO.parents[1] / "outputs"
DEFAULT_CONFIG = REPO / "configs/vlm_object_kl_gradient_v1.json"
STEPS = (25, 100, 175)


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


def number(value):
    if value is None:
        return "待完成 / 未定义"
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        return f"{value:.8g}" if math.isfinite(value) else "无效数值"
    return esc(value)


def link(path, output, label=None):
    path = Path(path).resolve()
    return (f'<a href="{esc(os.path.relpath(path, output.parent))}">{esc(label or path.name)}</a>'
            f' <small>（相对链接 / <a href="{esc(path)}">绝对路径</a>）</small>')


def table(headers, rows):
    return ("<div class='scroll'><table><tr>" + "".join(f"<th>{esc(h)}</th>" for h in headers)
            + "</tr>" + "".join("<tr>" + "".join(f"<td>{value}</td>" for value in row) + "</tr>" for row in rows)
            + "</table></div>")


def details(title, value):
    return "<details><summary>" + esc(title) + "</summary><pre>" + esc(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)) + "</pre></details>"


def source_link(path, output, needle):
    path = REPO / path
    lines = path.read_text().splitlines()
    line = next((i + 1 for i, text in enumerate(lines) if needle in text), 1)
    return link(path, output, f"{path.relative_to(REPO)}:{line}") + "<code>" + esc(f"{path}:{line}") + "</code>"


def point_files(root):
    result = {}
    for group, steps in (("early_late", (25, 175)), ("middle", (100,))):
        for step in steps:
            directory = root / f"groups/{group}/points/step_{step:06d}_block_{step:06d}"
            result[step] = directory
    return result


def load_points(root, analysis):
    directories = point_files(root)
    final = analysis and analysis.get("schema_version") == "rarevlm.object_kl.gradient.analysis.v1" and analysis.get("status") == "completed"
    summary = {point["checkpoint_step"]: point for point in analysis.get("points", [])} if final else {}
    points = []
    for step in STEPS:
        directory = directories[step]
        point = summary.get(step) or read(directory / "POINT.json")
        if (not point or point.get("checkpoint_step") != step or point.get("block_index") != step
                or not isinstance(point.get("geometry"), dict) or not isinstance(point.get("adam_shadow"), dict)):
            points.append({"checkpoint_step": step, "pending": True, "directory": str(directory)})
            continue
        parity = read(directory / "PARITY.json")
        points.append({"checkpoint_step": step, "pending": False, "directory": str(directory),
            "scope": "completed_analysis" if final else "saved_point_only; whole diagnostic pending",
            "geometry": point["geometry"], "adam_shadow": point["adam_shadow"],
            "object_CE": point.get("object_CE"), "replay_KL": point.get("replay_KL"), "costs": point.get("costs"),
            "parity": {key: parity.get(key) for key in ("actual_QA", "max_absolute_KL", "gradient_l2", "active_seconds", "optimizer_updates")} if parity else None,
            "initial_parity_max_absolute_KL": point.get("initial_parity_max_absolute_KL", (parity or {}).get("max_absolute_KL"))})
    return points, bool(final)


def measured_tables(points, output):
    rows = []
    for point in points:
        if point["pending"]:
            rows.append([str(point["checkpoint_step"])] + ["准备中"] * 7)
            continue
        geometry = point["geometry"]
        rows.append([str(point["checkpoint_step"])] + [number(geometry.get(key)) for key in (
            "object_gradient_l2", "KL_gradient_l2", "KL_to_object_norm_ratio", "initial_KL_gradient_floor_l2",
            "cosine_floor_guard_l2", "raw_cosine_if_nonzero", "interpretable_cosine")])
    value = "<h2>三个固定点：实测范数与数值底噪</h2>" + table([
        "checkpoint完成更新", "‖g_object‖", "‖g_KL‖", "KL/Object范数比", "初始KL梯度底噪", "cos解释guard", "原始cos", "可解释cos"], rows)
    value += "<p>guard=max(10×同批初始KL梯度范数, 1e−10)。KL范数≤guard或源梯度为零时，不解释夹角冲突；null保留为未定义，不填0。guard是事前工程数值规则，不是统计显著性阈值；三点不是独立重复。</p>"
    for point in points:
        if point["pending"]:
            continue
        value += details(f"step{point['checkpoint_step']}：实测initial parity与损失", {
            "scope": point["scope"], "object_CE": point["object_CE"], "replay_KL": point["replay_KL"],
            "initial_parity_max_absolute_KL": point["initial_parity_max_absolute_KL"], "initial_parity": point["parity"],
            "cosine_interpretation": point["geometry"].get("cosine_interpretation")})
        value += link(Path(point["directory"]) / "POINT.json", output, f"step{point['checkpoint_step']} POINT元数据")
    return value


def adam_tables(points):
    value = "<h2>真实Adam shadow表：条件一阶方向，不执行更新</h2>"
    value += "<p>读取实际保存的m/v、step、betas、eps和下一步LR，复制后CPU合并梯度、统一clip1并计算方向。这里两列条件于相同历史m/v；λ滑条不修改此表。Adam的坐标级预条件可能把方向带出两梯度张成平面，因此不在二维图中伪造Adam箭头。</p>"
    for point in points:
        if point["pending"]:
            continue
        shadow = point["adam_shadow"]
        rows = []
        for key, label in (("with_current_KL", "当前λ=1"), ("without_current_KL_same_history", "本步省略KL，历史m/v相同")):
            item = shadow[key]
            rows.append([label] + [number(item.get(field)) for field in (
                "delta_theta_l2", "joint_clip_coefficient", "object_gradient_dot_delta_theta", "KL_gradient_dot_delta_theta")])
        value += f"<h3>checkpoint{point['checkpoint_step']} → 条件下一步{shadow.get('next_update_one_based')}</h3>"
        value += table(["已保存shadow方案", "‖Δθ‖", "统一clip系数", "g_object·Δθ", "g_KL·Δθ"], rows)
        value += details("实际shadow超参与边界", {key: shadow.get(key) for key in (
            "next_learning_rate", "saved_betas", "saved_eps", "weight_decay", "clip_norm_cap",
            "current_KL_marginal_direction_difference_l2", "current_KL_marginal_difference_over_joint_norm", "method", "limitations")})
    if all(point["pending"] for point in points):
        value += "<p class='pending'>准备中：尚无可显示的实际shadow数值。</p>"
    return value + "<p>g·Δθ&lt;0只是一阶局部下降预测，&gt;0只是一阶局部上升预测；没有执行step或更新后forward。非线性、高阶项、后续batch和历史状态均未由该符号控制，不是实际损失变化、反事实训练轨迹或遗忘因果。</p>"


def images_panel(points, output):
    value = "<h2>固定block实际训练原图与问答（最多6图）</h2><p>只取每点ACTUAL_INPUTS中object首题和replay首题；没有读取dev选图。完整原图等比显示，非模型crop或注意力。teacher forcing使用原目标前缀，图像展示不是新生成评测。</p><div class='cases'>"
    count = 0
    for point in points:
        records = read(Path(point["directory"]) / "ACTUAL_INPUTS.json")
        if not records:
            continue
        for source in ("object", "replay"):
            if not isinstance(records.get(source), list) or not records[source]:
                continue
            row = records[source][0]
            if not all(key in row for key in ("image_path", "question", "target")):
                continue
            path = Path(row["image_path"])
            value += f"<article><h3>step{point['checkpoint_step']} / {esc(source)}首题</h3>"
            value += f"<img loading='lazy' src='{esc(os.path.relpath(path, output.parent))}' alt='固定训练block实际原图'>"
            value += "<p>" + esc(row["question"]) + "</p><p>训练target：" + esc(row["target"]) + "</p>"
            value += details("实际输入记录的选择字段", {key: row[key] for key in (
                "question_id", "image_id", "assistant_tokens", "input_tokens", "visual_tokens", "image_grid_thw", "input_ids_sha256") if key in row})
            value += link(path, output, "原图文件") + "</article>"
            count += 1
    value += "</div>"
    if not count:
        value += "<p class='pending'>准备中：ACTUAL_INPUTS尚未出现，不补入假设图片。</p>"
    return value, count


def costs_panel(root, analysis, complete, points, output):
    value = "<h2>阶段、操作计数与真实成本</h2>"
    value += "<p>事前计划：72题主诊断F/B +24题同批初始parity F/B =96题新训练QA前后向，2组模型加载；0 optimizer更新、0新生成评测、无heldout性能指标。计划不是当前实测完成量。</p>"
    if complete and complete.get("status") == "completed_diagnostic":
        value += "<p class='badge'>本轮已完成诊断；performance_evaluation=false、goal_complete=false。诊断完成不是性能成功。</p>" + details("本轮COMPLETION", complete)
    else:
        value += details("本轮当前已保存状态（缺失为null）", read(root / "STATUS.json"))
    if analysis and analysis.get("status") == "completed":
        value += details("完整ANALYSIS实际计数", {key: analysis.get(key) for key in (
            "completed_at", "new_training_qa_forward_backward", "main_qa_forward_backward", "initial_parity_qa_forward_backward",
            "optimizer_updates", "generation_questions", "performance_evaluation", "goal_complete", "limitations")})
        value += link(root / "ANALYSIS.json", output, "完整新ANALYSIS")
    rows = []
    for point in points:
        if point["pending"]:
            continue
        cost = point.get("costs") or {}
        rows.append([str(point["checkpoint_step"])] + [number(cost.get(key)) for key in (
            "object_forward_backward_seconds", "KL_forward_backward_seconds", "initial_parity_forward_backward_seconds",
            "CPU_geometry_and_Adam_shadow_seconds", "gradient_FP32_tensor_bytes")])
    value += table(["点", "Object F/B秒", "KL F/B秒", "初始parity F/B秒", "CPU几何/shadow秒", "保存梯度FP32字节"], rows)
    for group in ("early_late", "middle"):
        path = root / f"groups/{group}/RESULT.json"
        result = read(path)
        if not result or result.get("status") != "completed":
            value += "<p class='pending'>" + esc(group) + "组RESULT准备中。</p>"
            continue
        value += details(group + "：实际组成本和计数", {key: result.get(key) for key in (
            "new_training_qa_forward_backward", "main_qa_forward_backward", "initial_parity_qa_forward_backward", "optimizer_updates",
            "generation_questions", "model_loads_current_session", "model_load_seconds_current_session",
            "actual_model_load_starts_all_sessions", "actual_model_load_completions_all_sessions",
            "actual_completed_main_QA_all_sessions", "actual_completed_parity_QA_all_sessions", "source_call_starts_all_sessions",
            "source_call_completions_all_sessions", "session_wall_seconds", "peak_allocated_gib", "peak_reserved_gib", "performance_evaluation")})
        value += link(path, output, group + "组RESULT")
    return value + "<p>源阶段耗时包含编码/H2D/F/B/FP32梯度拷贝/运行状态恢复，不是纯GPU kernel时间；加载、CPU数学、组session和controller壁钟另列。并行组session不能相加当研究壁钟；没有FLOPs或加速结论。本学习页仅JSON读取，新增F/B、GPU、模型加载均为0。</p>"


def history_panel(config, output):
    root = Path(config["source_study_root"])
    path = root / "analysis/pilot_seed42/screen/ANALYSIS.json"
    data = read(path)
    value = "<h2>为什么做诊断：旧KL完整负结果仍保留</h2>"
    if data and data.get("status") == "complete":
        rows = []
        for task in ("object", "text"):
            for row in data["contrasts"][task]:
                if row["slice"] in ("overall", "rare_1_5", "frequent_gt20"):
                    rows.append([task + "/" + row["slice"], number(row["before_score_percent"]), number(row["after_score_percent"]), number(row["delta_pp"]), number(row["one_sided_98_75_lower_pp"])])
        value += table(["旧试验端点", "初始%", "KL后%", "差值pp", "98.75%下界pp"], rows)
    value += "<p>旧物体收益没有伴随Text非劣和rare/frequent点保护通过。低mean KL不等于小梯度，也不保证heldout生成保持；这次只测固定训练batch局部梯度，不重跑dev、外部或POPE。</p>"
    return value + link(REPO / "docs/VLM_OBJECT_KL_V1_RESULTS.md", output, "旧KL负结果") + " · " + link(root / "learning/学习页.html", output, "旧KL真实病例学习页")


INTERACTIVE = r"""
<h2>精确二维几何：两条梯度的张成空间</h2>
<p>x轴沿g_object。令a=‖g_object‖、b=‖g_KL‖、c=实际raw cosine，则g_object=(a,0)，g_KL=(bc,b√(1−c²))。
这是两向量等距几何表达，不是挑两个LoRA坐标或PCA投影；y方向正负只选定平面朝向。共线时空间退化为一维。图中两轴统一除以a并等比例缩放，保留范数比、内积和夹角，不单独放大KL。</p>
<label>实际点：<select id="point"></select></label>
<label>λ教学滑条（原训练λ=1）：<input id="lambda" type="range" min="0" max="5" step="0.05" value="1"><output id="lambdaValue">1</output></label>
<p class="badge">固定θ、固定batch、固定两条实测梯度，只计算g_object+λg_KL；不重算概率、梯度或Adam，不选λ、不新训练。正梯度箭头不是参数更新箭头。</p>
<p id="geometryStatus" aria-live="polite"></p>
<svg id="geometry" viewBox="0 0 900 420" role="img" aria-label="实际梯度张成空间的等比例几何图">
<defs><marker id="blueHead" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="#2864aa"/></marker>
<marker id="orangeHead" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="#b86814"/></marker>
<marker id="greenHead" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="#16806c"/></marker></defs>
<g id="drawing"></g></svg><div id="geometryNumbers"></div>
<p>蓝=g_object，橙=λg_KL，绿=合成梯度，虚线仅作平行四边形辅助。近底噪时，即使原始cos可代数重建，也不作冲突解释；小于像素的KL不夸大绘制。原始cos的±1附近仅作浮点舍入夹紧。</p>
<script>
const points=__POINTS__;
const select=document.getElementById('point'),slider=document.getElementById('lambda');
const finite=x=>typeof x==='number'&&Number.isFinite(x),f=x=>finite(x)?x.toPrecision(8):'未定义';
points.forEach((p,i)=>{let option=document.createElement('option');option.value=i;option.textContent='checkpoint '+p.checkpoint_step+(p.pending?'（准备中）':'（实测）');select.appendChild(option)});
const first=points.findIndex(p=>!p.pending);if(first>=0)select.value=first;
function draw(){const p=points[Number(select.value)],lambda=Number(slider.value);document.getElementById('lambdaValue').textContent=lambda.toFixed(2);
 const status=document.getElementById('geometryStatus'),drawing=document.getElementById('drawing'),numbers=document.getElementById('geometryNumbers');drawing.innerHTML='';numbers.innerHTML='';
 if(!p||p.pending){status.textContent='准备中：所选POINT/完整ANALYSIS尚无实际梯度，不填假设数值。';return}
 const g=p.geometry,a=g.object_gradient_l2,b=g.KL_gradient_l2,raw=g.raw_cosine_if_nonzero;
 if(!finite(a)||!finite(b)||a<=0||b<0||(b>0&&!finite(raw))){status.textContent='实际零范数/未定义夹角：不能把x轴设为非零g_object，不伪造几何。';return}
 const c=b===0?0:Math.max(-1,Math.min(1,raw)),r=b/a,kx=r*c,ky=r*Math.sqrt(Math.max(0,1-c*c)),sx=1+lambda*kx,sy=lambda*ky;
 const scale=285/Math.max(1,lambda*r,Math.hypot(sx,sy)),ox=400,oy=325,X=x=>ox+x*scale,Y=y=>oy-y*scale;
 const line=(x1,y1,x2,y2,color,marker,dashed=false)=>{const tiny=Math.hypot(x2-x1,y2-y1)*scale<10;return '<line x1="'+X(x1)+'" y1="'+Y(y1)+'" x2="'+X(x2)+'" y2="'+Y(y2)+'" stroke="'+color+'" stroke-width="2.5"'+(dashed?' stroke-dasharray="5 5"':'')+(!tiny&&marker?' marker-end="url(#'+marker+')"':'')+'/>'};
 const dot=(x,y,color)=>'<circle cx="'+X(x)+'" cy="'+Y(y)+'" r="3" fill="'+color+'"/>';
 let svg='<line x1="45" y1="'+oy+'" x2="855" y2="'+oy+'" stroke="#b8c5ce"/><line x1="'+ox+'" y1="25" x2="'+ox+'" y2="370" stroke="#b8c5ce"/><text x="730" y="'+(oy+25)+'">e₁：g_object方向</text><text x="410" y="25">e₂</text>';
 const uncertain=g.interpretable_cosine===null;
 svg+=line(0,0,1,0,'#2864aa','blueHead')+line(0,0,lambda*kx,lambda*ky,'#b86814','orangeHead',uncertain)+line(1,0,sx,sy,'#b86814',null,true)+line(lambda*kx,lambda*ky,sx,sy,'#2864aa',null,true)+line(0,0,sx,sy,'#16806c','greenHead')+dot(1,0,'#2864aa')+dot(lambda*kx,lambda*ky,'#b86814')+dot(sx,sy,'#16806c');
 svg+='<text x="'+X(1)+'" y="'+(Y(0)+45)+'" fill="#2864aa">g_object</text><text x="'+(X(lambda*kx)+8)+'" y="'+(Y(lambda*ky)-10)+'" fill="#b86814">λ g_KL</text><text x="'+(X(sx)+8)+'" y="'+(Y(sy)-28)+'" fill="#16806c">g_object+λg_KL</text>';
 drawing.innerHTML=svg;
 status.textContent=(uncertain?'方向处于底噪guard内或不可解释：图只作原始代数重建，不能判冲突。':'以上guard：局部实测梯度几何可描述，负cos仍非遗忘因果。')+(lambda*r*scale<3?' KL线段小于3像素，未单独放大；以数值为准。':'');
 const sumNorm=a*Math.hypot(sx,sy),objDot=a*a+lambda*a*b*c,klDot=a*b*c+lambda*b*b;
 numbers.innerHTML='<pre>横纵坐标单位：‖g_object‖，等比例缩放\ng_object=(1,0)\ng_KL=('+f(kx)+','+f(ky)+')\nλ='+f(lambda)+'，合成=('+f(sx)+','+f(sy)+')\n‖合成梯度‖='+f(sumNorm)+'\ng_object·合成='+f(objDot)+'\ng_KL·合成='+f(klDot)+'\n以上为固定向量合成推导，非新测量或实际Δθ。</pre>';
}
select.onchange=draw;slider.oninput=draw;draw();
</script>
"""


def build(config_path):
    config = read(config_path)
    if not config or config.get("experiment") != "vlm_object_kl_gradient_v1":
        raise ValueError("expected real registered gradient-v1 config")
    root = Path(config["output_root"]).resolve()
    if root != (OUTPUTS / "vlm_object_kl_gradient_v1").resolve():
        raise ValueError("writes only the independent gradient learning page")
    output = root / "learning/学习页.html"
    analysis, complete = read(root / "ANALYSIS.json"), read(root / "COMPLETION.json")
    points, final = load_points(root, analysis)
    body = "<h1>物体CE与固定教师KL：梯度诊断学习页</h1>"
    body += "<p class='badge'>0 optimizer更新，0新生成评测；无heldout性能指标。固定三点局部诊断，不是性能显著改善或目标完成。</p>"
    body += "<p>完整诊断已保存。" if final else "<p class='pending'>准备中 / 部分结果：尚无完整三点ANALYSIS。"
    body += "当前页快照UTC：" + esc(datetime.now(timezone.utc).isoformat()) + "。仅只读JSON元数据，不加载模型、处理器、.pt或GPU。</p>"
    body += "<p>刷新命令：</p><pre>source /opt/data/private/.tools/activate.sh\nsource /opt/data/private/.tools/vlm-a100-env.sh\n&quot;$VLM_A100_PYTHON&quot; " + esc(Path(__file__).resolve()) + "</pre><p>手动重建后刷新浏览器；没有启动watcher。</p>"
    body += measured_tables(points, output)
    data = json.dumps(points, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    body += INTERACTIVE.replace("__POINTS__", data)
    body += adam_tables(points)
    body += "<h2>对齐实际loss与因果边界</h2><pre>g_object = ∇θ [(1/16) Σ_QA (1/Tᵢ) Σ_answer_token CE]\ng_KL = ∇θ [(1/8) Σ_QA (1/Tⱼ) Σ_answer_token Σ_v q_initial(v) log(q_initial(v)/p_actor(v))]\ng_combined = g_object + λg_KL；原训练λ=1，T（蒸馏温度）=1。</pre><p>答案token数包含原结束token；prompt/padding label=−100，label t由logits t−1预测。KL先对完整151936词表求和，不再除词表大小；每题token均值后QA均值。小loss不推出小梯度，负cos只表示固定参数/批次两源梯度内积为负；既不是总体冲突比例，也不是遗忘因果。</p>"
    images, image_count = images_panel(points, output)
    body += images + costs_panel(root, analysis, complete, points, output) + history_panel(config, output)
    body += "<h2>真实源码与协议</h2><ul>"
    for path, needle in (("src/vlm_object_kl_gradient/math.py", "def gradient_geometry"),
                         ("src/vlm_object_kl_gradient/math.py", "def shadow_direction"),
                         ("src/vlm_object_kl_gradient/probe.py", "def initial_parity"),
                         ("src/vlm_object_kl_gradient/probe.py", "def save_parity"),
                         ("src/vlm_object_kl/loss.py", "def per_question_kl"),
                         ("tools/run_object_kl_gradient_study.py", "def aggregate")):
        body += "<li>" + source_link(path, output, needle) + "</li>"
    body += "</ul>" + link(config_path, output, "冻结梯度配置") + " · " + link(config["study_protocol"], output, "事前诊断协议")
    body += "<p>正式checkpoint25/100/175对应下一实际训练block25/100/175（0起始），更新26/101/176。输入和θ均固定后离线重算，不冒充旧训练原始逐源日志；不据本页自动挑λ、epoch、样本或启动训练。原KL负结果不变，真实提升目标仍active。</p>"
    style = "body{font:16px/1.65 system-ui,sans-serif;max-width:1200px;margin:auto;padding:26px;background:#f6f8fa;color:#173143}h2{margin-top:38px;border-bottom:1px solid #ccd8e1}table{border-collapse:collapse;width:100%;font-size:14px;background:white}td,th{border:1px solid #ccd8e1;padding:9px;text-align:left;vertical-align:top}th{background:#e7eef4}pre,code{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.5 monospace}pre{padding:14px;background:#e8eef3;max-height:400px;overflow:auto}code{display:block}details{padding:10px;margin:10px 0;background:white}summary{cursor:pointer}.badge,.pending{background:#fff0c8;padding:13px}.cases{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:18px}.cases article{padding:14px;background:white}.cases img{width:100%;height:340px;object-fit:contain;background:#eee}svg{width:100%;background:white;border:1px solid #ccd8e1}svg text{font:14px system-ui}a{color:#175f97}small{font-size:11px}.scroll{overflow:auto}input,select{margin:8px;font:inherit}input[type=range]{width:300px;vertical-align:middle}label{display:block}"
    page = "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>梯度诊断学习页</title><style>" + style + "</style><body>" + body + "</body></html>"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    print(json.dumps({"HTML": str(output), "bytes": output.stat().st_size,
        "complete_analysis": final, "saved_point_count": sum(not point["pending"] for point in points),
        "native_training_images": image_count, "model_loads": 0, "tensor_cache_reads": 0,
        "model_forwards": 0, "GPU_operations": 0, "watchers_started": 0}, ensure_ascii=False))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    build(args.config.resolve())


if __name__ == "__main__":
    main()
