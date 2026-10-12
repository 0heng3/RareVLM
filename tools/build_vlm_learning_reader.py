"""Build a self-contained Chinese learning reader from verified project evidence.

Only reads project/code/result files and embeds images in one offline HTML.
No model loading, generation, training, network access, or frozen-source changes.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools.layout_paths import project_file, verify_source
DEFAULT_ROOT = REPO.parents[1] / "outputs/vlm_learning_v1"


def esc(value):
    return html.escape(str(value), quote=True)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def number(value, decimals=3):
    if value is None or value == "":
        return "—"
    return f"{float(value):.{decimals}f}"


def table(headers, rows, caption=""):
    title = f"<caption>{esc(caption)}</caption>" if caption else ""
    return '<div class="table-scroll"><table>' + title + "<thead><tr>" + "".join(f'<th scope="col">{esc(c)}</th>' for c in headers) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in row) + "</tr>" for row in rows) + "</tbody></table></div>"


def paragraphs(values, cls=""):
    return "".join(f'<p class="{esc(cls)}">{esc(value)}</p>' for value in values)


def list_html(values):
    return "<ul>" + "".join(f"<li>{esc(value)}</li>" for value in values) + "</ul>"


def raw_json(value, title="展开原始证据 JSON"):
    return f'<details class="raw"><summary>{esc(title)}</summary><pre>{esc(json.dumps(value, ensure_ascii=False, indent=2))}</pre></details>'


CSS = r"""
:root{--ink:#193247;--muted:#526575;--green:#086d63;--blue:#255a90;--gold:#925019;--line:#d8e3e9;--paper:#fff;--wash:#f4f7f9;--code:#122638}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:96px}body{margin:0;background:var(--wash);color:var(--ink);font:18px/1.8 system-ui,-apple-system,"Segoe UI","Noto Sans CJK SC","Microsoft YaHei",sans-serif}a{color:var(--blue);text-underline-offset:3px}button,input,select,textarea{font:inherit}button{cursor:pointer}a:focus-visible,button:focus-visible,input:focus-visible,select:focus-visible,summary:focus-visible{outline:3px solid #d08628;outline-offset:4px}h1,h2,h3,h4{line-height:1.35;letter-spacing:.01em}h1{font-size:clamp(32px,4vw,48px);margin:12px 0 20px}h2{font-size:32px;margin:0 0 12px}h3{font-size:23px;margin:30px 0 12px}h4{font-size:19px;margin:12px 0}p{margin:0 0 16px}ul{padding-left:1.35em;margin:0 0 18px}li{margin:8px 0}code{font: .94em ui-monospace,SFMono-Regular,Consolas,monospace;background:#eaf0f4;border-radius:4px;padding:1px 4px}pre{margin:12px 0;background:var(--code);color:#e8f2f6;border-radius:12px;padding:18px;overflow:auto;font:14px/1.7 ui-monospace,SFMono-Regular,Consolas,monospace;tab-size:4;max-height:640px}pre code{background:none;padding:0;color:inherit}details{margin:16px 0}summary{cursor:pointer;color:var(--blue);font-weight:650}summary::marker{color:var(--green)}figure{margin:24px 0}figcaption{font-size:16px;color:var(--muted);padding-top:12px}img{display:block;width:100%;height:auto;border-radius:12px;border:1px solid var(--line);background:#fff}table{border-collapse:collapse;width:100%;font-size:16px;line-height:1.6}caption{text-align:left;font-weight:650;color:var(--ink);padding:0 0 10px}th,td{padding:11px 13px;border:1px solid var(--line);vertical-align:top;text-align:left}th{background:#eaf2f4;font-weight:650}tbody tr:nth-child(even){background:#f8fafb}.table-scroll{overflow-x:auto;margin:18px 0;border-radius:8px}.shell{max-width:1200px;margin:auto;padding:34px 26px 70px}.hero,.lesson,.case,.bridge,.evidence{background:var(--paper);border:1px solid var(--line);border-radius:20px;padding:34px;margin:0 0 26px;box-shadow:0 5px 18px #16344905}.eyebrow{color:var(--green);font-weight:750;font-size:15px;letter-spacing:.08em}.lead{font-size:21px;max-width:940px}.meta{color:var(--muted);font-size:15px;overflow-wrap:anywhere}.status{display:inline-block;font-size:14px;font-weight:700;line-height:1.5;padding:5px 11px;border-radius:30px;margin:0 0 14px}.measured{background:#dcf1e8;color:#075b45}.planned{background:#fff0da;color:#85500e}.mixed{background:#e9f0fb;color:#315b91}.callout{border-left:5px solid var(--green);background:#eef8f3;border-radius:0 12px 12px 0;padding:18px 21px;margin:20px 0}.boundary{border-left-color:#b06b22;background:#fff6e9}.question{font-size:22px;font-weight:650;margin:12px 0 22px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px}.stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:24px 0}.stat{border:1px solid var(--line);border-radius:12px;padding:16px;background:#f9fcfc}.stat strong{display:block;color:var(--green);font-size:26px;line-height:1.2}.stat span{display:block;color:var(--muted);font-size:14px;margin-top:10px}.nav{position:sticky;top:0;z-index:10;background:#f4f7f9f5;backdrop-filter:blur(8px);border-bottom:1px solid var(--line);padding:10px 0;overflow:auto}.nav-inner{max-width:1200px;margin:auto;display:flex;gap:8px;align-items:center;padding:0 26px;white-space:nowrap}.nav a{text-decoration:none;padding:5px 12px;border-radius:30px;border:1px solid var(--line);font-size:15px;background:#fff}.nav a:hover{background:#e7f3ef}.skip{position:absolute;left:20px;top:-80px;background:#fff;z-index:20;padding:8px}.skip:focus{top:12px}.formula{font-size:21px;line-height:1.9;background:#f0f5fa;border:1px solid #dbe6f0;border-radius:12px;padding:16px 20px;margin:12px 0;overflow:auto;white-space:normal}.formula sub,.formula sup{font-size:.68em}.code-card{border:1px solid var(--line);border-radius:12px;padding:20px;margin:18px 0}.source{font-size:14px;color:var(--muted);word-break:break-all}.hash{font:13px ui-monospace,monospace;word-break:break-all}.calculator{border:1px solid #b8d9d1;border-radius:14px;background:#f4faf7;padding:23px;margin:26px 0}.calculator label{display:block;margin:0 0 8px;font-weight:650;font-size:16px}.calculator input,.calculator select{width:100%;min-height:44px;border:1px solid #a7bbb8;border-radius:7px;padding:6px 10px;background:#fff;color:var(--ink)}.calculator .controls{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.calc-output{background:#fff;border:1px solid #d2e4df;border-radius:10px;padding:16px;margin-top:18px;font-size:17px}.calc-output strong{font-size:23px;color:var(--green)}.button{display:inline-block;border:1px solid #aac8c3;background:#e5f3ed;border-radius:8px;padding:7px 13px;color:#0c5e50;font-weight:650}.zoom-scroll{overflow:auto;max-height:800px;border:1px solid var(--line);border-radius:10px}.zoom-scroll img{max-width:none;width:1600px;border:0;border-radius:0}.token{display:inline-block;margin:3px;padding:4px 8px;border-radius:6px;font:15px ui-monospace,monospace;white-space:pre-wrap}.token.ignored{background:#e9eef2;color:#4f6575}.token.supervised{background:#daf2e5;color:#075b45;border:1px solid #91cbb0}.mask-bar{display:flex;width:100%;min-height:60px;margin:15px 0;border-radius:10px;overflow:hidden;border:1px solid #acc1c4}.mask-bar .ignored{background:#dfe8ee;flex:273;display:flex;align-items:center;padding:10px;font-size:16px}.mask-bar .supervised{background:#bbebd5;flex:4;min-width:110px;display:flex;align-items:center;justify-content:center;padding:10px;font-size:15px}.exercise{background:#f4f7fb;border-radius:12px;padding:18px 22px;margin:20px 0}.small{font-size:16px;color:var(--muted)}.portrait{max-width:980px;margin:auto}.raw pre{font-size:13px;max-height:420px}.path{overflow-wrap:anywhere}.footer{padding:20px 6px;color:var(--muted);font-size:15px}.flow{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px;margin:24px 0}.flow div{background:#edf5f3;border:1px solid #d4e5df;border-radius:10px;padding:12px;font-size:15px}.flow strong{display:block;color:var(--green)}.noscript{background:#fff2d9;padding:12px 18px;border-radius:10px}.note{font-size:16px}.result-intro{margin-top:20px}.references li{font-size:15px}.example-title{font-size:21px;color:var(--green)}.source-lesson{padding:18px 0;border-bottom:1px solid var(--line)}
@media(max-width:760px){body{font-size:17px;line-height:1.8}.shell{padding:20px 12px 40px}.hero,.lesson,.case,.bridge,.evidence{padding:23px 18px;border-radius:14px}.grid,.calculator .controls{grid-template-columns:1fr}.stats{grid-template-columns:repeat(2,minmax(0,1fr))}.flow{grid-template-columns:1fr}.flow div{display:flex;gap:12px;align-items:center}.flow strong{min-width:82px}h2{font-size:27px}h3{font-size:22px}.nav-inner{padding:0 12px}.formula{font-size:18px;padding:12px}pre{font-size:12px;padding:12px}.code-card{padding:14px}.lead{font-size:19px}.zoom-scroll img{width:1400px}.mask-bar .ignored{font-size:14px}.mask-bar .supervised{min-width:95px;font-size:13px}th,td{padding:9px;font-size:14px}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}@media print{.nav,.calculator,.button{display:none}.shell{max-width:none;padding:0}.hero,.lesson,.case,.bridge{box-shadow:none;break-inside:avoid}body{background:white;font-size:13px}pre{max-height:none;white-space:pre-wrap}details:not([open]){display:none}}
"""


JS = r"""
(() => {
  'use strict';
  const byId = id => document.getElementById(id);
  const format = value => new Intl.NumberFormat('zh-CN').format(value);
  function updateLoRA() {
    const rank = Number(byId('lora-rank').value);
    const targets = byId('lora-targets').value;
    const q = rank * (2048 + 2048);
    const v = rank * (2048 + 256);
    const layer = targets === 'qv' ? q + v : 2 * q + 2 * v;
    const total = layer * 36;
    byId('lora-shapes').textContent = `q: A[${rank}, 2048] / B[2048, ${rank}]；v: A[${rank}, 2048] / B[256, ${rank}]`;
    byId('lora-count').textContent = format(total);
    byId('lora-derivation').textContent = `每层 ${format(layer)} × 36 层 = ${format(total)} 个可训练参数。α=2r时，α/r=2。`;
    byId('lora-preset-note').textContent = targets === 'qv' && rank === 8
      ? '该尺寸与真实 qv_r8 checkpoint 对应；计算器仅复算参数，不训练模型。'
      : targets === 'qkvo' && rank === 4
      ? '该参数数量与真实 qkvo_r4 策略对应；计算器不产生新的实验结果。'
      : '该组合仅为教学尺寸计算，当前这里没有对应的新训练或准确率结果。';
  }
  function updateDPO() {
    const ids = ['dpo-chosen-model','dpo-chosen-ref','dpo-rejected-model','dpo-rejected-ref','dpo-beta'];
    const values = ids.map(id => Number(byId(id).value));
    if (!values.every(Number.isFinite) || values[4] <= 0 || values.slice(0,4).some(value => value > 0)) {
      byId('dpo-output').textContent = '请填写不大于0的序列log概率，并令 β>0。';
      return;
    }
    const [chosen, chosenRef, rejected, rejectedRef, beta] = values;
    const chosenRelative = chosen - chosenRef;
    const rejectedRelative = rejected - rejectedRef;
    const margin = beta * (chosenRelative - rejectedRelative);
    const loss = Math.max(-margin, 0) + Math.log1p(Math.exp(-Math.abs(margin)));
    const probability = margin >= 0 ? 1 / (1 + Math.exp(-margin)) : Math.exp(margin) / (1 + Math.exp(margin));
    byId('dpo-output').textContent = `优选相对reference变化 ${chosenRelative.toFixed(3)}；劣选变化 ${rejectedRelative.toFixed(3)}。偏好margin = ${margin.toFixed(4)}，σ(margin) = ${probability.toFixed(4)}，DPO loss = ${loss.toFixed(4)}。`;
    byId('dpo-interpretation').textContent = margin > 0
      ? '此教学输入中，优选相对reference的提升大于劣选。loss较低仅表示更符合这对偏好，不保证客观VQA准确率提升。'
      : margin < 0
      ? '此教学输入中，劣选相对reference的提升更大，DPO会惩罚这个排序。首先确认偏好标签本身正确。'
      : '优选与劣选的相对reference变化相等，margin=0，loss=ln(2)≈0.6931。';
  }
  byId('lora-rank').addEventListener('change', updateLoRA);
  byId('lora-targets').addEventListener('change', updateLoRA);
  ['dpo-chosen-model','dpo-chosen-ref','dpo-rejected-model','dpo-rejected-ref','dpo-beta'].forEach(id => byId(id).addEventListener('input', updateDPO));
  byId('copy-cpu').addEventListener('click', async () => {
    const command = byId('cpu-command').textContent;
    try {
      if (!navigator.clipboard || !navigator.clipboard.writeText) throw new Error('clipboard unavailable');
      await navigator.clipboard.writeText(command);
      byId('copy-status').textContent = '已复制CPU读取命令。';
    } catch (_) {
      const field = document.createElement('textarea');
      field.value = command;
      field.setAttribute('readonly', '');
      field.style.position = 'fixed'; field.style.top = '0'; field.style.left = '0';
      document.body.appendChild(field); field.select();
      const success = document.execCommand('copy');
      field.remove(); byId('copy-cpu').focus();
      byId('copy-status').textContent = success ? '已复制CPU读取命令。' : '浏览器限制剪贴板，请选中下方命令手动复制。';
    }
  });
  updateLoRA(); updateDPO();
})();
"""


FORMULAS = {
    "lora": ["ΔW = (α / r) B A", "y = W<sub>0</sub>x + (α / r) B(Ax)",
             "P<sub>LoRA</sub> = r (d<sub>in</sub> + d<sub>out</sub>)", "Attention(Q,K,V) = softmax(QK<sup>T</sup> / √d) V"],
    "qlora": ["Ŵ = dequantize(code<sub>4bit</sub>, scales)", "y ≈ Ŵx + (α / r) BAx"],
    "sft": ["L<sub>i</sub> = −(1 / T<sub>i</sub>) Σ<sub>t ∈ assistant</sub> log p<sub>θ</sub>(y<sub>t</sub> | 图像, 问题, y<sub>&lt;t</sub>)",
            "L<sub>batch</sub> = (1 / N) Σ<sub>i</sub> L<sub>i</sub>"],
    "ddp": ["B<sub>effective</sub> = world × micro × acc = 4 × 4 × 1 = 16",
            "L<sub>rank</sub> = (world / N<sub>actual</sub>) Σ<sub>i ∈ rank</sub> valid<sub>i</sub>L<sub>i</sub>",
            "DDP梯度平均 = ∇[(1 / N<sub>actual</sub>) Σ<sub>真实QA</sub> L<sub>i</sub>]"],
    "vision_budget": ["N<sub>visual</sub> = grid<sub>t</sub> × grid<sub>h</sub> × grid<sub>w</sub> / merge_size<sup>2</sup>",
                      "N<sub>visual</sub> ≈ H′W′ / (14<sup>2</sup> × 2<sup>2</sup>) = H′W′ / 784"],
    "data_selection": ["f(a) = Σ<sub>i ∈ train_10k</sub> 1[Normalizer(target<sub>i</sub>) = a]",
                       "切片得分 = (100 / |S|) Σ<sub>i ∈ S</sub> Consensus(pred<sub>i</sub>, answers<sub>i</sub>)",
                       "计划采样：p<sub>i</sub> ∝ (f(target<sub>i</sub>) + ε)<sup>−τ</sup>，τ ≥ 0"],
    "multitask_dpo": ["计划多任务：L<sub>mix</sub> = Σ<sub>k</sub> λ<sub>k</sub> E<sub>(x,y)∼Dₖ</sub>[L<sub>SFT</sub>(x,y)]",
                      "计划DPO：L<sub>DPO</sub> = −log σ{β[(log π<sub>θ</sub>(y<sup>+</sup>|x) − log π<sub>ref</sub>(y<sup>+</sup>|x)) − (log π<sub>θ</sub>(y<sup>−</sup>|x) − log π<sub>ref</sub>(y<sup>−</sup>|x))]}"],
}


class Reader:
    def __init__(self, root):
        self.root = root
        self.lessons = read(root / "code/LESSONS.json")
        self.tokens = read(root / "code/TOKEN_MASK.json")
        self.tensors = read(root / "code/LORA_TENSORS.json")
        self.cases = read(root / "cases/CASES.json")
        self.images = []
        self.sources = []
        self.inputs = []
        self.equation_count = 0

    def image(self, path, alt, zoom=False):
        from PIL import Image
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Required learning figure has not been produced: {path}")
        data = path.read_bytes()
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
        mime = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
        encoded = f"data:{mime};base64,{base64.b64encode(data).decode()}"
        self.images.append({"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
                            "width": width, "height": height, "alt": alt})
        result = f'<img src="{encoded}" width="{width}" height="{height}" alt="{esc(alt)}" loading="lazy">'
        if zoom:
            result += '<details><summary>展开大图，横向滚动查看文字与像素</summary><div class="zoom-scroll">' + result + '</div></details>'
        return result

    def figure(self, name, alt, caption, zoom=False):
        return '<figure>' + self.image(self.root / "figures" / name, alt, zoom) + f'<figcaption>{esc(caption)}</figcaption></figure>'

    def csv(self, name):
        path = self.root / "figures" / name
        self.inputs.append({"path": str(path), "sha256": sha(path)})
        return read_csv(path)

    def source_card(self, source):
        original = Path(source["path"])
        receipt = verify_source(original, source["sha256"], allow_current=True) if original.is_relative_to(REPO) else None
        path = Path(receipt['path']) if receipt else project_file(original)
        if receipt is None:
            assert sha(path) == source["sha256"], f"Source changed: {path}"
        lines = path.read_text().splitlines()
        snippet_lines = source["text"].splitlines()
        matches = [index for index in range(len(lines) - len(snippet_lines) + 1)
                   if lines[index:index + len(snippet_lines)] == snippet_lines]
        assert len(matches) == 1, f"Saved snippet absent or ambiguous after migration; re-extract learning source: {path}"
        start, end = matches[0] + 1, matches[0] + len(snippet_lines)
        offset = start - source["start_line"]
        first = source.get("display_start_line", source["start_line"]) + offset
        last = source.get("display_end_line", source["end_line"]) + offset
        display = "\n".join(f"{index}: {lines[index - 1]}" for index in range(first, last + 1))
        numbered = "\n".join(f"{index}: {lines[index - 1]}" for index in range(start, end + 1))
        current_sha = sha(path)
        self.sources.append({"path": str(path), "first_line": start, "last_line": end,
                             "sha256": current_sha, "verified_exact": True,
                             "historical_source_sha256": source["sha256"], "migration": receipt})
        block = f'<div class="code-card"><p class="source">{esc(path)}<br>重点阅读第 {first}–{last} 行</p><pre><code>{esc(display)}</code></pre>'
        if offset or current_sha != source["sha256"]:
            block += f'<p class="small">目录迁移已核对旧哈希与当前哈希；片段内容逐行相同。下列解释中的历史行号对应当前行号加 {offset}。</p>'
        block += list_html(source["explanations"])
        block += f'<details><summary>展开完整片段（第 {start}–{end} 行）与文件哈希</summary><pre><code>{esc(numbered)}</code></pre><p class="hash">当前文件 SHA256：{esc(current_sha)}<br>原记录 SHA256：{esc(source["sha256"])}</p></details></div>'
        return block

    def metrics(self, record, labels):
        values = record["values"]
        rows = []
        for key, label in labels:
            value = values.get(key)
            if isinstance(value, bool):
                value = "通过 / 是" if value else "否"
            elif isinstance(value, float):
                value = number(value, 6 if "error" in key or "nll" in key else 3)
            elif isinstance(value, int):
                value = f"{value:,}"
            rows.append([label, value])
        return table(["工程记录", "实际值"], rows) + f'<p class="source">来源：{esc(record["path"])}<br>SHA256：{esc(record["sha256"])}</p>'

    def lora_calculator(self):
        return '''<div class="calculator"><h3>亲手算：改变 rank 后，参数预算怎样变化？</h3><p class="small">只进行矩阵尺寸教学计算。没有新训练、没有新的准确率；输入维度来自当前真实Qwen模型。</p><div class="controls"><div><label for="lora-rank">LoRA rank</label><select id="lora-rank"><option value="8">r = 8</option><option value="4">r = 4</option></select></div><div><label for="lora-targets">语言侧插入位置</label><select id="lora-targets"><option value="qv">q / v</option><option value="qkvo">q / k / v / o</option></select></div></div><div class="calc-output" aria-live="polite"><p id="lora-shapes">q: A[8,2048]/B[2048,8]；v: A[8,2048]/B[256,8]</p><strong id="lora-count">1,843,200</strong> 个可训练参数<p id="lora-derivation">每层51,200 × 36层 = 1,843,200</p><p id="lora-preset-note">真实 qv_r8 checkpoint 的参数预算。</p></div></div>'''

    def dpo_calculator(self):
        fields = [("dpo-chosen-model", "优选回答 log πθ(y+ | x)", "-2"),
                  ("dpo-chosen-ref", "优选回答 log πref(y+ | x)", "-3"),
                  ("dpo-rejected-model", "劣选回答 log πθ(y− | x)", "-4"),
                  ("dpo-rejected-ref", "劣选回答 log πref(y− | x)", "-3.5"),
                  ("dpo-beta", "β（大于0）", "0.1")]
        controls = ""
        for key, label, value in fields:
            bounds = 'min="0.001"' if key == "dpo-beta" else 'max="0"'
            controls += f'<div><label for="{key}">{esc(label)}</label><input id="{key}" type="number" value="{value}" step="0.1" {bounds}></div>'
        return '<div class="calculator"><h3>亲手算：DPO到底比较什么？</h3><span class="status planned">教学数字 · 非实验结果</span><p>填写完整回答序列的 log 概率；数值为人为设定，尚未运行DPO。先比较优选与劣选各自相对reference的变化，再比较这两个变化。</p><div class="controls">' + controls + '</div><div class="calc-output" role="status" aria-live="polite"><p id="dpo-output">开启JavaScript后显示教学计算。</p><p id="dpo-interpretation"></p></div><p class="small">reference是固定参考策略。β缩放相对log概率margin；这个计算器没有模型、训练数据或优化器。</p></div>'

    def results(self, lesson):
        key, result = lesson["id"], lesson["results"]
        out = '<h3>四、真实结果与工程证据</h3>'
        if key == "lora":
            out += table(["真实策略", "语言rank / alpha", "语言参数", "merger参数", "总参数"],
                [[p["method"], f"{p['rank']} / {p['alpha']}", f"{p['parameter_groups']['language']:,}", f"{p['parameter_groups']['merger']:,}", f"{p['trainable_parameters']:,}"] for p in result["parameter_budgets"]])
            out += self.figure("lora_update.png", "真实qv_r8 seed0第0层q投影LoRA增量的64乘64热图，以及8个非零奇异值", "训练后的真实ΔW=(16/8)BA。热图只是2048×2048矩阵左上64×64；奇异值由thin QR后的8×8核心计算，没有对大矩阵做SVD。")
            out += table(["奇异值序号", "实际值"], [[i + 1, number(v, 6)] for i, v in enumerate(self.tensors["singular_values"])])
            out += self.metrics(result["engineering_evidence"], [("completed_updates", "完成更新"), ("actual_questions_seen", "真实QA曝光"), ("trainable_parameters", "可训练参数"), ("frozen_parameters_unchanged", "冻结权重哈希前后相同")])
            out += self.figure("stage1_results.png", "三个LoRA策略在内部dev与三个公开任务上的已完成种子结果，不完整策略单独标识", "点代表已完成seed；均值与seed标准差是训练重复的描述，不是总体置信区间。merger仅完成seed0，不能与三seed平均据此定胜负。")
            rows = self.csv("stage1_aggregates.csv")
            out += table(["策略", "任务 / 指标", "完成seed数", "均值 (%)", "seed样本SD"], [[r["method"], {"dev": "内部dev / TextVQA", "textvqa": "公开TextVQA", "docvqa": "DocVQA / ANLS", "mmstar": "MMStar / 子类平均准确率"}.get(r["task"], r["task"]), r["n_seeds"], number(r["mean_score_percent"]), number(r["sample_seed_sd"])] for r in rows])
        elif key == "qlora":
            out += self.metrics(result["training"], [("gpu_name", "训练GPU"), ("peak_gpu_allocated_gib", "单卡峰allocated (GiB)"), ("peak_gpu_reserved_gib", "单卡峰reserved (GiB)"), ("active_seconds", "训练与dev NLL阶段active秒"), ("input_tokens_per_second", "四卡合计输入tokens / active秒")])
            out += self.figure("numerical_paths.png", "原始FP32与语言NF4权重分别在无adapter和历史pilot adapter下的完整TextVQA得分", "数值路径的四条件受控评测使用相同公开TextVQA协议。图中的adapter是历史pilot；不等于本轮10k QA训练。权重量化、计算精度和adapter是不同实验轴。")
            points = {r["condition"]: r for r in self.csv("numerical_path_points.csv")}
            out += table(["权重路径 / compute FP32", "无adapter (%)", "历史pilot adapter (%)", "增量 (百分点)"],
                [[label, number(points[f"{prefix}_base"]["score_percent"]), number(points[f"{prefix}_pilot"]["score_percent"]), number(float(points[f"{prefix}_pilot"]["score_percent"]) - float(points[f"{prefix}_base"]["score_percent"]))] for prefix, label in (("original", "原始权重"), ("nf4_language_only", "仅语言NF4"))])
            out += '<p class="callout boundary">原始FP32预检峰14.199GiB与NF4训练峰5.848GiB来自不同执行过程。它们证明各自能运行；不能据此计算训练显存节省率或推断提速比例。</p>'
        elif key == "sft":
            tokens = self.tokens
            out += self.figure("sft_mask.png", "同一张Albertville路牌图生成277个输入token，273个前缀位置labels为负100，后4个assistant token参与teacher forcing损失", "可视化对应dev qid27482。图像234个token被纳入273个忽略目标的前缀，但仍参与attention；本次仅CPU检查编码，没有对dev样本训练。")
            out += '<div class="mask-bar" aria-label="273个忽略目标的前缀token和4个参与监督的assistant token；为便于阅读，绿色区宽度被放大"><div class="ignored">前缀273：labels = −100</div><div class="supervised">答案4：监督</div></div><p class="small">绿色区为可读性放大，宽度不是精确token比例。图像234、其他前缀39、assistant 4，总计277。</p>'
            supervised = [t for t in tokens["tokens"] if t["supervised"]]
            out += table(["监督位置（从0计）", "真实token", "token ID / label", "用于预测的前一位置"], [[t["position"], repr(t["decoded_token"]), t["token_id"], t["next_token_predicted_by_position"]] for t in supervised])
            out += '<div class="callout"><strong>Teacher forcing：</strong>预测“0”时，训练输入已经包含真实的“4”；自由生成时，“4”来自模型自身的上一轮输出。因此降低训练NLL与改善完整生成得分要分别验证。</div>'
            out += self.metrics(result["tail_head_equivalence"], [("passed", "tail head预检"), ("tail_head_loss_max_absolute_error", "完整head与切片head的loss最大绝对误差"), ("tail_head_gradient_max_absolute_error", "梯度最大绝对误差"), ("stress_sequence_tokens", "压力检查序列长度")])
            out += table(["同一真实样本", "值"], [["问题", tokens["question"]], ["监督target", tokens["target"]], ["监督解码", repr(tokens["supervised_decoded"])], ["input / prefix / answer token数", f"{tokens['input_length']} / {tokens['prefix_length']} / {tokens['supervised_token_count']}"]])
        elif key == "ddp":
            out += self.metrics(result["full_stage1"], [("world_size", "GPU进程数"), ("microbatch", "每卡QA数"), ("accumulation", "梯度累积次数"), ("effective_batch", "全局有效batch"), ("completed_updates", "完成更新"), ("actual_questions_seen", "实际有效QA"), ("skipped_updates", "跳过更新")])
            out += table(["真实33 QA契约 / 第几步", "计算槽位", "真实有效QA", "DDP归一化分母"], [[1, 16, 16, 16], [2, 16, 16, 16], [3, 16, 1, 1]])
            out += self.metrics(result["checkpoint_restore_contract"], [("passed", "恢复契约"), ("adapter_tensors_identical_after_restore", "adapter张量精确一致"), ("optimizer_restored", "optimizer恢复"), ("cpu_cuda_rng_restored", "CPU / CUDA RNG恢复"), ("checkpoint_updates", "checkpoint步数"), ("questions_seen", "真实QA"), ("restored_dev_nll", "恢复后dev NLL"), ("original_dev_nll", "原dev NLL")])
        elif key == "vision_budget":
            measured = result["measured_current_case"]
            out += table(["路牌样本的实际处理", "值"], [["原图 W×H", "1024 × 768"], ["预算缩放后 W×H", "504 × 364"], ["patch grid [T,H,W]", "[1,26,36]"], ["视觉编码器patch数", "1 × 26 × 36 = 936"], ["语言侧视觉token", "936 / 4 = 234"], ["max视觉token预算", "256；实际234"]])
            case = self.cases["cases"][0]
            out += '<div class="grid"><figure>' + self.image(self.root / "cases" / case["assets"]["original.jpg"], "Albertville路牌样本原图1024乘768") + '<figcaption>原始图像：1024×768</figcaption></figure><figure>' + self.image(self.root / "cases" / case["assets"]["preprocessed_rgb.png"], "实际processor缩放后RGB图像504乘364，未锐化，随后进行归一化和patch打包") + '<figcaption>真实processor resize后的RGB：504×364。展示的是归一化前像素，不是attention。</figcaption></figure></div>'
            out += '<h4>下一轮怎样检验？</h4>' + list_html(result["planned_controls"])
            out += '<p class="callout boundary">这里没有更高视觉预算或局部裁图的模型新预测。人类觉得图更清楚，是形成实验假设的依据；效果需要固定checkpoint、生成评测和计算成本对照。</p>'
        elif key == "data_selection":
            out += self.figure("answer_frequency_slices.png", "qv rank8三个训练seed在0、1至5、6至20、大于20次训练目标频次切片上的内部dev得分及各切片QA数量", "频次来自实际train_10k目标标准化后的QA次数；未见答案858/1617。误差条为seed样本SD，不包含切片混杂或多重比较校正。")
            labels = {"unseen_0": "0次（训练target未见）", "rare_1_5": "1–5次", "medium_6_20": "6–20次", "frequent_gt20": ">20次"}
            out += table(["训练目标频次", "dev QA数", "图像数", "qv_r8三seed均分 (%)"], [[labels[b], r["questions"], r["unique_images"], number(r["mean_score_percent"])] for b, r in result["slice_scores_qv_r8_three_seeds"].items()])
            out += '<p>例如店招答案“flesor\'s candy kitchen”在训练目标中出现较少或未见时，频次切片能描述这种输出覆盖；它无法单独判定是像素不足、字符识别、拼写先验还是标注问题。采样对照要同时记录独立图像与token曝光。</p>'
            out += self.metrics({"values": result["diagnostic_evidence"]["manifest"], "path": result["diagnostic_evidence"]["path"], "sha256": result["diagnostic_evidence"]["sha256"]}, [("training_questions", "训练目标统计QA"), ("unique_normalized_training_targets", "不同标准化训练目标"), ("dev_questions", "内部dev QA"), ("dev_unique_images", "内部dev图像")])
        elif key == "multitask_dpo":
            out += '<span class="status planned">尚无多任务混合训练或DPO实验结果</span>'
            out += self.figure("task_deltas.png", "同协议语言NF4底座为参照的TextVQA、DocVQA及MMStar适配分数变化，各指标分面单独显示", "这是单任务TextVQA适配后的跨任务诊断。每个任务使用自己的指标；百分点差不合并成总准确率。它既不是多任务训练结果，也不是DPO结果。")
            rows = self.csv("task_delta_aggregates.csv")
            out += table(["任务 / 指标", "策略", "完成seed数", "同路径底座得分 (%)", "适配增量 (百分点)"], [[{"textvqa": "TextVQA / 共识", "docvqa": "DocVQA / ANLS", "mmstar": "MMStar / 子类平均准确率"}.get(r["task"], r["task"]), r["method"], r["n_seeds"], number(r["baseline_score_percent"]), number(r["mean_delta_pp"])] for r in rows])
            out += '<h4>最少需要的对照</h4>' + table(["研究", "固定什么", "比较什么"], [["多任务混合", "同一SFT起点、候选训练池、评测协议", "纯TextVQA；总预算匹配的混合；TextVQA曝光匹配的混合"], ["DPO", "同一SFT checkpoint、偏好数据、预算、reference", "不追加训练；chosen-only继续SFT；DPO"]])
            out += self.dpo_calculator()
        out += f'<p class="result-intro">{esc(result["interpretation"])}</p>'
        out += raw_json(result, "展开该课的原始结果字段与来源哈希")
        return out

    def lesson(self, lesson):
        key = lesson["id"]
        if lesson["status"] == "implemented_and_measured":
            badge = '<span class="status measured">已有实现与工程结果</span>'
        elif lesson["status"] == "planned_not_implemented_or_trained":
            badge = '<span class="status planned">计划课 · 尚未实现或训练</span>'
        else:
            badge = '<span class="status mixed">已有诊断 · 算法干预待执行</span>'
        out = f'<section class="lesson" id="{key}" aria-labelledby="{key}-title">{badge}<h2 id="{key}-title">{esc(lesson["title"])}</h2><p class="question">{esc(lesson["question"])}</p>'
        out += '<h3>一、先建立直觉</h3>' + paragraphs([lesson["principle"]["intuition"]])
        out += paragraphs(lesson["principle"]["walkthrough"])
        for formula in FORMULAS[key]:
            self.equation_count += 1
            out += f'<div class="formula">{formula}</div>'
        if key == "lora":
            out += self.figure("mechanism_lora.png", "冻结W0与可训练A、B两条支路组成真实q和v投影；q输出2048维，GQA的v输出256维", "结构图按真实checkpoint形状绘制。蓝色底座冻结，绿色A/B可训练；视觉trunk保持冻结，merger策略仅更新指定merger LoRA。")
        out += '<h3>二、跟踪张量形状</h3>' + list_html(lesson["principle"]["shapes"])
        if key == "lora":
            out += self.lora_calculator()
        out += '<h3>三、读真实代码</h3>'
        if lesson["code"]:
            out += '<p class="small">以下均为当前项目或已安装依赖的原始代码。重点行先展开，完整函数和SHA256可按需查看。</p>'
            out += "".join(self.source_card(source) for source in lesson["code"])
        else:
            out += '<p class="callout boundary">冻结runner中没有多任务训练和DPO的实现。此课仅说明原理、数据契约和实验对照，不把未来公式展示为现有项目代码。</p>'
        out += self.results(lesson)
        out += '<h3>五、结果的边界</h3><div class="callout boundary">' + list_html(lesson["limits"]) + '</div>'
        out += '<div class="exercise"><h3>六、动手检查与练习</h3>' + list_html([x.get("task", x.get("purpose", "")) for x in lesson["exercises"]]) + '</div>'
        if lesson["references"]:
            out += '<p class="small">原论文：' + ' · '.join(f'<a href="{esc(url)}" target="_blank" rel="noopener noreferrer">{esc({"https://arxiv.org/abs/2106.09685":"LoRA", "https://arxiv.org/abs/2305.14314":"QLoRA", "https://arxiv.org/abs/2502.13923":"Qwen2.5-VL", "https://arxiv.org/abs/2305.18290":"DPO"}.get(url,url))}</a>' for url in lesson["references"]) + '</p>'
        return out + '</section>'

    def case_gallery(self):
        out = '<section id="cases" class="evidence"><span class="eyebrow">真实内部 DEV · 仅展示已保存的 SEED0 预测</span><h2>把算法重新放回图片里</h2><p>三个案例分别呈现数字对应、店招字符串和共同失败。它们由人工选作学习材料，不代表整个dev分布或方法排名。原始答案和预测都保留；没有新生成，也没有把局部放大图输入模型。</p></section>'
        for index, case in enumerate(self.cases["cases"], 1):
            out += f'<article class="case" id="case-{esc(case["question_id"])}"><h3>{esc(case["title"])}</h3><p class="meta">内部dev · question_id={esc(case["question_id"])} · 原始训练目标频次={case["train_10k_target_qa_frequency"]}次</p><p class="example-title">{esc(case["question_zh"])}</p><p>原始问题：{esc(case["raw_question"])}</p>'
            out += '<figure class="portrait">' + self.image(self.root / "cases" / case["assets"]["teaching_board.png"], f"案例{index}：{case['title']}。含原图、真实processor缩放、人工放大及三策略预测；人工区域未输入模型。", True) + '<figcaption>上方两幅是原图与真实processor resize；局部放大只供人类查看。图中裁剪不是attention热图、OCR预测或模型新增输入。</figcaption></figure>'
            out += table(["策略 / seed0", "保存预测", "共识得分 (%)", "匹配标准化人类答案数"], [[p["method"], p["raw_prediction"], number(100 * p["saved_consensus_score"], 1), p["normalized_reference_match_count"]] for p in case["saved_seed0_predictions"]])
            out += table(["人类标注编号", "原始答案"], [[i + 1, a] for i, a in enumerate(case["reference_answers"])], "十个人类答案：TextVQA共识评价使用全部答案，SFT target使用确定性多数值")
            prep = case["preprocessing"]
            out += table(["输入与展示", "真实情况"], [["原图W×H", " × ".join(map(str, case["original_dimensions_wh"]))], ["processor W×H", " × ".join(map(str, prep["actual_resized_dimensions_wh"]))], ["视觉patch / 语言视觉token", f"{prep['vision_encoder_patch_tokens']} / {prep['merged_visual_tokens']}"], ["参考答案是否进入生成processor", "否"], ["人工局部放大是否输入模型", "否"], ["裁剪区域依据", case["illustration_only_crop"]["selection_basis"]]])
            out += '<h4>观察 → 假设 → 下一步</h4>' + paragraphs([case["observation"], case["explanation"], case["next_experiment"]])
            out += '<div class="callout boundary">' + paragraphs([case["failure_boundary"]]) + '</div>'
            out += raw_json(case, "展开本例完整证据、图像与预测哈希") + '</article>'
        return out

    def bridge(self):
        methods = REPO / "src/methods.py"
        lines = methods.read_text().splitlines()
        self.inputs.append({"path": str(methods), "sha256": sha(methods)})
        out = '<section class="bridge" id="clip-bridge"><span class="status mixed">历史CLIP路线 · 独立数据与指标</span><h2>从CLIP长尾分类，到生成式VLM</h2><p>此前项目用冻结CLIP图像特征与文本类别原型做分类；当前Qwen使用图像与问题生成字符串。两条路线都能研究适配位置、有限预算和频次覆盖，但预测空间与评价契约不同。</p>'
        out += table(["阶段", "输入与输出", "主要损失", "评价单位"], [["历史CLIP", "图像特征[B,D] × 类别文本原型[C,D] → logits[B,C]", "类级CE、训练时LA、TailSpec-static", "ImageNet-LT等分类OA / Many / Medium / Few"], ["当前Qwen VLM", "图像patch + 问题tokens → assistant回答序列", "每QA内assistant-token CE", "TextVQA共识、DocVQA ANLS、MMStar子类平均准确率"]])
        out += '<div class="formula">CLIP分类：z = s · normalize(v) · normalize(T)<sup>T</sup></div><div class="formula">训练时LA：L = CE(z + τ log π, y)，π<sub>c</sub> = n<sub>c</sub> / Σ<sub>j</sub>n<sub>j</sub></div>'
        out += '<p>类别先验越小，训练时加入的log先验越负。模型要为尾部真类学到足够大的原始logit来抵消这一项。项目的LA在loss中调整，classifier的推理前向仍输出原始余弦logits。</p><p>当前VQA频次统计的是回答字符串的QA覆盖。字符串可能由多个token组成，还有多个有效人类答案；它不对应固定的C个互斥类别，因此不能把分类LA公式直接加到语言词表logits后声称完成长尾VLM算法。</p>'
        for first, last, title in ((67, 78, "余弦原型分类与普通CE"), (81, 94, "真实LogitAdjustmentLoss")):
            text = "\n".join(f"{i}: {lines[i - 1]}" for i in range(first, last + 1))
            self.sources.append({"path": str(methods), "first_line": first, "last_line": last, "sha256": sha(methods), "verified_exact": True})
            out += f'<h4>{title}</h4><p class="source">{esc(methods)} · 第{first}–{last}行</p><pre><code>{esc(text)}</code></pre>'
        out += '<figure>' + self.image(REPO / "assets/main_results.png", "项目历史CLIP长尾分类结果图，指标与当前生成式VLM评测独立") + '<figcaption>项目已有历史分类图。历史协议与当前VLM表独立，不把OA、TextVQA共识和ANLS混进同一列。</figcaption></figure>'
        historical_path = REPO / "results/main_results.csv"
        self.inputs.append({"path": str(historical_path), "sha256": sha(historical_path)})
        rows = [r for r in read_csv(historical_path) if r["setting"] == "imagenet_lt_b32" and r["method"] in {"zero_shot", "ce", "la"}]
        out += table(["历史ImageNet-LT / ViT-B32", "运行数", "OA (%)", "Many (%)", "Medium (%)", "Few (%)"], [[r["method"], r["n_runs"], number(r["oa"]), number(r["many"]), number(r["medium"]), number(r["few"])] for r in rows])
        out += '<div class="exercise"><strong>桥接练习：</strong>比较历史ResidualAdapter的“特征残差”与当前LoRA的“权重增量”。然后指出类频次先验与答案字符串频次各自的统计单位。</div></section>'
        return out

    def evidence_section(self):
        command = self.lessons["common_cpu_reproduction"][0]["command"]
        out = '<section class="evidence" id="reproduce"><h2>用CPU亲手核对这堂课</h2><p>下面只重读源码、已训练adapter和真实样本的processor编码，输出学习JSON。它不会加载Qwen底座、不生成新答案、不开始训练，也不会修改冻结实验协议。</p>'
        out += f'<button class="button" id="copy-cpu" type="button">复制CPU读取命令</button> <span id="copy-status" class="small" role="status" aria-live="polite"></span><pre id="cpu-command">{esc(command)}</pre>'
        out += '<h3>本材料的证据层次</h3>' + table(["内容", "证据", "使用边界"], [["LoRA / NF4 / SFT / DDP", "冻结源码、真实adapter、已完成工程JSON", "只陈述被核验的实现和运行条件"], ["定量图", "已完成的评测记录与对应CSV快照", "snapshot不会随着后台实验自动刷新"], ["真实图片案例", "内部dev原图、processor像素和已保存seed0预测", "人工选例，不代表整体排名"], ["预算、采样、多任务、DPO", "原理、诊断与最小实验对照", "尚无相应新训练或生成结论"], ["离线交互", "纯JavaScript数学计算", "教学计算，非训练结果"]])
        out += '<p class="small">构建检查包含HTML结构、内嵌图片解码、源码片段与哈希、Node语法。未进行浏览器渲染或交互运行测试；交互的运行效果须在浏览器打开本文件后查看。</p>'
        out += '<h3>论文与科学绘图参考</h3><ul class="references">'
        references = [("LoRA：低秩权重适配", "https://arxiv.org/abs/2106.09685"), ("QLoRA：NF4、double quant与低成本微调", "https://arxiv.org/abs/2305.14314"), ("Qwen2.5-VL技术报告：视觉与语言结构", "https://arxiv.org/abs/2502.13923"), ("DPO：直接偏好优化", "https://arxiv.org/abs/2305.18290"), ("Kassis等，2026：科学绘图软件参考", "https://doi.org/10.48550/arXiv.2609.00065")]
        out += "".join(f'<li><a href="{esc(url)}" target="_blank" rel="noopener noreferrer">{esc(title)}</a></li>' for title, url in references) + '</ul></section>'
        return out

    def build(self):
        sections = [self.lesson(lesson) for lesson in self.lessons["lessons"]]
        methods_map = {p["method"]: p for p in self.cases["cases"][0]["saved_seed0_predictions"]}
        initial_table = table(["同图同问题 / seed0", "已保存回答", "共识得分 (%)"], [[m, methods_map[m]["raw_prediction"], number(100 * methods_map[m]["saved_consensus_score"], 1)] for m in ("qv_r8", "qkvo_r4", "qv_r4_merger")])
        hero = '''<header class="hero" id="start"><span class="eyebrow">LONGTAIL VLM · 代码与图片图解课堂</span><h1>从一张路牌图，读懂你正在训练的VLM</h1><p class="lead">面对“到Albertville是多少？”这张真实图片，q/v rank8回答20，另外两种策略回答40。我们沿着图片→视觉token→LoRA投影→assistant loss→生成评分，解释实现怎样工作，以及现有结果能支持什么判断。</p><div class="flow"><div><strong>01 看错误</strong>同一张图的真实预测</div><div><strong>02 懂原理</strong>公式和张量形状</div><div><strong>03 读代码</strong>准确路径与行号</div><div><strong>04 看结果</strong>控制变量与证据</div><div><strong>05 做下一轮</strong>提出可检验假设</div></div><div class="stats"><div class="stat"><strong>10,000</strong><span>qv_r8 seed0有效训练QA</span></div><div class="stat"><strong>625</strong><span>完成优化器更新</span></div><div class="stat"><strong>1,843,200</strong><span>真实LoRA可训练参数</span></div><div class="stat"><strong>234</strong><span>这张路牌的视觉tokens</span></div></div>'''
        hero += initial_table + '<div class="callout boundary">一个案例让我们提出“读字”和“图文对应”的假设；它没有证明注意力定位的因果变化。完整内部dev中，qv_r8三seed均分反而高于qkvo_r4。请把局部案例与整体统计一起读。</div><p>适合已经会PyTorch并训练过模型的学习者。建议先读第1、3、5课，再看三个完整案例；第二遍补齐量化、DDP、数据采样和DPO。源码长片段与原始记录可按需展开。</p><p class="meta">材料快照：北京时间2026年10月10日 · 已完成记录为准，后台训练继续独立运行。</p></header>'
        nav_items = [("start", "入口"), ("lora", "1 LoRA"), ("qlora", "2 NF4"), ("sft", "3 SFT"), ("ddp", "4 DDP"), ("vision_budget", "5 视觉预算"), ("data_selection", "6 数据选择"), ("multitask_dpo", "7 多任务/DPO"), ("cases", "真实图片"), ("clip-bridge", "CLIP桥接"), ("reproduce", "CPU核对")]
        nav = '<nav class="nav" aria-label="课程导航"><div class="nav-inner">' + "".join(f'<a href="#{key}">{label}</a>' for key, label in nav_items) + '</div></nav>'
        body = hero + '<noscript><p class="noscript">静态图片、公式、代码与结果可离线阅读。开启JavaScript后可使用LoRA与DPO教学计算器；阅读不需要网络。</p></noscript>' + "".join(sections) + self.case_gallery() + self.bridge() + self.evidence_section()
        return '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="color-scheme" content="light"><title>图解课堂：从真实图片读懂LoRA、NF4与VLM训练</title><style>' + CSS + '</style></head><body><a class="skip" href="#main">跳转到课程内容</a>' + nav + '<main class="shell" id="main">' + body + '<footer class="footer">所有图片内嵌；没有外部脚本、样式或字体。算法原论文链接仅供按需联网阅读。学习计算器不会启动训练。</footer></main><script>' + JS + '</script></body></html>'


class StructureReview(HTMLParser):
    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ids, self.links, self.images, self.external_resources = [], [], [], []
        self.stack, self.errors, self.source_text = [], [], []
        self.tag_counts = Counter()

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        self.tag_counts[tag] += 1
        if attrs.get("id"):
            self.ids.append(attrs["id"])
        if tag == "a":
            self.links.append(attrs.get("href", ""))
        if tag == "img":
            self.images.append(attrs)
        if tag in {"script", "img", "iframe", "link", "source"}:
            for name in ("src", "href", "srcset"):
                if attrs.get(name) and not attrs[name].startswith("data:"):
                    self.external_resources.append({"tag": tag, "attribute": name, "value": attrs[name]})
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in self.VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"Unbalanced closing tag: {tag}; top={self.stack[-1] if self.stack else None}")
        else:
            self.stack.pop()


def review_document(document, reader, output):
    parser = StructureReview()
    parser.feed(document)
    parser.close()
    assert not parser.errors and not parser.stack, parser.errors or parser.stack
    assert len(parser.ids) == len(set(parser.ids)), "Duplicate element ID"
    missing = [href for href in parser.links if href.startswith("#") and href[1:] not in set(parser.ids)]
    assert not missing, missing
    assert not parser.external_resources, parser.external_resources
    assert all(image.get("alt") for image in parser.images), "Image alt missing"
    for attrs in parser.images:
        assert attrs["src"].startswith("data:image/")
        base64.b64decode(attrs["src"].split(",", 1)[1], validate=True)
    assert "fetch(" not in JS and "eval(" not in JS
    js_path = reader.root / "code/reader_interactions.js"
    js_path.write_text(JS)
    node = shutil.which("node")
    if node:
        result = subprocess.run([node, "--check", str(js_path)], capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr
        js_check = {"passed": True, "command": [node, "--check", str(js_path)], "scope": "JavaScript syntax only"}
    else:
        js_check = {"passed": False, "reason": "node unavailable; syntax check not performed"}
    return {"schema_version": "rarevlm.learning_reader_review.v1", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "html": {"path": str(output), "sha256": hashlib.sha256(document.encode()).hexdigest(), "bytes": len(document.encode())},
        "parser_structure_passed": True, "unique_ids_passed": True, "all_internal_anchors_resolve": True,
        "embedded_images_base64_valid": True, "image_elements": len(parser.images), "unique_embedded_image_files": len(reader.images),
        "image_alt_passed": True, "external_script_css_font_or_image_resources": [],
        "source_snippets_exact": True, "source_snippet_count": len(reader.sources), "source_snippets": reader.sources,
        "images": reader.images, "input_csv_and_historical_sources": reader.inputs,
        "input_json_sources": [{"path": str(path), "sha256": sha(path)} for path in
            [reader.root / "code/LESSONS.json", reader.root / "code/TOKEN_MASK.json",
             reader.root / "code/LORA_TENSORS.json", reader.root / "cases/CASES.json",
             reader.root / "figures/FIGURES_PROVENANCE.json", reader.root / "MECHANISMS_PROVENANCE.json"] if path.exists()],
        "javascript": js_check, "javascript_path": str(js_path), "javascript_sha256": sha(js_path),
        "native_details_count": parser.tag_counts["details"], "table_count": parser.tag_counts["table"],
        "equations_html_sub_sup_unicode": reader.equation_count,
        "no_fetch_or_model_execution": True,
        "browser_rendering_test_performed": False, "browser_interaction_test_performed": False,
        "limits": ["HTML parser and Node syntax checks do not prove browser layout or interaction runtime behavior.",
                   "All plots/cases are a frozen completed-evidence snapshot; reader does not auto-refresh live experiments.",
                   "LoRA and DPO calculators perform only pedagogical arithmetic, not training or inference."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learning-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.learning_root / "图解课堂.html"
    watched = sorted((REPO / "src" / "vlm_deep").rglob("*.py")) + [REPO / "configs/vlm_deep_v2.json"]
    before = {str(path): sha(path) for path in watched}
    reader = Reader(args.learning_root)
    document = reader.build()
    review = review_document(document, reader, output)
    assert before == {str(path): sha(path) for path in watched}, "Frozen experimental sources changed"
    review["frozen_training_sources_unchanged"] = True
    review["frozen_training_source_hashes"] = before
    output.write_text(document)
    (args.learning_root / "READER_REVIEW.json").write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"html": str(output), "html_mebibytes": round(review["html"]["bytes"] / 2**20, 3),
        "image_elements": review["image_elements"], "source_snippets": review["source_snippet_count"],
        "tables": review["table_count"], "node_syntax_passed": review["javascript"]["passed"],
        "frozen_sources_unchanged": True, "browser_test_performed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
