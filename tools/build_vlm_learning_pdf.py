#!/usr/bin/env python3
"""Build a Chinese, source-grounded teaching PDF with existing PyMuPDF (CPU only).

Reads code/LESSONS.json, saved source snapshots, figures, and three saved teaching
cases. Does not load model weights, evaluate, train, or modify the frozen runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pymupdf as fitz

PRIVATE = Path("/opt/data/private")
FONT_PATH = PRIVATE / ".tools/fonts/NotoSansCJKsc-Regular.otf"
WIDTH, HEIGHT, MARGIN, BOTTOM = 1000, 800, 45, 754
INK = (.10, .15, .22)
BLUE = (.03, .30, .48)
ORANGE = (.59, .27, .02)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def clean_formula(text: str) -> str:
    # Keep scientific symbols readable; these are explanation lines, not LaTeX.
    return text.replace("planned ", "【计划公式】 ").replace("₀", "0").replace("ᵀ", "^T").replace("∼", "~").replace("y_<t", "答案前缀").replace("Σ_", "Σ ")


class Book:
    def __init__(self, output: Path):
        self.output = output
        self.doc = fitz.open()
        self.font = fitz.Font(fontfile=str(FONT_PATH))
        self.mono = fitz.Font("cour")
        self.page = None
        self.y = 0
        self.section = ""
        self.layout = []
        self.toc = []
        self.sources = []
        self.images = []
        self.used_text = []

    def new_page(self, title: str, *, toc=False):
        self.section = title
        self.page = self.doc.new_page(width=WIDTH, height=HEIGHT)
        self.page.insert_font(fontname="zh", fontfile=str(FONT_PATH))
        self.page.insert_font(fontname="cour")
        self.page.insert_text((MARGIN, 27), "RareVLM 图解课堂 · 原理 / 真实代码 / 工程证据", fontname="zh", fontsize=9, color=(.40, .45, .51))
        self.page.draw_line((MARGIN, 36), (WIDTH-MARGIN, 36), color=(.80, .84, .88), width=.6)
        self.y = 52
        self.paragraph(title, size=22, color=INK, gap=14)
        if toc:
            self.toc.append([1, title, self.page.number+1])

    def ensure(self, height: float):
        if self.y + height > BOTTOM:
            self.new_page(self.section + "（续）")

    def wrap(self, text: str, size: float, font=None, width=None) -> list[str]:
        font = font or self.font
        width = width or WIDTH-2*MARGIN
        result = []
        for raw in text.split("\n"):
            if not raw:
                result.append("")
                continue
            current = ""
            for char in raw:
                if font.text_length(current+char, fontsize=size) > width and current:
                    result.append(current)
                    current = char
                else:
                    current += char
            result.append(current)
        return result

    def paragraph(self, text: str, *, size=12, color=INK, gap=7, indent=0, fontname="zh"):
        font = self.font if fontname == "zh" else self.mono
        leading = size * (1.45 if fontname == "zh" else 1.36)
        for line in self.wrap(text, size, font=font, width=WIDTH-2*MARGIN-indent):
            self.ensure(leading)
            baseline = self.y + font.ascender * size
            self.page.insert_text((MARGIN+indent, baseline), line, fontsize=size, fontname=fontname, color=color)
            self.used_text.append(line)
            self.layout.append({"page": self.page.number+1, "type": "text", "font_size": size,
                                "baseline": baseline, "text": line})
            self.y += leading
        self.y += gap

    def heading(self, text: str):
        self.ensure(42)
        self.paragraph(text, size=14, color=BLUE, gap=4)

    def bullets(self, texts: list[str], *, size=12):
        for text in texts:
            self.paragraph("• " + text, size=size, gap=4)

    def status(self, label: str, planned=False):
        self.paragraph(label, size=12, color=ORANGE if planned else BLUE, gap=12)

    def code(self, snippet: dict):
        numbered = snippet["display_numbered_text"].splitlines()
        if len(numbered) > 18:
            raise ValueError("Displayed source snippet must contain at most 18 original lines")
        self.ensure(60)
        path = snippet["path"]
        self.paragraph(f"来源：{path} · 原始行 {snippet['display_start_line']}–{snippet['display_end_line']}", size=9.5, gap=4)
        self.sources.append({"path": path, "start_line": snippet["display_start_line"],
                             "end_line": snippet["display_end_line"], "source_sha256": snippet["sha256"],
                             "snippet_sha256": snippet.get("snippet_sha256"), "original_lines_displayed": len(numbered)})
        wrapped = []
        for line in numbered:
            lines = self.wrap(line, 9.5, font=self.mono, width=WIDTH-2*MARGIN-24)
            wrapped.extend([lines[0]] + ["     " + continuation for continuation in lines[1:]])
        height = len(wrapped) * 13 + 20
        self.ensure(height)
        rect = fitz.Rect(MARGIN-5, self.y, WIDTH-MARGIN+5, self.y+height)
        self.page.draw_rect(rect, fill=(.955, .965, .976), color=(.82, .86, .90), width=.5)
        self.y += 8
        for line in wrapped:
            # Existing code lines are ASCII; Chinese comments use the same CJK font.
            fontname = "cour" if line.isascii() else "zh"
            f = self.mono if fontname == "cour" else self.font
            baseline = self.y + f.ascender * 9.5
            self.page.insert_text((MARGIN+6, baseline), line, fontname=fontname, fontsize=9.5, color=INK)
            self.used_text.append(line)
            self.layout.append({"page": self.page.number+1, "type": "code", "font_size": 9.5,
                                "source_path": path, "baseline": baseline, "text": line})
            self.y += 13
        self.y += 12

    def image_page(self, title: str, image: Path, caption: str, *, limits=(), exercises=(), planned=False):
        if not image.exists():
            raise FileNotFoundError(f"Required teaching asset is not ready: {image}")
        self.new_page(title)
        if planned:
            self.status("【待研究】该页讨论下一步实验；图片只反映当前已经保存的观察。", planned=True)
        image_top = self.y + 3
        # Maximum single-image height reserves room for readable source and caveats.
        image_rect = fitz.Rect(MARGIN, image_top, WIDTH-MARGIN, image_top+490)
        self.page.insert_image(image_rect, filename=str(image), keep_proportion=True)
        self.images.append({"path": str(image), "sha256": digest(image), "page": self.page.number+1,
                            "placement": list(image_rect), "display": "preserved aspect ratio; original PNG embedded"})
        self.y = image_rect.y1 + 7
        self.paragraph(caption, size=12, gap=5)
        if limits:
            self.paragraph("结论边界：" + "；".join(limits), size=12, color=(.36,.39,.44), gap=4)
        if exercises:
            self.paragraph("动手练习：" + "；".join(exercises), size=12, color=BLUE, gap=4)
        self.paragraph("图像来源：" + str(image), size=8.5, color=(.43,.47,.53), gap=0)

    def save(self):
        for page in self.doc:
            page.insert_text((MARGIN, HEIGHT-24), f"教学结果快照：2026-10-10 · 已执行与研究计划分开呈现", fontname="zh", fontsize=9, color=(.44,.49,.55))
            page.insert_text((WIDTH-MARGIN-45, HEIGHT-24), str(page.number+1), fontname="zh", fontsize=10, color=(.44,.49,.55))
        self.doc.set_toc(self.toc)
        self.doc.set_metadata({"title":"RareVLM 图解课堂：从代码到实验", "author":"RareVLM project learning materials", "subject":"Chinese CPU-only source-grounded algorithm tutorial", "keywords":"LoRA NF4 SFT DDP VLM 数据算法 DPO"})
        # Keep the complete CJK font: avoids CFF subsetting warnings and preserves
        # original glyph outlines. A single font is shared by the document pages.
        self.doc.save(self.output, garbage=4, deflate=True)
        self.doc.close()


def results_text(lesson: dict) -> list[str]:
    r = lesson["results"]
    name = lesson["id"]
    if name == "lora":
        values = r["engineering_evidence"]["values"]
        checkpoint = r["actual_checkpoint"]
        return [f"真实 qv_r8 / seed0 完成 {values['completed_updates']} 次更新、看到 {values['actual_questions_seen']:,} 个 QA；可训练参数 {values['trainable_parameters']:,}，冻结权重训练前后哈希一致。",
                f"q 的 A={checkpoint['q_proj']['A_shape']}，B={checkpoint['q_proj']['B_shape']}；v 的 B={checkpoint['v_proj']['B_shape']}。q 增量首尾奇异值约 {checkpoint['singular_values'][0]:.4f} / {checkpoint['singular_values'][-1]:.4f}，不等于准确率因果证据。"]
    if name == "qlora":
        v = r["training"]["values"]
        return [f"真实四卡 V100 16GiB 训练的单卡峰 allocated={v['peak_gpu_allocated_gib']:.3f} GiB，reserved={v['peak_gpu_reserved_gib']:.3f} GiB；活跃时间 {v['active_seconds']/60:.1f} 分钟。",
                "原始 FP32 预检峰 14.199 GiB 来自不同检查过程，不能直接计算训练显存节省比例。普通 AdamW、FP32 计算是本工程配方，未完整复现 QLoRA 论文的 paged optimizer 配方。"]
    if name == "sft":
        v = r["tail_head_equivalence"]["values"]
        real = r["real_case"]
        return [f"真实 dev 问题 {real['question_id']}：input_length={real['input_length']}，prefix={real['prefix_length']}，监督 {real['supervised_token_count']} 个 token；答案为“{real['target']}”。该 dev 样本只展示编码，没有执行优化。",
                f"尾部 lm_head 等价检查：loss 最大绝对误差 {v['tail_head_loss_max_absolute_error']:.2e}，梯度最大误差 {v['tail_head_gradient_max_absolute_error']:.2e}。训练 QA 均值 loss 与 dev token NLL 的加权方式不同。"]
    if name == "ddp":
        v = r["checkpoint_restore_contract"]["values"]
        return ["10000 QA ÷ 全局 batch16 = 625 次更新；真实末批契约用 33 QA 完成 3 次更新，有效 QA 数为 16、16、1，不重复或丢弃真实 QA。",
                f"恢复验证中 adapter、优化器和 CPU/CUDA RNG 通过恢复检查，dev NLL={v['restored_dev_nll']:.6f}，与原记录一致。这是工程契约证据，不是算法 benchmark 成绩。"]
    if name == "vision_budget":
        return ["真实路牌图片原始尺寸 1024×768，经同一 processor 缩放到 504×364，image_grid_thw=[1,26,36]；936 个 patch 经 2×2 merger 形成 234 个语言侧视觉 token。",
                "【未执行】256/512/1024 最大视觉预算对照、全图+预测 OCR 区域、全图+随机区域对照，当前均没有新生成成绩。下页局部放大只供人观察，未送给模型。"]
    if name == "data_selection":
        return ["qv_r8 三种子的“训练目标未见”切片平均 69.460%，常见答案切片平均 81.992%；未见桶占 858/1617 条 dev 问题。",
                "【未执行】随机 10k、质量+多样性 10k、答案频次覆盖 10k 的等预算采样训练，尚无受控收益。当前 stage1 是固定 SHA 图像顺序取前 10000 QA，并非频次均衡。"]
    return ["现有三任务结果来自单任务 TextVQA SFT：TextVQA 提升、DocVQA 因策略而变、MMStar 略降。请逐任务解释，不合成一个总分。",
            "【未实现 / 未训练】当前冻结 runner 无混合多任务 loader、无 DPO loss 或偏好训练结果。先从同一完整 SFT checkpoint 设计对照；下页图为当前单任务 SFT 观察。"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PRIVATE/"longtail_vlm/rarevlm_pilot/outputs/vlm_learning_v1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output or root/"图解课堂.pdf").resolve()
    if PRIVATE not in output.parents:
        raise ValueError("All task outputs must remain under /opt/data/private")
    source = root/"code/LESSONS.json"
    data = read_json(source)
    case_data = read_json(root/"cases/CASES.json")
    figure_data = read_json(root/"figures/FIGURES_PROVENANCE.json")
    book = Book(output)
    book.new_page("从一张路牌到一次可信的算法研究", toc=True)
    book.status("适用基础：会 Python/PyTorch、训练过模型；目标是从真实代码理解算法，再读实验结果。")
    book.heading("先明确你在学什么")
    book.paragraph("本读物沿着当前项目的真实路径学习：图像→processor→视觉 token→冻结底座与 LoRA 增量→assistant-only SFT→四卡训练与恢复→多任务评测。每一课回答一个具体问题，并用保存的代码与工程结果验证理解。")
    book.heading("建议阅读顺序")
    book.bullets(["第1–3课：理解参数更新、量化存储和 SFT 监督；先看机制图，再手算张量维度。",
                  "第4–6课：学习分布式归一化、视觉预算和数据切片；观察真实案例，再提出可被对照实验推翻的解释。",
                  "第7课：从当前单任务结果设计能力保持与 DPO；所有计划都显式标注，不能写成已有研究成果。"])
    book.heading("本次快照的证据范围")
    book.paragraph("图表冻结于 2026-10-10 10:48（北京时间）：q/v r8 和 q/k/v/o r4 各 3 个种子，q/v r4+merger 仅 seed0，共 7 个完整训练种子、28 条任务结果。图片和 PDF 不追读后台更新，不改变正在运行的训练或选型规则。")
    book.paragraph("TextVQA 使用十人参考答案的一致性评分；DocVQA 使用 ANLS；MMStar 使用子类别宏平均。图中的 SD 是已完成训练种子的样本标准差，不是问题/图像置信区间。三个手选案例供理解机制与提出假设，不是代表性样本。")
    book.heading("如何把练习变成工程经验")
    book.paragraph("每课完成一次手算或 CPU 阅读练习，写出“我预期看到什么、哪个结果会推翻解释”。不要改当前冻结配置。后续视觉预算、数据选择、能力保持和 DPO 实验会按单独协议执行；本读物中尚未得到它们的新训练/生成结果。")
    book.paragraph("完整代码片段、SHA 与可复现提取命令在 code/LESSONS.json；图表原始种子、源快照和方法记录在 figures/。本 PDF 是教学材料，不宣称发布、期刊规范认证或因果证明。", color=(.39,.43,.49))

    chosen_codes = {"lora": [0], "qlora": [0], "sft": [1,2], "ddp": [3], "vision_budget": [2], "data_selection": [3], "multitask_dpo": []}
    for lesson in data["lessons"]:
        name = lesson["id"]
        planned = name in {"vision_budget", "data_selection", "multitask_dpo"}
        book.new_page(lesson["title"], toc=True)
        if name == "multitask_dpo":
            book.status("【计划研究：未实现 / 未训练】多任务与 DPO 尚无当前项目代码或训练成绩。", planned=True)
        elif planned:
            book.status("【当前诊断已测量；算法干预待执行】以下观察与下一轮计划分别说明。", planned=True)
        else:
            book.status("【已有实现与工程测量】下面代码与结果来自现有冻结项目。")
        book.paragraph("要回答的问题：" + lesson["question"], color=BLUE)
        book.heading("基本原理")
        principle = lesson["principle"]
        book.paragraph(principle["intuition"])
        for equation in principle["equations"][:2]:
            book.paragraph(clean_formula(equation), size=12, color=BLUE, gap=3)
        shape_count = 2 if name == "lora" else 1
        book.bullets(principle.get("shapes", [])[:shape_count])
        book.heading("真实代码与读法" if chosen_codes[name] else "当前实现边界与最小对照")
        for index in chosen_codes[name]:
            snippet = lesson["code"][index]
            book.code(snippet)
            explanations = snippet["explanations"]
            chosen = explanations[:1] + (explanations[-1:] if len(explanations)>1 else [])
            book.bullets(chosen)
        if name == "multitask_dpo":
            book.paragraph("此处没有现有 DPO 代码可展示。上方是计划算法公式：DPO 比较同一输入的优选/劣选完整回答，相对固定参考策略改变对数概率比；序列 log 概率求和与现有每 QA 平均 token CE 不同。")
            book.bullets(["多任务：纯 TextVQA、混合训练且匹配总 QA/token 预算、混合训练且匹配 TextVQA 曝光。",
                          "DPO：同一 SFT 起点不追加训练、chosen-only 继续 SFT、DPO；匹配数据、预算、reference，并审核偏好对是否有效。"])
        book.heading("实际工程结果")
        book.bullets(results_text(lesson))
        if name == "ddp":
            book.paragraph("读法：world=4、末批 actual=1 时，本地唯一真样本的 loss 乘 4；DDP 再跨 4 卡求均值，最终得到该单样本梯度。sentinel 只保持计算路径，valid=False 使它不贡献梯度或曝光统计。")
            book.paragraph("结论边界：" + lesson["limits"][0])
            book.paragraph("动手练习：" + lesson["exercises"][0]["task"], color=BLUE)
        else:
            book.paragraph("下一页把这课对应到图片；看完图，再完成页末练习。", size=10, color=(.39,.43,.49))
        limits = lesson["limits"][:1]
        exercises = [lesson["exercises"][0]["task"]]
        if name == "lora":
            book.image_page("第1课图解：LoRA 的两条计算支路", root/"figures/mechanism_lora.png",
                            "先沿图走一遍 x→A→B，再与冻结的 W0x 相加。注意力 q/k/v/o 是不同投影；当前三策略匹配参数预算，但 rank 和位置同时变化。")
            book.image_page("第1课实物：checkpoint 中的低秩更新", root/"figures/lora_update.png",
                            "图中的 A/B 与 ΔW 来自真实保存的 qv_r8 / seed0 adapter。奇异值描述该层增量方向的大小；热图是实际子块，不是模型注意力图。", limits=limits, exercises=exercises)
        elif name == "qlora":
            book.image_page("第2课证据：同一 adapter 的四条推理路径", root/"figures/numerical_paths.png",
                            figure_data["figures"]["numerical_paths"]["description_zh"], limits=limits, exercises=exercises)
        elif name == "sft":
            book.image_page("第3课图解：输入保留，监督只算 assistant", root/"figures/sft_mask.png",
                            "labels=-100 只屏蔽对应位置的目标 loss，图像和问题仍在 input_ids 中作为条件。下一 token 预测需要错一位；答案结束模板 token 也参与监督。", limits=limits, exercises=exercises)
        elif name == "vision_budget":
            for index, case in enumerate(case_data["cases"], 1):
                book.image_page(f"第5课真实案例 {index}：{case['title']}", root/"cases"/case["assets"]["teaching_board.png"],
                                case["observation"] + " " + case["explanation"],
                                limits=[case["failure_boundary"]],
                                exercises=["写出能区分“没读清”与“问题关联错位”的一组固定 checkpoint 输入对照；当前图的局部放大未送入模型。"] if index == 3 else [])
        elif name == "data_selection":
            book.image_page("第6课证据：频次切片不是视觉长尾的因果结论", root/"figures/answer_frequency_slices.png",
                            figure_data["figures"]["answer_frequency_slices"]["description_zh"], limits=limits, exercises=exercises)
        elif name == "multitask_dpo":
            book.image_page("第7课观察：当前单任务 SFT 的四项指标", root/"figures/stage1_results.png",
                            figure_data["figures"]["stage1_results"]["description_zh"])
            book.image_page("第7课研究起点：把收益与代价分任务读清", root/"figures/task_deltas.png",
                            figure_data["figures"]["task_deltas"]["description_zh"], limits=limits, exercises=exercises)

    book.new_page("参考资料与复现入口", toc=True)
    book.heading("算法论文（原理来源；不是本项目性能的替代证据）")
    references = [
        ("LoRA: Low-Rank Adaptation of Large Language Models", "2021", "https://arxiv.org/abs/2106.09685"),
        ("QLoRA: Efficient Finetuning of Quantized LLMs", "2023", "https://arxiv.org/abs/2305.14314"),
        ("Qwen2.5-VL Technical Report", "2025", "https://arxiv.org/abs/2502.13923"),
        ("Direct Preference Optimization: Your Language Model is Secretly a Reward Model", "2023", "https://arxiv.org/abs/2305.18290"),
    ]
    for title, year, url in references:
        book.paragraph(f"{title}（arXiv，{year}）\n{url}", gap=10)
    book.heading("软件与绘图流程参考")
    book.paragraph("Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. arXiv:2609.00065。核验版本 v2（2026-09-02）；引用 DOI 不附版本后缀。\nhttps://doi.org/10.48550/arXiv.2609.00065")
    book.paragraph("本次使用 PyMuPDF 排版并嵌入 Noto Sans CJK SC，Matplotlib 绘制结果图。来源/变换/缺失值/原始种子的记录遵循 scientific-visualization 技能；这不构成期刊或无障碍合规认证。")
    book.heading("本地证据与可复现脚本")
    book.paragraph("code/LESSONS.json：完整原始片段、行号、SHA 与工程证据路径。\ncode/LORA_TENSORS.json：真实 A/B 形状、参数量与奇异值。\ncode/TOKEN_MASK.json：真实 input/label/token 对齐。\nfigures/snapshot/：原始结果表；figures/*.csv：散点与均值。\ncases/CASES.json：案例预测、预处理与人工裁图说明。")
    book.paragraph("构建 PDF：source /opt/data/private/.tools/activate.sh\npython tools/build_vlm_learning_pdf.py\n脚本只读上述教学材料，不加载模型、不使用 GPU、不重跑实验。", size=10)
    book.save()

    # One directly tied output audit: geometry, embedded images, readable text and fonts.
    pdf = fitz.open(output)
    pages = []
    overflow = []
    for page in pdf:
        span_count = 0
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    span_count += 1
                    rect = fitz.Rect(span["bbox"])
                    if rect.x0 < 0 or rect.y0 < 0 or rect.x1 > WIDTH+.2 or rect.y1 > HEIGHT+.2:
                        overflow.append({"page":page.number+1,"bbox":list(rect),"text":span["text"]})
        pages.append({"page":page.number+1,"text_spans":span_count,"images":len(page.get_images(full=True)),
                      "fonts":[f[3] for f in page.get_fonts(full=True)], "text_characters":len(page.get_text())})
    used = set("".join(book.used_text))
    missing = sorted(c for c in used if not c.isspace() and not book.font.has_glyph(ord(c)))
    # Chinese text is covered by the embedded CJK font; ASCII code uses Courier.
    chinese_missing = [c for c in missing if ord(c)>127]
    def canonical_text(value):
        # CJK glyph aliases may extract as Unicode compatibility ideographs.
        value = unicodedata.normalize("NFKC", value)
        for hyphen in "‐‑‒–−":
            value = value.replace(hyphen, "-")
        return "".join(value.split())
    all_text = canonical_text("".join(page.get_text() for page in pdf))
    title_check = {l["id"]:canonical_text(l["title"]) in all_text for l in data["lessons"]}
    if overflow or chinese_missing or not all(title_check.values()):
        raise ValueError(f"PDF output check failed: overflow={overflow}, missing={chinese_missing}, lessons={title_check}")
    qa = root/"qa"
    qa.mkdir(exist_ok=True)
    render_numbers = [2, next(i["page"] for i in book.images if i["path"].endswith("sft_mask.png")),
                      next(i["page"] for i in book.images if i["path"].endswith("03_shared_clock_failure/teaching_board.png"))]
    rendered = []
    for number in render_numbers:
        path = qa/f"pdf_page_{number:02d}.png"
        pdf[number-1].get_pixmap(matrix=fitz.Matrix(1.4,1.4), alpha=False).save(path)
        rendered.append(str(path))
    review = {"schema_version":"rarevlm.learning_pdf_review.v1", "created_at_asia_shanghai":datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "pdf":str(output),"pdf_sha256":digest(output),"page_count":len(pdf),"page_size_pt":[WIDTH,HEIGHT],
              "body_size_pt":12,"code_size_pt":9.5,"lesson_titles_found":title_check,
              "text_search_normalization":"NFKC + whitespace/dash normalization for CJK font glyph aliases",
              "embedded_image_placements":len(book.images),"images":book.images,
              "text_bbox_overflow":overflow,"missing_non_ascii_glyphs":chinese_missing,
              "font":str(FONT_PATH),"font_sha256":digest(FONT_PATH),"pages":pages,
              "source_lessons":{"path":str(source),"sha256":digest(source)},"displayed_code":book.sources,
              "other_sources":[{"path":str(p),"sha256":digest(p)} for p in [root/"cases/CASES.json",root/"figures/FIGURES_PROVENANCE.json"]],
              "rendered_qa_pages":rendered,"manual_render_review":"pending actual view_image review",
              "scope":"teaching output QA, not journal/accessibility compliance certification; no model execution",
              "script":{"path":str(Path(__file__).resolve()),"sha256":digest(Path(__file__))}}
    (root/"PDF_REVIEW.json").write_text(json.dumps(review,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    pdf.close()
    print(json.dumps({"pdf":str(output),"pages":review["page_count"],"images":len(book.images),"qa_pages":rendered},ensure_ascii=False))


if __name__ == "__main__":
    main()
