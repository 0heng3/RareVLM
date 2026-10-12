"""Build CPU-only TextVQA learning cases from saved internal-dev predictions.

No model weights are loaded, and no prediction or training is run. The manually
specified rectangles below were selected by looking at original images. They
are illustrations for human readers, never additional model inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

# Keep imports from creating bytecode in frozen experiment source directories.
sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("MPLBACKEND", "Agg")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from vlm_deep.common import NORMALIZER, consensus, content_sha, encode, read_manifest
from vlm_sft.common import sha_file

WORKSPACE = Path("/opt/data/private")
METHODS = ("qv_r8", "qkvo_r4", "qv_r4_merger")
LABELS = {
    "qv_r8": "q/v · rank 8",
    "qkvo_r4": "q/k/v/o · rank 4",
    "qv_r4_merger": "q/v · rank 4 + merger",
}
# Rectangles use original-image x0,y0,x1,y1 pixels and are fixed teaching
# annotations, not OCR detections, attention maps, or label-driven localization.
CASE_SPECS = (
    {
        "question_id": "27482", "slug": "01_road_number",
        "title": "先看真实错误：路牌有多个数字，问题到底指哪一行？",
        "question_zh": "到 Albertville 是多少英里？（沿用原问题；按标注 40 评分）",
        "crop_xyxy": [510, 236, 970, 355],
        "crop_selection_basis": "人工查看原图，选择右侧箭头路牌的两行文字；包含邻近的 20 与 ALBERTVILLE 40，未用答案自动定位。",
        "observation": "原图右侧路牌上方是 20，下方 ALBERTVILLE 是 40。三个模型输入相同，输出却不同。",
        "explanation": "这个例子让你提出两种假设：是否没读清数字？是否把问题对应到错误的文字行？仅凭输出不能分辨。",
        "next_experiment": "下一轮固定 checkpoint，再改视觉预算检验信息瓶颈；本图没有运行更高预算或裁剪推理。",
        "failure_boundary": "q/k/v/o 与 merger 在本例纠错，不证明它们在整个 dev 上占优，也不证明注意力权重定位改变。",
    },
    {
        "question_id": "10476", "slug": "02_shop_string",
        "title": "只差一个字母：细粒度识别与严格答案匹配",
        "question_zh": "这家店叫什么名字？",
        "crop_xyxy": [35, 34, 241, 215],
        "crop_selection_basis": "人工查看原图，选择左上角醒目的店招完整区域；不是 OCR 模型结果，也没有使用答案自动选框。",
        "observation": "店招写着 FLESOR'S CANDY KITCHEN；q/v 与 q/k/v/o 将 r 输出成了 n，merger 版本在本例匹配参考答案。",
        "explanation": "TextVQA 的标准分数按规范化字符串计算；近似拼写没有编辑距离部分分。模型为何写错字母仍需要输入干预。",
        "next_experiment": "该代表训练目标在本轮 10k QA 中出现 0 次；这不意味着预训练未见，也不能把单例错误归因于答案频次。",
        "failure_boundary": "本例只有 seed0；不能据此宣布 merger 更擅长 OCR。原图局部放大仅供学习，并未送入模型。",
    },
    {
        "question_id": "12257", "slug": "03_shared_clock_failure",
        "title": "共同失败例：看懂场景不代表读对小数字",
        "question_zh": "炉灶钟表上显示的时间是多少？",
        "crop_xyxy": [194, 266, 359, 334],
        "crop_selection_basis": "人工查看原图，选择炉灶控制面板上肉眼可见的时钟区域；固定像素矩形，不自动按参考答案定位。",
        "observation": "原图时钟显示 12:58。三种 LoRA 策略 seed0 分别输出 1:53、1:58、1:59，标准共识分都为 0。",
        "explanation": "显示区域在整图中很小，缩放后字形信息更少；这是待验证的瓶颈假设。增加参数并不保证恢复输入丢失的细节。",
        "next_experiment": "先在相同 checkpoint 下提高视觉预算：如果纠错，再研究同预算的区域感知；若仍失败，继续检查数据与字符串生成。",
        "failure_boundary": "共同失败是观察结果；视觉分辨率、标注歧义与模型行为的因果解释都不能仅凭这个示例确定。",
    },
)


def bound_json(path: Path):
    data = path.read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def require_persistent_output(path: Path):
    resolved = path.resolve()
    if not resolved.is_relative_to(WORKSPACE):
        raise ValueError("output must remain under /opt/data/private")


def read_run(root: Path, method: str, manifest, config):
    directory = root / f"stage1/{method}/seed0/eval_dev"
    evaluation_path = directory / "EVALUATION.json"
    predictions_path = directory / "predictions.json"
    evaluation, evaluation_sha = bound_json(evaluation_path)
    predictions, prediction_sha = bound_json(predictions_path)
    if evaluation["task"] != "textvqa" or evaluation["split"] != "dev":
        raise ValueError("not a completed internal-dev evaluation")
    if evaluation["manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("manifest binding mismatch")
    if evaluation["config_sha256"] != content_sha(config):
        raise ValueError("config binding mismatch")
    if evaluation["predictions_sha256"] != prediction_sha:
        raise ValueError("prediction file changed since scoring")
    dev_ids = {row["question_id"] for row in manifest["splits"]["dev"]}
    lookup = {row["question_id"]: row for row in predictions}
    if len(lookup) != len(predictions) or set(lookup) != dev_ids:
        raise ValueError("incomplete or duplicated dev prediction set")
    if evaluation["n"] != len(dev_ids):
        raise ValueError("evaluation count mismatch")
    generation = evaluation["generation"]
    if generation["min_pixels"] != config["min_pixels"] or generation["max_pixels"] != config["max_pixels"]:
        raise ValueError("saved prediction image budget differs")
    return lookup, {
        "method": method, "seed": 0,
        "evaluation_path": str(evaluation_path.resolve()),
        "evaluation_sha256": evaluation_sha,
        "predictions_path": str(predictions_path.resolve()),
        "predictions_sha256": prediction_sha,
        "config_sha256": evaluation["config_sha256"],
        "source_sha256": evaluation["source_sha256"],
        "completed_internal_dev_score_percent": evaluation["score_percent"],
    }


def inverse_unnormalized_patches(flatten, grid, processor):
    """Invert patch packing only; do not invert normalization or sharpen pixels."""
    import numpy as np

    t, h, w = map(int, grid)
    ip = processor.image_processor
    m, p, temporal = int(ip.merge_size), int(ip.patch_size), int(ip.temporal_patch_size)
    packed = flatten.reshape(t, h // m, w // m, m, m, 3, temporal, p, p)
    images = packed.transpose(0, 6, 5, 1, 3, 7, 2, 4, 8).reshape(t * temporal, 3, h * p, w * p)
    return np.asarray(images[0].transpose(1, 2, 0), dtype=np.uint8)


def create_board(case, original, resized, crop_original, crop_resized, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Rectangle
    import textwrap

    font = Path("/opt/data/private/.tools/fonts/NotoSansCJKsc-Regular.otf")
    if not font.exists():
        raise FileNotFoundError("the locally prepared Chinese font is missing")
    font_manager.fontManager.addfont(str(font))
    family = font_manager.FontProperties(fname=str(font)).get_name()
    style = {"font.family": family, "axes.unicode_minus": False, "font.size": 14,
             "figure.facecolor": "#ffffff", "text.color": "#172b3a"}
    with plt.rc_context(style):
        fig = plt.figure(figsize=(14, 16), dpi=160)
        fig.text(.045, .969, case["title"], fontsize=24, weight="bold", va="top")
        fig.text(.045, .936, "问题：" + case["question_zh"], fontsize=18, va="top")
        fig.text(.045, .912, "Raw question: " + case["raw_question"], fontsize=14, va="top")
        fig.text(.045, .890, f"内部 dev · qid={case['question_id']} · 同一图像/问题 · 三种策略 seed0 · 示例挑选，不代表总体表现", fontsize=13, color="#52616b")
        ax_original = fig.add_axes([.045, .584, .438, .276])
        ax_resized = fig.add_axes([.518, .584, .438, .276])
        for ax, bitmap in [(ax_original, original), (ax_resized, resized)]:
            ax.imshow(bitmap, interpolation="nearest")
            ax.axis("off")
        x0, y0, x1, y1 = case["illustration_only_crop"]["original_xyxy"]
        ax_original.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, edgecolor="#0072b2", linewidth=2.8))
        rx0, ry0, rx1, ry1 = case["illustration_only_crop"]["resized_xyxy"]
        ax_resized.add_patch(Rectangle((rx0, ry0), rx1-rx0, ry1-ry0, fill=False, edgecolor="#0072b2", linewidth=2.8))
        p = case["preprocessing"]
        ax_original.set_title(f"A  原图：{original.width} × {original.height} px", fontsize=17, loc="left", pad=10)
        ax_resized.set_title(f"B  真实预处理缩放：{resized.width} × {resized.height} px", fontsize=17, loc="left", pad=10)
        fig.text(.518, .573, f"最多 256 视觉 token；本图实际 {p['merged_visual_tokens']} 个", fontsize=14, color="#00608f")
        fig.text(.045, .550, "B 显示归一化与 patchify 前的 RGB 缩放结果；显示尺寸放大不会新增信息。蓝框是人工展示区域。", fontsize=13)
        ax_co = fig.add_axes([.045, .372, .438, .130])
        ax_cr = fig.add_axes([.518, .372, .438, .130])
        for ax, bitmap in [(ax_co, crop_original), (ax_cr, crop_resized)]:
            ax.imshow(bitmap, interpolation="nearest")
            ax.axis("off")
        ax_co.set_title(f"C  原图局部：{crop_original.width} × {crop_original.height} px", fontsize=17, loc="left", pad=10)
        ax_cr.set_title(f"D  缩放图同一区域：{crop_resized.width} × {crop_resized.height} px", fontsize=17, loc="left", pad=10)
        fig.text(.045, .360, "C/D 仅供人类观察：固定像素裁剪、最近邻显示，无锐化；本轮没有把裁剪图输入模型。", fontsize=14, color="#8a4c12")
        fig.text(.045, .329, "已保存预测（标准共识分 0–1；不是人工新打分）", fontsize=17, weight="bold")
        for i, prediction in enumerate(case["saved_seed0_predictions"]):
            y = .300 - i*.025
            score = prediction["saved_consensus_score"]
            fig.text(.055, y, LABELS[prediction["method"]], fontsize=16)
            fig.text(.368, y, prediction["raw_prediction"], fontsize=17, weight="bold")
            fig.text(.820, y, f"score = {score:.1f}", fontsize=16, color="#006b5d" if score == 1 else "#9b341c")
        ref_count = Counter(case["reference_answers"])
        refs = "；".join(f"{answer!r} ×{count}" for answer, count in ref_count.items())
        fig.text(.045, .213, "10 位人类参考答案：" + refs, fontsize=13)
        fig.text(.045, .187, "观察：" + case["observation"], fontsize=14)
        # Chinese characters are wrapped by character count, retaining spaces in
        # any inline English strings. Explicit positions prevent hidden clipping.
        for start, prefix, text in [(.158, "怎么理解：", case["explanation"]),
                                    (.116, "下一步：", case["next_experiment"]),
                                    (.074, "结论边界：", case["failure_boundary"])]:
            wrapped = textwrap.wrap(prefix + text, width=67, break_long_words=True, break_on_hyphens=False)
            fig.text(.045, start, "\n".join(wrapped), fontsize=13, va="top", linespacing=1.45)
        fig.savefig(output, facecolor="white", dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=WORKSPACE / "longtail_vlm/rarevlm_pilot/outputs/vlm_deep_v2")
    parser.add_argument("--run-root", type=Path, default=WORKSPACE / "longtail_vlm/rarevlm_pilot/outputs/vlm_deep_v2/runs")
    parser.add_argument("--model-root", type=Path, default=WORKSPACE / "longtail_vlm/rarevlm_pilot/outputs/vlm_sft_textvqa_v1/model")
    parser.add_argument("--config", type=Path, default=REPO / "configs/vlm_deep_v2.json")
    parser.add_argument("--output", type=Path, default=WORKSPACE / "longtail_vlm/rarevlm_pilot/outputs/vlm_learning_v1/cases")
    args = parser.parse_args()
    require_persistent_output(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import PIL
    from PIL import Image
    import transformers
    from transformers import AutoProcessor
    from transformers.image_transforms import resize
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

    manifest_path = args.asset_root / "prepared/manifest.json"
    manifest = read_manifest(manifest_path, verify_images=False)
    config, config_file_sha = bound_json(args.config)
    runs = {method: read_run(args.run_root, method, manifest, config) for method in METHODS}
    dev = {row["question_id"]: row for row in manifest["splits"]["dev"]}
    frequencies = Counter(NORMALIZER(row["target"]) for row in manifest["splits"]["train_10k"])
    processor = AutoProcessor.from_pretrained(str(args.model_root), local_files_only=True,
                                             min_pixels=config["min_pixels"], max_pixels=config["max_pixels"], use_fast=False)
    processor.tokenizer.padding_side = "left"
    ip = processor.image_processor
    cases = []
    for spec in CASE_SPECS:
        row = dev[spec["question_id"]]
        if len(row["answers"]) != 10:
            raise ValueError("expected exactly ten reference answers")
        image_info = manifest["images"][row["image_id"]]
        original_path = manifest_path.parent / image_info["path"]
        original_sha = sha_file(original_path)
        if original_sha != image_info["sha256"]:
            raise ValueError("original image file changed")
        original = Image.open(original_path).convert("RGB")
        # Reuse the exact label-free encode path of the ongoing experiment.
        encoded = encode(processor, row, original_path, supervised=False)
        grid = list(map(int, encoded["image_grid_thw"][0].tolist()))
        t, gh, gw = grid
        height, width = gh * int(ip.patch_size), gw * int(ip.patch_size)
        merged_tokens = t * gh * gw // (int(ip.merge_size) ** 2)
        smart_h, smart_w = smart_resize(original.height, original.width, factor=int(ip.patch_size)*int(ip.merge_size),
                                       min_pixels=config["min_pixels"], max_pixels=config["max_pixels"])
        if (height, width) != (smart_h, smart_w) or merged_tokens > config["max_pixels"] // (int(ip.patch_size)*int(ip.merge_size))**2:
            raise ValueError("actual processor dimensions or visual token budget differ")
        resized_pixels = resize(np.asarray(original), size=(height, width), resample=ip.resample,
                                input_data_format="channels_last")
        # Check the illustrative RGB bitmap against the actual image processor's
        # patch packing with only rescale/normalize disabled; no model is loaded.
        raw_preprocessed = ip(original, do_rescale=False, do_normalize=False, return_tensors="np")
        unpacked = inverse_unnormalized_patches(raw_preprocessed["pixel_values"], grid, processor)
        if not np.array_equal(resized_pixels, unpacked):
            raise ValueError("displayed resize differs from actual processor RGB pixels")
        resized = Image.fromarray(resized_pixels)
        directory = args.output / spec["slug"]
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original_path, directory / "original.jpg")
        if sha_file(directory / "original.jpg") != original_sha:
            raise ValueError("original image copy differs")
        resized.save(directory / "preprocessed_rgb.png")
        x0, y0, x1, y1 = spec["crop_xyxy"]
        if not (0 <= x0 < x1 <= original.width and 0 <= y0 < y1 <= original.height):
            raise ValueError("manual illustration rectangle is outside the image")
        resized_box = [math.floor(x0*width/original.width), math.floor(y0*height/original.height),
                       math.ceil(x1*width/original.width), math.ceil(y1*height/original.height)]
        crop_original = original.crop(spec["crop_xyxy"])
        crop_resized = resized.crop(resized_box)
        crop_original.save(directory / "human_zoom_original.png")
        crop_resized.save(directory / "human_zoom_preprocessed.png")
        predictions = []
        normalized_refs = [NORMALIZER(a) for a in row["answers"]]
        for method in METHODS:
            saved = runs[method][0][row["question_id"]]
            if saved["image_id"] != row["image_id"] or saved["image_sha256"] != original_sha or saved["question_sha256"] != content_sha(row["question"]):
                raise ValueError("saved prediction/image/question binding mismatch")
            computed = consensus(saved["prediction"], row["answers"])
            if not math.isclose(computed, saved["score"], abs_tol=1e-12):
                raise ValueError("saved standard consensus score differs from recomputation")
            normalized = NORMALIZER(saved["prediction"])
            loo = [min(sum(normalized == answer for j, answer in enumerate(normalized_refs) if j != i) / 3, 1)
                   for i in range(10)]
            predictions.append({"method": method, "seed": 0, "raw_prediction": saved["prediction"],
                                "normalized_prediction": normalized, "saved_consensus_score": saved["score"],
                                "recomputed_consensus_score": computed,
                                "normalized_reference_match_count": sum(normalized == answer for answer in normalized_refs),
                                "leave_one_annotator_out_scores": loo,
                                "provenance": runs[method][1]})
        case = {key: value for key, value in spec.items() if key not in ("crop_xyxy", "crop_selection_basis")}
        case.update({
            "image_id": row["image_id"], "raw_question": row["question"],
            "reference_answers": row["answers"], "normalized_reference_answers": normalized_refs,
            "raw_training_target": row["target"], "normalized_training_target": NORMALIZER(row["target"]),
            "train_10k_target_qa_frequency": frequencies[NORMALIZER(row["target"])],
            "target_frequency_scope": "frequency among actual 10k QA training targets, not pretraining exposure",
            "source_image_path": str(original_path.resolve()), "source_image_sha256": original_sha,
            "source_decoded_rgb_sha256": hashlib.sha256(np.asarray(original).tobytes()).hexdigest(),
            "original_dimensions_wh": list(original.size), "saved_seed0_predictions": predictions,
            "preprocessing": {
                "input_path": "vlm_deep.common.encode(processor, row, image_path, supervised=False)",
                "references_supplied_to_processor": False,
                "min_pixels": config["min_pixels"], "max_pixels": config["max_pixels"],
                "max_merged_visual_token_budget": config["max_pixels"] // (int(ip.patch_size)*int(ip.merge_size))**2,
                "actual_resized_dimensions_wh": [width, height], "image_grid_thw": grid,
                "vision_encoder_patch_tokens": t*gh*gw, "merged_visual_tokens": merged_tokens,
                "patch_size": int(ip.patch_size), "merge_size": int(ip.merge_size),
                "temporal_patch_size": int(ip.temporal_patch_size), "resample": "PIL BICUBIC (processor setting)",
                "display_pixels_verified_against_processor_patches": True,
                "bitmap_scope": "RGB after the actual processor resize, before rescaling/normalization and patch packing; not model attention or predicted OCR",
            },
            "illustration_only_crop": {
                "original_xyxy": spec["crop_xyxy"], "resized_xyxy": resized_box,
                "selection_basis": spec["crop_selection_basis"], "sent_to_model": False,
                "coordinate_system": "top-left origin, original or resized pixel coordinates; right/bottom exclusive",
                "pixel_processing": "exact pixel crop; no sharpening, contrast adjustment, generative redraw, or OCR; board displays use nearest interpolation",
            },
            "assets": {name: f"{spec['slug']}/{name}" for name in
                       ("original.jpg", "preprocessed_rgb.png", "human_zoom_original.png", "human_zoom_preprocessed.png", "teaching_board.png")},
            "selection_warning": "hand-selected teaching example, not a representative sample or aggregate method ranking",
        })
        create_board(case, original, resized, crop_original, crop_resized, directory / "teaching_board.png")
        case["asset_sha256"] = {name: sha_file(directory/name) for name in case["assets"]}
        write_json(directory / "CASE.json", case)
        cases.append(case)

    note = (
        "# 真实 TextVQA 内部 dev 教学案例\n\n"
        "这些是人工挑选的学习示例，不代表整体表现。只读取已有 seed0 预测与原图，没有加载模型、使用 GPU、生成新预测或改变冻结实验。\n\n"
        "每例都给出原始问题、十个人类参考答案、三种策略的已保存预测及标准共识分，并重新检查预测/评测/数据/原图哈希绑定。\n\n"
        "原图复制保持原文件不变。`preprocessed_rgb.png` 复用当前 label-free encode 路径的实际尺寸，且逐像素验证其与实际 CPU 图像处理器去掉 rescale/normalize 后的 patch 还原结果一致；它只是归一化及 patchify 前的 RGB 位图。\n\n"
        "C/D 区域仅为人类观察：人工查看原图后预先写入矩形坐标，只做原像素裁剪，未锐化、重绘或调整对比度；板上最近邻放大不会新增细节。局部图没有作为本轮模型输入，不是裁剪推理实验结果，也不是注意力/OCR 检测图。\n\n"
        "最多 256 个 merger 后视觉 token，不保证每张图正好 256；本例两张横图实际为 234，方图为 256。图像仍会进一步 rescale、normalize、patchify 后送入视觉编码器。\n\n"
        "共识评分先用 EvalAIAnswerProcessor 规范化预测与参考答案，再对每位标注者留一，计算 min(其余9位的匹配数/3, 1)，最后取10项平均。仅字符串相似不获得编辑距离部分分。\n\n"
    )
    for case in cases:
        note += f"## {case['title']}\n\nqid={case['question_id']}，image_id={case['image_id']}。\n\n"
        note += f"原始问题：{case['raw_question']}\n\n中文问题：{case['question_zh']}\n\n十个人类答案：{json.dumps(case['reference_answers'], ensure_ascii=False)}\n\n"
        for prediction in case["saved_seed0_predictions"]:
            note += f"- {prediction['method']} / seed0：{prediction['raw_prediction']!r}，saved consensus={prediction['saved_consensus_score']}。\n"
        note += f"\n{case['observation']}\n\n{case['explanation']}\n\n{case['next_experiment']}\n\n{case['failure_boundary']}\n\n"
        note += f"展示区域坐标：{case['illustration_only_crop']['original_xyxy']}；{case['illustration_only_crop']['selection_basis']}\n\n"
    note += "复现：先 source /opt/data/private/.tools/activate.sh，使用 /opt/data/private/.venvs/rarevlm-vlm/bin/python tools/build_vlm_learning_cases.py。全部输出都在 /opt/data/private 下。\n"
    (args.output / "说明.md").write_text(note)
    processor_source = Path(inspect.getsourcefile(type(ip)))
    record = {
        "schema_version": "rarevlm.learning_cases.v1", "scope": "CPU-only saved internal-dev illustration; no new inference",
        "selection": "three manually selected examples: a numeric correction, an unseen training-target string correction, and a shared failure; not representative",
        "manifest_path": str(manifest_path.resolve()), "manifest_file_sha256": sha_file(manifest_path),
        "manifest_content_sha256": manifest["manifest_sha256"], "config_path": str(args.config.resolve()),
        "config_file_sha256": config_file_sha, "config_content_sha256": content_sha(config),
        "script_path": str(Path(__file__).resolve()), "script_sha256": sha_file(Path(__file__)),
        "model_weights_loaded": False, "new_predictions_generated": False, "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "processor": {"model_root": str(args.model_root.resolve()), "class": type(processor).__name__,
                      "image_processor_class": type(ip).__name__, "image_processor_source_path": str(processor_source),
                      "image_processor_source_sha256": sha_file(processor_source),
                      "transformers_version": transformers.__version__, "pillow_version": PIL.__version__,
                      "preprocessor_config_sha256": sha_file(args.model_root / "preprocessor_config.json")},
        "cases": cases,
    }
    write_json(args.output / "CASES.json", record)
    print(json.dumps({"output": str(args.output.resolve()), "cases": len(cases),
                      "question_ids": [c["question_id"] for c in cases],
                      "actual_merged_visual_tokens": [c["preprocessing"]["merged_visual_tokens"] for c in cases]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
