"""Read-only CPU extraction of source-grounded lessons and real adapter tensors.

Writes only the requested learning directory. Never loads model weights, starts
training, edits frozen code/configuration, or initializes a CUDA context.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

sys.dont_write_bytecode = True
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
DEFAULT_RUN = REPO.parents[1] / "outputs/vlm_deep_v2"
DEFAULT_MODEL = REPO.parents[1] / "outputs/vlm_sft_textvqa_v1/model"
DEFAULT_OUT = REPO.parents[1] / "outputs/vlm_learning_v1/code"
PAPERS = {
    "lora": "https://arxiv.org/abs/2106.09685",
    "qlora": "https://arxiv.org/abs/2305.14314",
    "qwen": "https://arxiv.org/abs/2502.13923",
    "dpo": "https://arxiv.org/abs/2305.18290",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def evidence(path, fields=None):
    value = read(path)
    if fields is not None:
        value = {k: value[k] for k in fields}
    return {"path": str(Path(path).resolve()), "sha256": sha(path), "values": value}


def snippet(relative, first, last, explanations):
    path = REPO / relative
    if not path.exists() and not Path(relative).is_absolute():
        path = REPO / "src" / relative
    lines = path.read_text().splitlines()
    old_first, old_last = first, last
    offset = sum(line.startswith('from rarevlm_paths import ') for line in lines[:35])
    first, last = first + offset, last + offset
    if relative == "vlm_deep/train.py":
        anchors = {
            (43, 49): ("def schedule(n,seed,world,micro,acc):", "    return order,padded"),
            (102, 105): ("    micro=c['training_microbatch'];acc=c['training_accumulation'];effective=world*micro*acc",
                         "    order,padded=schedule(len(rows),args.seed,world,micro,acc);steps=len(padded)//effective"),
            (110, 111): ("    params=[p for p in model.parameters() if p.requires_grad]",
                         "    optimizer=torch.optim.AdamW(params,lr=optimizer_lr(0,steps,c),weight_decay=c['weight_decay'])"),
            (130, 155): ("    resume=None;latest=args.output_dir/'LATEST.json'",
                         "    if resume:restore_rng(resume['rank_states'][rank]['rng'])"),
            (157, 195): ("    def save_resume(completed):", "            save_resume(step+1)"),
        }
        start_anchor, end_anchor = anchors[(old_first, old_last)]
        starts = [index + 1 for index, line in enumerate(lines) if line == start_anchor]
        ends = [index + 1 for index, line in enumerate(lines) if line == end_anchor]
        assert len(starts) == len(ends) == 1, f"Teaching source anchors absent or ambiguous: {relative}"
        first, last = starts[0], ends[0]
        assert last - first == old_last - old_first, f"Teaching source span changed; review explanations: {relative}"
        offset = first - old_first
    if offset:
        explanations = [f'目录迁移使片段位置后移{offset}行；下列解释保留历史行号，当前定位加{offset}行。', *explanations]
    assert 1 <= first <= last <= len(lines)
    text = "\n".join(lines[first - 1:last])
    focus = {
        ("vlm_deep/model.py", 19): (38, 52),
        ("vlm_deep/model.py", 5): (9, 25),
        ("vlm_deep/common.py", 64): (95, 102),
        ("vlm_deep/train.py", 130): (130, 147),
        ("vlm_deep/train.py", 157): (174, 190),
        ("tools/analyze_vlm_dev_slices.py", 85): (103, 119),
    }
    display_first, display_last = focus.get((relative, old_first), (old_first, min(last - offset, old_first + 17)))
    display_first, display_last = display_first + offset, display_last + offset
    assert first <= display_first <= display_last <= last
    return {"path": str(path.resolve()), "start_line": first, "end_line": last,
            "sha256": sha(path), "snippet_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "text": text, "numbered_text": "\n".join(f"{i}: {lines[i - 1]}" for i in range(first, last + 1)),
            "display_start_line": display_first, "display_end_line": display_last,
            "display_text": "\n".join(lines[display_first - 1:display_last]),
            "display_numbered_text": "\n".join(f"{i}: {lines[i - 1]}" for i in range(display_first, display_last + 1)),
            "explanations": explanations, "kind": "existing_project_source"}


def tensor_evidence(run_root, output):
    import torch
    from safetensors import safe_open

    torch.set_num_threads(2)
    adapter = run_root / "runs/stage1/qv_r8/seed0/train/adapter/adapter_model.safetensors"
    config_path = adapter.parent / "adapter_config.json"
    cfg = read(config_path)
    scaling = cfg["lora_alpha"] / cfg["r"]
    with safe_open(adapter, framework="pt", device="cpu") as store:
        shapes = {key: store.get_slice(key).get_shape() for key in store.keys()}
        actual = sum(math.prod(shape) for shape in shapes.values())
        assert actual == 1843200
        projections = {}
        arrays = {}
        for projection in ("q_proj", "v_proj"):
            prefix = f"base_model.model.model.layers.0.self_attn.{projection}"
            a_name, b_name = f"{prefix}.lora_A.weight", f"{prefix}.lora_B.weight"
            a, b = store.get_tensor(a_name), store.get_tensor(b_name)
            arrays[projection] = (a.double(), b.double())
            projections[projection] = {"A_name": a_name, "B_name": b_name,
                "A_shape": list(a.shape), "B_shape": list(b.shape), "storage_dtype": str(a.dtype),
                "delta_shape": [b.shape[0], a.shape[1]], "parameters": a.numel() + b.numel(),
                "A_frobenius_norm": float(a.double().norm()), "B_frobenius_norm": float(b.double().norm())}
    a, b = arrays["q_proj"]
    # BA = Q_B (R_B R_A^T) Q_A^T, so the nonzero singular values only
    # require the 8x8 middle matrix; no large 2048x2048 SVD is performed.
    _, rb = torch.linalg.qr(b, mode="reduced")
    _, ra = torch.linalg.qr(a.T, mode="reduced")
    singular = torch.linalg.svdvals(scaling * rb @ ra.T)
    crop = scaling * b[:64, :] @ a[:, :64]
    record = {"schema_version": "rarevlm.learning_lora_tensors.v1", "scope": "real completed stage1 adapter; CPU-only inspection",
        "adapter": {"path": str(adapter), "sha256": sha(adapter)},
        "adapter_config": {"path": str(config_path), "sha256": sha(config_path)},
        "method": "qv_r8", "seed": 0, "layer": 0, "rank": cfg["r"],
        "alpha": cfg["lora_alpha"], "scaling_alpha_over_r": scaling,
        "adapter_tensor_count": len(shapes), "trainable_parameters_from_safetensors": actual,
        "count_derivation": "36 * (8*(2048+2048) + 8*(2048+256)) = 1843200",
        "projections": projections,
        "singular_values": singular.tolist(), "q_proj_nonzero_singular_values": singular.tolist(),
        "rank_upper_bound": 8,
        "singular_value_algorithm": "thin QR of B and A.T; SVD of (alpha/r)*R_B@R_A.T in float64 on CPU",
        "q_proj_delta_frobenius_norm": float(singular.norm()),
        "delta_crop": {"projection": "q_proj", "row_indices": [0, 63], "column_indices": [0, 63],
            "full_delta_shape": [2048, 2048], "shape": [64, 64], "values": crop.tolist(),
            "minimum": float(crop.min()), "maximum": float(crop.max()),
            "note": "Actual leading 64x64 submatrix of delta W=(16/8)BA; illustrative crop, not the full matrix or a rank measurement"},
        "limits": ["低秩约束属于权重增量 ΔW，不代表冻结底座 W0 的秩只有8。",
                   "奇异值衡量该层更新的方向与幅度，不能直接证明模型准确率或泛化能力。",
                   "v_proj输出256维来自grouped-query attention的KV投影，不是代码错误。"],
    }
    write(output / "LORA_TENSORS.json", record)
    return record


def token_evidence(run_root, model_root, output, qid):
    from PIL import Image
    from transformers import AutoProcessor
    from vlm_deep.common import encode, read_manifest

    config = read(REPO / "configs/vlm_deep_v2.json")
    manifest_path = run_root / "prepared/manifest.json"
    manifest = read_manifest(manifest_path)
    matches = [row for row in manifest["splits"]["dev"] if str(row["question_id"]) == str(qid)]
    assert len(matches) == 1
    row = matches[0]
    image = manifest_path.parent / manifest["images"][row["image_id"]]["path"]
    processor = AutoProcessor.from_pretrained(str(model_root), local_files_only=True,
        min_pixels=config["min_pixels"], max_pixels=config["max_pixels"], use_fast=False)
    processor.tokenizer.padding_side = "left"
    batch = encode(processor, row, image, supervised=True)
    prefix = encode(processor, row, image, supervised=False)
    ids = batch["input_ids"][0].tolist()
    labels = batch["labels"][0].tolist()
    grid = batch["image_grid_thw"].tolist()
    merge = processor.image_processor.merge_size
    visual_count = sum(math.prod(g) // (merge ** 2) for g in grid)
    image_token_id = read(model_root / "config.json")["image_token_id"]
    assert ids.count(image_token_id) == visual_count
    prefix_length = prefix["input_ids"].shape[1]
    supervised = [i for i, label in enumerate(labels) if label != -100]
    assert min(supervised) == prefix_length
    tokens = [{"position": i, "token_id": token,
        "token_text": processor.tokenizer.convert_ids_to_tokens(token),
        "decoded_token": processor.tokenizer.decode([token], skip_special_tokens=False),
        "label": label, "supervised": label != -100, "is_visual_token": token == image_token_id,
        "next_token_predicted_by_position": i - 1 if label != -100 else None}
        for i, (token, label) in enumerate(zip(ids, labels))]
    img = Image.open(image)
    processor_source = Path(inspect.getsourcefile(processor.image_processor.__class__))
    token_record = {"schema_version": "rarevlm.learning_token_mask.v1",
        "scope": "real internal-dev question, CPU AutoProcessor and existing common.encode; no model load or generation",
        "question_id": str(row["question_id"]), "image_id": row["image_id"],
        "question": row["question"], "target": row["target"], "human_answers": row["answers"],
        "image": {"path": str(image), "sha256": sha(image), "original_size_wh": list(img.size)},
        "manifest": {"path": str(manifest_path), "sha256": sha(manifest_path), "manifest_sha256": manifest["manifest_sha256"]},
        "source_sha256": {"vlm_deep/common.py": sha(REPO / "src" / "vlm_deep/common.py"),
                           "vlm_sft/common.py": sha(REPO / "src" / "vlm_sft/common.py")},
        "processor_source": {"path": str(processor_source), "sha256": sha(processor_source)},
        "min_pixels": config["min_pixels"], "max_pixels": config["max_pixels"],
        "input_length": len(ids), "prefix_length": prefix_length,
        "supervised_token_count": len(supervised), "supervised_positions": supervised,
        "image_grid_thw": grid, "visual_token_count": visual_count, "merge_size": merge,
        "pixel_values_shape": list(batch["pixel_values"].shape),
        "resized_image_hw_from_grid": [[g[1] * processor.image_processor.patch_size, g[2] * processor.image_processor.patch_size] for g in grid],
        "supervised_decoded": processor.tokenizer.decode([labels[i] for i in supervised], skip_special_tokens=False),
        "tokens": tokens,
        "limits": ["labels=-100只屏蔽loss；图像与问题仍作为条件参与attention。",
                   "assistant内容后的模板结束token也受监督，因此监督token数不等于答案的单词数。",
                   "此为训练编码检查，不是对该dev问答追加训练；标签不会进入正式生成输入。"],
    }
    write(output / "TOKEN_MASK.json", token_record)
    return token_record, processor


def dependency_resize_snippet(processor):
    import transformers
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

    source, first = inspect.getsourcelines(smart_resize)
    path = Path(inspect.getsourcefile(smart_resize))
    display_index = next(i for i, line in enumerate(source) if "h_bar = round(" in line)
    display_source = source[display_index:]
    return {"path": str(path), "start_line": first, "end_line": first + len(source) - 1,
        "sha256": sha(path), "text": "".join(source).rstrip(),
        "numbered_text": "\n".join(f"{first + i}: {line.rstrip()}" for i, line in enumerate(source)),
        "display_start_line": first + display_index, "display_end_line": first + len(source) - 1,
        "display_text": "".join(display_source).rstrip(),
        "display_numbered_text": "\n".join(f"{first + display_index + i}: {line.rstrip()}" for i, line in enumerate(display_source)),
        "kind": "installed_transformers_dependency_source", "dependency_version": transformers.__version__,
        "explanations": ["项目common.encode直接读取原图，具体尺寸约束由AutoProcessor的smart_resize实现；项目没有自定义bounded_resize函数。",
            "factor=patch_size*merge_size=14*2=28：高度与宽度按28对齐，之后每2×2 patch合为一个语言侧视觉token。",
            "高于max_pixels时用sqrt(HW/max_pixels)缩小，再向下对齐；低于min_pixels时相应放大并向上对齐。",
            "长宽比分布和round/floor/ceil会让实际token数低于上限，不能假设每张图恰好256 tokens。"]}


def collect_lessons(run_root, model_root, output, tensors, tokens, processor):
    train_path = run_root / "runs/stage1/qv_r8/seed0/train/TRAIN_RESULT.json"
    train = read(train_path)
    cfg = read(REPO / "configs/vlm_deep_v2.json")
    diag_path = run_root / "learning_dev_diagnostics/DEV_SLICES.json"
    diag = read(diag_path)
    slices = {}
    for bucket in ("unseen_0", "rare_1_5", "medium_6_20", "frequent_gt20"):
        rows = [r for r in diag["slice_metrics"] if r["method"] == "qv_r8" and r["dimension"] == "answer_frequency" and r["slice"] == bucket]
        assert sorted(r["seed"] for r in rows) == [0, 42, 200]
        slices[bucket] = {"questions": rows[0]["n"], "unique_images": rows[0]["unique_images"],
            "mean_score_percent": mean(r["score_percent"] for r in rows),
            "seed_scores_percent": {str(r["seed"]): r["score_percent"] for r in rows}}
    evaluations = []
    for method in ("qv_r8", "qkvo_r4", "qv_r4_merger"):
        for seed in (0, 42, 200):
            for folder in ("eval_dev", "eval_textvqa", "eval_docvqa", "eval_mmstar"):
                path = run_root / f"runs/stage1/{method}/seed{seed}/{folder}/EVALUATION.json"
                if path.exists():
                    item = read(path)
                    evaluations.append({"method": method, "seed": seed, "task": item["task"], "split": item["split"],
                        "n": item["n"], "score_percent": item["score_percent"],
                        "path": str(path), "sha256": sha(path), "quantization": item["quantization"]})
    parameters = []
    for method in cfg["methods"]:
        path = run_root / f"preflight/{method}.json"
        item = read(path)
        parameters.append({"method": method, "rank": cfg["methods"][method]["rank"],
            "alpha": cfg["methods"][method]["alpha"], "projections": cfg["methods"][method]["projections"],
            "merger_ranks": cfg["methods"][method].get("merger_ranks", {}),
            "parameter_groups": item["parameter_groups"], "trainable_parameters": item["trainable_parameters"],
            "evidence_path": str(path), "sha256": sha(path)})
    common_exercises = [
        {"type": "cpu_read", "command": f"source /opt/data/private/.tools/activate.sh\nPROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONDONTWRITEBYTECODE=1 /opt/data/private/.venvs/rarevlm-vlm/bin/python {REPO / 'tools/extract_vlm_learning_code.py'} --output-dir {output}",
         "purpose": "只在CPU重新提取冻结代码、真实adapter与样本mask；不会加载底座、训练或改协议。"}]
    lessons = [
        {"id": "lora", "title": "第1课：读一个真正训练后的LoRA更新", "status": "implemented_and_measured",
         "question": "只训练184万参数，模型怎样改变一层2048维的注意力投影？",
         "principle": {"intuition": "原权重W0冻结，输入走原线性层，也走A降维→B升维的增量支路，两路相加。r限制增量可用的独立方向，不限制底座自身的表达。",
             "equations": ["ΔW=(α/r)BA", "y=W0x+(α/r)B(Ax)", "P_LoRA=r(d_in+d_out)",
                 "Attention(Q,K,V)=softmax(QKᵀ/√d)V"],
             "shapes": ["本层x[...,2048]；q:A[8,2048]→Ax[...,8]→B(Ax)[...,2048]；v:A[8,2048]→Ax[...,8]→B(Ax)[...,256]。", 
                 "q参数8*(2048+2048)=32768；v参数8*(2048+256)=18432；每层51200×36层=1843200。",
                 "36层×(q/k/v/o的r4参数)同样=1843200；merger方案语言921600+merger913408=1835008。"],
             "walkthrough": ["q/k改变匹配分数，v改变被加权的内容，o将多头信息映回隐藏空间。并非把这些模块分别理解成独立视觉或语言技能。",
                 "PEFT处理前向的A/B支路；项目明确匹配语言层名字，避免误把视觉trunk同名投影纳入训练。",
                 "训练的lora_dropout=0.05作用于增量支路输入；评测时关闭dropout，ΔW可以按BA直接分析。",
                 "merger连接视觉patch特征与语言隐藏维度。只给merger加LoRA不等于解冻视觉编码器。"]},
         "code": [snippet("vlm_deep/model.py", 19, 53, ["19行冻结所有底座参数。", "30–35行按完整模块名选36层语言目标，再追加指定merger线性层。",
             "38–42行设置merger专属rank/alpha，语言使用全局rank/alpha；α/r均为2。", "46–52行只允许LoRA可训练，禁止视觉trunk解冻，并核对可训练参数数量。"]),
             snippet("vlm_deep/model.py", 64, 75, ["整个模型设train后，把visual设eval；merger方案单独将merger设train，避免视觉dropout引入额外不公平因素。", "parameter_groups分别统计语言与merger预算。"]),
             snippet("configs/vlm_deep_v2.json", 10, 15, ["这是当前冻结的三个策略。r与位置在前两个条件中同时变化，所以不能分别归因于rank或位置。"])] ,
         "results": {"parameter_budgets": parameters, "actual_checkpoint": {"path": str(output / "LORA_TENSORS.json"), "sha256": sha(output / "LORA_TENSORS.json"),
             "q_proj": tensors["projections"]["q_proj"], "v_proj": tensors["projections"]["v_proj"],
             "singular_values": tensors["singular_values"], "adapter_parameters": tensors["trainable_parameters_from_safetensors"]},
             "engineering_evidence": evidence(train_path, ["completed_updates", "actual_questions_seen", "trainable_parameters", "frozen_parameters_unchanged", "mean_question_loss", "peak_gpu_allocated_gib"]),
             "interpretation": "实际保存的144个A/B张量精确合计1843200参数；QR后只对8×8核心做SVD，获得q_proj增量的8个非零奇异值。热图展示实际ΔW的64×64子块。"},
         "limits": ["奇异值与热图是学过什么更新的线索，不是准确率的原因证明。", "qv_r8与qkvo_r4匹配参数预算，但同时改变插入位置与每层rank。", "当前未训练全参数SFT，不能说LoRA已优于全量微调。"],
         "exercises": [{"type": "reasoning", "task": "用真实q/v shape手算每层参数，再回答为什么v_proj不是2048×2048。"},
             {"type": "cpu_read", "task": "打开LORA_TENSORS.json，核对alpha/r=2、rank<=8，并比较第一与第八奇异值。"}],
         "references": [PAPERS["lora"], PAPERS["qwen"]]},
        {"id": "qlora", "title": "第2课：NF4省在哪里，FP32又用在哪里", "status": "implemented_and_measured",
         "question": "为什么NF4语言权重与FP32计算可以同时存在？",
         "principle": {"intuition": "权重存储与计算dtype是两件事。冻结的语言线性层用NF4代码与缩放统计量存储；计算时由量化算子使用FP32。可训练A/B、视觉权重和lm_head保持FP32。",
             "equations": ["W≈dequantize(code_4bit, scales)", "y≈W_quantized_dequantized x+(α/r)BAx"],
             "shapes": ["LoRA仍是FP32 A/B；4-bit语言底座不产生其自身的可训练梯度。", "double quantization进一步量化量化统计量，不把视觉输入像素量化成NF4。"],
             "walkthrough": ["NF4选用针对近似正态权重分布的非均匀量化等级；它是近似，不能保证生成结果与原始权重相同。",
                 "被冻结的W0仍参与前向与对LoRA输入的反向链式传播；只是不优化W0。",
                 "总显存还包括视觉FP32、激活、梯度、AdamW状态、padding与CUDA allocator缓存，因此不能用参数数×0.5字节预测全部显存。",
                 "项目用torch.optim.AdamW，并未实现论文中的paged optimizer，也没有把BF16当作计算默认。"]},
         "code": [snippet("vlm_deep/model.py", 5, 26, ["11–13行语言权重4bit NF4、double quant、compute FP32；skip visual与lm_head。",
             "15–18行其余权重加载为FP32并用sdpa。", "20–23行检查vision没有Linear4bit、language确实存在Linear4bit。", "24–25行处理器图像预算独立于语言权重量化。"]),
             snippet("vlm_deep/model.py", 46, 60, ["50行强制可训练LoRA参数FP32。", "59–60行检查所有视觉普通参数FP32。"]),
             snippet("vlm_deep/train.py", 110, 111, ["使用普通AdamW；不得把本工程称为逐项复现QLoRA论文训练配方。"])] ,
         "results": {"training": evidence(train_path, ["gpu_name", "peak_gpu_allocated_gib", "peak_gpu_reserved_gib", "active_seconds", "training_input_tokens", "input_tokens_per_second"]),
             "original_weight_preflight": evidence(run_root / "preflight/original_fp32.json"),
             "interpretation": "真实qv_r8 seed0在V100 16GiB四卡训练完成，单卡峰allocated5.848GiB，reserved10.254GiB。原始FP32预检峰14.199GiB属于另一个检查过程，不能用两者直接声称训练显存节省比例或速度提升。"},
         "limits": ["NF4与原始权重的准确率必须在同一协议下分别评测，不能默认数值近似无影响。", "这是QLoRA的NF4+double quant+LoRA机制选用，非论文默认配方完整复现。", "峰reserved包含缓存分配，不等于张量当前占用；二者都要报告。"],
         "exercises": [{"type": "reasoning", "task": "找出代码里三个独立dtype决策：量化存储、量化层compute、视觉/LoRA权重。解释为什么不能全部写成‘4bit训练’。"}],
         "references": [PAPERS["qlora"]]},
        {"id": "sft", "title": "第3课：同一张图如何变成assistant-only SFT", "status": "implemented_and_measured",
         "question": "模型看整段输入，为什么loss只算答案？",
         "principle": {"intuition": "因果语言模型根据图像、问题和已给出的答案前缀预测下一个token。训练提供真实答案作为teacher forcing。labels=-100屏蔽问题和图像位置的目标，仍保留它们作为条件输入。",
             "equations": ["L_i=-(1/T_i) Σ_(t∈assistant_i) log pθ(y_t|image, question, y_<t)", "L_batch=(1/N) Σ_i L_i"],
             "shapes": ["单QA input_ids[1,L]→拼批[B,Lmax]；labels同形，左侧padding和用户前缀均=-100。", "tail head保留max(T_i)+1个隐状态，logits[B,K,V]；[:, :-1]预测labels[:,1:]。", 
                 f"真实dev例qid{tokens['question_id']}：input_length={tokens['input_length']}，prefix={tokens['prefix_length']}，assistant监督{tokens['supervised_token_count']}个token。"],
             "walkthrough": ["先编码generation prefix，再编码带assistant的full对话；assert检查prefix恰好是full的开头，避免chat模板错位。",
                 "counts计算每个QA有效答案token数，CE reduction='none'保留每token loss，先在每QA内平均，再按有效QA平均。",
                 "长答案不会仅因token更多而获得更大的QA权重；这与把全batch所有答案token一起平均不同。",
                 "dev_token_nll属于token加权报告，训练mean_question_loss属于QA平均；两者语义不同，不能混用。",
                 "答案结束模板token也参与监督。正式generation使用encode(...,False)，不把target传入模型。"]},
         "code": [snippet("vlm_deep/common.py", 48, 61, ["53–54行生成只包含图像问题的prefix。", "57–60行追加真实assistant target，并用prefix计算labels。"]),
             snippet("vlm_sft/common.py", 94, 100, ["98行检查模板prefix一致。", "99行只把prefix对应labels设=-100。input_ids完全保留。"]),
             snippet("vlm_deep/common.py", 64, 102, ["69–73行左padding，labels填充=-100，使答案落在共同尾部。", "85–94行临时切lm_head输入尾部，减少不参与loss的词表logits，finally移除hook。",
                 "96–102行因果错一位、忽略-100、每QA归一化；断言确保没有丢失监督token。"])] ,
         "results": {"real_case": {"path": str(output / "TOKEN_MASK.json"), "sha256": sha(output / "TOKEN_MASK.json"),
             "question_id": tokens["question_id"], "question": tokens["question"], "target": tokens["target"],
             "input_length": tokens["input_length"], "prefix_length": tokens["prefix_length"], "supervised_token_count": tokens["supervised_token_count"]},
             "tail_head_equivalence": evidence(run_root / "preflight/qv_r8.json", ["passed", "tail_head_loss_max_absolute_error", "tail_head_gradient_max_absolute_error", "stress_sequence_tokens"]),
             "training": evidence(train_path, ["mean_question_loss", "internal_dev_token_nll", "training_answer_tokens", "dev_answer_tokens"]),
             "interpretation": "真实答案mask数据与图像、question ID和manifest绑定；head切片预检loss最大绝对误差约2.18e-6，梯度最大误差约8.35e-7。"},
         "limits": ["SFT NLL更低不自动意味着自由生成TextVQA得分更高，正式得分需要generate+共识评价。", "监督target是十人答案中的确定性多数原字符串，不是每个有效人类答案都被训练过。", "标签图仅展示编码和目标选择；没有对该dev样本执行优化。"],
         "exercises": [{"type": "cpu_read", "task": "打开TOKEN_MASK.json，找第一处label!=-100，并检查它由前一个位置的logits预测；比较target单词数与监督token数。"},
             {"type": "reasoning", "task": "解释问题的labels被屏蔽后，LoRA为什么仍然能学到如何利用问题和图像。"}],
         "references": [PAPERS["qwen"]]},
        {"id": "ddp", "title": "第4课：四卡训练与可恢复实验契约", "status": "implemented_and_measured",
         "question": "最后一批只有一个真实问题，四卡怎样保持准确的全局平均梯度？",
         "principle": {"intuition": "四卡各处理microbatch=4，梯度由DDP平均。用valid标记填充的sentinel样本并让其贡献为0；本地loss乘world/actual，DDP平均后恰好得到全局有效QA均值。",
             "equations": ["B_effective=world×micro×acc=4×4×1=16", "L_rank=(world/N_actual) Σ_(i in rank) valid_i L_i", "DDP_mean=(1/world)Σ_rank∇L_rank=∇[(1/N_actual)Σ_real_i L_i]", "updates=ceil(N_QA/16)"],
             "shapes": ["10000 QA恰好分625次更新，没有末批填充。", "真实33 QA契约验证补到48计算槽：3 updates，有效QA数为16、16、1。", "runner要求acc=1；当前配置的effective_batch=16不包含尚未实现的梯度累积。"],
             "walkthrough": ["schedule固定种子生成全局question permutation，随后补-1槽位；每张卡读取该步自己的连续4个槽。",
                 "-1槽位借用本步一个真实输入以运行计算，但valid=False使loss和统计量贡献为0；不丢弃末批真样本。",
                 "每50步以及末步保存adapter、optimizer、每rank RNG与样本统计；LATEST使用文件hash指向完整checkpoint。",
                 "恢复检查配置、manifest、全局样本顺序和adapter/state哈希，恢复优化器与RNG，接着从完成的step开始。",
                 "真正恢复等价还包括数据、更新顺序和RNG，不是仅能重新load一个adapter。当前契约验证restored dev NLL一致，未证明所有未来中断的逐step比特一致。"]},
         "code": [snippet("vlm_deep/train.py", 43, 49, ["统一生成全局顺序，以-1补齐effective batch倍数。"]),
             snippet("vlm_deep/train.py", 102, 105, ["用world*micro*acc算effective batch；明确拒绝acc!=1。"]),
             snippet("vlm_deep/train.py", 130, 155, ["校验resume adapter/state和protocol hash。", "145–147行将sentinel替换为可编码的fallback，但保留valid=False；数据按rank切分。", "155行恢复每rank RNG。"]),
             snippet("vlm_deep/train.py", 157, 195, ["保存所有rank RNG/statistics、adapter、optimizer与已完成步数。", "179行actual是本步真实QA数量。", "184行乘world/actual抵消DDP梯度平均，只对真实QA归一化。", "188–190行valid也控制曝光统计。"])] ,
         "results": {"full_stage1": evidence(train_path, ["world_size", "microbatch", "accumulation", "effective_batch", "question_count", "completed_updates", "actual_questions_seen", "skipped_updates", "resumed_at_update"]),
             "real_partial_batch_contract": evidence(run_root / "ddp_contract/TRAIN_RESULT.json", ["question_count", "completed_updates", "actual_questions_seen", "padding_policy"]),
             "checkpoint_restore_contract": evidence(run_root / "CHECKPOINT_RESTORE_CONTRACT.json"),
             "interpretation": "真实四卡33 QA小批契约完成3次更新，33个有效QA各见一次；adapter、optimizer、CPU/CUDA RNG均成功恢复，dev NLL=0.18981747088186882与原结果相同。它是工程契约证据，不是算法benchmark成绩。"},
         "limits": ["不能用四卡数量直接推断四倍提速；I/O、加载、数据准备、通信均有开销。", "恢复契约属于具体已验证条件，不能泛化为任何系统故障都无需重跑。", "不要把sentinel计算次数当作真实训练样本曝光。"],
         "exercises": [{"type": "reasoning", "task": "代入末批actual=1，说明唯一真样本所在rank的loss为何乘4，DDP平均后为何等于单样本梯度。"},
             {"type": "cpu_read", "task": "读CHECKPOINT_RESTORE_CONTRACT.json，区分已验证的恢复字段与尚未证明的逐step重启等价。"}],
         "references": []},
        {"id": "vision_budget", "title": "第5课：把图像预算落实到一张真实图片", "status": "current_processor_measured_future_ablation_planned",
         "question": "224×224、448×448与最多256个视觉token是什么关系？",
         "principle": {"intuition": "模型不会直接把整个高分辨率原图交给语言层。processor按预算缩放并按14像素patch编码，merger将相邻2×2 patch组合，再映射到2048维语言隐藏空间。总像素预算约等于28²×视觉token数。",
             "equations": ["N_visual=(grid_t×grid_h×grid_w)/(merge_size²)", "N_visual≈(H'×W')/(14²×2²)=(H'×W')/784", "50176=64×28²; 200704=256×28²"],
             "shapes": [f"真实case原图{tokens['image']['original_size_wh']}，processor grid={tokens['image_grid_thw']}，语言视觉token={tokens['visual_token_count']}。",
                 f"实际pixel_values shape={tokens['pixel_values_shape']}；静态图复制temporal_patch_size=2次组成一个时间patch，不代表输入是真视频。",
                 "图像占据连续image_pad token槽；merger输出替换这些槽的词嵌入，随后与问题token共同进入语言模型。"],
             "walkthrough": ["common.encode直接PIL读取原图，project没有单独bounded_resize；真正预算函数是安装依赖smart_resize。",
                 "保持长宽比并按28对齐，所以相同max_pixels不代表每图固定256 token，也不代表每图448×448。",
                 "更高预算可能保住小字细节，也增加prefill长度、激活、显存和延迟；是否值得必须用真实生成结果验证。",
                 "比较全图与全图+区域时要匹配总视觉token预算，并设置随机区域控制，才能区分聚焦与新增像素带来的收益。"]},
         "code": [snippet("vlm_deep/model.py", 24, 26, ["训练和评测通过相同min/max_pixels实例化processor。"]),
             snippet("vlm_deep/common.py", 48, 59, ["原图交给processor，resize/patchify来自真实依赖。"]),
             dependency_resize_snippet(processor)],
         "results": {"measured_current_case": {"path": str(output / "TOKEN_MASK.json"), "sha256": sha(output / "TOKEN_MASK.json"),
             "original_wh": tokens["image"]["original_size_wh"], "resized_hw": tokens["resized_image_hw_from_grid"],
             "grid_thw": tokens["image_grid_thw"], "visual_token_count": tokens["visual_token_count"]},
             "planned_controls": ["固定同一个完整SFT checkpoint，内部dev比较max视觉token预算256/512/1024，记录真实token与得分/延迟/显存。",
                 "在约相同总token下比较全图、全图+预测OCR区域、全图+随机区域；OCR不使用参考答案定位。"],
             "interpretation": "当前只确认processor预算与真实样本token；预算干预尚无新训练或生成结果，静态图片放大示意不等价于模型已经获益。"},
         "limits": ["局部裁图结果未执行，不能把读得更清楚的示意图标成算法提升。", "提高分辨率不能恢复原图已缺失的像素信息。", "256/512/1024是max预算，不是每图token常数。"],
         "exercises": [{"type": "reasoning", "task": "用真实image_grid_thw算视觉token数，再与TOKEN_MASK里的image_pad数量核对。"},
             {"type": "research_design", "task": "写出全图+OCR区域的随机区域控制，并列出要匹配的总token预算与checkpoint。"}],
         "references": [PAPERS["qwen"]]},
        {"id": "data_selection", "title": "第6课：从答案频次切片走向可检验的数据算法", "status": "diagnostics_measured_sampling_ablation_planned",
         "question": "未见答案切片较差，能否通过多采罕见答案改善？",
         "principle": {"intuition": "先把真实训练目标标准化后统计QA出现频次，再按dev目标的频次分组。它描述输出字符串覆盖，不等于视觉类别的长尾，也没有证明频次本身导致错误。下一步通过受控采样干预检验。",
             "equations": ["f(a)=Σ_(i∈train_10k) 1[Normalizer(target_i)=a]", "score_slice=(100/|S|)Σ_(i∈S)Consensus(pred_i, answers_i)",
                 "planned p_i∝(f(target_i)+ε)^(-τ), τ≥0（需冻结候选池与采样种子）"],
             "shapes": ["实际target为十个人类原答案多数值；并列时按字符串排序确定。", "frequency bins：0、1–5、6–20、>20；单位是QA次数。",
                 "dev有1617 QA/1024图；同图问题相关，因此差异不应按独立QA重采样来估计置信区间。"],
             "walkthrough": ["当前stage1在固定SHA图像顺序中取前10000 QA；不是已执行频次均衡算法，也不是严格的均匀QA抽样。",
                 "diagnostic从已完成dev predictions读取saved consensus score，不新生成、不从final改变选型。",
                 "同seed匹配相同question/image ID，bootstrap按image整簇重采样，保留该图所有切片QA。",
                 "稀有答案可能是人名、长字符串、难图或标注噪声。若按频次采样改进，仍应检查是否靠重复图像与常见能力下降换来。",
                 "最低限度比较随机10k、质量+多样性10k、频次覆盖10k；固定更新数并报告独立图像与真实token曝光。"]},
         "code": [snippet("vlm_deep/common.py", 20, 24, ["训练target基于原始字符串多数；不是用预测或dev成绩挑选目标。"]),
             snippet("vlm_deep/prepare.py", 96, 108, ["按固定SHA选image-disjoint dev，保留train_all，然后取前10000 QA。"]),
             snippet("tools/analyze_vlm_dev_slices.py", 49, 66, ["实际dev raw target检查后normalize，按train_10k出现数、字符长度和数字形式分组。"]),
             snippet("tools/analyze_vlm_dev_slices.py", 129, 140, ["频次只从真实train_10k target计算；dev映射只用于诊断。"]),
             snippet("tools/analyze_vlm_dev_slices.py", 85, 119, ["相同QA成对比较；按图像保留整簇问题进行2000次bootstrap；不包含训练seed或多重比较不确定性。"])] ,
         "results": {"slice_scores_qv_r8_three_seeds": slices,
             "diagnostic_evidence": {"path": str(diag_path), "sha256": sha(diag_path), "manifest": diag["manifest"], "definitions": diag["definitions"]},
             "interpretation": f"当前qv_r8三seed未见答案均分{slices['unseen_0']['mean_score_percent']:.2f}%，常见答案均分{slices['frequent_gt20']['mean_score_percent']:.2f}%；未见目标{slices['unseen_0']['questions']}/1617个QA。这是关联观察，频次均衡/质量选择尚无受控训练结果。"},
         "limits": ["训练目标未见≠预训练未见；公开底座预训练重叠未知。", "切片按代表target字符串划分，不是所有十人答案并集，也不是实体/视觉类。", "bootstrap区间条件于已训练seed、未校正多重切片，因此保持探索性。"],
         "exercises": [{"type": "reasoning", "task": "列出三个频次与错误关联的混杂因素，并说明怎样用采样对照与图像/token曝光记录加以检验。"}],
         "references": []},
        {"id": "multitask_dpo", "title": "第7课：多任务与DPO该怎样接上现有SFT", "status": "planned_not_implemented_or_trained",
         "question": "怎样把通用能力保持和偏好优化做成可信研究，而不是只新增名词？",
         "principle": {"intuition": "多任务改变监督训练分布；DPO利用同一输入的优选/劣选回答，改变两者相对概率。两者都应从同一个完整SFT checkpoint出发，并配套最低限度对照。当前项目没有多任务混合训练或DPO训练结果。",
             "equations": ["planned L_mix=Σ_k λ_k E_(x,y∼D_k)[L_SFT(x,y)]",
                 "planned L_DPO=-E log σ[β(log πθ(y+|x)-log πref(y+|x)-log πθ(y-|x)+log πref(y-|x))]"],
             "shapes": ["VLM x=(image, question)，y+与y-是同一x对应的完整回答序列。", "DPO用序列log概率和；它与现有SFT每QA内mean token CE的归一化方式不同，不可直接复制loss。"],
             "walkthrough": ["先检查TextVQA提升是否伴随DocVQA/MMStar变化；分任务报告自己的指标，不把异质分数混成一个‘准确率’。",
                 "多任务对照至少包含纯TextVQA、通用图文混合的总预算匹配、TextVQA曝光匹配；公开通用数据必须避免评测泄漏。",
                 "DPO最低对照：相同起点不追加训练、chosen-only继续SFT、DPO；控制训练数据、预算、答案长度偏好与reference。",
                 "十人有效答案不应随意划为负例。先做偏好正确性审核，排除仅字符串不同但同样有效的回答。",
                 "只有数据与SFT可靠后才加DPO；当前代码没有DPO loader/loss/training实现，不能展示为已实现能力。"]},
         "code": [],
         "results": {"existing_single_task_eval_snapshot": evaluations,
             "planned_minimal_controls": {"multitask": ["pure TextVQA SFT", "mixed SFT matched total QA/token budget", "mixed SFT matched TextVQA exposure"],
                 "dpo": ["fixed SFT checkpoint, no further training", "chosen-only continued SFT", "DPO from the same SFT checkpoint"]},
             "implementation_status": "No multitask mixed-training implementation/results and no DPO implementation/results in the current frozen VLM runner.",
             "interpretation": "现有TextVQA、DocVQA、MMStar生成评测是任务适配的诊断证据；不能当作已验证的多任务训练或DPO收益。"},
         "limits": ["MMStar得分波动尚不能直接证明灾难性遗忘，需相同底座/量化/评测协议与多seed对照。", "DPO公式不是现有项目代码。", "加入训练目标前先建立偏好数据与对照，不能保证DPO必然改善客观VQA。"],
         "exercises": [{"type": "research_design", "task": "为同一真实图像问题写一个有效chosen/rejected候选，再解释如何确认rejected不是另一个可接受答案。"}],
         "references": [PAPERS["dpo"]]},
    ]
    return {"schema_version": "rarevlm.learning_code_lessons.v1", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "audience": "会Python/PyTorch、训练过模型，希望理解基础算法与真实工程的学习者",
        "scope": "read-only CPU learning extraction; existing results and future algorithms explicitly separated",
        "repo": str(REPO), "run_root": str(run_root), "model_root": str(model_root),
        "common_cpu_reproduction": common_exercises, "lessons": lessons,
        "artifacts": {name: {"path": str(output / name), "sha256": sha(output / name)} for name in ("LORA_TENSORS.json", "TOKEN_MASK.json")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--question-id", default="27482")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    watched = sorted((REPO / "src" / "vlm_deep").rglob("*.py")) + [REPO / "configs/vlm_deep_v2.json"]
    before = {str(path): sha(path) for path in watched}
    tensors = tensor_evidence(args.run_root, args.output_dir)
    tokens, processor = token_evidence(args.run_root, args.model_root, args.output_dir, args.question_id)
    lessons = collect_lessons(args.run_root, args.model_root, args.output_dir, tensors, tokens, processor)
    after = {str(path): sha(path) for path in watched}
    assert before == after, "frozen training/evaluation sources changed"
    lessons["verification"] = {"frozen_sources_unchanged": True, "frozen_source_hashes": before,
        "cpu_only": True, "adapter_parameter_count_passed": True,
        "visual_token_count_matches_image_pad": True, "mask_prefix_contract_passed": True,
        "source_snippets_exact": True}
    write(args.output_dir / "LESSONS.json", lessons)
    print(json.dumps({"lessons": len(lessons["lessons"]), "adapter_parameters": tensors["trainable_parameters_from_safetensors"],
        "question_id": tokens["question_id"], "input_tokens": tokens["input_length"],
        "visual_tokens": tokens["visual_token_count"], "supervised_tokens": tokens["supervised_token_count"],
        "frozen_sources_unchanged": True, "output": str(args.output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
