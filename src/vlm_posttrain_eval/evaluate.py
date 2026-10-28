"""Fixed original-BF16/1024 internal-dev evaluation; externally admitted GPU only.

The public preference training prompt may differ. Evaluation always retains the
original short-answer suffix, ten-reference post-hoc scoring and greedy32.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT
from vlm_deep.common import PROMPT_SUFFIX, collate, encode, to_device
from vlm_deep.evaluate import aggregate, score_prediction
from vlm_formal.evaluate import complete_cache as frozen_complete_cache
from vlm_formal.evaluate import question_prompt_sha, stress_rows, summarize_task
from vlm_formal.evaluate import source_hashes as frozen_source_hashes
from vlm_joint.evaluate import model_metadata_binding, visible_gpu_binding
from vlm_precision.evaluate import PATH_SPECS, load_path
from vlm_probe.evaluate import (content_sha, describe_input, lock_file, now,
    package_versions, private_path, processor_for, read_json, read_manifest,
    repo_path, sha_file, verify_dev_images, write_json)

CONDITIONS = ("initial", "sft", "dpo")
METRIC = "TextVQA leave-one-out consensus percent"
BUDGET = 1024
FIXED = {
    "numerical_path": "original_bf16", "precision": "bf16",
    "vision_precision": "bf16", "quantization": "original",
    "visual_token_budget": 1024, "min_pixels": 50176, "max_pixels": 802816,
    "batch_size": 4, "max_new_tokens": 32, "do_sample": False,
    "attention_implementation": "sdpa", "allow_tf32": False,
    "input_truncation": False, "split": "dev",
    "expected_questions": 1617, "expected_images": 1024,
    "prompt_suffix": PROMPT_SUFFIX,
}


def model_key(condition, seed):
    return f"{condition}_seed{seed}"


def source_hashes():
    result = frozen_source_hashes()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        result[str(path.relative_to(SOURCE_ROOT))] = sha_file(path)
    return result


def config_path(config, name, default=None):
    value = config.get(name, default)
    if value is None:
        raise ValueError(f"configuration requires {name}")
    return private_path(repo_path(value))


def initial_adapter_path(config):
    value = config.get("initial_adapter", config.get("initial_adapter_dir"))
    if value is None:
        raise ValueError("configuration requires initial_adapter")
    path = private_path(repo_path(value))
    if config.get("initial_adapter_dir") is not None:
        if path != private_path(repo_path(config["initial_adapter_dir"])):
            raise ValueError("initial_adapter aliases disagree")
    return path


def validate_config(config, reference, batch_size=4):
    evaluation = config.get("evaluation", {})
    for key, value in FIXED.items():
        if evaluation.get(key, value) != value:
            raise ValueError(f"fixed posttrain evaluation differs: {key}")
    for key, value in (("questions", 1617), ("images", 1024)):
        if evaluation.get(key, value) != value:
            raise ValueError(f"fixed posttrain evaluation differs: {key}")
    if batch_size != 4:
        raise ValueError("fixed evaluation requires batch4")
    if (reference["precision"] != "fp32"
        or reference["training_quantization"] != "nf4_language_only"
        or reference["min_pixels"] != 50176 or reference["max_pixels"] != 256 * 784
        or reference["max_new_tokens"] != 32 or reference["do_sample"]):
        raise ValueError("frozen original reference contract differs")
    return {**FIXED, **evaluation}


def adapter_files(path):
    path = private_path(path)
    files = {p.name: sha_file(p) for p in sorted(path.iterdir()) if p.is_file()}
    if "adapter_config.json" not in files or "adapter_model.safetensors" not in files:
        raise ValueError("complete safetensors PEFT adapter required")
    config = read_json(path / "adapter_config.json")
    if (config.get("peft_type") != "LORA" or config.get("r") != 8
        or config.get("lora_alpha") != 16 or config.get("bias") != "none"
        or set(config.get("target_modules", [])) != {"q_proj", "v_proj"}
        or config.get("modules_to_save") or config.get("use_dora") or config.get("use_rslora")):
        raise ValueError("fixed language q/v rank8 alpha16 LoRA required")
    return files


def adapter_binding(config, condition, seed, adapter_dir, model_root):
    """Own binding: never relax a frozen historical runner's adapter identity."""
    adapter_dir = private_path(adapter_dir)
    initial = initial_adapter_path(config)
    initial_files = adapter_files(initial)
    initial_train_path = initial.parent / "TRAIN_RESULT.json"
    initial_train = read_json(initial_train_path)
    if initial_train.get("adapter_files_sha256") != initial_files:
        raise ValueError("initial adapter differs from its completed training")
    if (initial_train.get("method") != "qv_r8"
        or initial_train.get("split") != "train_all"
        or initial_train.get("completed_updates") != 2062
        or initial_train.get("planned_updates") != 2062
        or initial_train.get("actual_questions_seen") != 32985
        or initial_train.get("frozen_parameters_unchanged") is not True):
        raise ValueError("initial must be the fixed complete full-data qv_r8 SFT")
    provenance_sha = sha_file(model_root.parent / "model_provenance.json")
    if initial_train.get("model_provenance_sha256") != provenance_sha:
        raise ValueError("initial and evaluation base-model provenance differ")
    expected_initial_sha = config.get("initial_train_result_sha256")
    if expected_initial_sha and expected_initial_sha != sha_file(initial_train_path):
        raise ValueError("predeclared initial training result changed")
    files = adapter_files(adapter_dir)
    train_path = adapter_dir.parent / "TRAIN_RESULT.json"
    train = read_json(train_path)
    if train.get("adapter_files_sha256") != files:
        raise ValueError("adapter bytes differ from completed training")
    if train.get("model_provenance_sha256") != provenance_sha:
        raise ValueError("adapter and evaluation base-model provenance differ")
    if condition == "initial":
        if adapter_dir != initial or files != initial_files:
            raise ValueError("initial condition must use predeclared initializer")
    elif condition in ("sft", "dpo"):
        expected = {"condition": condition, "seed": seed, "status": "completed",
            "frozen_parameters_unchanged": True, "config_sha256": content_sha(config),
            "initial_adapter_files_sha256": initial_files, "trainable_parameters": 1843200,
            "numerical_path": "original_bf16", "visual_token_budget": 1024}
        for key, value in expected.items():
            if train.get(key) != value:
                raise ValueError(f"completed posttrain binding differs: {key}")
        if (not isinstance(train.get("planned_updates"), int) or train["planned_updates"] <= 0
            or train.get("completed_updates") != train["planned_updates"]
            or train.get("actual_questions_seen", 0) <= 0):
            raise ValueError("posttraining did not complete the fixed schedule")
        if adapter_dir == initial:
            raise ValueError("new posttraining condition cannot reuse initial directory")
    else:
        raise ValueError("unknown posttraining condition")
    return files, {
        "path": str(train_path), "sha256": sha_file(train_path),
        "condition": condition, "evaluation_seed": seed, "training_seed": train.get("seed"),
        "initial_adapter_dir": str(initial), "initial_adapter_files_sha256": initial_files,
        "initial_train_result_path": str(initial_train_path),
        "initial_train_result_sha256": sha_file(initial_train_path),
        "completed_updates": train.get("completed_updates"),
        "actual_questions_seen": train.get("actual_questions_seen"),
        "training_prompt_suffix": PROMPT_SUFFIX if condition == "initial" else config.get("prompt_suffix", ""),
        "training_numerical_path": train.get("numerical_path", "nf4_language_only_fp32" if condition == "initial" else None),
        "training_visual_token_budget": train.get("visual_token_budget", 256 if condition == "initial" else config.get("visual_token_budget")),
        "training_lora_parameter_precision": train.get("lora_parameter_precision", "fp32"),
        "eval_adapter_conversion": "all floating weights including LoRA cast to BF16 by frozen load_path",
        "scope": "fixed final checkpoint; no dev/official benchmark checkpoint selection",
    }


def dev_data(asset_root):
    path = private_path(asset_root) / "prepared/manifest.json"
    manifest = read_manifest(path)
    rows = manifest["splits"]["dev"]
    if len(rows) != 1617 or len({r["image_id"] for r in rows}) != 1024:
        raise ValueError("full frozen internal dev requires 1617QA/1024images")
    if any(len(r["answers"]) != 10 for r in rows):
        raise ValueError("ten post-hoc references required")
    if any(r.get("prompt_suffix", PROMPT_SUFFIX) != PROMPT_SUFFIX for r in rows):
        raise ValueError("dev prompt suffix differs from original short-answer rule")
    return rows, manifest["images"], path.parent, {
        "task": "textvqa", "split": "dev", "n": len(rows), "metric": METRIC,
        "source_split": "frozen internal training-side dev",
        "manifest_sha256": manifest["manifest_sha256"], "manifest_file_sha256": sha_file(path),
        "manifest_path": str(path), "expected_questions": 1617, "expected_images": 1024,
        "question_order_sha256": content_sha([r["question_id"] for r in rows]),
        "question_prompt_order_sha256": content_sha([question_prompt_sha(r) for r in rows]),
        "pretraining_overlap": "unknown; public preference sources may overlap historical pretraining",
    }


def inference_row(row):
    return {**{k: row[k] for k in ("question_id", "image_id", "question")},
        "prompt_suffix": PROMPT_SUFFIX}


def tensor_binding(tensor):
    import torch
    cpu = tensor.detach().cpu().contiguous()
    return {"sha256": hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest(),
        "shape": list(cpu.shape), "dtype": str(cpu.dtype).removeprefix("torch.")}


def input_details(sample, row, images, processor, image_token_id=151655):
    details = describe_input(sample, row, images, processor, image_token_id, BUDGET)
    ids = sample["input_ids"][0].tolist()
    mask = sample["attention_mask"][0].tolist()
    details.update(input_ids=ids, attention_mask=mask,
        input_ids_sha256=content_sha(ids), attention_mask_sha256=content_sha(mask),
        pixel_values=tensor_binding(sample["pixel_values"]),
        image_grid_thw_tensor=tensor_binding(sample["image_grid_thw"]))
    return details


def evaluate_batch(model, processor, chunk, images, image_root, device, check_logits=False):
    """Frozen encode/collate/generate semantics plus exact CPU input/output audit."""
    import torch
    begun = time.monotonic()
    samples = [encode(processor, inference_row(r), image_root / images[r["image_id"]]["path"], False)
        for r in chunk]
    if any("labels" in sample for sample in samples):
        raise ValueError("generation input unexpectedly contains answer labels")
    details = [input_details(s, r, images, processor) for s, r in zip(samples, chunk)]
    cpu_batch = collate(samples, processor.tokenizer.pad_token_id)
    preprocessing = time.monotonic() - begun
    torch.cuda.synchronize(device)
    begun = time.monotonic()
    batch = to_device(cpu_batch, device)
    torch.cuda.synchronize(device)
    transfer = time.monotonic() - begun
    torch.cuda.reset_peak_memory_stats(device)
    finite = None
    if check_logits:
        output = model(**batch, use_cache=False)
        finite = bool(torch.isfinite(output.logits).all())
        del output
        if not finite:
            raise ValueError("non-finite preflight logits")
    torch.cuda.synchronize(device)
    begun = time.monotonic()
    generated = model.generate(**batch, max_new_tokens=32, do_sample=False, use_cache=True)
    torch.cuda.synchronize(device)
    elapsed = time.monotonic() - begun
    padded = int(batch["input_ids"].shape[1])
    answers = []
    for i in range(len(chunk)):
        tail = generated[i, padded:]
        prediction = processor.tokenizer.decode(tail, skip_special_tokens=True,
            clean_up_tokenization_spaces=False).strip()
        generated_ids = tail.tolist()
        answers.append({"prediction": prediction, "generated_ids": generated_ids,
            "pad_token_id": int(processor.tokenizer.pad_token_id),
            "generated_ids_sha256": content_sha(generated_ids),
            "generated_nonpad_tokens": int((tail != processor.tokenizer.pad_token_id).sum()),
            "generated_tail_slots": int(tail.numel()),
            "padded_input_ids_sha256": content_sha(cpu_batch["input_ids"][i].tolist()),
            "padded_attention_mask_sha256": content_sha(cpu_batch["attention_mask"][i].tolist()),
            **details[i]})
    resource = {"batch_size": len(chunk), "padded_input_tokens": padded,
        "batch_generation_seconds": elapsed, "batch_preprocessing_seconds": preprocessing,
        "batch_h2d_seconds": transfer,
        "batch_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "batch_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        "finite_logits": finite}
    return answers, resource


def validate_input_record(item):
    for value_key, sha_key in (("input_ids", "input_ids_sha256"),
        ("attention_mask", "attention_mask_sha256"), ("generated_ids", "generated_ids_sha256")):
        value = item[value_key]
        if (not isinstance(value, list) or any(type(v) is not int for v in value)
            or content_sha(value) != item[sha_key]):
            raise ValueError(f"actual token audit differs: {value_key}")
    if (len(item["input_ids"]) != item["input_tokens"]
        or item["attention_mask"] != [1] * item["input_tokens"]
        or item["input_ids"].count(151655) != item["visual_tokens"]
        or len(item["generated_ids"]) != item["generated_tail_slots"]
        or item["generated_tail_slots"] > 32):
        raise ValueError("actual input/output token lengths differ")
    padding = item["padded_input_tokens"] - item["input_tokens"]
    if (padding < 0 or item["generated_nonpad_tokens"] != sum(v != item["pad_token_id"] for v in item["generated_ids"])
        or content_sha([item["pad_token_id"]] * padding + item["input_ids"]) != item["padded_input_ids_sha256"]
        or content_sha([0] * padding + item["attention_mask"]) != item["padded_attention_mask_sha256"]):
        raise ValueError("actual padded input/generated nonpad contract differs")
    pixels = item["pixel_values"]
    if pixels["dtype"] != "float32" or pixels["shape"] != [item["premerge_patch_tokens"], 1176]:
        raise ValueError("actual pixel tensor shape/dtype differs")
    for key in ("pixel_values", "image_grid_thw_tensor"):
        sha = item[key]["sha256"]
        if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError("actual pixel/grid SHA invalid")
    if item["image_grid_thw_tensor"]["shape"] != [1, 3]:
        raise ValueError("actual image-grid tensor shape differs")
    import torch
    expected_grid = tensor_binding(torch.tensor([item["image_grid_thw"]], dtype=torch.int64))
    if item["image_grid_thw_tensor"] != expected_grid:
        raise ValueError("actual image-grid tensor bytes differ")


def complete_cache(path, rows, images, batch_size, lock, processor=None):
    # Frozen helper checks prediction IDs, hashes, rescoring, visual contract,
    # scientific/runtime lock and resource values before repairing a final tail.
    items, repairs = frozen_complete_cache(path, rows, images, batch_size, "textvqa", lock)
    for item in items:
        if item.get("condition") != lock["condition"]:
            raise ValueError("cached condition differs")
        validate_input_record(item)
        if processor is not None:
            decoded = processor.tokenizer.decode(item["generated_ids"], skip_special_tokens=True,
                clean_up_tokenization_spaces=False).strip()
            if decoded != item["prediction"] or item["pad_token_id"] != processor.tokenizer.pad_token_id:
                raise ValueError("cached actual generated IDs/decoded prediction differ")
    return items, repairs


def run_preflight(model, processor, args, lock, rows, images, image_root, device):
    selected = stress_rows(rows, images, processor)
    directory = args.output_root / "preflight" / args.condition / f"seed{args.seed}"
    prelock = {**lock, "scope": "engineering_preflight_not_scored",
        "selection": "eight deterministic input-only stress examples",
        "question_ids": [r["question_id"] for r in selected]}
    lock_file(directory / "INPUT_LOCK.json", prelock)
    result_path = directory / "PREFLIGHT.json"
    if result_path.exists():
        result = read_json(result_path)
        if result.get("input_lock_sha256") != content_sha(prelock) or result.get("status") != "passed":
            raise ValueError("completed preflight differs")
        return result
    import torch
    records, batches = [], []
    with torch.inference_mode():
        for begin in range(0, len(selected), 4):
            chunk = selected[begin:begin + 4]
            answers, resource = evaluate_batch(model, processor, chunk, images, image_root, device, True)
            batches.append(resource)
            records.extend({"question_id": r["question_id"], "image_id": r["image_id"], **a}
                for r, a in zip(chunk, answers))
    result = {"schema_version": "rarevlm.posttrain.preflight.v1", "status": "passed",
        "completed_at": now(), "input_lock_sha256": content_sha(prelock),
        "records": records, "batches": batches, "scope": "finite logits; no references scored"}
    write_json(result_path, result)
    return result


def run_cell(model, processor, args, lock, rows, images, image_root, device, update):
    directory = args.output_root / "runs" / args.condition / f"seed{args.seed}" / "dev"
    lock_file(directory / "INPUT_LOCK.json", lock)
    path = directory / "predictions.jsonl"
    items, repaired = complete_cache(path, rows, images, 4, lock, processor)
    result_path = directory / "EVALUATION.json"
    if result_path.exists():
        result = read_json(result_path)
        value, _ = aggregate(items, "textvqa")
        if (len(items) != len(rows) or result.get("predictions_sha256") != sha_file(path)
            or result.get("input_lock_sha256") != content_sha(lock)
            or not math.isclose(value, result["score_percent"], abs_tol=1e-12, rel_tol=0)):
            raise ValueError("completed result binding/score differs")
        update(len(items), "completed")
        return result
    import torch
    start, begun = len(items), time.monotonic()
    session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    session = {"session_id": session_id, "resumed_questions": start,
        "cache_repair": repaired, "started_at": now()}
    with path.open("a") as stream, torch.inference_mode():
        for begin in range(start, len(rows), 4):
            chunk = rows[begin:begin + 4]
            answers, resource = evaluate_batch(model, processor, chunk, images, image_root, device)
            batch = []
            for row, answer in zip(chunk, answers):
                item = {"question_id": row["question_id"], "image_id": row["image_id"],
                    "question_sha256": content_sha(row["question"]),
                    "question_prompt_sha256": question_prompt_sha(row),
                    "image_sha256": images[row["image_id"]]["sha256"], **answer,
                    "score": score_prediction(answer["prediction"], row, "textvqa"),
                    "task": "textvqa", "subcategory": "", "category": "",
                    "model_key": args.model_key, "condition": args.condition,
                    "seed": args.seed, "metric": METRIC,
                    "model_revision": lock["model_revision"],
                    "adapter_files_sha256": lock["adapter_files_sha256"],
                    "source_binding_sha256": content_sha(lock["source_sha256"]),
                    "reference_score_source": "all ten references used only after label-free generation",
                    **{k: v for k, v in resource.items() if k != "finite_logits"},
                    "batch_start": begin, "session_id": session_id,
                    "timing_warmup": begin == start, "visual_token_budget": BUDGET,
                    "numerical_path": "original_bf16", "precision": "bf16", "quantization": "original",
                    "input_lock_sha256": content_sha(lock)}
                validate_input_record(item)
                batch.append(item)
            stream.write("".join(json.dumps(i, ensure_ascii=False, allow_nan=False) + "\n" for i in batch))
            stream.flush()
            os.fsync(stream.fileno())
            items.extend(batch)
            elapsed = time.monotonic() - begun
            write_json(directory / "PROGRESS.json", {"status": "running", "updated_at": now(),
                "condition": args.condition, "seed": args.seed, "completed_questions": len(items),
                "total_questions": len(rows), "session_id": session_id,
                "session_elapsed_seconds": elapsed, "last_batch": resource,
                "remaining_estimate_seconds": (len(rows) - len(items)) * elapsed / (len(items) - start)})
            update(len(items), "running")
            if begin == start or ((begin - start) // 4 + 1) % 25 == 0:
                print("GENERATED", args.model_key, len(items), "/", len(rows), flush=True)
    session.update(completed_at=now(), new_questions=len(items) - start, wall_seconds=time.monotonic() - begun)
    result = summarize_task(items, lock, path, session)
    result.update(schema_version="rarevlm.posttrain.dev_cell.v1", condition=args.condition,
        seed=args.seed, scope="fixed completed checkpoint; full internal dev; exploratory; no formal-set feedback")
    write_json(result_path, result)
    write_json(directory / "PROGRESS.json", {"status": "completed", "updated_at": now(),
        "completed_questions": len(items), "total_questions": len(rows)})
    update(len(items), "completed")
    print("CELL COMPLETE", args.model_key, result["score_percent"], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 42, 200), default=42)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    for name in ("config", "adapter_dir", "output_root"):
        setattr(args, name, private_path(getattr(args, name)))
    config = read_json(args.config)
    for name in ("asset_root", "model_root"):
        override = getattr(args, name)
        path = private_path(override) if override else config_path(config, name)
        if override and config.get(name) and path != config_path(config, name):
            raise ValueError(f"CLI override differs from predeclared {name}")
        setattr(args, name, path)
    args.model_key = model_key(args.condition, args.seed)
    reference_path = config_path(config, "reference_config", "configs/vlm_deep_v2.json")
    reference = read_json(reference_path)
    validate_config(config, reference, args.batch_size)
    files, training = adapter_binding(config, args.condition, args.seed, args.adapter_dir, args.model_root)
    rows, images, image_root, metadata = dev_data(args.asset_root)
    model_metadata = model_metadata_binding(args.model_root, reference)
    # Hash current weight bytes once; provenance alone would only bind a prior check.
    for name, expected in model_metadata["model_weight_provenance_sha256"].items():
        if sha_file(args.model_root / name) != expected:
            raise ValueError(f"current original model weight differs: {name}")
    import torch
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or not torch.cuda.is_available():
        raise ValueError("one externally admitted CUDA worker required")
    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda", 0)
    props = torch.cuda.get_device_properties(device)
    if "A100" not in props.name or props.total_memory < 70 * 2**30:
        raise ValueError("fixed pilot requires admitted A10080GB")
    hardware = {**visible_gpu_binding(), "name": props.name, "total_memory_bytes": props.total_memory,
        "compute_capability": [props.major, props.minor], "torch_cuda_version": torch.version.cuda}
    lock = {"schema_version": "rarevlm.posttrain.dev_input.v1", "condition": args.condition,
        "seed": args.seed, "model_key": args.model_key, "role": "adapted",
        "posttrain_config_sha256": sha_file(args.config), "posttrain_config_content_sha256": content_sha(config),
        "reference_config_sha256": sha_file(reference_path), **model_metadata, **metadata,
        "current_original_weight_bytes_verified": True,
        "adapter_dir": str(args.adapter_dir), "adapter_files_sha256": files, "training_binding": training,
        "source_sha256": source_hashes(), "packages": package_versions(), "hardware": hardware,
        **FIXED, **PATH_SPECS["original_bf16"], "tf32_matmul": False, "tf32_cudnn": False,
        "head_last_position_only": True, "autocast": False, "adapter_autocast_dtype": False,
        "explicit_lora_bf16_conversion": True,
        "input_audit_hashing_included_in_preprocessing_seconds": True,
        "input": "image+original question+fixed short-answer suffix; no references/targets/OCR",
        "training_eval_prompt_difference": "public preference training may use no suffix and longer answers; eval remains original short-answer greedy32",
        "statistical_unit": "image cluster; checkpoints are not independent image observations",
        "scope": "single-checkpoint exploratory internal dev; formal benchmarks forbidden for tuning"}
    status_path = args.output_root / "runs" / args.condition / f"seed{args.seed}" / "EVAL_STATUS.json"
    status = {"condition": args.condition, "seed": args.seed, "started_at": now(),
        "phase": "starting", "hardware": hardware, "completed_questions": 0, "total_questions": len(rows)}
    def update(count=None, state=None):
        if count is not None:
            status["completed_questions"] = count
        if state is not None:
            status["cell_status"] = state
        status["updated_at"] = now()
        write_json(status_path, status)
    model, hook = None, None
    try:
        status["phase"] = "loading_model"; update()
        begun = time.monotonic()
        model, dtypes = load_path(args.model_root, reference, device, "original_bf16", args.adapter_dir)
        status["model_load_seconds"] = time.monotonic() - begun
        lock["actual_parameter_dtypes"] = dtypes
        if int(model.config.image_token_id) != 151655:
            raise ValueError("image token ID differs")
        hook = model.get_output_embeddings().register_forward_pre_hook(lambda module, inputs: (inputs[0][:, -1:, :],))
        processor = processor_for(args.model_root, reference, BUDGET)
        lock_file(args.output_root / "runs" / args.condition / f"seed{args.seed}" / "dev" / "INPUT_LOCK.json", lock)
        status["phase"] = "validating_images"; update()
        status["image_validation"] = verify_dev_images(rows, images, image_root)
        status["phase"] = "preflight"; update()
        run_preflight(model, processor, args, lock, rows, images, image_root, device)
        if not args.preflight_only:
            status["phase"] = "evaluating"; update()
            run_cell(model, processor, args, lock, rows, images, image_root, device, update)
        status.update(phase="preflight_completed" if args.preflight_only else "completed", completed_at=now())
        update()
    except Exception as error:
        status.update(phase="failed", error_type=type(error).__name__, error=str(error)); update()
        raise
    finally:
        if hook is not None:
            hook.remove()
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
