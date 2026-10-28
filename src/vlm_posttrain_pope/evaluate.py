"""Once-only original POPE generation diagnostic; external queue admits GPU work."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT
from vlm_deep.common import collate, encode, to_device
from vlm_formal.evaluate import stress_rows
from vlm_joint.evaluate import model_metadata_binding, visible_gpu_binding
from vlm_posttrain_eval import evaluate as fixed_text
from vlm_precision.evaluate import load_path
from vlm_probe.evaluate import (content_sha, lock_file, now, package_versions, private_path,
    processor_for, read_json, sha_file, verify_dev_images, write_json)
from .prepare import (ANNOTATIONS, CATEGORIES, MIRROR_REVISION, MIRROR_SHA,
    OFFICIAL_COMMIT, PROMPT_SUFFIX, SCORER_SHA, coco_id, content_sha as manifest_sha)
from .scoring import metrics, official_prediction, validity_diagnostic

FIXED = {"numerical_path": "original_bf16", "precision": "bf16", "vision_precision": "bf16",
    "quantization": "original", "visual_token_budget": 1024, "min_pixels": 50176, "max_pixels": 802816,
    "batch_size": 4, "max_new_tokens": 32, "do_sample": False,
    "attention_implementation": "sdpa", "tf32_matmul": False, "tf32_cudnn": False,
    "input_truncation": False, "prompt_suffix": PROMPT_SUFFIX}


def source_hashes():
    result = fixed_text.source_hashes()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        result[str(path.relative_to(SOURCE_ROOT))] = sha_file(path)
    path = PROJECT_ROOT / "tools/prepare_posttrain_pope.py"
    result["tools/prepare_posttrain_pope.py"] = sha_file(path)
    return result


def load_data(data_root):
    data_root = private_path(data_root)
    path = data_root / "manifest.json"
    manifest = read_json(path)
    if manifest_sha({k: v for k, v in manifest.items() if k != "manifest_sha256"}) != manifest["manifest_sha256"]:
        raise ValueError("POPE data manifest changed")
    if (manifest["revision"] != OFFICIAL_COMMIT
        or manifest["mirror_audit"]["mirror_revision"] != MIRROR_REVISION
        or manifest["mirror_audit"]["mirror_file_sha256"] != MIRROR_SHA
        or sha_file(data_root / "official_evaluate.py") != SCORER_SHA):
        raise ValueError("POPE fixed official/mirror/scorer source differs")
    rows, images = manifest["records"], manifest["images"]
    if (manifest["schema_version"] != "rarevlm.posttrain.pope.data.v1" or len(rows) != 9000
        or len(images) != 500 or len({r["question_id"] for r in rows}) != 9000
        or {r["image_id"] for r in rows} != set(images)
        or any(r["prompt_suffix"] != PROMPT_SUFFIX for r in rows)):
        raise ValueError("expected original 9000-question/500-image POPE")
    for category in CATEGORIES:
        selected = [r for r in rows if r["category"] == category]
        if (len(selected) != 3000 or len({r["image_id"] for r in selected}) != 500
            or sum(r["label"] == "yes" for r in selected) != 1500
            or sum(r["label"] == "no" for r in selected) != 1500):
            raise ValueError("POPE category coverage/labels changed")
        annotation_path = data_root / f"coco_pope_{category}.json"
        if sha_file(annotation_path) != ANNOTATIONS[category][0]:
            raise ValueError("current original POPE annotations changed")
        originals = [json.loads(line) for line in annotation_path.read_bytes().splitlines()]
        for row, original in zip(selected, originals):
            expected = {"question_id": f"pope_{category}_{original['question_id']}",
                "source_question_id": str(original["question_id"]),
                "image_id": f"COCO_val2014_{coco_id(original['image'])}",
                "question": original["text"], "label": original["label"],
                "category": category, "prompt_suffix": PROMPT_SUFFIX}
            if row != expected:
                raise ValueError("POPE question/label/image record differs from pinned original")
    return rows, images, data_root, {"data_root": str(data_root),
        "pope_manifest_sha256": manifest["manifest_sha256"], "pope_manifest_file_sha256": sha_file(path),
        "official_commit": manifest["revision"], "official_source": manifest["source"],
        "mirror_source": manifest["mirror_audit"], "questions": 9000, "unique_images": 500,
        "question_order_sha256": content_sha([r["question_id"] for r in rows]),
        "question_prompt_order_sha256": content_sha([content_sha(r["question"] + PROMPT_SUFFIX) for r in rows]),
        "pretraining_overlap": manifest["pretraining_overlap"]}


def inference_row(row):
    return {**{k: row[k] for k in ("question_id", "image_id", "question")}, "prompt_suffix": PROMPT_SUFFIX}


def evaluate_batch(model, processor, chunk, images, image_root, device, check_logits=False):
    import torch
    begun = time.monotonic()
    samples = [encode(processor, inference_row(r), image_root / images[r["image_id"]]["path"], False) for r in chunk]
    if any("labels" in sample for sample in samples):
        raise ValueError("POPE generation input contains answer labels")
    details = [fixed_text.input_details(s, r, images, processor) for s, r in zip(samples, chunk)]
    cpu_batch = collate(samples, processor.tokenizer.pad_token_id)
    preprocessing = time.monotonic() - begun
    torch.cuda.synchronize(device); begun = time.monotonic()
    batch = to_device(cpu_batch, device)
    torch.cuda.synchronize(device); transfer = time.monotonic() - begun
    torch.cuda.reset_peak_memory_stats(device)
    finite = None
    if check_logits:
        output = model(**batch, use_cache=False)
        finite = bool(torch.isfinite(output.logits).all()); del output
        if not finite:
            raise ValueError("non-finite POPE preflight logits")
    torch.cuda.synchronize(device); begun = time.monotonic()
    generated = model.generate(**batch, max_new_tokens=32, do_sample=False, use_cache=True)
    torch.cuda.synchronize(device); elapsed = time.monotonic() - begun
    padded = int(batch["input_ids"].shape[1]); answers = []
    for index in range(len(chunk)):
        tail = generated[index, padded:]
        ids = tail.tolist()
        answers.append({"prediction": processor.tokenizer.decode(tail, skip_special_tokens=True,
                clean_up_tokenization_spaces=False).strip(),
            "generated_ids": ids, "generated_ids_sha256": content_sha(ids),
            "pad_token_id": int(processor.tokenizer.pad_token_id),
            "generated_nonpad_tokens": int((tail != processor.tokenizer.pad_token_id).sum()),
            "generated_tail_slots": len(ids), **details[index],
            "padded_input_ids_sha256": content_sha(cpu_batch["input_ids"][index].tolist()),
            "padded_attention_mask_sha256": content_sha(cpu_batch["attention_mask"][index].tolist())})
    return answers, {"batch_size": len(chunk), "padded_input_tokens": padded,
        "batch_generation_seconds": elapsed, "batch_preprocessing_seconds": preprocessing,
        "batch_h2d_seconds": transfer, "finite_logits": finite,
        "batch_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "batch_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30}


def complete_cache(path, rows, images, lock, processor=None, *, repair=True):
    if not path.exists():
        return [], None
    lines = path.read_bytes().splitlines(keepends=True); repairs = []
    if lines and not lines[-1].endswith(b"\n"):
        if not repair:
            raise ValueError("completed POPE predictions have interrupted tail")
        lines.pop(); repairs.append("discarded interrupted final JSONL line")
    items = [json.loads(line) for line in lines]
    if len(items) > len(rows):
        raise ValueError("too many POPE cached predictions")
    if items:
        start = (len(items) - 1) // 4 * 4
        if len(items) - start != min(4, len(rows) - start):
            if not repair:
                raise ValueError("completed POPE predictions contain incomplete batch")
            items, lines = items[:start], lines[:start]
            repairs.append("discarded incomplete final batch")
    for index, (item, row) in enumerate(zip(items, rows)):
        expected = {"question_id": row["question_id"], "image_id": row["image_id"],
            "question_sha256": content_sha(row["question"]), "question_prompt_sha256": content_sha(row["question"] + PROMPT_SUFFIX),
            "image_sha256": images[row["image_id"]]["sha256"], "label": row["label"], "category": row["category"],
            "condition": lock["condition"], "seed": lock["seed"], "input_lock_sha256": content_sha(lock),
            "source_binding_sha256": content_sha(lock["source_sha256"]),
            "adapter_files_sha256": lock["adapter_files_sha256"], "model_revision": lock["model_revision"],
            "batch_start": index // 4 * 4, "batch_size": min(4, len(rows) - index // 4 * 4)}
        for key, value in expected.items():
            if item.get(key) != value:
                raise ValueError(f"POPE prediction binding differs: {key}")
        parsed = official_prediction(item["prediction"])
        if item["official_prediction"] != parsed or item["score"] != float(parsed == row["label"]):
            raise ValueError("POPE official parser/label/score differs")
        if item["validity_diagnostic"] != validity_diagnostic(item["prediction"]):
            raise ValueError("POPE invalid-output diagnostic differs")
        fixed_text.validate_input_record(item)
        grid = item["image_grid_thw"]
        visual = math.prod(grid) // 4
        if (len(grid) != 3 or grid[0] != 1 or math.prod(grid) % 4 or visual > 1024
            or visual != item["visual_tokens"] or visual != item["counted_image_tokens"] or item["input_truncated"]):
            raise ValueError("POPE actual visual token contract differs")
        for key in ("batch_generation_seconds", "batch_preprocessing_seconds", "batch_h2d_seconds",
            "batch_peak_allocated_gib", "batch_peak_reserved_gib"):
            if not math.isfinite(item[key]) or item[key] < 0:
                raise ValueError("invalid POPE resource observation")
        if processor is not None:
            decoded = processor.tokenizer.decode(item["generated_ids"], skip_special_tokens=True,
                clean_up_tokenization_spaces=False).strip()
            if decoded != item["prediction"] or item["pad_token_id"] != processor.tokenizer.pad_token_id:
                raise ValueError("POPE actual generated IDs/decoded output differ")
    if repairs:
        with path.open("wb") as stream:
            stream.write(b"".join(lines)); stream.flush(); os.fsync(stream.fileno())
    return items, repairs or None


def run_preflight(model, processor, args, lock, rows, images, image_root, device):
    selected = stress_rows(rows, images, processor)
    directory = args.output_root / "preflight_pope" / args.condition / f"seed{args.seed}"
    prelock = {**lock, "scope": "input-only engineering preflight; no scored diagnostic",
        "question_ids": [r["question_id"] for r in selected]}
    lock_file(directory / "INPUT_LOCK.json", prelock)
    path = directory / "PREFLIGHT.json"
    if path.exists():
        saved = read_json(path)
        if saved.get("input_lock_sha256") != content_sha(prelock) or saved.get("status") != "passed":
            raise ValueError("POPE preflight binding/status differs")
        return saved
    import torch
    records, batches = [], []
    with torch.inference_mode():
        for begin in range(0, len(selected), 4):
            chunk = selected[begin:begin + 4]
            answers, resource = evaluate_batch(model, processor, chunk, images, image_root, device, True)
            records.extend({"question_id": r["question_id"], "image_id": r["image_id"], **a} for r, a in zip(chunk, answers))
            batches.append(resource)
    result = {"status": "passed", "completed_at": now(), "input_lock_sha256": content_sha(prelock),
        "records": records, "batches": batches, "scope": "finite logits; no label scoring/cache reuse"}
    write_json(path, result)
    return result


def run_cell(model, processor, args, lock, rows, images, image_root, device, update):
    directory = args.output_root / "runs" / args.condition / f"seed{args.seed}" / "pope"
    lock_file(directory / "INPUT_LOCK.json", lock)
    path = directory / "predictions.jsonl"
    result_path = directory / "EVALUATION.json"
    items, repaired = complete_cache(path, rows, images, lock, processor, repair=not result_path.exists())
    if result_path.exists():
        saved = read_json(result_path)
        if (len(items) != len(rows) or saved.get("predictions_sha256") != sha_file(path)
            or saved.get("input_lock_sha256") != content_sha(lock) or saved.get("metrics") != metrics(items)):
            raise ValueError("completed POPE result differs")
        update(len(items), "completed"); return saved
    import torch
    start, begun = len(items), time.monotonic()
    session = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    with path.open("a") as stream, torch.inference_mode():
        for begin in range(start, len(rows), 4):
            chunk = rows[begin:begin + 4]
            answers, resource = evaluate_batch(model, processor, chunk, images, image_root, device)
            batch = []
            for row, answer in zip(chunk, answers):
                parsed = official_prediction(answer["prediction"])
                item = {"question_id": row["question_id"], "image_id": row["image_id"],
                    "question_sha256": content_sha(row["question"]),
                    "question_prompt_sha256": content_sha(row["question"] + PROMPT_SUFFIX),
                    "image_sha256": images[row["image_id"]]["sha256"], **answer,
                    "label": row["label"], "category": row["category"],
                    "official_prediction": parsed, "score": float(parsed == row["label"]),
                    "validity_diagnostic": validity_diagnostic(answer["prediction"]),
                    "condition": args.condition, "seed": args.seed, "model_revision": lock["model_revision"],
                    "adapter_files_sha256": lock["adapter_files_sha256"],
                    "source_binding_sha256": content_sha(lock["source_sha256"]),
                    "input_lock_sha256": content_sha(lock), "batch_start": begin,
                    "session_id": session, "timing_warmup": begin == start,
                    "label_use": "official labels scored only after label-free generation",
                    **{k: v for k, v in resource.items() if k != "finite_logits"}}
                fixed_text.validate_input_record(item); batch.append(item)
            stream.write("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in batch))
            stream.flush(); os.fsync(stream.fileno()); items.extend(batch)
            write_json(directory / "PROGRESS.json", {"status": "running", "updated_at": now(),
                "completed_questions": len(items), "total_questions": 9000,
                "session_id": session, "session_elapsed_seconds": time.monotonic() - begun,
                "last_batch": resource})
            update(len(items), "running")
            if begin == start or ((begin - start) // 4 + 1) % 25 == 0:
                print("POPE GENERATED", args.condition, args.seed, len(items), "/", len(rows), flush=True)
    batches = {(r["session_id"], r["batch_start"]): r for r in items}
    timed = [r for r in batches.values() if not r["timing_warmup"]]
    result = {"schema_version": "rarevlm.posttrain.pope.cell.v1", "status": "completed",
        "completed_at": now(), "condition": args.condition, "seed": args.seed,
        "input_lock_sha256": content_sha(lock), "predictions_sha256": sha_file(path),
        "questions": len(items), "unique_images": len({r["image_id"] for r in items}),
        "metrics": metrics(items), "session_id": session, "resumed_questions": start,
        "cache_repair": repaired, "current_session_wall_seconds": time.monotonic() - begun,
        "resources": {"timed_questions": sum(r["batch_size"] for r in timed),
            "model_generate_seconds": sum(r["batch_generation_seconds"] for r in timed),
            "preprocessing_h2d_generation_seconds": sum(r["batch_preprocessing_seconds"] + r["batch_h2d_seconds"] + r["batch_generation_seconds"] for r in timed),
            "peak_allocated_gib": max(r["batch_peak_allocated_gib"] for r in batches.values()),
            "peak_reserved_gib": max(r["batch_peak_reserved_gib"] for r in batches.values()),
            "scope": "cache VLM stage; input audit hashing included in preprocessing; loading/preflight/validation/writer excluded; no serving-speed claim"},
        "scope": "once-only full external object-hallucination diagnostic; fixed checkpoint; no POPE feedback tuning"}
    write_json(result_path, result)
    update(len(items), "completed")
    print("POPE COMPLETE", args.condition, args.seed, json.dumps(result["metrics"]["overall"]), flush=True)
    return result


def read_cell(root, condition, seed, config_path, config):
    directory = root / "runs" / condition / f"seed{seed}" / "pope"
    path = directory / "EVALUATION.json"
    if not path.exists():
        return {"complete": False, "condition": condition, "seed": seed, "path": str(directory)}
    lock, result = read_json(directory / "INPUT_LOCK.json"), read_json(path)
    expected = {**FIXED, "condition": condition, "seed": seed,
        "training_config_sha256": sha_file(config_path), "training_config_content_sha256": content_sha(config),
        "source_sha256": source_hashes()}
    for key, value in expected.items():
        if lock.get(key) != value:
            raise ValueError(f"POPE completed-cell fixed binding differs: {key}")
    model_root = fixed_text.config_path(config, "model_root")
    files, binding = fixed_text.adapter_binding(config, condition, seed, Path(lock["adapter_dir"]), model_root)
    if files != lock["adapter_files_sha256"] or binding != lock["training_binding"]:
        raise ValueError("POPE completed-cell adapter/training changed")
    rows, images, image_root, metadata = load_data(Path(lock["data_root"]))
    for key, value in metadata.items():
        if lock.get(key) != value:
            raise ValueError(f"POPE completed-cell dataset differs: {key}")
    reference = read_json(fixed_text.config_path(config, "reference_config", "configs/vlm_deep_v2.json"))
    processor = processor_for(model_root, reference, 1024)
    records, repairs = complete_cache(directory / "predictions.jsonl", rows, images, lock, processor, repair=False)
    if (len(records) != 9000 or repairs or result.get("status") != "completed"
        or result.get("predictions_sha256") != sha_file(directory / "predictions.jsonl")
        or result.get("input_lock_sha256") != content_sha(lock) or result.get("metrics") != metrics(records)):
        raise ValueError("POPE completed-cell scores/hash/count differ")
    return {"complete": True, "condition": condition, "seed": seed, "rows": records,
        "lock": lock, "evaluation": result,
        "provenance": {"evaluation_path": str(path), "evaluation_sha256": sha_file(path),
            "predictions_sha256": result["predictions_sha256"], "input_lock_sha256": content_sha(lock)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "adapter-dir", "output-root", "data-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--condition", choices=fixed_text.CONDITIONS, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 42, 200), default=42)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    for name in ("config", "adapter_dir", "output_root", "data_root"):
        setattr(args, name, private_path(getattr(args, name)))
    config = read_json(args.config)
    model_root = fixed_text.config_path(config, "model_root")
    reference_path = fixed_text.config_path(config, "reference_config", "configs/vlm_deep_v2.json")
    reference = read_json(reference_path)
    fixed_text.validate_config(config, reference)
    files, training = fixed_text.adapter_binding(config, args.condition, args.seed, args.adapter_dir, model_root)
    rows, images, image_root, metadata = load_data(args.data_root)
    model_metadata = model_metadata_binding(model_root, reference)
    for name, expected in model_metadata["model_weight_provenance_sha256"].items():
        if sha_file(model_root / name) != expected:
            raise ValueError(f"POPE original base-model bytes changed: {name}")
    import torch
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or not torch.cuda.is_available():
        raise ValueError("one external-queue-admitted CUDA worker required")
    torch.cuda.set_device(0); torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda", 0); props = torch.cuda.get_device_properties(device)
    if "A100" not in props.name or props.total_memory < 70 * 2**30:
        raise ValueError("fixed POPE pilot requires admitted A10080GB")
    hardware = {**visible_gpu_binding(), "name": props.name, "total_memory_bytes": props.total_memory,
        "compute_capability": [props.major, props.minor], "torch_cuda_version": torch.version.cuda}
    lock = {"schema_version": "rarevlm.posttrain.pope.input.v1", **FIXED,
        "condition": args.condition, "seed": args.seed,
        "training_config_sha256": sha_file(args.config), "training_config_content_sha256": content_sha(config),
        "reference_config_sha256": sha_file(reference_path), **model_metadata, **metadata,
        "current_original_weight_bytes_verified": True, "adapter_dir": str(args.adapter_dir),
        "adapter_files_sha256": files, "training_binding": training,
        "source_sha256": source_hashes(), "packages": package_versions(), "hardware": hardware,
        "head_last_position_only": True, "explicit_lora_bf16_conversion": True,
        "input": "image + original POPE question + Answer yes or no.; no labels/references/target/OCR",
        "statistical_unit": "500 image clusters shared across all three categories, not 1500 independent images",
        "scope": "once-only frozen external diagnostic; no checkpoint/prompt/setting selection"}
    status_path = args.output_root / "runs" / args.condition / f"seed{args.seed}" / "POPE_STATUS.json"
    status = {"condition": args.condition, "seed": args.seed, "phase": "starting", "started_at": now(), "hardware": hardware}
    def update(count=None, state=None):
        if count is not None:
            status["completed_questions"] = count
        if state is not None:
            status["cell_status"] = state
        status["updated_at"] = now(); write_json(status_path, status)
    model, hook = None, None
    try:
        status["phase"] = "loading_model"; update()
        model, dtypes = load_path(model_root, reference, device, "original_bf16", args.adapter_dir)
        lock["actual_parameter_dtypes"] = dtypes
        if int(model.config.image_token_id) != 151655:
            raise ValueError("POPE image token ID differs")
        hook = model.get_output_embeddings().register_forward_pre_hook(lambda module, inputs: (inputs[0][:, -1:, :],))
        processor = processor_for(model_root, reference, 1024)
        lock_file(args.output_root / "runs" / args.condition / f"seed{args.seed}" / "pope" / "INPUT_LOCK.json", lock)
        status["phase"] = "validating_images"; update()
        status["image_validation"] = {**verify_dev_images(rows, images, image_root), "scope": "all 500 POPE image bytes and dimensions verified"}
        status["phase"] = "preflight"; update()
        run_preflight(model, processor, args, lock, rows, images, image_root, device)
        if not args.preflight_only:
            status["phase"] = "evaluating"; update()
            run_cell(model, processor, args, lock, rows, images, image_root, device, update)
        status.update(phase="preflight_completed" if args.preflight_only else "completed", completed_at=now()); update()
    except Exception as error:
        status.update(phase="failed", error_type=type(error).__name__, error=str(error)); update(); raise
    finally:
        if hook is not None:
            hook.remove()
        del model; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
