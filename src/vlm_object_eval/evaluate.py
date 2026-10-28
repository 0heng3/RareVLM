"""Independent BF16/1024, batch4 evaluation of fixed object-SFT adapters.

External study queue alone admits GPU work. The full new validation and original
Text dev screen precedes POPE confirmation. No reference enters generation.
"""
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

from rarevlm_paths import SOURCE_ROOT
from vlm_deep.common import PROMPT_SUFFIX as TEXT_SUFFIX, collate, encode, to_device
from vlm_deep.evaluate import score_prediction
from vlm_formal.evaluate import stress_rows
from vlm_joint.evaluate import model_metadata_binding, visible_gpu_binding
from vlm_posttrain_eval import evaluate as frozen_text
from vlm_posttrain_pope.evaluate import load_data as frozen_pope_data
from vlm_precision.evaluate import load_path
from vlm_probe.evaluate import (content_sha, lock_file, now, package_versions, private_path,
    processor_for, read_json, sha_file, verify_dev_images, write_json)
from .scoring import PROMPT_SUFFIX, diagnostics, metrics, official_prediction

CONDITIONS = ("initial", "object_only", "object_replay")
CANDIDATES = CONDITIONS[1:]
TASKS = ("object", "text", "pope")
OBJECT_MANIFEST_SCHEMAS = ("rarevlm.object_presence.manifest.v1", "rarevlm.object_presence.manifest.v2")
FIXED = {"numerical_path": "original_bf16", "precision": "bf16", "vision_precision": "bf16",
    "quantization": "original", "visual_token_budget": 1024, "min_pixels": 50176,
    "max_pixels": 802816, "batch_size": 4, "max_new_tokens": 32, "do_sample": False,
    "attention_implementation": "sdpa", "tf32_matmul": False, "tf32_cudnn": False,
    "input_truncation": False}


def source_hashes():
    result = frozen_text.source_hashes()
    from vlm_posttrain_pope.evaluate import source_hashes as pope_hashes
    result.update(pope_hashes())
    for path in sorted(Path(__file__).parent.glob("*.py")):
        result[str(path.relative_to(SOURCE_ROOT))] = sha_file(path)
    for path in sorted((SOURCE_ROOT / "vlm_object_study").glob("*.py")):
        result[str(path.relative_to(SOURCE_ROOT))] = sha_file(path)
    return result


config_path = frozen_text.config_path
initial_adapter_path = frozen_text.initial_adapter_path
adapter_files = frozen_text.adapter_files
validate_input_record = frozen_text.validate_input_record


def validate_config(config, reference, batch_size=4):
    if tuple(config.get("candidates", CANDIDATES)) != CANDIDATES:
        raise ValueError("both object_only and object_replay must be predeclared in this order")
    evaluation = config.get("evaluation", {})
    for key, expected in FIXED.items():
        if evaluation.get(key, expected) != expected:
            raise ValueError(f"fixed object evaluation differs: {key}")
    if batch_size != 4:
        raise ValueError("fixed evaluation requires batch4")
    # Preserve frozen Text/scorer/processor contracts without imposing its old arms.
    frozen_text.validate_config({"evaluation": {}}, reference, batch_size)
    analysis = config.get("analysis", {})
    for key, expected in (("bootstrap_replicates", 6000), ("bootstrap_seed", 20261015)):
        if analysis.get(key, expected) != expected:
            raise ValueError(f"fixed analysis differs: {key}")


def training_batch_contract(config):
    microbatch = config.get("training_microbatch")
    accumulation = config.get("training_accumulation")
    effective = config.get("object_effective_batch")
    if (type(microbatch) is not int or microbatch <= 0 or type(accumulation) is not int
        or accumulation <= 0 or type(effective) is not int or effective != 16
        or microbatch * accumulation != effective):
        raise ValueError("explicit positive training microbatch × accumulation must equal object_effective_batch16")
    return {"training_microbatch_questions": microbatch,
        "training_accumulation": accumulation, "object_effective_batch": effective}


def adapter_binding(config, condition, seed, adapter_dir, model_root):
    """Validate the NEW trainer schema without modifying old binding rules."""
    if condition not in CONDITIONS or seed not in (0, 42, 200):
        raise ValueError("unknown object condition or seed")
    initial = initial_adapter_path(config)
    # Frozen initial binding checks the absolute original SFT path, all bytes,
    # complete 32985QA/2062 updates, frozen tensors, and base provenance.
    initial_files, initial_binding = frozen_text.adapter_binding(
        config, "initial", 42, initial, model_root)
    adapter_dir = private_path(adapter_dir)
    if condition == "initial":
        if adapter_dir != initial:
            raise ValueError("initial must use the single fixed SFT42 adapter")
        return initial_files, {**initial_binding, "evaluation_seed": seed,
            "shared_initial_checkpoint": True}
    if adapter_dir == initial:
        raise ValueError("candidate cannot reuse the initial adapter directory")
    files = adapter_files(adapter_dir)
    path = adapter_dir.parent / "TRAIN_RESULT.json"
    train = read_json(path)
    manifest_path = config_path(config, "train_manifest")
    object_questions = len(read_json(manifest_path)["rows"])
    planned_updates = math.ceil(object_questions / 16)
    replay_exposures = planned_updates * 8 if condition == "object_replay" else 0
    expected = {"schema_version": "rarevlm.vlm_object_train.v1",
        "condition": condition, "seed": seed, "status": "completed",
        "config_sha256": content_sha(config), "adapter_files_sha256": files,
        "initial_adapter_files_sha256": initial_files, "trainable_parameters": 1843200,
        "numerical_path": "original_bf16", "visual_token_budget": 1024,
        "frozen_parameters_unchanged": True,
        "initial_adapter_dir": str(initial), "train_manifest_sha256": sha_file(manifest_path),
        "train_manifest_file_sha256": sha_file(manifest_path), "lora_parameter_precision": "fp32",
        **training_batch_contract(config),
        "replay_questions_per_update": 8 if condition == "object_replay" else 0,
        "replay_lambda": 1. if condition == "object_replay" else 0.,
        "planned_updates": planned_updates, "completed_updates": planned_updates,
        "actual_object_exposures": object_questions, "actual_replay_exposures": replay_exposures,
        "actual_questions_seen": object_questions + replay_exposures,
        "model_provenance_sha256": sha_file(model_root.parent / "model_provenance.json")}
    for key, expected_value in expected.items():
        if train.get(key) != expected_value:
            raise ValueError(f"completed object training binding differs: {key}")
    if (type(train.get("planned_updates")) is not int or train["planned_updates"] <= 0
        or train.get("completed_updates") != train["planned_updates"]
        or train.get("actual_questions_seen", 0) <= 0
        or train.get("actual_object_exposures", 0) <= 0
        or train.get("actual_questions_seen") != train.get("actual_object_exposures", 0) + train.get("actual_replay_exposures", 0)
        or (condition == "object_only" and train.get("actual_replay_exposures") != 0)
        or (condition == "object_replay" and train.get("actual_replay_exposures", 0) <= 0)):
        raise ValueError("object training has not completed its declared schedule")
    return files, {"path": str(path), "sha256": sha_file(path), "condition": condition,
        "evaluation_seed": seed, "training_seed": train["seed"],
        "initial_adapter_dir": str(initial), "initial_adapter_files_sha256": initial_files,
        "initial_train_result_path": initial_binding["initial_train_result_path"],
        "initial_train_result_sha256": initial_binding["initial_train_result_sha256"],
        "completed_updates": train["completed_updates"],
        "actual_questions_seen": train["actual_questions_seen"],
        "scope": "fixed Final adapter; no eval-based checkpoint selection",
        "eval_adapter_conversion": "all floating weights including LoRA cast to BF16 by frozen load_path"}


def validate_object_schema(manifest, expected_schema=None, expected_protocol_sha256=None):
    schema = manifest.get("schema_version")
    if schema not in OBJECT_MANIFEST_SCHEMAS or (expected_schema is not None and schema != expected_schema):
        raise ValueError("object manifest schema differs from declared v1/v2 data source")
    if schema == OBJECT_MANIFEST_SCHEMAS[1]:
        if (not manifest.get("protocol_sha256") or (expected_protocol_sha256 is not None
            and manifest["protocol_sha256"] != expected_protocol_sha256)):
            raise ValueError("v2 object manifest protocol differs from current config")
        if any(not str(row["question_id"]).startswith("object_presence_v2:") for row in manifest["rows"]):
            raise ValueError("v2 manifest must retain its real v2 question namespace")
    elif any(str(row["question_id"]).startswith("object_presence_v2:") for row in manifest["rows"]):
        raise ValueError("v2 rows cannot masquerade as a v1 manifest")
    return schema


def object_data(path, expected_schema=None, expected_protocol_sha256=None):
    path = private_path(path)
    if path.is_dir():
        path = path / "VAL_MANIFEST.json"
    manifest = read_json(path)
    if content_sha({k: v for k, v in manifest.items() if k != "manifest_sha256"}) != manifest.get("manifest_sha256"):
        raise ValueError("object validation manifest changed")
    schema = validate_object_schema(manifest, expected_schema, expected_protocol_sha256)
    if manifest.get("split") not in ("val", "validation", "heldout"):
        raise ValueError("object evaluation must use held-out validation, never training")
    if manifest.get("prompt_suffix") != PROMPT_SUFFIX:
        raise ValueError("object validation yes/no suffix differs")
    rows, images = [], manifest["images"]
    for raw in manifest["rows"]:
        label = raw.get("label", raw.get("target"))
        if label not in ("yes", "no") or not isinstance(raw.get("question"), str):
            raise ValueError("object question/label invalid")
        if raw.get("prompt_suffix", PROMPT_SUFFIX) != PROMPT_SUFFIX:
            raise ValueError("object row suffix differs")
        if raw["image_id"] not in images:
            raise ValueError("object row image absent")
        info = images[raw["image_id"]]
        if (str(private_path(raw["image_path"])) != str(private_path(info["path"]))
            or raw.get("image_sha256", raw.get("file_sha256")) != info["sha256"]):
            raise ValueError("object row image bytes/path disagree")
        rows.append({**raw, "label": label, "category": str(raw.get("category_name", "object"))})
    if (not rows or len({r["question_id"] for r in rows}) != len(rows)
        or {r["image_id"] for r in rows} != set(images)):
        raise ValueError("object validation requires full unique question/image coverage")
    return rows, images, path.parent, {"data_manifest_path": str(path),
        "data_manifest_schema_version": schema,
        "data_manifest_sha256": manifest["manifest_sha256"], "data_manifest_file_sha256": sha_file(path),
        "questions": len(rows), "unique_images": len(images),
        "question_order_sha256": content_sha([r["question_id"] for r in rows]),
        "question_prompt_order_sha256": content_sha([content_sha(r["question"] + PROMPT_SUFFIX) for r in rows]),
        "source_split": "new fixed image-disjoint COCO object validation",
        "pretraining_overlap": "unknown; full POPE was exposed in the previous study; not blind or pretrain-disjoint"}


def load_data(config, task, data_root=None):
    if task == "object":
        return object_data(data_root or config_path(config, "val_manifest", config.get("object_data_root")),
            config.get("object_manifest_schema_version"),
            sha_file(config_path(config, "dataset_protocol")) if config.get("dataset_protocol") else None)
    if task == "text":
        rows, images, root, metadata = frozen_text.dev_data(config_path(config, "asset_root"))
        return rows, images, root, {**{key: value for key, value in metadata.items() if key != "task"},
            "original_task": "textvqa",
            "data_manifest_path": metadata["manifest_path"],
            "data_manifest_sha256": metadata["manifest_sha256"],
            "data_manifest_file_sha256": metadata["manifest_file_sha256"],
            "questions": len(rows), "unique_images": len({r["image_id"] for r in rows})}
    if task == "pope":
        rows, images, root, metadata = frozen_pope_data(data_root or config_path(config, "pope_data_root"))
        return rows, images, root, {**metadata,
            "data_manifest_path": str(root / "manifest.json"),
            "data_manifest_sha256": metadata["pope_manifest_sha256"],
            "data_manifest_file_sha256": metadata["pope_manifest_file_sha256"]}
    raise ValueError("unknown task")


def suffix(task):
    return TEXT_SUFFIX if task == "text" else PROMPT_SUFFIX


def inference_row(row, task):
    return {**{k: row[k] for k in ("question_id", "image_id", "question")},
        "prompt_suffix": suffix(task)}


def evaluate_batch(model, processor, chunk, images, image_root, device, task, check_logits=False):
    # The frozen generator strips reference fields and stores actual IDs/pixels/grid.
    # Calling it with sanitized rows and the requested suffix is deliberately local
    # rather than changing the frozen Text/POPE module globals.
    import torch
    begun = time.monotonic()
    samples = [encode(processor, inference_row(row, task), image_root / images[row["image_id"]]["path"], False)
        for row in chunk]
    if any("labels" in item for item in samples):
        raise ValueError("generation input unexpectedly contains labels")
    details = [frozen_text.input_details(item, row, images, processor) for item, row in zip(samples, chunk)]
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
            raise ValueError("non-finite engineering-preflight logits")
    torch.cuda.synchronize(device)
    begun = time.monotonic()
    generated = model.generate(**batch, max_new_tokens=32, do_sample=False, use_cache=True)
    torch.cuda.synchronize(device)
    elapsed = time.monotonic() - begun
    padded = int(batch["input_ids"].shape[1])
    answers = []
    for index, row in enumerate(chunk):
        tail = generated[index, padded:]
        ids = tail.tolist()
        answers.append({"prediction": processor.tokenizer.decode(tail, skip_special_tokens=True,
                clean_up_tokenization_spaces=False).strip(), "generated_ids": ids,
            "generated_ids_sha256": content_sha(ids), "pad_token_id": int(processor.tokenizer.pad_token_id),
            "generated_nonpad_tokens": int((tail != processor.tokenizer.pad_token_id).sum()),
            "generated_tail_slots": len(ids), **details[index],
            "padded_input_ids_sha256": content_sha(cpu_batch["input_ids"][index].tolist()),
            "padded_attention_mask_sha256": content_sha(cpu_batch["attention_mask"][index].tolist())})
    return answers, {"batch_size": len(chunk), "padded_input_tokens": padded,
        "batch_generation_seconds": elapsed, "batch_preprocessing_seconds": preprocessing,
        "batch_h2d_seconds": transfer, "finite_logits": finite,
        "batch_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "batch_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30}


def scoring_fields(prediction, row, task):
    if task == "text":
        return {"score": score_prediction(prediction, row, "textvqa")}
    parsed = official_prediction(prediction)
    return {"label": row["label"], "category": row["category"], "official_prediction": parsed,
        "score": float(parsed == row["label"]), "validity_diagnostic": diagnostics(prediction)}


def complete_cache(path, rows, images, lock, processor=None, *, repair=True):
    """Recover interrupted tail only; reject any completed-batch identity drift."""
    if not path.exists():
        return [], None
    lines = path.read_bytes().splitlines(keepends=True)
    repairs = []
    if lines and not lines[-1].endswith(b"\n"):
        if not repair:
            raise ValueError("completed predictions have interrupted tail")
        lines.pop()
        repairs.append("discarded interrupted final JSONL line")
    items = [json.loads(line) for line in lines]
    if len(items) > len(rows):
        raise ValueError("too many cached predictions")
    if items:
        start = (len(items) - 1) // 4 * 4
        if len(items) - start != min(4, len(rows) - start):
            if not repair:
                raise ValueError("completed predictions contain incomplete batch")
            items, lines = items[:start], lines[:start]
            repairs.append("discarded incomplete final batch")
    for index, (item, row) in enumerate(zip(items, rows)):
        expected = {"question_id": row["question_id"], "image_id": row["image_id"],
            "question_sha256": content_sha(row["question"]),
            "question_prompt_sha256": content_sha(row["question"] + suffix(lock["task"])),
            "image_sha256": images[row["image_id"]]["sha256"], "task": lock["task"],
            "condition": lock["condition"], "seed": lock["seed"], "input_lock_sha256": content_sha(lock),
            "source_binding_sha256": content_sha(lock["source_sha256"]),
            "adapter_files_sha256": lock["adapter_files_sha256"], "model_revision": lock["model_revision"],
            "batch_start": index // 4 * 4, "batch_size": min(4, len(rows) - index // 4 * 4),
            **scoring_fields(item["prediction"], row, lock["task"])}
        for key, value in expected.items():
            if item.get(key) != value:
                raise ValueError(f"cached prediction binding differs: {key}")
        validate_input_record(item)
        grid = item["image_grid_thw"]
        visual = math.prod(grid) // 4
        if (len(grid) != 3 or grid[0] != 1 or math.prod(grid) % 4 or visual > 1024
            or visual != item["visual_tokens"] or visual != item["counted_image_tokens"]
            or item["input_truncated"]):
            raise ValueError("actual visual tokens/truncation contract differs")
        for key in ("batch_generation_seconds", "batch_preprocessing_seconds", "batch_h2d_seconds",
            "batch_peak_allocated_gib", "batch_peak_reserved_gib"):
            if not math.isfinite(item[key]) or item[key] < 0:
                raise ValueError("invalid resource observation")
        if processor is not None:
            decoded = processor.tokenizer.decode(item["generated_ids"], skip_special_tokens=True,
                clean_up_tokenization_spaces=False).strip()
            if decoded != item["prediction"] or item["pad_token_id"] != processor.tokenizer.pad_token_id:
                raise ValueError("actual generated IDs differ from decoded prediction")
    if repairs:
        with path.open("wb") as stream:
            stream.write(b"".join(lines))
            stream.flush()
            os.fsync(stream.fileno())
    return items, repairs or None


def summary(items, task):
    if task == "text":
        return {"score_percent": 100 * sum(r["score"] for r in items) / len(items),
            "metric": "original ten-reference leave-one-out TextVQA consensus percent"}
    values = metrics(items)
    return {"score_percent": 100 * values["overall"]["accuracy"], "metrics": values,
        "metric": "pinned official POPE-parser accuracy percent; strict parsing separate"}


def directory_for(root, condition, seed, task):
    return root / "runs" / condition / f"seed{seed}" / task


def run_preflight(model, processor, args, lock, rows, images, image_root, device):
    selected = stress_rows(rows, images, processor)
    directory = args.output_root / "preflight" / args.condition / f"seed{args.seed}" / args.task
    prelock = {**lock, "scope": "engineering finite-logit/input check only; no scored labels",
        "question_ids": [r["question_id"] for r in selected]}
    lock_file(directory / "INPUT_LOCK.json", prelock)
    path = directory / "PREFLIGHT.json"
    if path.exists():
        saved = read_json(path)
        if saved.get("input_lock_sha256") != content_sha(prelock) or saved.get("status") != "passed":
            raise ValueError("preflight binding differs")
        return saved
    import torch
    records, batches = [], []
    with torch.inference_mode():
        for begin in range(0, len(selected), 4):
            chunk = selected[begin:begin + 4]
            answers, resource = evaluate_batch(model, processor, chunk, images, image_root, device, args.task, True)
            records.extend({"question_id": r["question_id"], "image_id": r["image_id"], **answer}
                for r, answer in zip(chunk, answers))
            batches.append(resource)
    peak = max(max(batch["batch_peak_allocated_gib"], batch["batch_peak_reserved_gib"]) for batch in batches)
    result = {"status": "passed" if peak <= 65 else "failed", "input_lock_sha256": content_sha(prelock),
        "completed_at": now(), "records": records, "batches": batches,
        "memory_cap_gib": 65, "memory_cap_passed": peak <= 65}
    if peak > 65:
        result["error"] = "preflight_memory_cap_exceeded"
    write_json(path, result)
    if peak > 65:
        raise RuntimeError("preflight_memory_cap_exceeded")
    return result


def run_cell(model, processor, args, lock, rows, images, image_root, device, update):
    directory = directory_for(args.output_root, args.condition, args.seed, args.task)
    lock_file(directory / "INPUT_LOCK.json", lock)
    path, result_path = directory / "predictions.jsonl", directory / "EVALUATION.json"
    items, repaired = complete_cache(path, rows, images, lock, processor, repair=not result_path.exists())
    if result_path.exists():
        saved = read_json(result_path)
        if (len(items) != len(rows) or saved.get("predictions_sha256") != sha_file(path)
            or saved.get("input_lock_sha256") != content_sha(lock)
            or any(saved.get(key) != value for key, value in summary(items, args.task).items())):
            raise ValueError("completed result binding/score differs")
        update(len(items), "completed")
        return saved
    import torch
    start, begun = len(items), time.monotonic()
    session = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    with path.open("a") as stream, torch.inference_mode():
        for begin in range(start, len(rows), 4):
            chunk = rows[begin:begin + 4]
            answers, resource = evaluate_batch(model, processor, chunk, images, image_root, device, args.task)
            batch = []
            for row, answer in zip(chunk, answers):
                item = {"question_id": row["question_id"], "image_id": row["image_id"],
                    "question_sha256": content_sha(row["question"]),
                    "question_prompt_sha256": content_sha(row["question"] + suffix(args.task)),
                    "image_sha256": images[row["image_id"]]["sha256"], **answer,
                    **scoring_fields(answer["prediction"], row, args.task), "task": args.task,
                    "condition": args.condition, "seed": args.seed, "model_revision": lock["model_revision"],
                    "adapter_files_sha256": lock["adapter_files_sha256"],
                    "source_binding_sha256": content_sha(lock["source_sha256"]),
                    "input_lock_sha256": content_sha(lock), "batch_start": begin,
                    "session_id": session, "timing_warmup": begin == start,
                    "label_use": "references/labels used only after label-free generation",
                    **{k: v for k, v in resource.items() if k != "finite_logits"}}
                validate_input_record(item)
                batch.append(item)
            stream.write("".join(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n" for item in batch))
            stream.flush()
            os.fsync(stream.fileno())
            items.extend(batch)
            elapsed = time.monotonic() - begun
            write_json(directory / "PROGRESS.json", {"status": "running", "updated_at": now(),
                "completed_questions": len(items), "total_questions": len(rows), "session_id": session,
                "session_elapsed_seconds": elapsed, "last_batch": resource,
                "remaining_estimate_seconds": (len(rows) - len(items)) * elapsed / (len(items) - start)})
            update(len(items), "running")
            if begin == start or ((begin - start) // 4 + 1) % 25 == 0:
                print("GENERATED", args.task, args.condition, args.seed, len(items), "/", len(rows), flush=True)
    batches = {(r["session_id"], r["batch_start"]): r for r in items}
    timed = [r for r in batches.values() if not r["timing_warmup"]]
    result = {"schema_version": "rarevlm.object_presence.eval_cell.v1", "status": "completed",
        "completed_at": now(), "condition": args.condition, "seed": args.seed, "task": args.task,
        "input_lock_sha256": content_sha(lock), "predictions_sha256": sha_file(path),
        "questions": len(items), "unique_images": len({r["image_id"] for r in items}),
        **summary(items, args.task), "session_id": session, "resumed_questions": start,
        "cache_repair": repaired, "current_session_wall_seconds": time.monotonic() - begun,
        "resources": {"timed_questions": sum(r["batch_size"] for r in timed),
            "model_generate_seconds": sum(r["batch_generation_seconds"] for r in timed),
            "preprocessing_h2d_generation_seconds": sum(r["batch_preprocessing_seconds"] + r["batch_h2d_seconds"]
                + r["batch_generation_seconds"] for r in timed),
            "peak_allocated_gib": max(r["batch_peak_allocated_gib"] for r in batches.values()),
            "peak_reserved_gib": max(r["batch_peak_reserved_gib"] for r in batches.values()),
            "scope": "cached VLM stage; hashing included; loading/validation/preflight/writer separate; no FP32 or serving-speed claim"},
        "scope": "fixed Final checkpoint; whole predeclared split; no benchmark-based tuning"}
    write_json(result_path, result)
    update(len(items), "completed")
    print("COMPLETE", args.task, args.condition, args.seed, result["score_percent"], flush=True)
    return result


def read_cell(root, condition, seed, task, config_path_value, config, *, processor=None):
    directory = directory_for(root, condition, seed, task)
    path = directory / "EVALUATION.json"
    if not path.exists():
        if condition == "initial" and seed == 42 and task in ("text", "pope") and config.get("reference_study_root"):
            return historical_initial(root, task, config_path_value, config)
        return {"complete": False, "condition": condition, "seed": seed, "task": task, "path": str(directory)}
    lock, result = read_json(directory / "INPUT_LOCK.json"), read_json(path)
    rows, images, image_root, metadata = load_data(config, task)
    for key, value in {**FIXED, **metadata, "condition": condition, "seed": seed, "task": task,
        "study_config_sha256": sha_file(config_path_value), "study_config_content_sha256": content_sha(config),
        "source_sha256": source_hashes(), "prompt_suffix": suffix(task)}.items():
        if lock.get(key) != value:
            raise ValueError(f"completed-cell fixed binding differs: {key}")
    files, binding = adapter_binding(config, condition, seed, Path(lock["adapter_dir"]), config_path(config, "model_root"))
    if files != lock["adapter_files_sha256"] or binding != lock["training_binding"]:
        raise ValueError("completed-cell adapter/training bytes changed")
    predictions = directory / "predictions.jsonl"
    records, repairs = complete_cache(predictions, rows, images, lock, processor, repair=False)
    if (len(records) != len(rows) or repairs or result.get("status") != "completed"
        or result.get("predictions_sha256") != sha_file(predictions)
        or result.get("input_lock_sha256") != content_sha(lock)
        or any(result.get(key) != value for key, value in summary(records, task).items())):
        raise ValueError("completed-cell count/score/hash differs")
    return {"complete": True, "condition": condition, "seed": seed, "task": task,
        "rows": records, "lock": lock, "evaluation": result,
        "provenance": {"predictions_file": str(predictions), "predictions_sha256": sha_file(predictions),
            "evaluation_file": str(path), "evaluation_sha256": sha_file(path),
            "input_lock_file": str(directory / "INPUT_LOCK.json"),
            "input_lock_sha256": sha_file(directory / "INPUT_LOCK.json"),
            "training_result_sha256": binding["sha256"]}}


def historical_initial(root, task, config_file, config):
    """Read original bytes through unchanged old validators; never recast a run."""
    from vlm_posttrain_eval.analysis import read_cell as old_text_cell
    from vlm_posttrain_pope.evaluate import read_cell as old_pope_cell
    old_root = config_path(config, "reference_study_root")
    old_config_file = config_path(config, "reference_study_config", "configs/vlm_public_posttrain_diagnostic_v1.json")
    old_config = read_json(old_config_file)
    study_file = old_root / "STUDY_INPUT_LOCK.json"
    study = read_json(study_file)
    if study.get("config_sha256") != sha_file(old_config_file):
        raise ValueError("historical study config changed")
    for name, digest in study["sources"].items():
        if sha_file(SOURCE_ROOT / name) != digest:
            raise ValueError(f"historical study source changed: {name}")
    for name, digest in study["data_sha256"].items():
        if sha_file(private_path(name)) != digest:
            raise ValueError(f"historical study data changed: {name}")
    for name in ("model_root", "asset_root", "initial_adapter", "reference_config", "pope_data_root"):
        if config_path(config, name) != config_path(old_config, name):
            raise ValueError(f"historical reference scientific path differs: {name}")
    model_root = config_path(config, "model_root")
    reference = read_json(config_path(config, "reference_config"))
    metadata = model_metadata_binding(model_root, reference)
    for name, digest in metadata["model_weight_provenance_sha256"].items():
        if sha_file(model_root / name) != digest:
            raise ValueError(f"historical reference original model byte changed: {name}")
    processor = processor_for(model_root, reference, 1024)
    rows, images, image_root, data_metadata = load_data(config, task)
    if task == "text":
        cell = old_text_cell(old_root, "initial", 42, rows, images, old_config_file, old_config, processor)
        old_task = "dev"
    else:
        cell = old_pope_cell(old_root, "initial", 42, old_config_file, old_config)
        old_task = "pope"
    if not cell["complete"]:
        raise ValueError("historical initial reference is incomplete")
    old_directory = old_root / "runs" / "initial" / "seed42" / old_task
    old_lock = cell["lock"]
    files, binding = adapter_binding(config, "initial", 42, initial_adapter_path(config), model_root)
    if (old_lock["adapter_files_sha256"] != files or old_lock["training_binding"]["sha256"] != binding["sha256"]):
        raise ValueError("historical reference is not this exact initial SFT42 checkpoint")
    for key, expected in {**FIXED, "prompt_suffix": suffix(task),
        "reference_config_sha256": sha_file(config_path(config, "reference_config")),
        **{key: metadata[key] for key in ("model_config_sha256", "model_provenance_sha256",
            "model_revision", "model_weight_provenance_sha256", "processor_and_metadata_sha256")}}.items():
        if old_lock.get(key) != expected:
            raise ValueError(f"historical reference scientific setting differs: {key}")
    original_lock_file = old_directory / "INPUT_LOCK.json"
    provenance = {"predictions_file": str(old_directory / "predictions.jsonl"),
        "predictions_sha256": sha_file(old_directory / "predictions.jsonl"),
        "evaluation_file": str(old_directory / "EVALUATION.json"),
        "evaluation_sha256": sha_file(old_directory / "EVALUATION.json"),
        "input_lock_file": str(original_lock_file), "input_lock_sha256": sha_file(original_lock_file),
        "training_result_sha256": binding["sha256"], "historical_reference": True}
    declaration = {"schema_version": "rarevlm.object_presence.historical_reference.v1",
        "task": task, "condition": "initial", "seed": 42, "new_generation_questions": 0,
        "new_study_config_sha256": sha_file(config_file),
        "historical_study_root": str(old_root), "historical_config_path": str(old_config_file),
        "historical_config_sha256": sha_file(old_config_file),
        "historical_study_lock_path": str(study_file), "historical_study_lock_sha256": sha_file(study_file),
        "all_historical_source_data_and_current_original_model_bytes_rehashed": True,
        "original_source_and_hardware_lock_retained": True, "source_data": provenance,
        "scope": "explicit historical initial reference; no new run/prediction/source/hardware claim; actual inputs compared with each new candidate"}
    lock_file(directory_for(root, "initial", 42, task) / "HISTORICAL_REFERENCE.json", declaration)
    # comparison_metadata supplies only dataset aliases. The original source,
    # config and hardware stay in the raw old lock and are never overwritten.
    return {**cell, "task": task, "historical_reference": declaration,
        "comparison_metadata": data_metadata, "provenance": provenance}


def screen_admission(path, config_file, condition, seed, training):
    if path is None:
        raise ValueError("POPE requires the completed predeclared object+Text screen")
    path = private_path(path)
    value = read_json(path)
    if (value.get("schema_version") != "rarevlm.object_presence.screen.v1"
        or value.get("status") != "complete" or value.get("config_sha256") != sha_file(config_file)):
        raise ValueError("POPE screen schema/config/status differs")
    passed = value.get("eligible_candidates", [])
    selected = value.get("selected_candidate")
    if (condition == "initial" and not passed) or (condition != "initial" and condition not in passed):
        raise ValueError("POPE blocked because candidate did not pass object+Text screen")
    if (value.get("seeds") != [0, 42, 200] or value.get("stage") != "replicated_internal_selection"
        or selected not in passed or (condition != "initial" and condition != selected)):
        raise ValueError("POPE requires the one internally selected candidate after all three order seeds")
    if value.get("source_sha256") != source_hashes():
        raise ValueError("POPE screen scientific source changed")
    key = f"{condition}_{42 if condition == 'initial' else seed}_text"
    provenance = value.get("source_data", {}).get(key)
    if not provenance or provenance.get("training_result_sha256") != training["sha256"]:
        raise ValueError("POPE adapter differs from screen Text checkpoint")
    verify_source_data(value.get("source_data", {}))
    return {"screen_analysis_path": str(path), "screen_analysis_sha256": sha_file(path),
        "eligible_candidates": passed, "selected_candidate": selected,
        "screen_checkpoint_binding_verified": True}


def verify_source_data(source_data):
    """Rebind every scored file used for screening without altering its source."""
    if not source_data:
        raise ValueError("screen source-data provenance missing")
    for source in source_data.values():
        for kind in ("predictions", "evaluation", "input_lock"):
            target = private_path(source[f"{kind}_file"])
            if sha_file(target) != source[f"{kind}_sha256"]:
                raise ValueError(f"screen source file changed: {kind}")


def gpu_admission(path, config_file, condition, seed, task, hardware):
    path = private_path(path)
    value = read_json(path)
    expected = {"schema_version": "rarevlm.object_presence.gpu_admission.v1", "status": "admitted",
        "condition": condition, "seed": seed, "task": task,
        "gpu_uuid": hardware["uuid"], "config_sha256": sha_file(config_file)}
    for key, item in expected.items():
        if value.get(key) != item:
            raise ValueError(f"external GPU admission differs: {key}")
    return {"gpu_admission_path": str(path), "gpu_admission_sha256": sha_file(path),
        "gpu_admission_verified": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "adapter-dir", "output-root", "admission"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 42, 200), default=42)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--screen-analysis", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    for name in ("config", "adapter_dir", "output_root"):
        setattr(args, name, private_path(getattr(args, name)))
    config = read_json(args.config)
    model_root = config_path(config, "model_root")
    reference_path = config_path(config, "reference_config", "configs/vlm_deep_v2.json")
    reference = read_json(reference_path)
    validate_config(config, reference, args.batch_size)
    if args.data_root:
        expected = config_path(config, "object_data_root" if args.task == "object" else "pope_data_root")
        if private_path(args.data_root) != expected:
            raise ValueError("data-root override differs from predeclared dataset")
    files, training = adapter_binding(config, args.condition, args.seed, args.adapter_dir, model_root)
    admission = screen_admission(args.screen_analysis, args.config, args.condition, args.seed, training) if args.task == "pope" else {}
    rows, images, image_root, metadata = load_data(config, args.task, args.data_root)
    model_metadata = model_metadata_binding(model_root, reference)
    for name, expected in model_metadata["model_weight_provenance_sha256"].items():
        if sha_file(model_root / name) != expected:
            raise ValueError(f"current original model weight differs: {name}")
    import torch
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or not torch.cuda.is_available():
        raise ValueError("one external-queue-admitted CUDA worker required")
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
    queue_binding = gpu_admission(args.admission, args.config, args.condition, args.seed, args.task, hardware)
    lock = {"schema_version": "rarevlm.object_presence.eval_input.v1", **FIXED,
        "condition": args.condition, "seed": args.seed, "task": args.task,
        "study_config_sha256": sha_file(args.config), "study_config_content_sha256": content_sha(config),
        "reference_config_sha256": sha_file(reference_path), **model_metadata, **metadata,
        "current_original_weight_bytes_verified": True, "adapter_dir": str(args.adapter_dir),
        "adapter_files_sha256": files, "training_binding": training, "source_sha256": source_hashes(),
        "packages": package_versions(), "hardware": hardware, "prompt_suffix": suffix(args.task),
        "head_last_position_only": True, "explicit_lora_bf16_conversion": True,
        "input": "image + original question + fixed task suffix; no references/target/evidence/OCR",
        "statistical_unit": "image clusters; POPE three categories share 500 images",
        "scope": "predeclared full split; public/exposed POPE not blind or pretrain-disjoint", **admission, **queue_binding}
    status_path = directory_for(args.output_root, args.condition, args.seed, args.task) / "STATUS.json"
    status = {"condition": args.condition, "seed": args.seed, "task": args.task,
        "phase": "starting", "started_at": now(), "hardware": hardware, "total_questions": len(rows)}
    def update(count=None, state=None):
        if count is not None:
            status["completed_questions"] = count
        if state is not None:
            status["cell_status"] = state
        status["updated_at"] = now()
        write_json(status_path, status)
    model, hook = None, None
    try:
        status["phase"] = "loading_model"
        update()
        begun = time.monotonic()
        model, dtypes = load_path(model_root, reference, device, "original_bf16", args.adapter_dir)
        status["model_load_seconds"] = time.monotonic() - begun
        lock["actual_parameter_dtypes"] = dtypes
        if int(model.config.image_token_id) != 151655:
            raise ValueError("image token ID differs")
        hook = model.get_output_embeddings().register_forward_pre_hook(lambda module, inputs: (inputs[0][:, -1:, :],))
        processor = processor_for(model_root, reference, 1024)
        lock_file(directory_for(args.output_root, args.condition, args.seed, args.task) / "INPUT_LOCK.json", lock)
        status["phase"] = "validating_images"
        update()
        status["image_validation"] = verify_dev_images(rows, images, image_root)
        status["phase"] = "preflight"
        update()
        run_preflight(model, processor, args, lock, rows, images, image_root, device)
        if not args.preflight_only:
            status["phase"] = "evaluating"
            update()
            run_cell(model, processor, args, lock, rows, images, image_root, device, update)
        status.update(phase="preflight_completed" if args.preflight_only else "completed", completed_at=now())
        update()
    except Exception as error:
        status.update(phase="failed", error_type=type(error).__name__, error=str(error))
        update()
        raise
    finally:
        if hook is not None:
            hook.remove()
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
