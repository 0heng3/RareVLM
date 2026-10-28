"""New KL/data/cache identities; historical runs keep their original bytes."""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT
from vlm_object_eval import evaluate as generation
from vlm_posttrain_eval import evaluate as public_text
from vlm_probe.evaluate import content_sha, private_path, read_json, sha_file

CANDIDATE = "object_kl"
CONDITIONS = ("initial", CANDIDATE)
TASKS = ("object", "text", "external", "pope")
FIXED = dict(generation.FIXED)


def path(config, name, default=None):
    value = config.get(name, default)
    if value is None or not Path(value).is_absolute():
        raise ValueError(f"explicit absolute path required: {name}")
    return private_path(value)


def source_hashes():
    from vlm_object_kl.train import source_hashes as training_sources
    result = generation.source_hashes()
    result.update(training_sources())
    for directory in ("vlm_object_kl_eval", "vlm_object_kl_study", "vlm_object_external_data"):
        for file in sorted((SOURCE_ROOT / directory).glob("*.py")):
            result[str(file.relative_to(SOURCE_ROOT))] = sha_file(file)
    for name in ("research_runtime/queue.py", "vlm_joint/pipeline.py", "vlm_precision/pipeline.py"):
        result[name] = sha_file(SOURCE_ROOT / name)
    launcher = PROJECT_ROOT / "tools/run_object_kl_study.py"
    if launcher.exists():
        result["tools/run_object_kl_study.py"] = sha_file(launcher)
    return dict(sorted(result.items()))


def canonical_rows(rows, task):
    if task != "text":
        return rows, {"normalized_categories": 0, "all_fields_unchanged": True}
    copied = deepcopy(rows)
    counts = {"missing": 0, "none": 0, "empty_string": 0}
    for before, after in zip(rows, copied):
        if "category" not in before:
            counts["missing"] += 1
        elif before["category"] is None:
            counts["none"] += 1
        elif type(before["category"]) is str and before["category"] == "":
            counts["empty_string"] += 1
        else:
            raise ValueError("nonempty Text category cannot be canonicalized")
        after.pop("category", None)
    original = [{key: value for key, value in row.items() if key != "category"} for row in rows]
    if original != copied:
        raise ValueError("Text canonicalization changed a noncategory field")
    return copied, {"aliases": counts, "normalized_categories": counts["none"] + counts["empty_string"],
        "noncategory_fields_sha256_before": content_sha(original),
        "noncategory_fields_sha256_after": content_sha(copied), "scores_and_inputs_unchanged": True}


def external_admission(config):
    root = path(config, "external_data_root")
    files = {"external_manifest_sha256": path(config, "external_manifest", str(root / "MANIFEST.json")),
        "external_class_balanced_manifest_sha256": path(config, "external_class_balanced_manifest", str(root / "CLASS_BALANCED_MANIFEST.json")),
        "external_dataset_protocol_sha256": path(config, "external_dataset_protocol"),
        "frozen_audit_sha256": path(config, "external_frozen_audit", str(root / "semantic_audit/FROZEN_64.json")),
        "semantic_review_sha256": path(config, "external_semantic_review"),
        "cpu_completion_sha256": path(config, "external_cpu_completion", str(root / "COMPLETION.json"))}
    admission_path = path(config, "evaluation_data_admission")
    admission = read_json(admission_path)
    if (admission.get("schema_version") != "rarevlm.object_external.evaluation_admission.v1"
        or admission.get("status") != "admitted" or admission.get("quality_gate_passed") is not True
        or admission.get("scope") != "object_external_confirmation"
        or admission.get("automatic_diagnostic_bypass_allowed") is not False):
        raise ValueError("external evaluation requires root factual-quality admission; no diagnostic bypass")
    for key, file in files.items():
        if admission.get(key) != sha_file(file):
            raise ValueError(f"external admission byte binding differs: {key}")
    review = read_json(files["semantic_review_sha256"])
    full_sha = sha_file(files["external_manifest_sha256"])
    if (review.get("schema_version") != "rarevlm.object_external.semantic_review.v1"
        or review.get("gate", {}).get("passed") is not True
        or review.get("source_audit_sha256") != sha_file(files["frozen_audit_sha256"])
        or review.get("external_manifest_sha256") != full_sha
        or ("source_full_manifest_sha256" in review and review["source_full_manifest_sha256"] != full_sha)):
        raise ValueError("external semantic review is not bound to this new full dataset/audit")
    completion = read_json(files["cpu_completion_sha256"])
    if (completion.get("complete") is not True or completion.get("data_only") is not True
        or completion.get("optimizer_updates") != 0 or completion.get("training_allowed") is not False):
        raise ValueError("full external CPU preparation must complete before any prediction")
    encoding = completion.get("encoding", {})
    n = len(read_json(files["external_manifest_sha256"])["rows"])
    if (encoding.get("model_forward_or_gpu_used") is not False or encoding.get("questions") != n
        or encoding.get("fresh_actual_cpu_encodes") != n or encoding.get("old_encoding_records_reused") != 0
        or encoding.get("input_ids_labels_and_pixel_hash_saved") is not True or encoding.get("truncation") is not False
        or encoding.get("journal_sha256") != sha_file(root / "ENCODING_CHECKS.jsonl")):
        raise ValueError("external full fresh CPU actual-input verification is incomplete")
    return {"evaluation_data_admission_path": str(admission_path),
        "evaluation_data_admission_sha256": sha_file(admission_path),
        "external_admission_files_sha256": {str(file): sha_file(file) for file in files.values()}}


def external_data(config, *, balanced=False):
    root = path(config, "external_data_root")
    file = path(config, "external_class_balanced_manifest" if balanced else "external_manifest",
        str(root / ("CLASS_BALANCED_MANIFEST.json" if balanced else "MANIFEST.json")))
    manifest = read_json(file)
    if (manifest.get("schema_version") != "rarevlm.object_external.manifest.v1"
        or manifest.get("split") != ("external_class_balanced" if balanced else "external_confirmation")
        or manifest.get("evaluation_requires_external_admission") is not True
        or manifest.get("training_allowed") is not False
        or manifest.get("prompt_suffix") != generation.PROMPT_SUFFIX
        or manifest.get("protocol_sha256") != sha_file(path(config, "external_dataset_protocol"))
        or content_sha({key: value for key, value in manifest.items() if key != "manifest_sha256"}) != manifest.get("manifest_sha256")):
        raise ValueError("external full/balanced manifest schema/protocol/digest differs")
    source = read_json(root / "SOURCE_LOCK.json")
    if manifest.get("source_lock_sha256") != source.get("source_lock_sha256"):
        raise ValueError("external source lineage differs")
    images, rows = manifest["images"], []
    for raw in manifest["rows"]:
        if (not str(raw["question_id"]).startswith("object_external_v1:")
            or raw.get("target") not in ("yes", "no") or raw.get("prompt_suffix") != generation.PROMPT_SUFFIX
            or raw["image_id"] not in images):
            raise ValueError("external question namespace/label/prompt differs")
        image = images[raw["image_id"]]
        if (private_path(raw["image_path"]) != private_path(image["path"])
            or raw["image_sha256"] != image["sha256"] or raw["pixel_sha256"] != image["pixel_sha256"]
            or image.get("native_original_jpeg") is not True or image.get("source_image_resized_or_cropped") is not False):
            raise ValueError("external original JPEG/image byte identity differs")
        rows.append({**raw, "label": raw["target"], "category": raw["category_name"]})
    if (not rows or len({row["question_id"] for row in rows}) != len(rows)
        or {row["image_id"] for row in rows} != set(images)):
        raise ValueError("external full unique coverage required")
    return rows, images, file.parent, {"data_manifest_path": str(file),
        "data_manifest_sha256": manifest["manifest_sha256"], "data_manifest_file_sha256": sha_file(file),
        "data_manifest_schema_version": manifest["schema_version"], "questions": len(rows), "unique_images": len(images),
        "question_order_sha256": content_sha([row["question_id"] for row in rows]),
        "question_prompt_order_sha256": content_sha([content_sha(row["question"] + generation.PROMPT_SUFFIX) for row in rows]),
        "dataset_scope": "new public external confirmation; not blind/pretraining-disjoint; never training",
        "external_source_lock_file_sha256": sha_file(root / "SOURCE_LOCK.json")}


def load_data(config, task):
    if task == "external":
        return external_data(config)
    return generation.load_data(config, task)


def validate_teacher_cache(train, train_dir, config):
    from vlm_object_kl.cache import load_entry, verify_frozen_cache
    cache_dir = private_path(train["teacher_cache_directory"])
    if cache_dir != train_dir / "teacher_cache":
        raise ValueError("teacher cache must belong to this completed training")
    manifest_file = cache_dir / "MANIFEST.json"
    digest = sha_file(manifest_file)
    if (train.get("teacher_cache_manifest_sha256") != digest
        or train.get("teacher_cache_manifest_sha256_after") != digest
        or train.get("teacher_cache_unchanged") is not True):
        raise ValueError("fixed teacher cache before/after SHA differs")
    manifest = verify_frozen_cache(cache_dir, digest)
    binding = read_json(cache_dir / "INPUT_LOCK.json")
    if content_sha(binding) != train.get("teacher_cache_binding_sha256"):
        raise ValueError("teacher cache actual-input/teacher binding differs")
    for key, value in binding.items():
        if key in ("teacher_role", "replay_microbatch_order_sha256", "teacher_distribution", "teacher_forward_path"):
            continue
        if train.get(key) != value:
            raise ValueError(f"teacher training identity differs: {key}")
    if (binding.get("teacher_role") != "fixed_initial_before_optimizer"
        or manifest.get("teacher_parameters_sha256") != train["initial_adapter_tensor_sha256"]
        or manifest.get("frozen_parameters_sha256") != train["frozen_parameter_sha256_before"]
        or manifest.get("teacher_forward_questions") != train["actual_replay_exposures"]
        or manifest.get("initial_parameters_rng_buffers_rope_restored") is not True
        or train.get("teacher_cache_summary") != {key: value for key, value in manifest.items() if key != "entries"}):
        raise ValueError("teacher is not the complete fixed initial full-vocabulary cache")
    inputs_file = train_dir / "TRAINED_INPUT_CONTRACTS.json"
    if sha_file(inputs_file) != train["trained_input_contracts_file_sha256"]:
        raise ValueError("actual trained object/replay input contracts changed")
    inputs = read_json(inputs_file)
    for name in ("object", "replay"):
        if content_sha(inputs[name]) != train[f"{name}_input_contract_sha256"]:
            raise ValueError("actual trained per-item input contract differs")
    replay = inputs["replay"]
    count = 0
    model_config = read_json(path(config, "model_root") / "config.json")
    vocab = model_config.get("vocab_size", model_config.get("text_config", {}).get("vocab_size"))
    if type(vocab) is not int or vocab <= 1:
        raise ValueError("actual model vocabulary must be explicitly bound")
    for key, entry in sorted(manifest["entries"].items()):
        identity = entry["identity"]
        start = identity["update_index"] * 8 + identity["replay_microbatch_index"] * 4
        contracts = replay[start:start + len(identity["question_ids"]) ]
        if identity["input_contracts"] != contracts:
            raise ValueError("teacher cache and actual actor padded replay inputs differ")
        payload = load_entry(cache_dir / f"{key}.pt", identity, entry)
        if list(payload["logprobs"].shape) != entry["shape"] or entry["shape"][1] != vocab:
            raise ValueError("teacher cache is not complete full model vocabulary")
        count += len(contracts)
    if count != train["actual_replay_exposures"]:
        raise ValueError("teacher cache complete-batch exposure count differs")
    return {"manifest_path": str(manifest_file), "manifest_sha256": digest,
        "full_cache_files_and_tensors_verified": True, "actual_padded_actor_inputs_matched": True,
        "teacher_forward_questions": count, "vocabulary_size": vocab}


def validate_parameter_budget(train):
    names, dtypes = train.get("trainable_names", {}), train.get("dtypes", {})
    if (len(names) != 144 or sum(names.values()) != 1843200
        or any(".visual." in name or not any(part in name for part in (".q_proj.", ".v_proj."))
            or not any(part in name for part in ("lora_A", "lora_B")) for name in names)
        or dtypes.get("lora_elements_by_dtype") != {"float32": 1843200}
        or dtypes.get("lora_tensors_by_dtype") != {"float32": 144} or dtypes.get("params4bit_tensors") != 0):
        raise ValueError("KL must train exactly144 FP32 language q/v LoRA tensors/1843200 parameters")


def adapter_binding(config, condition, seed, adapter_dir, model_root, *, config_file):
    initial = path(config, "initial_adapter")
    initial_files, initial_binding = public_text.adapter_binding(config, "initial", 42, initial, model_root)
    adapter_dir = private_path(adapter_dir)
    if condition == "initial":
        if seed != 42 or adapter_dir != initial:
            raise ValueError("one shared fixed SFT42 initializer required")
        return initial_files, {**initial_binding, "shared_initial_checkpoint": True}
    if condition != CANDIDATE or seed not in (0, 42, 200) or adapter_dir == initial:
        raise ValueError("new object_kl condition/order seed/adapter required")
    files = generation.adapter_files(adapter_dir)
    train_dir, file = adapter_dir.parent, adapter_dir.parent / "TRAIN_RESULT.json"
    train = read_json(file)
    n = len(read_json(path(config, "train_manifest"))["rows"])
    updates = math.ceil(n / 16)
    expected = {"schema_version": "rarevlm.object_kl.train.v1", "condition": CANDIDATE, "seed": seed,
        "status": "completed", "config_sha256": content_sha(config), "config_file_sha256": sha_file(config_file),
        "adapter_files_sha256": files, "initial_adapter_files_sha256": initial_files,
        "initial_adapter_dir": str(initial), "trainable_parameters": 1843200,
        "training_microbatch_questions": 4, "training_accumulation": 4, "object_effective_batch": 16,
        "actual_object_exposures": n, "actual_replay_exposures": updates * 8, "actual_questions_seen": n + updates * 8,
        "planned_updates": updates, "completed_updates": updates, "skipped_updates": 0,
        "numerical_path": "original_bf16", "visual_token_budget": 1024, "lora_parameter_precision": "fp32",
        "frozen_parameters_unchanged": True, "kl_direction": "initial_to_actor", "kl_temperature": 1.,
        "teacher_logprob_dtype": "float32", "teacher_full_vocabulary": True, "replay_hard_ce_backward": False,
        "replay_lambda": 1., "train_manifest_file_sha256": sha_file(path(config, "train_manifest")),
        "model_provenance_sha256": sha_file(model_root.parent / "model_provenance.json")}
    for key, value in expected.items():
        if value is not None and train.get(key) != value:
            raise ValueError(f"completed KL training binding differs: {key}")
    validate_parameter_budget(train)
    old_dir = path(config, "historical_object_study_root") / "runs/object_replay/seed42/train"
    old = read_json(old_dir / "TRAIN_RESULT.json")
    if (train["initial_adapter_tensor_sha256"] != old["initial_adapter_tensor_sha256"]
        or train["trainable_names"] != old["trainable_names"]
        or train["frozen_parameter_sha256_before"] != old["frozen_parameter_sha256_before"]
        or train["frozen_parameter_sha256_after"] != old["frozen_parameter_sha256_after"]
        or train["train_manifest_file_sha256"] != old["train_manifest_file_sha256"]):
        raise ValueError("KL initializer/data/base differs from the fixed historical numerical path")
    if seed == 42:
        for key in ("object_order_sha256", "replay_order_sha256", "object_microbatch_order_sha256",
            "object_input_contract_sha256", "replay_input_contract_sha256"):
            if train.get(key) != old.get(key):
                raise ValueError(f"matched CEReplay/KL actual input contract differs: {key}")
    teacher = validate_teacher_cache(train, train_dir, config)
    return files, {"path": str(file), "sha256": sha_file(file), "condition": condition,
        "training_seed": seed, "initial_train_result_sha256": initial_binding["sha256"],
        "initial_adapter_files_sha256": initial_files, "initial_adapter_tensor_sha256": train["initial_adapter_tensor_sha256"],
        "teacher_cache": teacher, "scope": "fixed Final object KL; frozen initial teacher; not on-policy/RL"}
