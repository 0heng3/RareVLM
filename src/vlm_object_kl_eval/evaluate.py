"""New KL final-adapter runner; frozen low-level generation, originalBF16/batch4."""
from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import time

from vlm_joint.evaluate import model_metadata_binding, visible_gpu_binding
from vlm_object_eval import evaluate as generation
from vlm_precision.evaluate import load_path
from vlm_probe.evaluate import content_sha, lock_file, now, package_versions, processor_for, read_json, sha_file, verify_dev_images, write_json
from .contracts import (CANDIDATE, CONDITIONS, FIXED, TASKS, adapter_binding, external_admission,
    load_data, path, source_hashes)
from .read import directory, phase_admission


def validate_config(config):
    if config.get("candidates", [CANDIDATE]) != [CANDIDATE]:
        raise ValueError("exactly one new object_kl candidate is preregistered")
    for key, value in FIXED.items():
        if config.get("evaluation", {}).get(key, value) != value:
            raise ValueError(f"fixed BF16 generation setting differs: {key}")
    for key, value in (("bootstrap_replicates", 6000), ("bootstrap_seed", 20261015), ("one_sided_confidence", .9875)):
        if config.get("analysis", {}).get(key, value) != value:
            raise ValueError(f"four-endpoint family differs: {key}")


def progression(args, config, binding):
    result = {}
    if args.task == "external":
        result.update(external_admission(config))
        if args.condition == "initial":
            return {**result, "initial_external_baseline_only": True,
                "scope_of_early_external": "initial only, parallel pilot permitted; never read by internal selection"}
        if args.screen_analysis is None:
            raise ValueError("external generation requires completed internal three-seed screen")
        artifact = phase_admission(args.screen_analysis, args.config, "screen")
        source = artifact["source_data"][f"{args.condition}_{42 if args.condition == 'initial' else args.seed}_text"]
    elif args.task == "pope":
        if args.external_analysis is None:
            raise ValueError("POPE requires completed external three-seed confirmation")
        artifact = phase_admission(args.external_analysis, args.config, "external")
        source = artifact["source_data"][f"{args.condition}_{42 if args.condition == 'initial' else args.seed}_external"]
    else:
        return result
    if source["training_result_sha256"] != binding["sha256"]:
        raise ValueError("phase progression checkpoint differs")
    file = args.screen_analysis if args.task == "external" else args.external_analysis
    result.update(prior_phase_analysis_file=str(file), prior_phase_analysis_sha256=sha_file(file))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "adapter-dir", "output-root", "admission"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 42, 200), required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--screen-analysis", type=Path)
    parser.add_argument("--external-analysis", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config)
    validate_config(config)
    external_admission(config)
    if args.output_root.resolve() != path(config, "output_root"):
        raise ValueError("output root differs from predeclared KL study")
    model_root, reference_file = path(config, "model_root"), path(config, "reference_config")
    reference = read_json(reference_file)
    import torch
    torch.set_num_threads(2)
    files, binding = adapter_binding(config, args.condition, args.seed, args.adapter_dir, model_root, config_file=args.config)
    progress_binding = progression(args, config, binding)
    rows, images, image_root, metadata = load_data(config, args.task)
    model_metadata = model_metadata_binding(model_root, reference)
    for name, digest in model_metadata["model_weight_provenance_sha256"].items():
        if sha_file(model_root / name) != digest:
            raise ValueError("current original base weight bytes differ")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or not torch.cuda.is_available():
        raise ValueError("single external-queue-admitted CUDA worker required")
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda", 0)
    props = torch.cuda.get_device_properties(device)
    if "A100" not in props.name or props.total_memory < 70 * 2**30:
        raise ValueError("admitted A10080GB required")
    hardware = {**visible_gpu_binding(), "name": props.name, "total_memory_bytes": props.total_memory,
        "compute_capability": [props.major, props.minor], "torch_cuda_version": torch.version.cuda}
    admission = read_json(args.admission)
    for key, expected in {"schema_version": "rarevlm.object_kl.gpu_admission.v1", "status": "admitted",
        "condition": args.condition, "seed": args.seed, "task": args.task, "gpu_uuid": hardware["uuid"],
        "config_sha256": sha_file(args.config)}.items():
        if admission.get(key) != expected:
            raise ValueError(f"new KL GPU admission differs: {key}")
    lock = {"schema_version": "rarevlm.object_kl.eval_input.v1", **FIXED, **model_metadata, **metadata,
        "condition": args.condition, "seed": args.seed, "task": args.task,
        "study_config_sha256": sha_file(args.config), "study_config_content_sha256": content_sha(config),
        "reference_config_sha256": sha_file(reference_file), "source_sha256": source_hashes(),
        "packages": package_versions(), "hardware": hardware, "adapter_dir": str(args.adapter_dir.resolve()),
        "adapter_files_sha256": files, "training_binding": binding, "prompt_suffix": generation.suffix(args.task),
        "current_original_weight_bytes_verified": True, "head_last_position_only": True,
        "explicit_lora_bf16_conversion": True, "gpu_admission_sha256": sha_file(args.admission),
        "input": "original image+question+fixed task suffix; no reference/label/target/OCR", **progress_binding}
    place = directory(args.output_root, args.condition, args.seed, args.task)
    status = {"condition": args.condition, "seed": args.seed, "task": args.task, "phase": "loading_model", "started_at": now()}
    def update(count=None, state=None):
        if count is not None: status["completed_questions"] = count
        if state is not None: status["cell_status"] = state
        status["updated_at"] = now()
        write_json(place / "STATUS.json", status)
    model, hook = None, None
    try:
        update()
        started = time.monotonic()
        model, dtypes = load_path(model_root, reference, device, "original_bf16", args.adapter_dir)
        status["model_load_seconds"] = time.monotonic() - started
        lock["actual_parameter_dtypes"] = dtypes
        if int(model.config.image_token_id) != 151655:
            raise ValueError("image token ID differs")
        hook = model.get_output_embeddings().register_forward_pre_hook(lambda module, inputs: (inputs[0][:, -1:, :],))
        processor = processor_for(model_root, reference, 1024)
        lock_file(place / "INPUT_LOCK.json", lock)
        status["phase"] = "validating_images"; update()
        status["image_validation"] = verify_dev_images(rows, images, image_root)
        status["phase"] = "engineering_preflight"; update()
        generation.run_preflight(model, processor, args, lock, rows, images, image_root, device)
        if not args.preflight_only:
            status["phase"] = "evaluating"; update()
            result = generation.run_cell(model, processor, args, lock, rows, images, image_root, device, update)
            result.update(schema_version="rarevlm.object_kl.eval_cell.v1", scope="fixed Final KL checkpoint; predeclared full split; no new selection on confirmation tasks")
            write_json(place / "EVALUATION.json", result)
        status.update(phase="preflight_completed" if args.preflight_only else "completed", completed_at=now()); update()
    except BaseException as error:
        status.update(phase="failed", error_type=type(error).__name__, error=str(error)); update()
        raise
    finally:
        if hook is not None: hook.remove()
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
