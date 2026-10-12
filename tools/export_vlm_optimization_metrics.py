"""Export completed diagnostic aggregates and existing figures without predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_record(path: Path) -> dict:
    return {"file": str(path), "sha256": sha256(path)}


def budget_phase_timing(path: Path, expected: dict, prediction_sha256: str) -> dict:
    """Deduplicate already recorded batch timings; no generation or rescoring."""
    batches = {}
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            row = json.loads(line)
            key = (row["session_id"], row["batch_start"])
            batches.setdefault(key, row)
    assert digest.hexdigest() == prediction_sha256, path
    timed = [r for r in batches.values() if not r["timing_warmup"]]
    count = sum(r["batch_size"] for r in timed)
    generation = sum(r["batch_generation_seconds"] for r in timed)
    preprocessing = sum(r["batch_preprocessing_seconds"] for r in timed)
    h2d = sum(r["batch_h2d_seconds"] for r in timed)
    assert count == expected["timed_questions"]
    assert len(timed) == expected["timed_batches"]
    assert math.isclose(generation, expected["model_generate_seconds_excluding_warmup"], abs_tol=1e-6)
    return {
        "timed_questions": count,
        "timed_batches": len(timed),
        "generation_seconds_excluding_warmup": generation,
        "preprocessing_seconds_excluding_warmup": preprocessing,
        "h2d_seconds_excluding_warmup": h2d,
        "generation_preprocessing_h2d_questions_per_second": count / (generation + preprocessing + h2d),
        "scope": "Recorded steady-state generation+preprocessing+H2D phases. Deduplicate (session_id,batch_start), exclude each session first warmup batch. Excludes model load, preflight, image validation, disk writes and controller overhead; not end-to-end or isolated single-request latency.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--outputs-root", type=Path)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    outputs = (args.outputs_root or repo.parents[1] / "outputs").resolve()
    budget_root = outputs / "vlm_budget_probe_v1"
    precision_root = outputs / "vlm_precision_bridge_v1"
    budget_path = budget_root / "analysis/ANALYSIS.json"
    precision_path = precision_root / "analysis/ANALYSIS.json"
    budget = read_json(budget_path)
    precision = read_json(precision_path)
    assert budget["complete"] and precision["complete"]
    assert budget["completed_cells"] == precision["completed_cells"] == 6
    assert budget["all_saved_consensus_scores_independently_recomputed"]
    assert all(precision["checks"][k] for k in (
        "same_encoded_inputs_and_batch_groups", "same_within_role_checkpoint_hardware",
        "source_package_generation_contracts_match", "batch_timing_deduplicated_and_warmup_checked"))

    cells = []
    fields = (
        "role", "visual_token_budget", "score_percent", "questions", "unique_images",
        "visual_tokens", "input_tokens", "input_exceeds_training_768_questions",
        "input_exceeds_training_768_percent", "pure_generation_questions_per_second",
        "mean_amortized_generation_seconds_per_question", "peak_allocated_gib", "peak_reserved_gib",
    )
    for cell in budget["cell_summaries"]:
        compact = {k: cell[k] for k in fields}
        provenance = budget["source_provenance"][f"{cell['role']}:{cell['visual_token_budget']}"]
        compact["phase_timing"] = budget_phase_timing(
            Path(provenance["prediction_file"]), cell, provenance["prediction_file_snapshot_sha256"])
        cells.append(compact)

    precision_fields = (
        "role", "numerical_path", "precision", "quantization", "score_percent", "questions",
        "unique_images", "visual_token_budget", "visual_tokens", "input_tokens", "timed_questions",
        "timed_batches", "pure_generation_questions_per_second",
        "generation_preprocessing_h2d_questions_per_second", "peak_allocated_gib", "peak_reserved_gib",
        "hardware", "timing_scope", "resource_timing_status", "sampled_contention_observations",
        "exclusive_gpu_benchmark",
    )
    precision_cells = [{k: c[k] for k in precision_fields} for c in precision["summaries"]]
    adapted = {c["numerical_path"]: c for c in precision_cells if c["role"] == "adapted"}
    speed_ratios = {
        "original_bf16_vs_original_fp32_generation": adapted["original_bf16"]["pure_generation_questions_per_second"] / adapted["original_fp32"]["pure_generation_questions_per_second"],
        "original_bf16_vs_nf4_fp32_generation": adapted["original_bf16"]["pure_generation_questions_per_second"] / adapted["nf4_fp32"]["pure_generation_questions_per_second"],
        "original_bf16_vs_original_fp32_generation_preprocessing_h2d": adapted["original_bf16"]["generation_preprocessing_h2d_questions_per_second"] / adapted["original_fp32"]["generation_preprocessing_h2d_questions_per_second"],
        "original_bf16_vs_original_fp32_peak_allocated_reduction_percent": 100 * (1 - adapted["original_bf16"]["peak_allocated_gib"] / adapted["original_fp32"]["peak_allocated_gib"]),
        "scope": "Fixed seed42 adapter, 256 visual budget, same A100 worker and batch4; observed steady-state ratios, not end-to-end production speed or precision equivalence.",
    }
    base_eval = read_json(budget_root / "runs/base/budget_256/EVALUATION.json")
    adapter_eval = read_json(budget_root / "runs/adapted/budget_256/EVALUATION.json")
    cases = []
    for question_id in ("27482", "10476", "12257"):
        path = budget_root / f"cases/{question_id}/CASE.json"
        case = read_json(path)
        cases.append({
            "question_id": question_id,
            "source": source_record(path),
            "question": case["question"],
            "sft_target_answer_frequency": case["feature"]["sft_target_answer_frequency"],
            "train_all_sft_target_qa_frequency": case["feature"]["train_all_sft_target_qa_frequency"],
            "outcomes": [{k: row[k] for k in ("role", "budget", "score", "visual_tokens", "input_tokens")} for row in case["predictions"]],
            "teaching_board": case["teaching_board"],
            "scope": case["scope"],
        })

    asset_dir = repo / "assets/vlm_optimization_v1"
    asset_dir.mkdir(parents=True, exist_ok=True)
    figure_exports = []
    for source, name, original_provenance in (
        (budget_root / "analysis/quality_cost.png", "budget_quality_cost.png", budget_root / "analysis/FIGURE_PROVENANCE.json"),
        (precision_root / "analysis/quality_resources.png", "precision_quality_cost.png", precision_root / "analysis/FIGURE_PROVENANCE.json"),
    ):
        destination = asset_dir / name
        shutil.copyfile(source, destination)
        assert sha256(source) == sha256(destination)
        figure_exports.append({
            "file": str(destination.relative_to(repo)), "sha256": sha256(destination),
            "original_file": str(source), "original_sha256": sha256(source),
            "original_figure_provenance": source_record(original_provenance),
            "transformation": "Exact byte copy; no resizing, edits or redraw.",
        })
    (asset_dir / "FIGURE_PROVENANCE.json").write_text(json.dumps({
        "schema_version": "rarevlm.optimization.figure_export.v1", "figures": figure_exports,
        "scope": "Existing scientific visualization exports from completed internal-dev experiments; original provenance remains authoritative.",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    document = {
        "schema_version": "rarevlm.optimization.aggregate_export.v1",
        "status": "complete",
        "export_source_sha256": sha256(Path(__file__)),
        "sources": {"visual_budget": source_record(budget_path), "numerical_path": source_record(precision_path)},
        "scope": {
            "dataset": "TextVQA internal dev", "questions_per_cell": 1617, "unique_images_per_cell": 1024,
            "completed_cells": 12, "completed_predictions": 19404, "training_runs_added": 0,
            "model": "Qwen2.5-VL-3B-Instruct", "model_revision": base_eval["model_revision"],
            "model_provenance_sha256": base_eval["model_provenance_sha256"],
            "adapter": "stage2 qv_r8 seed42", "fixed_adapter_seed": 42,
            "adapter_choice": budget["adapter_choice"], "adapter_files_sha256": adapter_eval["adapter_files_sha256"],
            "training_binding": adapter_eval["training_binding"], "hardware": base_eval["hardware"],
            "batch_size": 4, "attention_implementation": "sdpa", "do_sample": False,
            "max_new_tokens": 32, "input_truncation": False, "ocr_in_prompt": False,
            "uncertainty": "Image-cluster paired 2000-replicate pointwise percentile intervals conditional on fixed weights. No training-seed uncertainty, multiple-comparison correction or equivalence margin.",
            "do_not_mix_with": "The prior 15-training/67-evaluation multi-seed official-val study is separately exported as results/vlm_deep_v2_metrics.json. These 12 cells are single-checkpoint internal-dev diagnostics.",
        },
        "visual_budget": {
            "precision": "fp32", "vision_precision": "fp32", "quantization": "nf4_language_only",
            "source_provenance": budget["source_provenance"], "cells": cells,
            "paired_contrasts": budget["paired_contrasts"],
            "sft_target_frequency_definition": budget["sft_target_frequency_definition"],
            "sft_target_slices": [c for c in budget["slice_scores"] if c["dimension"] == "sft_target_answer_frequency"],
            "sft_target_slice_scope": "Training supervised-target exposure only, not model pretraining exposure, a visual-class definition, or causal evidence of a long-tail algorithm.",
            "teaching_cases": cases, "limitations": budget["limitations"],
        },
        "numerical_path": {
            "visual_token_budget": 256, "source_provenance": precision["source_data"],
            "cells": precision_cells, "adapted_speed_ratios": speed_ratios,
            "paired_contrasts": [{k: v for k, v in c.items() if k != "score_transition_matrix"} for c in precision["paired_contrasts"]],
            "checks": precision["checks"], "resource_contention": precision["resource_contention"],
            "limitations": precision["limitations"],
        },
        "unmeasured_joint_configuration": {
            "original_bf16_at_visual_budget_512_or_1024": "not measured in these two completed studies",
            "interpretation": "Do not combine the NF4/FP32 score at 1024 with the original/BF16 speed measured at 256 as one realized deployment configuration.",
        },
        "figures": figure_exports,
        "software_reference": precision["software_reference"],
    }
    target = repo / "results/vlm_optimization_v1_metrics.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"file": str(target), "sha256": sha256(target), "size_bytes": target.stat().st_size,
                      "visual_budget_cells": len(cells), "numerical_path_cells": len(precision_cells),
                      "figures": figure_exports, "adapted_speed_ratios": speed_ratios}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
