"""Prepare a fixed-checkpoint, zero-update diagnostic cohort.

Selection uses only saved INITIAL generation, never Final scores. Whole source
four-question batches retain the initial generator's padding/grouping. Derived
initial predictions are provenance records, NOT a cache for a new evaluation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re


PACKAGE = Path(__file__).resolve().parents[1]
OUTPUTS = PACKAGE.parents[1] / "outputs"
SOURCE = OUTPUTS / "vlm_complementary_rank_v1"
DESTINATION = OUTPUTS / "vlm_fixed_error_readout_v1" / "data"
CORRECT_SALT = "fixed_error_readout_v1_correct_training_anchors"
PAIR_KEYS = ("positive_question_id", "negative_question_id")


def read(path):
    return json.loads(Path(path).read_text())


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def reference(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": sha_file(path)}


def freeze(path, value, jsonl=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = ("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                       for row in value) if jsonl else
               json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    if path.exists():
        if path.read_text() != encoded:
            raise ValueError(f"Refusing to change an existing cohort artifact: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded)
    temporary.replace(path)


def checked_manifest(path):
    data = read(path)
    if data["schema_version"] != "rarevlm.vlm_data.v2":
        raise ValueError("Wrong manifest schema")
    if data["manifest_sha256"] != content_sha({k: v for k, v in data.items()
                                               if k != "manifest_sha256"}):
        raise ValueError(f"Manifest identity changed: {path}")
    return data


def generation(path, manifest_path, rows, expected_lora):
    path = Path(path)
    lock = read(path.parent / "INPUT_LOCK.json")
    receipt = read(path.parent / "COMPLETE.json")
    if (lock["manifest_sha256"] != sha_file(manifest_path)
            or lock["lora_sha256"] != expected_lora
            or lock["task"] != "human" or lock["budget"] != 1024
            or lock["batch_size"] != 4 or lock["max_new_tokens"] != 32
            or lock["numerics"] != "BF16 base/FP32 LoRA/no global AMP"):
        raise ValueError(f"Source generation profile changed: {path}")
    if receipt["predictions_sha256"] != sha_file(path) or receipt["qa"] != len(rows):
        raise ValueError(f"Incomplete source generation: {path}")
    predictions = [json.loads(line) for line in path.read_text().splitlines()]
    if [r["question_id"] for r in rows] != [r["question_id"] for r in predictions]:
        raise ValueError("Source generation order/population differs")
    images = read(manifest_path)["images"]
    for index, (row, prediction) in enumerate(zip(rows, predictions)):
        match = re.match(r"^(yes|no)\b", prediction["prediction"].strip(),
                         flags=re.IGNORECASE | re.ASCII)
        binary = match.group(1).lower() if match else None
        if (prediction["image_id"] != row["image_id"]
                or prediction["question_sha256"] != content_sha(row["question"])
                or prediction["image_sha256"] != images[row["image_id"]]["sha256"]
                or prediction["batch_start"] != (index // 4) * 4
                or prediction["batch_size"] != 4
                or prediction["primary_score"] != float(binary == row["target"])):
            raise ValueError(f"Source row identity/scoring/grouping differs: {index}")
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-data", type=Path, default=DESTINATION)
    args = parser.parse_args()
    destination = args.output_data.resolve()
    if not destination.is_relative_to(Path("/opt/data/private")):
        raise ValueError("All artifacts must remain under /opt/data/private")

    train_path, screen_path = (SOURCE / "data" / name for name in
                               ("TRAIN_EVAL.json", "SCREEN_EVAL.json"))
    train = checked_manifest(train_path)
    screen = checked_manifest(screen_path)
    catalog_path = SOURCE / "data/TRAINING_MANIFEST.json"
    catalog = checked_manifest(catalog_path)
    train_rows, screen_rows = train["splits"]["dev"], screen["splits"]["dev"]
    if len(train_rows) != 1200 or len(screen_rows) != 512:
        raise ValueError("The registered source cohort must be used intact")
    train_pairs = catalog["complementary_pairs"]["train"]
    screen_pairs = catalog["complementary_pairs"]["screen"]
    if len(train_pairs) != 600 or len(screen_pairs) != 256:
        raise ValueError("Source pair population differs")

    baseline_config_path = PACKAGE / "configs/vlm_human_sft_baseline_v1.json"
    baseline = read(baseline_config_path)
    initial_lock_path = SOURCE / "initial/train/human/INPUT_LOCK.json"
    initial_lora = read(initial_lock_path)["lora_sha256"]
    initial_train_path = SOURCE / "initial/train/human/predictions.jsonl"
    initial_train = generation(initial_train_path, train_path, train_rows, initial_lora)
    initial_map = {row["question_id"]: row for row in initial_train}
    gold_map = {row["question_id"]: row for row in train_rows}
    source_index = {row["question_id"]: index for index, row in enumerate(train_rows)}

    def pair_correct(pair):
        return all(initial_map[pair[key]]["primary_score"] == 1 for key in PAIR_KEYS)

    def image_keys(pair):
        image_ids = {gold_map[pair[key]]["image_id"] for key in PAIR_KEYS}
        # IDs, source file hashes and decoded pixels prevent duplicate anchors.
        return {(kind, value) for image_id in image_ids for kind, value in
                (("id", image_id), ("file", train["images"][image_id]["sha256"]),
                 ("pixels", train["images"][image_id]["pixel_sha256"]))}

    wrong = [pair for pair in train_pairs if not pair_correct(pair)]
    if len(wrong) != 45:
        raise ValueError("Expected all 45 initial-not-fully-correct training pairs")
    occupied = set().union(*(image_keys(pair) for pair in wrong))
    correct = []
    for pair in sorted((pair for pair in train_pairs if pair_correct(pair)),
                       key=lambda pair: content_sha([CORRECT_SALT, pair["pair_id"]])):
        keys = image_keys(pair)
        if keys & occupied:
            continue
        correct.append(pair)
        occupied.update(keys)
        if len(correct) == len(wrong):
            break
    if len(correct) != 45:
        raise ValueError("Insufficient image-distinct initially correct anchors")
    anchor_groups = {pair["pair_id"]: label for label, pairs in
                     (("initial_wrong", wrong), ("initial_correct", correct)) for pair in pairs}
    starts = sorted({(source_index[pair[key]] // 4) * 4
                     for pair in wrong + correct for key in PAIR_KEYS})
    selected_indices = [index for start in starts for index in range(start, start + 4)]
    selected_rows = [deepcopy(train_rows[index]) for index in selected_indices]
    selected_qids = {row["question_id"] for row in selected_rows}
    selected_pairs = []
    for pair in train_pairs:
        present = [pair[key] in selected_qids for key in PAIR_KEYS]
        if any(present) and not all(present):
            raise ValueError("Source batch closure split a complementary pair")
        if all(present):
            selected_pairs.append({**pair, "anchor_group": anchor_groups.get(pair["pair_id"], "batch_neighbor"),
                                   "initial_pair_correct": pair_correct(pair),
                                   "source_batch_start": (source_index[pair[PAIR_KEYS[0]]] // 4) * 4})
    images = {row["image_id"]: deepcopy(train["images"][row["image_id"]]) for row in selected_rows}
    for item in images.values():
        item["path"] = str((train_path.parent / item["path"]).resolve(strict=True))
    manifest = {"schema_version": "rarevlm.vlm_data.v2",
                "purpose": "Zero-update fixed Final200 diagnostic only; original batch-four closure of 45 initial-wrong and 45 image-distinct initial-correct anchor pairs; not an unbiased training accuracy estimate",
                "images": images,
                "splits": {"train_all": [], "train_10k": [], "dev": selected_rows, "final": []}}
    manifest["manifest_sha256"] = content_sha(manifest)
    manifest_path = destination / "TRAIN_SELECTED_EVAL.json"
    freeze(manifest_path, manifest)

    derived_predictions = [initial_train[index] for index in selected_indices]
    derived_path = destination / "INITIAL_TRAIN_SELECTED_PREDICTIONS.jsonl"
    freeze(derived_path, derived_predictions, jsonl=True)
    mapping = {row["question_id"]: {
        "selected_row_index": index, "selected_batch_start": (index // 4) * 4,
        "source_row_index": selected_indices[index],
        "source_batch_start": (selected_indices[index] // 4) * 4,
        "pair_id": row["complementary_pair_id"],
        "anchor_group": anchor_groups.get(row["complementary_pair_id"], "batch_neighbor"),
        "image_id": row["image_id"], "target": row["target"]}
        for index, row in enumerate(selected_rows)}
    for start in range(0, len(selected_rows), 4):
        old_start = starts[start // 4]
        if ([r["question_id"] for r in selected_rows[start:start + 4]] !=
                [r["question_id"] for r in train_rows[old_start:old_start + 4]]):
            raise ValueError("Selected four-question grouping changed")

    models = {}
    generation_sources = {}
    for role in ("initial", "ce", "ce_rank"):
        if role == "initial":
            adapter = Path(baseline["initial_adapter"])
            lora = initial_lora
            screen_prediction_path = SOURCE / "initial/screen/human/predictions.jsonl"
            receipt_ref = reference(SOURCE / "initial/train/human/COMPLETE.json")
        else:
            receipt_path = SOURCE / f"runs/{role}/COMPLETE.json"
            receipt = read(receipt_path)
            if receipt["updates"] != 200 or receipt["status"] != "completed":
                raise ValueError("Only completed registered Final200 adapters are permitted")
            adapter = Path(receipt["train"]["adapter"])
            lora = receipt["train"]["final_lora_sha256"]
            screen_prediction_path = SOURCE / f"runs/{role}/human/predictions.jsonl"
            receipt_ref = reference(receipt_path)
        screen_predictions = generation(screen_prediction_path, screen_path, screen_rows, lora)
        if len(screen_predictions) != 512:
            raise ValueError("Screen must reuse the entire original population")
        models[role] = {"adapter": str(adapter.resolve(strict=True)), "lora_sha256": lora,
                        "adapter_weights": reference(adapter / "adapter_model.safetensors"),
                        "adapter_config": reference(adapter / "adapter_config.json"),
                        "source_receipt": receipt_ref,
                        "updates_for_this_diagnostic": 0,
                        "checkpoint_selection": "original SFT42" if role == "initial" else "registered Final200"}
        generation_sources[role] = {
            "screen": {**reference(screen_prediction_path), "qa": 512, "reused": True},
            "train_selected": ({**reference(derived_path), "qa": len(selected_rows), "reused": True,
                                "source": reference(initial_train_path),
                                "preserved_source_batch_start": True,
                                "cache_scope": "sidecar provenance only; do not install as a new evaluator cache"}
                               if role == "initial" else
                               {"qa": len(selected_rows), "reused": False,
                                "required": "new generation only for this fixed Final200 adapter"})}

    qid_component = {pair[key]: pair["component_id"] for pair in selected_pairs for key in PAIR_KEYS}
    counts = {"train_selected": {
        "qa": len(selected_rows), "pairs": len(selected_pairs), "images": len(images),
        "components": len(set(qid_component.values())), "source_batches": len(starts),
        "anchor_groups": dict(Counter(pair["anchor_group"] for pair in selected_pairs))},
        "screen": {"qa": len(screen_rows), "pairs": len(screen_pairs),
                   "images": len({row["image_id"] for row in screen_rows}),
                   "components": len({pair["component_id"] for pair in screen_pairs})}}
    cohort = {
        "status": "prepared", "study": "vlm_fixed_error_readout_v1", "schema_version": 1,
        "scope": "Exploratory fixed-checkpoint failure attribution. Zero training updates; confirm is not evaluated. No capability-gain claim.",
        "selection": {"uses": "initial training generation only", "initial_wrong_pairs": 45,
                      "initial_correct_pairs": 45, "correct_anchor_hash_salt": CORRECT_SALT,
                      "correct_anchor_hash": "content_sha([salt, pair_id]) ascending; reject anchor image/file/pixel overlap",
                      "batch_closure": "retain each source ordered four-question batch intact and sort by original batch_start",
                      "final_scores_used_for_selection": False,
                      "correct_anchors_image_disjoint_from_wrong_and_each_other": True},
        "manifests": {"train_selected": {**reference(manifest_path), "qa": len(selected_rows)},
                      "screen": {**reference(screen_path), "qa": len(screen_rows)}},
        "pairs": {"train_selected": selected_pairs, "screen": screen_pairs},
        "anchor_pair_ids": {"initial_wrong": [p["pair_id"] for p in wrong],
                            "initial_correct": [p["pair_id"] for p in correct]},
        "source_batch_starts": starts,
        "qid_mapping": {"train_selected": mapping,
                        "screen": {row["question_id"]: {"row_index": index,
                                    "source_batch_start": (index // 4) * 4,
                                    "pair_id": row["complementary_pair_id"],
                                    "image_id": row["image_id"], "target": row["target"]}
                                   for index, row in enumerate(screen_rows)}},
        "initial_predictions": {task: generation_sources["initial"][task]
                                for task in ("train_selected", "screen")},
        "generation_sources": generation_sources,
        "models": models,
        "baseline_config": reference(baseline_config_path),
        "model_root": baseline["model_root"],
        "reference_config": reference(baseline["reference_config"]),
        "generation_profile": {"evaluation_budget": 1024, "evaluation_batch_size": 4,
                               "max_new_tokens": 32, "generation": "greedy",
                               "numerics": "BF16 base/FP32 LoRA/no global AMP"},
        "counts": counts,
        "source_references": {"train_eval": reference(train_path), "screen_eval": reference(screen_path),
                              "pair_catalog": reference(catalog_path),
                              "initial_input_lock": reference(initial_lock_path),
                              "prepare_source": reference(Path(__file__))},
        "cache_reuse_limit": "Original row dicts retain ORIGINAL batch_start and timing. Only same-four-QA membership/order permits derived initial reuse. Use selected/new batch indices from qid_mapping for new generation; never reinterpret old timings as fresh inference. Screen generations are reused from the intact original screen only.",
    }
    freeze(destination / "COHORT.json", cohort)
    print(json.dumps({"status": "prepared", "cohort": str(destination / "COHORT.json"),
                      "manifests": cohort["manifests"], "counts": counts,
                      "reuse_generation": {role: {task: values["reused"] for task, values in tasks.items()}
                                           for role, tasks in generation_sources.items()},
                      "training_updates": 0, "new_gpu_calls_by_preparation": 0},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
