"""Fixed-CE-checkpoint prior-offset test, tuned only on ImageNet-LT validation.

This test directly asks how much of CE's frequency bias can be recovered while
keeping its adapted image features unchanged. It is run only after the main
ImageNet-LT replication; its alpha grid is frozen in the V3 protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from imagenet_lt_manifest import content_sha256
from methods import PrototypeClassifier, ResidualAdapter
from run_imagenet_lt import (
    EXPECTED_CLIP_WEIGHT_SHA256, EXPECTED_PROMPT_TEMPLATE, classify_metrics,
    load_bank, save_json,
)


ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
SEEDS = (0, 42, 200)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_adapter(path: Path, manifest_hash: str, seed: int, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["manifest_sha256"] != manifest_hash or payload["run_id"] != f"imagenet_lt_seed{seed}_ce":
        raise ValueError(f"CE checkpoint provenance mismatch: {path}")
    adapter = ResidualAdapter(512, bottleneck=64).to(device)
    adapter.load_state_dict(payload["adapter_state_dict"], strict=True)
    return adapter.eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--main-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(4)
    manifest = json.loads(args.manifest.read_text())
    manifest_hash = manifest.get("manifest_sha256")
    if content_sha256({k: v for k, v in manifest.items() if k != "manifest_sha256"}) != manifest_hash:
        raise ValueError("manifest hash mismatch")
    prototype_bank = torch.load(args.feature_dir / "text_prototypes.pt", map_location="cpu", weights_only=False)
    if (prototype_bank["manifest_sha256"] != manifest_hash
            or prototype_bank.get("label_to_wnid") != manifest["label_to_wnid"]
            or prototype_bank.get("model_weight_sha256") != EXPECTED_CLIP_WEIGHT_SHA256
            or prototype_bank.get("prompt_template") != EXPECTED_PROMPT_TEMPLATE):
        raise ValueError("prototype bank provenance mismatch")
    metadata = json.loads((args.feature_dir / "metadata.json").read_text())
    if (metadata.get("manifest_sha256") != manifest_hash
            or metadata.get("model_weight_sha256") != EXPECTED_CLIP_WEIGHT_SHA256
            or metadata.get("prompt_template") != EXPECTED_PROMPT_TEMPLATE):
        raise ValueError("feature metadata provenance mismatch")
    device = torch.device(args.device)
    classifier = PrototypeClassifier(
        prototype_bank["text_prototypes"], prototype_bank["logit_scale"]
    ).to(device).eval()
    prior = torch.tensor(manifest["class_counts"], device=device, dtype=torch.float32)
    log_prior = (prior / prior.sum()).log()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Finish and persist all validation-only selections before opening test.pt.
    val_x, val_y = load_bank(
        args.feature_dir / "val.pt", "val", manifest_hash, manifest["splits"]["val"], device
    )
    selections = {}
    for seed in SEEDS:
        ce_result_path = args.main_results / f"imagenet_lt_seed{seed}_ce.json"
        ce_checkpoint = args.main_results / f"imagenet_lt_seed{seed}_ce_adapter.pt"
        ce_result = json.loads(ce_result_path.read_text())
        if ce_result["manifest_sha256"] != manifest_hash or ce_result["optimizer"] != {
            "name": "AdamW", "lr": 0.001, "weight_decay": 0.01,
            "batch_size": 256, "epochs": 20,
        } or ce_result.get("checkpoint_sha256") != sha256_file(ce_checkpoint):
            raise ValueError(f"CE result is not the frozen V3 protocol: seed {seed}")
        if any(ce_result.get("bank_metadata", {}).get(key) != value
               for key, value in metadata.items()):
            raise ValueError(f"CE result uses a different feature bank: seed {seed}")
        adapter = load_adapter(ce_checkpoint, manifest_hash, seed, device)
        with torch.inference_mode():
            logits = classifier(adapter(val_x))
            grid = []
            for alpha in ALPHAS:
                adjusted = logits - alpha * log_prior
                oa = float(adjusted.argmax(dim=1).eq(val_y).float().mean().item())
                grid.append({"alpha": alpha, "val_oa": oa})
        chosen = max(grid, key=lambda row: row["val_oa"])
        selections[str(seed)] = {
            "grid": grid, "selected_alpha": chosen["alpha"],
            "selected_val_oa": chosen["val_oa"],
            "ce_checkpoint_sha256": sha256_file(ce_checkpoint),
            "ce_best_epoch": ce_result["selection"]["best_epoch"],
        }
        print(f"seed={seed}: selected alpha={chosen['alpha']} valOA={chosen['val_oa']:.4f}", flush=True)
    feature_artifact_sha256 = {
        name: sha256_file(args.feature_dir / name)
        for name in ("train.pt", "val.pt", "test.pt", "text_prototypes.pt")
    }
    for seed in SEEDS:
        ce_result = json.loads((args.main_results / f"imagenet_lt_seed{seed}_ce.json").read_text())
        if ce_result["bank_metadata"].get("feature_artifact_sha256") != feature_artifact_sha256:
            raise ValueError(f"CE run uses different feature artifact bytes: seed {seed}")
    selection_payload = {
        "manifest_sha256": manifest_hash,
        "feature_metadata": metadata,
        "feature_artifact_sha256": feature_artifact_sha256,
        "selection_rule": "highest balanced validation OA; smallest alpha on tie",
        "alpha_grid": list(ALPHAS),
        "seeds": selections,
    }
    selection_path = args.output_dir / "validation_selections.json"
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection_payload:
        raise ValueError("existing post-hoc selections disagree with this run")
    if not selection_path.exists():
        save_json(selection_path, selection_payload)

    test_x, test_y = load_bank(
        args.feature_dir / "test.pt", "test", manifest_hash, manifest["splits"]["test"], device
    )
    for seed in SEEDS:
        output_json = args.output_dir / f"imagenet_lt_seed{seed}_ce_posthoc.json"
        output_pred = args.output_dir / f"imagenet_lt_seed{seed}_ce_posthoc_pred.npz"
        if output_json.exists() or output_pred.exists():
            if not (output_json.exists() and output_pred.exists()):
                raise ValueError(f"incomplete post-hoc output for seed {seed}")
            old = json.loads(output_json.read_text())
            if (old.get("manifest_sha256") != manifest_hash
                    or old.get("selected_alpha") != selections[str(seed)]["selected_alpha"]
                    or old.get("ce_checkpoint_sha256") != selections[str(seed)]["ce_checkpoint_sha256"]
                    or old.get("feature_artifact_sha256") != feature_artifact_sha256
                    or old.get("prediction_sha256") != sha256_file(output_pred)):
                raise ValueError(f"existing post-hoc result uses other inputs: seed {seed}")
            print(f"skip complete seed={seed}", flush=True)
            continue
        checkpoint = args.main_results / f"imagenet_lt_seed{seed}_ce_adapter.pt"
        adapter = load_adapter(checkpoint, manifest_hash, seed, device)
        with torch.inference_mode():
            adapted = adapter(test_x)
            logits = classifier(adapted)
            adjusted = logits - selections[str(seed)]["selected_alpha"] * log_prior
            metrics, predictions = classify_metrics(
                adjusted, adapted, test_x, test_y, classifier.text_prototypes, manifest["groups"]
            )
        result = {
            "method": "ce_fixed_checkpoint_posthoc_prior",
            "train_seed": seed,
            "manifest_sha256": manifest_hash,
            "ce_checkpoint_sha256": selections[str(seed)]["ce_checkpoint_sha256"],
            "feature_artifact_sha256": feature_artifact_sha256,
            "selected_alpha": selections[str(seed)]["selected_alpha"],
            "selection": selections[str(seed)],
            "test": metrics,
            "formula": "CE logits - alpha * log(actual train class prior)",
        }
        np.savez_compressed(output_pred, predictions=predictions,
                            targets=test_y.cpu().numpy().astype(np.uint16))
        result["prediction_sha256"] = sha256_file(output_pred)
        save_json(output_json, result)
        print(f"seed={seed}: posthoc OA={metrics['oa']:.4f} "
              f"Few={metrics['groups']['few']['accuracy']:.4f}", flush=True)


if __name__ == "__main__":
    main()
