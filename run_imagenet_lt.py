"""Frozen-CLIP ImageNet-LT replication: zero-shot, adapter CE and adapter LA.

This runner is intentionally separate from the CIFAR permutation pilot. It
selects each trained adapter epoch on the disjoint ImageNet-LT validation list
and reads official-test labels only when evaluating the selected checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from imagenet_lt_manifest import content_sha256
from methods import PrototypeClassifier, ResidualAdapter, build_loss

EXPECTED_CLIP_WEIGHT_SHA256 = "a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f"
EXPECTED_PROMPT_TEMPLATE = "a photo of a {}."


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_bank(path: Path, split: str, manifest_hash: str, manifest_records: list[dict], device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["manifest_sha256"] != manifest_hash or payload["split"] != split:
        raise ValueError(f"feature bank provenance mismatch: {path}")
    features = payload["features"]
    targets = payload["targets"].long()
    expected = torch.tensor([int(item["label"]) for item in manifest_records], dtype=torch.long)
    if features.shape != (len(expected), 512) or not torch.equal(targets, expected):
        raise ValueError(f"feature bank shape/labels disagree with manifest: {split}")
    if not bool(torch.isfinite(features).all()):
        raise ValueError(f"feature bank has non-finite values: {split}")
    return features.float().to(device), targets.to(device)


def classify_metrics(logits, adapted, frozen, labels, prototypes, groups):
    """Vectorized metrics; targets are consumed only after logits exist."""
    n_classes = prototypes.shape[0]
    if logits.shape != (len(labels), n_classes):
        raise ValueError("invalid logit shape")
    predictions = logits.argmax(dim=1)
    correct = predictions.eq(labels)
    counts = torch.bincount(labels, minlength=n_classes).float()
    predicted_counts = torch.bincount(predictions, minlength=n_classes).float()
    true_positives = torch.bincount(labels[correct], minlength=n_classes).float()
    if bool((counts <= 0).any()):
        raise ValueError("evaluation split omits a class")
    class_accuracy = true_positives / counts
    class_f1 = 2.0 * true_positives / (counts + predicted_counts).clamp_min(1.0)

    adapted_unit = F.normalize(adapted.float(), dim=1)
    frozen_unit = F.normalize(frozen.float(), dim=1)
    sample_alignment = (adapted_unit * prototypes[labels]).sum(dim=1)
    sample_drift = 1.0 - (adapted_unit * frozen_unit).sum(dim=1)
    top2 = logits.topk(2, dim=1).values
    true_logit = logits.gather(1, labels[:, None]).squeeze(1)
    max_other = torch.where(correct, top2[:, 1], top2[:, 0])
    sample_margin = true_logit - max_other

    centroids = torch.zeros_like(prototypes)
    centroids.index_add_(0, labels, adapted_unit)
    class_centroid_alignment = (F.normalize(centroids, dim=1) * prototypes).sum(dim=1)
    class_alignment = torch.bincount(labels, weights=sample_alignment, minlength=n_classes) / counts
    class_drift = torch.bincount(labels, weights=sample_drift, minlength=n_classes) / counts
    class_margin = torch.bincount(labels, weights=sample_margin, minlength=n_classes) / counts

    probability = logits.softmax(dim=1)
    confidence = probability.amax(dim=1)
    ece = 0.0
    for bin_id in range(15):
        lo, hi = bin_id / 15, (bin_id + 1) / 15
        mask = (confidence >= lo) & (confidence < hi if bin_id < 14 else confidence <= hi)
        if bool(mask.any()):
            ece += float(mask.float().mean().item()) * abs(
                float(correct[mask].float().mean().item()) - float(confidence[mask].mean().item())
            )

    group_results = {}
    for name, class_ids in groups.items():
        ids = torch.tensor(class_ids, device=labels.device, dtype=torch.long)
        sample_mask = torch.isin(labels, ids)
        group_results[name] = {
            "accuracy": float(correct[sample_mask].float().mean().item()),
            "macro_f1": float(class_f1[ids].mean().item()),
            "alignment": float(class_alignment[ids].mean().item()),
            "centroid_alignment": float(class_centroid_alignment[ids].mean().item()),
            "drift": float(class_drift[ids].mean().item()),
            "margin": float(class_margin[ids].mean().item()),
            "n": int(sample_mask.sum().item()),
            "correct": int(correct[sample_mask].sum().item()),
        }
    metrics = {
        "oa": float(correct.float().mean().item()),
        "macro_f1": float(class_f1.mean().item()),
        "nll": float(F.cross_entropy(logits, labels).item()),
        "ece15": ece,
        "n": len(labels),
        "correct": int(correct.sum().item()),
        "groups": group_results,
        "class_accuracy": class_accuracy.cpu().tolist(),
        "class_f1": class_f1.cpu().tolist(),
        "class_alignment": class_alignment.cpu().tolist(),
        "class_centroid_alignment": class_centroid_alignment.cpu().tolist(),
        "class_drift": class_drift.cpu().tolist(),
        "class_margin": class_margin.cpu().tolist(),
    }
    return metrics, predictions.cpu().numpy().astype(np.uint16)


def evaluate(adapter, classifier, x, y, groups):
    with torch.inference_mode():
        adapted = adapter(x) if adapter is not None else x
        logits = classifier(adapted)
        return classify_metrics(logits, adapted, x, y, classifier.text_prototypes, groups)


def train_one(
    *, seed, method, classifier, train_x, train_y, val_x, val_y, test_x, test_y,
    manifest, output_dir, epochs, batch_size, lr, weight_decay, bank_metadata,
):
    run_id = f"imagenet_lt_seed{seed}_{method}"
    result_file = output_dir / f"{run_id}.json"
    prediction_file = output_dir / f"{run_id}_pred.npz"
    checkpoint_file = output_dir / f"{run_id}_adapter.pt"
    optimizer_config = {"name": "AdamW", "lr": lr, "weight_decay": weight_decay,
                        "batch_size": batch_size, "epochs": epochs}
    if result_file.exists() and prediction_file.exists() and checkpoint_file.exists():
        result = json.loads(result_file.read_text())
        if (result.get("manifest_sha256") == manifest["manifest_sha256"]
                and result.get("optimizer") == optimizer_config
                and result.get("bank_metadata") == bank_metadata
                and result.get("method") == method
                and result.get("train_seed") == seed
                and result.get("checkpoint_sha256") == sha256_file(checkpoint_file)
                and result.get("prediction_sha256") == sha256_file(prediction_file)):
            print(f"skip complete {run_id}", flush=True)
            return result
        raise ValueError(f"existing result uses different config, features, or manifest: {run_id}")

    torch.manual_seed(seed)
    adapter = ResidualAdapter(train_x.shape[1], bottleneck=64).to(train_x.device)
    criterion = (build_loss("ce") if method == "ce" else
                 build_loss("la", manifest["class_counts"], tau=1.0)).to(train_x.device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=weight_decay)
    assert not any(parameter.requires_grad for parameter in classifier.parameters())
    assert not train_x.requires_grad and not val_x.requires_grad and not test_x.requires_grad
    assert {id(p) for group in optimizer.param_groups for p in group["params"]} == {
        id(p) for p in adapter.parameters()
    }
    rng = torch.Generator().manual_seed(seed + 100_000)
    best_oa, best_epoch, best_state = -math.inf, None, None
    history = []
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        adapter.train()
        order = torch.randperm(len(train_x), generator=rng)
        total_loss = 0.0
        for position in range(0, len(order), batch_size):
            take = order[position:position + batch_size].to(train_x.device)
            logits = classifier(adapter(train_x[take]))
            loss = criterion(logits, train_y[take])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite loss: {run_id}, epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()  # One and only one optimizer update per batch.
            total_loss += float(loss.detach().item()) * len(take)
        adapter.eval()
        with torch.inference_mode():
            val_logits = classifier(adapter(val_x))
            val_oa = float(val_logits.argmax(dim=1).eq(val_y).float().mean().item())
            val_nll = float(F.cross_entropy(val_logits, val_y).item())
        history.append({"epoch": epoch, "train_loss": total_loss / len(train_x),
                        "val_oa": val_oa, "val_nll": val_nll})
        if val_oa > best_oa:
            best_oa, best_epoch = val_oa, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"{run_id}: epoch={epoch} train_loss={total_loss/len(train_x):.4f} "
                  f"val_oa={val_oa:.4f} best={best_oa:.4f}@{best_epoch}", flush=True)
    assert best_state is not None
    adapter.load_state_dict(best_state)
    adapter.eval()
    metrics, predictions = evaluate(adapter, classifier, test_x, test_y, manifest["groups"])
    if not all(math.isfinite(metrics[key]) for key in ("oa", "nll", "ece15", "macro_f1")):
        raise FloatingPointError(f"non-finite test metric: {run_id}")
    result = {
        "run_id": run_id,
        "manifest_sha256": manifest["manifest_sha256"],
        "method": method, "train_seed": seed,
        "model": "fixed-text-prototype CLIP ViT-B/32 + LN/GELU adapter (512-64-512)",
        "loss": {"ce": {}, "la": {"tau": 1.0}}[method],
        "optimizer": optimizer_config,
        "selection": {"criterion": "highest balanced-validation OA; earliest tie",
                      "best_epoch": best_epoch, "best_val_oa": best_oa},
        "test": metrics,
        "history": history,
        "bank_metadata": bank_metadata,
        "duration_seconds": round(time.monotonic() - started, 2),
    }
    torch.save({"adapter_state_dict": best_state, "manifest_sha256": manifest["manifest_sha256"],
                "run_id": run_id}, checkpoint_file)
    np.savez_compressed(prediction_file, predictions=predictions,
                        targets=test_y.cpu().numpy().astype(np.uint16))
    result["checkpoint_sha256"] = sha256_file(checkpoint_file)
    result["prediction_sha256"] = sha256_file(prediction_file)
    save_json(result_file, result)
    print(f"{run_id}: test OA={metrics['oa']:.4f} macroF1={metrics['macro_f1']:.4f} "
          f"Many={metrics['groups']['many']['accuracy']:.4f} "
          f"Medium={metrics['groups']['medium']['accuracy']:.4f} "
          f"Few={metrics['groups']['few']['accuracy']:.4f}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--train-seeds", nargs="+", type=int, default=[0, 42, 200])
    parser.add_argument("--methods", nargs="+", choices=["ce", "la"], default=["ce", "la"])
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch size must be positive")
    torch.set_num_threads(4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    if content_sha256({k: v for k, v in manifest.items() if k != "manifest_sha256"}) != manifest.get("manifest_sha256"):
        raise ValueError("manifest content hash mismatch")
    if manifest.get("schema_version") != "rarevlm.imagenet_lt.v1":
        raise ValueError("unsupported ImageNet-LT manifest schema")
    device = torch.device(args.device)
    banks = {
        split: load_bank(
            args.feature_dir / f"{split}.pt", split, manifest["manifest_sha256"],
            manifest["splits"][split], device
        ) for split in ("train", "val", "test")
    }
    prototype_bank = torch.load(args.feature_dir / "text_prototypes.pt", map_location="cpu", weights_only=False)
    if prototype_bank["manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("text prototypes use a different manifest")
    if (tuple(prototype_bank["text_prototypes"].shape) != (1000, 512)
            or len(prototype_bank["class_names"]) != 1000
            or prototype_bank.get("label_to_wnid") != manifest["label_to_wnid"]
            or prototype_bank.get("model_weight_sha256") != EXPECTED_CLIP_WEIGHT_SHA256
            or prototype_bank.get("prompt_template") != EXPECTED_PROMPT_TEMPLATE):
        raise ValueError("text prototype order, model weight, or prompt mismatch")
    classifier = PrototypeClassifier(
        prototype_bank["text_prototypes"], prototype_bank["logit_scale"]
    ).to(device).eval()
    bank_metadata = json.loads((args.feature_dir / "metadata.json").read_text())
    if (bank_metadata.get("model_weight_sha256") != EXPECTED_CLIP_WEIGHT_SHA256
            or bank_metadata.get("prompt_template") != EXPECTED_PROMPT_TEMPLATE
            or bank_metadata.get("label_to_wnid") != manifest["label_to_wnid"]
            or bank_metadata.get("manifest_sha256") != manifest["manifest_sha256"]):
        raise ValueError("feature bank metadata disagrees with frozen protocol")
    bank_metadata["feature_artifact_sha256"] = {
        name: sha256_file(args.feature_dir / name)
        for name in ("train.pt", "val.pt", "test.pt", "text_prototypes.pt")
    }

    zero_file = args.output_dir / "zero_shot.json"
    zero_prediction_file = args.output_dir / "zero_shot_pred.npz"
    if zero_file.exists() or zero_prediction_file.exists():
        if not (zero_file.exists() and zero_prediction_file.exists()):
            raise ValueError("incomplete zero-shot output; use a fresh output directory")
        existing_zero = json.loads(zero_file.read_text())
        if (existing_zero.get("manifest_sha256") != manifest["manifest_sha256"]
                or existing_zero.get("bank_metadata") != bank_metadata
                or existing_zero.get("prediction_sha256") != sha256_file(zero_prediction_file)):
            raise ValueError("existing zero-shot output uses different manifest or features")
    else:
        metrics, predictions = evaluate(None, classifier, *banks["test"], manifest["groups"])
        np.savez_compressed(zero_prediction_file, predictions=predictions,
                            targets=banks["test"][1].cpu().numpy().astype(np.uint16))
        save_json(zero_file, {
            "method": "zero_shot", "manifest_sha256": manifest["manifest_sha256"],
            "test": metrics, "bank_metadata": bank_metadata,
            "prediction_sha256": sha256_file(zero_prediction_file),
        })
        print(f"zero-shot: OA={metrics['oa']:.4f} macroF1={metrics['macro_f1']:.4f}", flush=True)
    for seed in args.train_seeds:
        for method in args.methods:
            train_one(
                seed=seed, method=method, classifier=classifier,
                train_x=banks["train"][0], train_y=banks["train"][1],
                val_x=banks["val"][0], val_y=banks["val"][1],
                test_x=banks["test"][0], test_y=banks["test"][1],
                manifest=manifest, output_dir=args.output_dir,
                epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                weight_decay=args.weight_decay, bank_metadata=bank_metadata,
            )


if __name__ == "__main__":
    main()
