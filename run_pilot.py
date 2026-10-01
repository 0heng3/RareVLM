"""Paired frozen-CLIP adapter experiment over three CIFAR frequency permutations.

All model choices use a disjoint validation subset of CIFAR's official train
set. The official test set is evaluated only after an adapter epoch is chosen.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from methods import PrototypeClassifier, ResidualAdapter, build_loss
from splits import build_manifests, validate_manifest


LOSS_CONFIGS = {
    "ce": {},
    "la": {"tau": 1.0},
    # GALE's three static components retained; its 70/0.1 logit transform is
    # rescaled for CLIP's already-large fixed 100x cosine logits.
    "ts": {
        "max_margin": 0.1,
        "max_logit_scale": 1.2,
        "temperature": 1.0,
        "inverse_weight_power": 0.5,
        "class_weight_gain": 2.0,
        "class_weight_cap": 2.0,
    },
}


def group_metrics(logits, adapted, frozen, targets, prototypes, groups):
    targets = targets.long()
    prediction = logits.argmax(dim=1)
    correct = prediction.eq(targets)
    confidence = logits.softmax(dim=1).amax(dim=1)
    adapted = F.normalize(adapted.float(), dim=1)
    frozen = F.normalize(frozen.float(), dim=1)
    alignment = (adapted * prototypes[targets]).sum(dim=1)
    drift = 1.0 - (adapted * frozen).sum(dim=1)
    others = logits.clone()
    others[torch.arange(len(targets), device=targets.device), targets] = -torch.inf
    margin = logits.gather(1, targets[:, None]).squeeze(1) - others.amax(dim=1)
    class_accuracy = []
    class_centroid_alignment = []
    class_margin = []
    class_drift = []
    for c in range(prototypes.shape[0]):
        mask = targets.eq(c)
        if not bool(mask.any()):
            raise ValueError(f"evaluation set has no instances of class {c}")
        class_accuracy.append(float(correct[mask].float().mean().item()))
        centroid = F.normalize(adapted[mask].mean(dim=0), dim=0)
        class_centroid_alignment.append(float((centroid * prototypes[c]).sum().item()))
        class_margin.append(float(margin[mask].mean().item()))
        class_drift.append(float(drift[mask].mean().item()))

    group_data = {}
    for name, ids in groups.items():
        mask = torch.isin(targets, torch.tensor(ids, device=targets.device))
        group_data[name] = {
            "accuracy": float(correct[mask].float().mean().item()),
            "alignment": float(alignment[mask].mean().item()),
            "centroid_alignment": float(np.mean([class_centroid_alignment[c] for c in ids])),
            "drift": float(drift[mask].mean().item()),
            "margin": float(margin[mask].mean().item()),
            "correct": int(correct[mask].sum().item()),
            "n": int(mask.sum().item()),
        }
    ece = 0.0
    for b in range(15):
        low, high = b / 15, (b + 1) / 15
        mask = (confidence >= low) & (confidence < high if b < 14 else confidence <= high)
        if bool(mask.any()):
            ece += float(mask.float().mean().item()) * abs(
                float(correct[mask].float().mean().item()) - float(confidence[mask].mean().item())
            )
    return {
        "oa": float(correct.float().mean().item()),
        "correct": int(correct.sum().item()),
        "n": len(targets),
        "nll": float(F.cross_entropy(logits, targets).item()),
        "ece15": ece,
        "groups": group_data,
        "class_accuracy": class_accuracy,
        "class_centroid_alignment": class_centroid_alignment,
        "class_margin": class_margin,
        "class_drift": class_drift,
    }, prediction.cpu().numpy().astype(np.uint8)


def evaluate(adapter, classifier, features, labels, groups):
    with torch.inference_mode():
        adapted = adapter(features) if adapter is not None else features
        logits = classifier(adapted)
        return group_metrics(
            logits, adapted, features, labels, classifier.text_prototypes, groups
        )


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def run_one(
    *, manifest, seed, method, classifier, train_features, train_targets,
    test_features, test_targets, output_dir, epochs, batch_size, lr, weight_decay,
    feature_metadata,
):
    run_id = f"perm{manifest['permutation_seed']}_seed{seed}_{method}"
    result_path = output_dir / f"{run_id}.json"
    prediction_path = output_dir / f"{run_id}_pred.npz"
    checkpoint_path = output_dir / f"{run_id}_adapter.pt"
    if result_path.exists() and prediction_path.exists() and checkpoint_path.exists():
        existing = json.loads(result_path.read_text())
        if existing.get("manifest_sha256") == manifest["manifest_sha256"]:
            print(f"skip complete {run_id}", flush=True)
            return existing
        raise RuntimeError(f"existing run has different manifest: {run_id}")

    torch.manual_seed(seed)
    device = train_features.device
    adapter = ResidualAdapter(train_features.shape[1], bottleneck=64).to(device)
    criterion = build_loss(method, manifest["class_counts"], **LOSS_CONFIGS[method]).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=weight_decay)
    assert all(not p.requires_grad for p in classifier.parameters())
    assert all(not tensor.requires_grad for tensor in (train_features, test_features))
    assert set(id(p) for group in optimizer.param_groups for p in group["params"]) == {
        id(p) for p in adapter.parameters()
    }

    train_idx = torch.tensor(manifest["train_indices"], dtype=torch.long, device=device)
    val_idx = torch.tensor(manifest["val_indices"], dtype=torch.long, device=device)
    x_train, y_train = train_features[train_idx], train_targets[train_idx]
    x_val, y_val = train_features[val_idx], train_targets[val_idx]
    best_oa, best_epoch, best_state = -math.inf, None, None
    history = []
    order_rng = torch.Generator().manual_seed(seed + 100_000)
    start_time = time.monotonic()
    for epoch in range(1, epochs + 1):
        adapter.train()
        order = torch.randperm(len(x_train), generator=order_rng)
        total_loss = 0.0
        for position in range(0, len(x_train), batch_size):
            take = order[position:position + batch_size].to(device)
            logits = classifier(adapter(x_train[take]))
            loss = criterion(logits, y_train[take])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite loss in {run_id}, epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()  # Exactly one update for this batch.
            total_loss += float(loss.detach().item()) * len(take)
        adapter.eval()
        with torch.inference_mode():
            val_logits = classifier(adapter(x_val))
            val_oa = float(val_logits.argmax(dim=1).eq(y_val).float().mean().item())
            val_nll = float(F.cross_entropy(val_logits, y_val).item())
        history.append({"epoch": epoch, "train_loss": total_loss / len(x_train),
                        "val_oa": val_oa, "val_nll": val_nll})
        if val_oa > best_oa:
            best_oa, best_epoch = val_oa, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in adapter.state_dict().items()}
    assert best_state is not None
    adapter.load_state_dict(best_state)
    adapter.eval()
    test_metrics, prediction = evaluate(
        adapter, classifier, test_features, test_targets, manifest["groups"]
    )
    if not all(math.isfinite(test_metrics[key]) for key in ("oa", "nll", "ece15")):
        raise FloatingPointError(f"non-finite test metric for {run_id}")
    result = {
        "run_id": run_id,
        "manifest_sha256": manifest["manifest_sha256"],
        "permutation_seed": manifest["permutation_seed"],
        "train_seed": seed,
        "method": method,
        "loss_config": LOSS_CONFIGS[method],
        "model": "fixed-text-prototype CLIP ViT-B/32 + LN/GELU adapter (512-64-512)",
        "optimizer": {"name": "AdamW", "lr": lr, "weight_decay": weight_decay,
                      "batch_size": batch_size, "epochs": epochs},
        "selection": {"criterion": "highest balanced-validation OA; earliest tie",
                      "best_epoch": best_epoch, "best_val_oa": best_oa},
        "test": test_metrics,
        "history": history,
        "feature_metadata": feature_metadata,
        "duration_seconds": round(time.monotonic() - start_time, 2),
    }
    torch.save({"adapter_state_dict": best_state, "manifest_sha256": manifest["manifest_sha256"],
                "run_id": run_id}, checkpoint_path)
    np.savez_compressed(prediction_path, predictions=prediction,
                        targets=test_targets.cpu().numpy().astype(np.uint8))
    atomic_json(result_path, result)
    print(f"{run_id}: val={best_oa:.4f} epoch={best_epoch} "
          f"test={test_metrics['oa']:.4f} many={test_metrics['groups']['many']['accuracy']:.4f} "
          f"medium={test_metrics['groups']['medium']['accuracy']:.4f} "
          f"few={test_metrics['groups']['few']['accuracy']:.4f}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[0, 42, 200])
    parser.add_argument("--methods", nargs="+", choices=list(LOSS_CONFIGS), default=list(LOSS_CONFIGS))
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch size must be positive")
    torch.set_num_threads(4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bank = torch.load(args.features, map_location="cpu", weights_only=False)
    train_targets = bank["train_targets"].long()
    test_targets = bank["test_targets"].long()
    manifests = build_manifests(
        train_targets.tolist(), test_targets.tolist(), args.output_dir / "manifests"
    )
    for manifest in manifests:
        validate_manifest(manifest, train_targets.tolist(), test_targets.tolist())
    assert len(manifests) == 3 and all(
        max(m["class_counts"]) / min(m["class_counts"]) == 100 for m in manifests
    )
    device = torch.device(args.device)
    train_features = bank["train_features"].float().to(device)
    test_features = bank["test_features"].float().to(device)
    train_targets = train_targets.to(device)
    test_targets = test_targets.to(device)
    classifier = PrototypeClassifier(bank["text_prototypes"], bank["logit_scale"]).to(device)
    classifier.eval()

    zero_path = args.output_dir / "zero_shot.json"
    if not zero_path.exists():
        zero_data = {"model": bank["metadata"], "permutations": {}}
        for manifest in manifests:
            metrics, prediction = evaluate(
                None, classifier, test_features, test_targets, manifest["groups"]
            )
            zero_data["permutations"][str(manifest["permutation_seed"])] = {
                "manifest_sha256": manifest["manifest_sha256"], "test": metrics
            }
        np.savez_compressed(args.output_dir / "zero_shot_pred.npz", predictions=prediction,
                            targets=test_targets.cpu().numpy().astype(np.uint8))
        atomic_json(zero_path, zero_data)
        print(f"zero-shot test OA={metrics['oa']:.4f}", flush=True)

    for manifest in manifests:
        for seed in args.train_seeds:
            for method in args.methods:
                run_one(
                    manifest=manifest, seed=seed, method=method, classifier=classifier,
                    train_features=train_features, train_targets=train_targets,
                    test_features=test_features, test_targets=test_targets,
                    output_dir=args.output_dir, epochs=args.epochs,
                    batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
                    feature_metadata=bank["metadata"],
                )


if __name__ == "__main__":
    main()
