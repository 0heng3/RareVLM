"""P2: paired online train views for frozen CLIP ViT-B/32 ImageNet-LT.

CE and LA start from identical adapter weights and consume the same augmented
image tensors and the same frozen CLIP feature on every optimizer step. Their
only training difference is the loss. Validation/test use the original cached
center-crop features; test labels are used only after validation selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from extract_features import DEFAULT_SNAPSHOT
from extract_imagenet_lt_features import load_manifest
from methods import PrototypeClassifier, ResidualAdapter, build_loss
from online_aug_dataset import OnlineAugmentedTrainDataset, sampler_order_sha256
from run_imagenet_lt import (
    EXPECTED_CLIP_WEIGHT_SHA256,
    EXPECTED_PROMPT_TEMPLATE,
    evaluate,
    load_bank,
    save_json,
    sha256_file,
)


RUN_SCHEMA = "rarevlm.imagenet_lt.p2_online_aug.v1"
METHODS = ("ce", "la")
FIXED_EPOCHS = 20
FIXED_BATCH_SIZE = 256
FIXED_LR = 0.001
FIXED_WEIGHT_DECAY = 0.01
AUDIT_PREVIEW_COUNT = 8


class EpochOrderSampler(Sampler[int]):
    """Expose the audited order to one persistent-worker DataLoader."""

    def __init__(self, length: int):
        self.length = length
        self.order: list[int] = []

    def set_order(self, order: torch.Tensor) -> None:
        values = order.tolist()
        if len(values) != self.length or len(set(values)) != self.length:
            raise ValueError("epoch sampler is not a full permutation")
        self.order = values

    def __iter__(self):
        if len(self.order) != self.length:
            raise RuntimeError("set sampler order before iterating the DataLoader")
        return iter(self.order)

    def __len__(self) -> int:
        return self.length


def save_torch_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def save_predictions_atomic(path: Path, predictions: np.ndarray, targets: torch.Tensor) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            predictions=predictions,
            targets=targets.cpu().numpy().astype(np.uint16),
        )
    temporary.replace(path)


def check_frozen_inputs(manifest: dict, feature_dir: Path, snapshot: Path) -> tuple[dict, dict]:
    """Bind online training to the original audited B/32 image/text checkpoint."""
    if sha256_file(snapshot / "pytorch_model.bin") != EXPECTED_CLIP_WEIGHT_SHA256:
        raise ValueError("CLIP B/32 weight SHA-256 differs from the frozen protocol")
    metadata = json.loads((feature_dir / "metadata.json").read_text(encoding="utf-8"))
    if (metadata.get("model_weight_sha256") != EXPECTED_CLIP_WEIGHT_SHA256
            or metadata.get("prompt_template") != EXPECTED_PROMPT_TEMPLATE
            or metadata.get("label_to_wnid") != manifest["label_to_wnid"]
            or metadata.get("manifest_sha256") != manifest["manifest_sha256"]
            or metadata.get("train_augmentation") is not False
            or metadata.get("tta") is not False):
        raise ValueError("original feature metadata disagrees with frozen P2 protocol")
    prototype_bank = torch.load(feature_dir / "text_prototypes.pt", map_location="cpu", weights_only=False)
    if (prototype_bank.get("manifest_sha256") != manifest["manifest_sha256"]
            or prototype_bank.get("model_weight_sha256") != EXPECTED_CLIP_WEIGHT_SHA256
            or prototype_bank.get("prompt_template") != EXPECTED_PROMPT_TEMPLATE
            or prototype_bank.get("label_to_wnid") != manifest["label_to_wnid"]
            or tuple(prototype_bank["text_prototypes"].shape) != (1000, 512)
            or len(prototype_bank["class_names"]) != 1000):
        raise ValueError("prototype bank is not the original B/32 text bank")
    metadata["feature_artifact_sha256"] = {
        name: sha256_file(feature_dir / name)
        for name in ("train.pt", "val.pt", "test.pt", "text_prototypes.pt")
    }
    return metadata, prototype_bank


def paired_preflight(dataset: OnlineAugmentedTrainDataset, manifest: dict,
                     seed: int, epochs: int) -> dict:
    """Independently replay CE/LA sampler and crop streams before training."""
    peer = OnlineAugmentedTrainDataset(manifest, dataset.image_root, seed)
    ce_rng = torch.Generator().manual_seed(seed + 100_000)
    la_rng = torch.Generator().manual_seed(seed + 100_000)
    plans = []
    for epoch in range(1, epochs + 1):
        ce_order = torch.randperm(len(dataset), generator=ce_rng)
        la_order = torch.randperm(len(peer), generator=la_rng)
        if not torch.equal(ce_order, la_order):
            raise AssertionError(f"CE/LA sampler order differs at epoch {epoch}")
        ce_audit = dataset.audit_manifest(epoch, ce_order, AUDIT_PREVIEW_COUNT)
        la_audit = peer.audit_manifest(epoch, la_order, AUDIT_PREVIEW_COUNT)
        if ce_audit != la_audit:
            raise AssertionError(f"CE/LA augmentation stream differs at epoch {epoch}")
        plans.append(ce_audit)
    probe = list(range(min(AUDIT_PREVIEW_COUNT, len(dataset))))
    first = [dataset.params_sha256(index, 1) for index in probe]
    second = [dataset.params_sha256(index, 2) for index in probe]
    if first == second:
        raise AssertionError("different epochs unexpectedly produce the same crop/flip probes")
    return {
        "schema_version": RUN_SCHEMA,
        "manifest_sha256": manifest["manifest_sha256"],
        "seed": seed,
        "sampler_seed": seed + 100_000,
        "worker_seed_policy": dataset.augmentation_config["worker_seed_policy"],
        "paired_view_policy": "one shared image batch and one frozen CLIP encoding for CE and LA",
        "ce_la_independent_preflight_equal": True,
        "cross_epoch_probe_indices": probe,
        "cross_epoch_probe_changed": True,
        "epochs": plans,
    }


def validate_or_save_audit(path: Path, audit: dict) -> str:
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != audit:
            raise ValueError(f"existing pairing audit differs: {path}")
    else:
        save_json(path, audit)
    return sha256_file(path)


def run_paths(output_dir: Path, seed: int, method: str) -> tuple[str, Path, Path, Path]:
    run_id = f"imagenet_lt_seed{seed}_aug_{method}"
    return (run_id, output_dir / f"{run_id}.json",
            output_dir / f"{run_id}_adapter.pt", output_dir / f"{run_id}_pred.npz")


def check_existing(output_dir: Path, seed: int, manifest: dict,
                   metadata: dict, optimizer_config: dict, audit_sha256: str,
                   augmentation_config: dict, micro_batch_size: int) -> bool:
    complete = []
    for method in METHODS:
        run_id, result_path, checkpoint_path, prediction_path = run_paths(output_dir, seed, method)
        files = (result_path, checkpoint_path, prediction_path)
        if not any(path.exists() for path in files):
            complete.append(False)
            continue
        if not all(path.exists() for path in files):
            raise ValueError(f"incomplete existing output for {run_id}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (result.get("schema_version") != RUN_SCHEMA
                or result.get("run_id") != run_id
                or result.get("manifest_sha256") != manifest["manifest_sha256"]
                or result.get("train_seed") != seed
                or result.get("method") != method
                or result.get("optimizer") != optimizer_config
                or result.get("effective_batch_size") != FIXED_BATCH_SIZE
                or result.get("physical_micro_batch_size") != micro_batch_size
                or result.get("augmentation") != augmentation_config
                or result.get("bank_metadata") != metadata
                or result.get("pair_audit_sha256") != audit_sha256
                or result.get("checkpoint_sha256") != sha256_file(checkpoint_path)
                or result.get("prediction_sha256") != sha256_file(prediction_path)):
            raise ValueError(f"existing output has different protocol or hash: {run_id}")
        complete.append(True)
    if any(complete) and not all(complete):
        raise ValueError("one paired method exists without the other; use a fresh output directory")
    return all(complete)


def load_frozen_clip(snapshot: Path, device: torch.device):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import CLIPModel

    model = CLIPModel.from_pretrained(
        str(snapshot), local_files_only=True, use_safetensors=False
    ).eval().to(device)
    if model.config.projection_dim != 512:
        raise ValueError("frozen CLIP projection dimension must be 512")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("CLIP has trainable parameters")
    return model


def encode_online(model, images: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Mirror the original CLIP extraction path, then return float32 features."""
    autocast = (torch.autocast(device_type="cuda", dtype=torch.float16)
                if device.type == "cuda" else nullcontext())
    with torch.no_grad():
        with autocast:
            encoded = model.get_image_features(pixel_values=images.to(device, non_blocking=True))
        features = F.normalize(encoded.float(), dim=-1)
    if features.dtype != torch.float32 or features.requires_grad or not bool(torch.isfinite(features).all()):
        raise FloatingPointError("invalid frozen CLIP online features")
    return features


def train_pair(*, manifest: dict, dataset: OnlineAugmentedTrainDataset, audit: dict,
               audit_sha256: str, model, classifier: PrototypeClassifier,
               val_x: torch.Tensor, val_y: torch.Tensor,
               test_x: torch.Tensor, test_y: torch.Tensor,
               bank_metadata: dict, output_dir: Path, seed: int,
               batch_size: int, micro_batch_size: int, workers: int,
               lr: float, weight_decay: float) -> None:
    device = val_x.device
    torch.manual_seed(seed)
    adapters = {"ce": ResidualAdapter(512, bottleneck=64).to(device)}
    adapters["la"] = ResidualAdapter(512, bottleneck=64).to(device)
    adapters["la"].load_state_dict(adapters["ce"].state_dict())
    if not all(torch.equal(adapters["ce"].state_dict()[key], adapters["la"].state_dict()[key])
               for key in adapters["ce"].state_dict()):
        raise AssertionError("CE/LA adapters do not share the same initialization")
    losses = {
        "ce": build_loss("ce").to(device),
        "la": build_loss("la", manifest["class_counts"], tau=1.0).to(device),
    }
    optimizers = {
        method: torch.optim.AdamW(adapters[method].parameters(), lr=lr, weight_decay=weight_decay)
        for method in METHODS
    }
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("the frozen CLIP model became trainable")
    for method in METHODS:
        actual = {id(p) for group in optimizers[method].param_groups for p in group["params"]}
        expected = {id(p) for p in adapters[method].parameters()}
        if actual != expected:
            raise AssertionError(f"{method}: optimizer updates parameters other than its adapter")

    sampler = EpochOrderSampler(len(dataset))
    worker_generator = torch.Generator().manual_seed(seed + 300_000)
    loader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, num_workers=workers,
        pin_memory=device.type == "cuda", persistent_workers=workers > 0,
        generator=worker_generator,
    )
    train_targets = torch.tensor([int(record["label"]) for record in manifest["splits"]["train"]])
    order_rng = torch.Generator().manual_seed(seed + 100_000)
    histories: dict[str, list[dict]] = {method: [] for method in METHODS}
    best = {method: {"oa": -math.inf, "epoch": None, "state": None} for method in METHODS}
    started = time.monotonic()
    for epoch in range(1, FIXED_EPOCHS + 1):
        dataset.set_epoch(epoch)
        order = torch.randperm(len(dataset), generator=order_rng)
        expected_audit = audit["epochs"][epoch - 1]
        if sampler_order_sha256(order) != expected_audit["sampler_order_sha256"]:
            raise AssertionError(f"sampler order changed since paired preflight: epoch {epoch}")
        current_audit = dataset.audit_manifest(epoch, order, AUDIT_PREVIEW_COUNT)
        if current_audit != expected_audit:
            raise AssertionError(f"augmentation stream changed since preflight: epoch {epoch}")
        sampler.set_order(order)
        for method in METHODS:
            adapters[method].train()
        running_loss = {method: 0.0 for method in METHODS}
        seen = 0
        for images, labels, original_indices in loader:
            count = len(labels)
            expected_indices = order[seen:seen + count]
            if (not torch.equal(original_indices, expected_indices)
                    or not torch.equal(labels, train_targets[original_indices])):
                raise AssertionError(f"DataLoader labels/order disagree with manifest: epoch {epoch}")
            for method in METHODS:
                optimizers[method].zero_grad(set_to_none=True)
            for start in range(0, count, micro_batch_size):
                stop = min(count, start + micro_batch_size)
                features = encode_online(model, images[start:stop], device)
                y = labels[start:stop].to(device, non_blocking=True)
                weight = (stop - start) / count
                for method in METHODS:
                    logits = classifier(adapters[method](features))
                    loss = losses[method](logits, y)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError(f"non-finite {method} loss at epoch {epoch}")
                    (loss * weight).backward()
                    running_loss[method] += float(loss.detach().item()) * (stop - start)
            for method in METHODS:
                optimizers[method].step()  # Exactly one update per effective batch.
            seen += count
        if seen != len(dataset):
            raise AssertionError(f"epoch {epoch} trained on {seen}/{len(dataset)} images")
        status = []
        for method in METHODS:
            adapters[method].eval()
            with torch.inference_mode():
                logits = classifier(adapters[method](val_x))
                val_oa = float(logits.argmax(dim=1).eq(val_y).float().mean().item())
                val_nll = float(F.cross_entropy(logits, val_y).item())
            histories[method].append({
                "epoch": epoch,
                "train_loss": running_loss[method] / len(dataset),
                "val_oa": val_oa,
                "val_nll": val_nll,
                "sampler_order_sha256": expected_audit["sampler_order_sha256"],
                "preview_stream_sha256": expected_audit["preview_stream_sha256"],
            })
            if val_oa > best[method]["oa"]:  # strict: earliest epoch wins ties
                best[method]["oa"] = val_oa
                best[method]["epoch"] = epoch
                best[method]["state"] = {
                    key: value.detach().cpu().clone()
                    for key, value in adapters[method].state_dict().items()
                }
            status.append(f"{method} loss={running_loss[method]/len(dataset):.4f} "
                          f"val={val_oa:.4f} best={best[method]['oa']:.4f}@{best[method]['epoch']}")
        print(f"P2 seed{seed} epoch={epoch} " + " | ".join(status), flush=True)

    optimizer_config = {
        "name": "AdamW", "lr": lr, "weight_decay": weight_decay,
        "batch_size": batch_size, "epochs": FIXED_EPOCHS,
    }
    for method in METHODS:
        chosen = best[method]
        if chosen["state"] is None:
            raise AssertionError(f"no selected {method} adapter")
        adapters[method].load_state_dict(chosen["state"])
        adapters[method].eval()
        metrics, predictions = evaluate(
            adapters[method], classifier, test_x, test_y, manifest["groups"]
        )
        if not all(math.isfinite(metrics[key]) for key in ("oa", "nll", "ece15", "macro_f1")):
            raise FloatingPointError(f"non-finite {method} test metrics")
        run_id, result_path, checkpoint_path, prediction_path = run_paths(output_dir, seed, method)
        save_torch_atomic(checkpoint_path, {
            "adapter_state_dict": chosen["state"],
            "manifest_sha256": manifest["manifest_sha256"],
            "run_id": run_id,
        })
        save_predictions_atomic(prediction_path, predictions, test_y)
        result = {
            "schema_version": RUN_SCHEMA,
            "run_id": run_id,
            "manifest_sha256": manifest["manifest_sha256"],
            "method": method,
            "train_seed": seed,
            "model": "fixed-text-prototype CLIP ViT-B/32 + LN/GELU adapter (512-64-512); online train views",
            "loss": {"ce": {}, "la": {"tau": 1.0}}[method],
            "optimizer": optimizer_config,
            "effective_batch_size": batch_size,
            "physical_micro_batch_size": micro_batch_size,
            "train_feature_path": "raw image -> random resized crop/flip -> frozen CLIP -> normalized float32",
            "augmentation": dataset.augmentation_config,
            "pair_audit_sha256": audit_sha256,
            "selection": {
                "criterion": "highest balanced-validation OA; earliest tie",
                "best_epoch": chosen["epoch"], "best_val_oa": chosen["oa"],
            },
            "test": metrics,
            "history": histories[method],
            "bank_metadata": bank_metadata,
            "duration_seconds": round(time.monotonic() - started, 2),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "prediction_sha256": sha256_file(prediction_path),
        }
        save_json(result_path, result)
        print(f"{run_id}: OA={metrics['oa']:.4f} Many={metrics['groups']['many']['accuracy']:.4f} "
              f"Medium={metrics['groups']['medium']['accuracy']:.4f} "
              f"Few={metrics['groups']['few']['accuracy']:.4f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=FIXED_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=FIXED_BATCH_SIZE)
    parser.add_argument("--micro-batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=FIXED_LR)
    parser.add_argument("--weight-decay", type=float, default=FIXED_WEIGHT_DECAY)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if (args.epochs != FIXED_EPOCHS or args.batch_size != FIXED_BATCH_SIZE
            or args.lr != FIXED_LR or args.weight_decay != FIXED_WEIGHT_DECAY):
        parser.error("P2 freezes 20 epochs, effective batch 256, AdamW lr=.001, weight decay=.01")
    if args.seed < 0 or args.workers < 0 or not 1 <= args.micro_batch_size <= args.batch_size:
        parser.error("seed/workers must be nonnegative; micro batch must be within 1..256")
    torch.set_num_threads(4)
    image_root = args.image_root.resolve(strict=True)
    snapshot = args.model_snapshot.resolve(strict=True)
    manifest = load_manifest(args.manifest)
    if image_root != Path(manifest["image_root"]).resolve(strict=True):
        raise ValueError("image root differs from audited manifest")
    bank_metadata, prototype_bank = check_frozen_inputs(manifest, args.feature_dir, snapshot)
    dataset = OnlineAugmentedTrainDataset(manifest, image_root, args.seed)
    audit = paired_preflight(dataset, manifest, args.seed, args.epochs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.output_dir / f"online_aug_seed{args.seed}_audit.json"
    audit_sha256 = validate_or_save_audit(audit_path, audit)
    print(f"paired sampler/crop preflight passed: {audit_path} ({audit_sha256})", flush=True)
    if args.preflight_only:
        return

    optimizer_config = {
        "name": "AdamW", "lr": args.lr, "weight_decay": args.weight_decay,
        "batch_size": args.batch_size, "epochs": args.epochs,
    }
    if check_existing(args.output_dir, args.seed, manifest, bank_metadata,
                      optimizer_config, audit_sha256,
                      dataset.augmentation_config, args.micro_batch_size):
        print(f"skip complete paired P2 seed{args.seed}", flush=True)
        return
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    val_x, val_y = load_bank(
        args.feature_dir / "val.pt", "val", manifest["manifest_sha256"],
        manifest["splits"]["val"], device,
    )
    test_x, test_y = load_bank(
        args.feature_dir / "test.pt", "test", manifest["manifest_sha256"],
        manifest["splits"]["test"], device,
    )
    classifier = PrototypeClassifier(
        prototype_bank["text_prototypes"], prototype_bank["logit_scale"]
    ).to(device).eval()
    model = load_frozen_clip(snapshot, device)
    train_pair(
        manifest=manifest, dataset=dataset, audit=audit, audit_sha256=audit_sha256,
        model=model, classifier=classifier, val_x=val_x, val_y=val_y,
        test_x=test_x, test_y=test_y, bank_metadata=bank_metadata,
        output_dir=args.output_dir, seed=args.seed,
        batch_size=args.batch_size, micro_batch_size=args.micro_batch_size,
        workers=args.workers, lr=args.lr, weight_decay=args.weight_decay,
    )


if __name__ == "__main__":
    main()
