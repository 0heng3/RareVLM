"""P1 ImageNet-LT replication with a full CLIP ViT-B/16 checkpoint.

The B/16 feature bank must carry its own audited model and preprocessing
provenance. This runner never reuses the historical B/32 feature bank.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from imagenet_lt_manifest import content_sha256
from extract_imagenet_lt_features import load_manifest
from methods import PrototypeClassifier, ResidualAdapter, build_loss

EXPECTED_MODEL_ID = "openai/clip-vit-base-patch16"
EXPECTED_MODEL_REVISION = "57c216476eefef5ab752ec549e440a49ae4ae5f3"
EXPECTED_MANIFEST_SHA256: str | None = None
EXPECTED_CLASS_NAMES_SHA256 = "62fff941ecff3f19de9128c6ca9c2097807c6b7bafc5552d7589a219431221ed"
EXPECTED_WNID_NAME_MAPPING_SHA256 = "f40692d20defb014047d9eca1e21ba53dc42d6fef1e0508c84bee1b19cf29ba2"
B32_WEIGHT_SHA256 = "a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f"
FROZEN_B16_WEIGHT_SHA256 = "ec89c7b09c749a60aae3c9cd910516f24b58214a7df060b48962d14c469cfbf0"
EXPECTED_PROMPT_TEMPLATE = "a photo of a {}."
EXPECTED_SEEDS = [0, 42, 200]
EXPECTED_METHODS = ["ce", "la"]
EXPECTED_SNAPSHOT_SHA256 = {
    "config.json": "eaf1c9089a8553c913d27ea66407f8bfc2be9989c80c9f331ddb3d63d4c5e8ad",
    "merges.txt": "9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a",
    "preprocessor_config.json": "910e70b3956ac9879ebc90b22fb3bc8a75b6a0677814500101a4c072bd7857bd",
    "pytorch_model.bin": FROZEN_B16_WEIGHT_SHA256,
    "special_tokens_map.json": "f8c0d6c39aee3f8431078ef6646567b0aba7f2246e9c54b8b99d55c22b707cbf",
    "tokenizer.json": "a83e0809aa4c3af7208b2df632a7a69668c6d48775b3c3fe4e1b1199d1f8b8f4",
    "tokenizer_config.json": "3f74b6a4d97fee70b3b96921a687fc5e3f2fb583f5f81e7c178c0c22281beade",
    "vocab.json": "3f0c4f7d2086b61b38487075278ea9ed04edb53a03cbb045b86c27190fa8fb69",
}
EXPECTED_CONFIG_HASHES = {
    "model_config_sha256": "config.json",
    "preprocessor_config_sha256": "preprocessor_config.json",
    "tokenizer_config_sha256": "tokenizer_config.json",
    "tokenizer_sha256": "tokenizer.json",
    "vocab_sha256": "vocab.json",
    "merges_sha256": "merges.txt",
    "special_tokens_map_sha256": "special_tokens_map.json",
}


def require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def is_224_size(value: object, *, crop: bool) -> bool:
    if value == 224:
        return True
    if crop:
        return value == {"height": 224, "width": 224}
    return value in ({"shortest_edge": 224}, {"height": 224, "width": 224})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def save_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_bank(path: Path, split: str, manifest_hash: str, manifest_records: list[dict],
              expected_weight_sha: str, expected_tensor_sha: str, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (payload.get("manifest_sha256") != manifest_hash
            or payload.get("split") != split
            or payload.get("model_weight_sha256") != expected_weight_sha):
        raise ValueError(f"feature bank provenance mismatch: {path}")
    features = payload["features"]
    targets = payload["targets"].long()
    expected = torch.tensor([int(item["label"]) for item in manifest_records], dtype=torch.long)
    if (features.shape != (len(expected), 512) or features.dtype != torch.float16
            or not torch.equal(targets, expected)):
        raise ValueError(f"feature bank shape/labels disagree with manifest: {split}")
    if not bool(torch.isfinite(features).all()):
        raise ValueError(f"feature bank has non-finite values: {split}")
    if tensor_sha256(features) != expected_tensor_sha:
        raise ValueError(f"feature bank tensor values disagree with metadata: {split}")
    return features.float().to(device), targets.to(device)


def validate_provenance(metadata: dict, prototype_bank: dict, manifest: dict,
                        manifest_path: Path, expected_weight_sha: str) -> None:
    if (metadata.get("schema_version") != "rarevlm.imagenet_lt.features.v1"
            or metadata.get("model") != EXPECTED_MODEL_ID
            or metadata.get("model_id") != EXPECTED_MODEL_ID
            or metadata.get("model_revision") != EXPECTED_MODEL_REVISION
            or metadata.get("snapshot") != EXPECTED_MODEL_REVISION
            or metadata.get("weight_file") != "pytorch_model.bin"
            or metadata.get("model_weight_sha256") != expected_weight_sha
            or metadata.get("weight_sha256") != expected_weight_sha
            or metadata.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256
            or metadata.get("manifest_file_sha256") != sha256_file(manifest_path)
            or metadata.get("label_to_wnid") != manifest["label_to_wnid"]
            or metadata.get("prompt_template") != EXPECTED_PROMPT_TEMPLATE
            or metadata.get("prompt_name_rule") != "first comma-separated torchvision category synonym, stripped"
            or metadata.get("split_source_sha256") != manifest["split_sha256"]
            or metadata.get("split_sizes") != {s: len(manifest["splits"][s]) for s in ("train", "val", "test")}
            or metadata.get("split_records_sha256") != {
                s: content_sha256(manifest["splits"][s]) for s in ("train", "val", "test")
            }
            or metadata.get("class_counts_sha256") != content_sha256(manifest["class_counts"])
            or metadata.get("class_names_sha256") != EXPECTED_CLASS_NAMES_SHA256
            or metadata.get("wnid_name_mapping_sha256") != EXPECTED_WNID_NAME_MAPPING_SHA256
            or metadata.get("feature_dtype") != "float16 stored, float32 used in adapter"
            or metadata.get("train_augmentation") is not False
            or metadata.get("tta") is not False):
        raise ValueError("B/16 metadata differs from frozen data, model, or protocol")
    if (metadata.get("class_names") != prototype_bank.get("class_names")
            or metadata.get("class_names_sha256") != content_sha256(prototype_bank["class_names"])):
        raise ValueError("class names differ between metadata and B/16 text prototypes")
    snapshot_hashes = metadata.get("snapshot_file_sha256")
    if not isinstance(snapshot_hashes, dict):
        raise ValueError("B/16 snapshot file hashes are missing")
    snapshot_path = Path(metadata.get("model_snapshot", ""))
    if not snapshot_path.is_dir() or snapshot_path.name != EXPECTED_MODEL_REVISION:
        raise ValueError("B/16 metadata does not point to the frozen local snapshot")
    for filename, digest in EXPECTED_SNAPSHOT_SHA256.items():
        if (snapshot_hashes.get(filename) != digest
                or sha256_file(snapshot_path / filename) != digest):
            raise ValueError(f"B/16 snapshot hash mismatch: {filename}")
    for field, filename in EXPECTED_CONFIG_HASHES.items():
        if metadata.get(field) != EXPECTED_SNAPSHOT_SHA256[filename]:
            raise ValueError(f"B/16 config hash mismatch: {field}")
    preproc = metadata.get("image_preprocessing")
    if (not isinstance(preproc, dict)
            or not is_224_size(preproc.get("crop_size"), crop=True)
            or not is_224_size(preproc.get("size"), crop=False)
            or preproc.get("resample") != 3
            or preproc.get("do_center_crop") is not True
            or preproc.get("do_resize") is not True
            or preproc.get("do_normalize") is not True
            or preproc.get("image_mean") != [0.48145466, 0.4578275, 0.40821073]
            or preproc.get("image_std") != [0.26862954, 0.26130258, 0.27577711]):
        raise ValueError("B/16 image preprocessing differs from the frozen 224-pixel protocol")


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
    run_id = f"imagenet_lt_b16_seed{seed}_{method}"
    result_file = output_dir / f"{run_id}.json"
    prediction_file = output_dir / f"{run_id}_pred.npz"
    checkpoint_file = output_dir / f"{run_id}_adapter.pt"
    optimizer_config = {"name": "AdamW", "lr": lr, "weight_decay": weight_decay,
                        "batch_size": batch_size, "epochs": epochs}
    existing = [result_file.exists(), prediction_file.exists(), checkpoint_file.exists()]
    if any(existing) and not all(existing):
        raise ValueError(f"incomplete B/16 run output; inspect before resuming: {run_id}")
    if all(existing):
        result = json.loads(result_file.read_text())
        if (result.get("manifest_sha256") == manifest["manifest_sha256"]
                and result.get("optimizer") == optimizer_config
                and result.get("bank_metadata") == bank_metadata
                and result.get("method") == method
                and result.get("train_seed") == seed
                and result.get("model_id") == EXPECTED_MODEL_ID
                and result.get("model_revision") == EXPECTED_MODEL_REVISION
                and result.get("model_weight_sha256") == FROZEN_B16_WEIGHT_SHA256
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
        "model": "fixed-text-prototype CLIP ViT-B/16 + LN/GELU adapter (512-64-512)",
        "model_id": EXPECTED_MODEL_ID,
        "model_revision": EXPECTED_MODEL_REVISION,
        "model_weight_sha256": FROZEN_B16_WEIGHT_SHA256,
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
                "model_weight_sha256": FROZEN_B16_WEIGHT_SHA256,
                "model_revision": EXPECTED_MODEL_REVISION,
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
    global EXPECTED_MANIFEST_SHA256
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-weight-sha", required=True,
                        help="SHA-256 of the frozen B/16 pytorch_model.bin")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--train-seeds", nargs="+", type=int, default=EXPECTED_SEEDS)
    parser.add_argument("--methods", nargs="+", choices=EXPECTED_METHODS, default=EXPECTED_METHODS)
    args = parser.parse_args()
    expected_weight_sha = require_sha256(args.expected_weight_sha, "--expected-weight-sha")
    if expected_weight_sha in (B32_WEIGHT_SHA256,) or expected_weight_sha != FROZEN_B16_WEIGHT_SHA256:
        raise ValueError("--expected-weight-sha must be the audited B/16 checkpoint SHA-256")
    if (args.epochs != 20 or args.batch_size != 256 or args.lr != 0.001
            or args.weight_decay != 0.01 or args.train_seeds != EXPECTED_SEEDS
            or args.methods != EXPECTED_METHODS):
        raise ValueError("P1 training hyperparameters, seeds, and CE/LA matrix are frozen")
    b32_output_root = (Path(__file__).resolve().parent / "artifacts" / "imagenet_lt").resolve()
    if (args.output_dir.resolve().is_relative_to(b32_output_root)
            or args.feature_dir.resolve().is_relative_to(b32_output_root)
            or args.output_dir.resolve().is_relative_to(args.feature_dir.resolve())
            or args.feature_dir.resolve().is_relative_to(args.output_dir.resolve())):
        raise ValueError("B/16 features/results must have distinct directories outside B/32 history")
    torch.set_num_threads(4)
    manifest = load_manifest(args.manifest)
    EXPECTED_MANIFEST_SHA256 = manifest["manifest_sha256"]
    if (content_sha256({k: v for k, v in manifest.items() if k != "manifest_sha256"})
            != EXPECTED_MANIFEST_SHA256
            or manifest.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256
            or manifest.get("schema_version") != "rarevlm.imagenet_lt.v1"):
        raise ValueError("P1 requires the exact frozen ImageNet-LT manifest")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    metadata_path = args.feature_dir / "metadata.json"
    raw_metadata = json.loads(metadata_path.read_text())
    prototype_bank = torch.load(args.feature_dir / "text_prototypes.pt", map_location="cpu", weights_only=False)
    validate_provenance(raw_metadata, prototype_bank, manifest, args.manifest, expected_weight_sha)
    if (tuple(prototype_bank["text_prototypes"].shape) != (1000, 512)
            or len(prototype_bank["class_names"]) != 1000
            or prototype_bank.get("label_to_wnid") != manifest["label_to_wnid"]
            or prototype_bank.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256
            or prototype_bank.get("model_weight_sha256") != expected_weight_sha
            or prototype_bank.get("prompt_template") != EXPECTED_PROMPT_TEMPLATE):
        raise ValueError("text prototype order, model weight, or prompt mismatch")
    if (raw_metadata.get("text_prototype_sha256") != tensor_sha256(prototype_bank["text_prototypes"])
            or raw_metadata.get("text_prototype_artifact_sha256")
            != sha256_file(args.feature_dir / "text_prototypes.pt")):
        raise ValueError("B/16 text prototype hashes differ from feature metadata")
    split_artifact_hashes = raw_metadata.get("feature_artifact_sha256")
    split_tensor_hashes = raw_metadata.get("split_feature_sha256")
    if (not isinstance(split_artifact_hashes, dict)
            or not isinstance(split_tensor_hashes, dict)
            or set(split_artifact_hashes) != {"train", "val", "test"}
            or set(split_tensor_hashes) != {"train", "val", "test"}):
        raise ValueError("B/16 feature metadata is incomplete")
    for split in ("train", "val", "test"):
        require_sha256(split_tensor_hashes[split], f"{split} tensor SHA-256")
        if split_artifact_hashes[split] != sha256_file(args.feature_dir / f"{split}.pt"):
            raise ValueError(f"B/16 {split} artifact differs from feature metadata")
    banks = {
        split: load_bank(
            args.feature_dir / f"{split}.pt", split, EXPECTED_MANIFEST_SHA256,
            manifest["splits"][split], expected_weight_sha, split_tensor_hashes[split], device
        ) for split in ("train", "val", "test")
    }
    classifier = PrototypeClassifier(
        prototype_bank["text_prototypes"], prototype_bank["logit_scale"]
    ).to(device).eval()
    bank_metadata = dict(raw_metadata)
    bank_metadata["extractor_feature_artifact_sha256"] = split_artifact_hashes
    bank_metadata["feature_artifact_sha256"] = {
        name: sha256_file(args.feature_dir / name)
        for name in ("train.pt", "val.pt", "test.pt", "text_prototypes.pt")
    }
    bank_metadata["metadata_file_sha256"] = sha256_file(metadata_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    zero_file = args.output_dir / "zero_shot.json"
    zero_prediction_file = args.output_dir / "zero_shot_pred.npz"
    if zero_file.exists() or zero_prediction_file.exists():
        if not (zero_file.exists() and zero_prediction_file.exists()):
            raise ValueError("incomplete zero-shot output; use a fresh output directory")
        existing_zero = json.loads(zero_file.read_text())
        if (existing_zero.get("manifest_sha256") != manifest["manifest_sha256"]
                or existing_zero.get("model_id") != EXPECTED_MODEL_ID
                or existing_zero.get("model_revision") != EXPECTED_MODEL_REVISION
                or existing_zero.get("model_weight_sha256") != expected_weight_sha
                or existing_zero.get("bank_metadata") != bank_metadata
                or existing_zero.get("prediction_sha256") != sha256_file(zero_prediction_file)):
            raise ValueError("existing zero-shot output uses different manifest or features")
    else:
        metrics, predictions = evaluate(None, classifier, *banks["test"], manifest["groups"])
        np.savez_compressed(zero_prediction_file, predictions=predictions,
                            targets=banks["test"][1].cpu().numpy().astype(np.uint16))
        save_json(zero_file, {
            "method": "zero_shot", "manifest_sha256": manifest["manifest_sha256"],
            "model_id": EXPECTED_MODEL_ID,
            "model_revision": EXPECTED_MODEL_REVISION,
            "model_weight_sha256": expected_weight_sha,
            "test": metrics, "bank_metadata": bank_metadata,
            "prediction_sha256": sha256_file(zero_prediction_file),
        })
        print(f"zero-shot: OA={metrics['oa']:.4f} macroF1={metrics['macro_f1']:.4f}", flush=True)
    for method in args.methods:
        for seed in args.train_seeds:
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
