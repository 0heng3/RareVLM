"""Cache ImageNet-LT features from one complete, pinned CLIP ViT-B/16 snapshot.

The manifest, image transform, category names, and prompt match the B/32
pilot. Output files use the existing runner's payload schema, but live in a
separate directory and carry B/16 model provenance. Each artifact is saved
atomically so a stopped extraction can resume safely.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

# This environment has an older ONNX protobuf binding imported by torchvision.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from project_paths import ROOT

import torch
from torchvision import transforms
from torchvision.models import ResNet50_Weights
from torchvision.transforms import InterpolationMode

from extract_features import PROMPT, sha256_file
from extract_imagenet_lt_features import (
    ManifestImageDataset,
    SPLITS,
    atomic_json_save,
    atomic_torch_save,
    check_samples,
    content_sha256,
    encode_images,
    encode_text,
    load_manifest,
)


MODEL_ID = "openai/clip-vit-base-patch16"
NUM_CLASSES = 1000
# The B/32 pilot's frozen checkpoint must never enter the B/16 bank.
B32_WEIGHT_SHA256 = "a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f"
SNAPSHOT_FILES = (
    "config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
)
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash tensor values independently of torch.save's serialization."""
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def build_transform_b16(preproc: dict):
    size = preproc.get("size")
    if isinstance(size, dict):
        size = size.get("shortest_edge")
    crop_size = preproc.get("crop_size")
    if isinstance(crop_size, dict):
        if crop_size.get("height") != crop_size.get("width"):
            raise ValueError("B/16 center crop must be square")
        crop_size = crop_size.get("height")
    if (size != 224 or crop_size != 224 or preproc.get("resample") != 3
            or any(preproc.get(name) is not True
                   for name in ("do_resize", "do_center_crop", "do_normalize"))
            or preproc.get("image_mean") != CLIP_MEAN
            or preproc.get("image_std") != CLIP_STD):
        raise ValueError("B/16 preprocessing differs from the frozen B/32 224-pixel protocol")
    return transforms.Compose([
        transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])


def read_snapshot(snapshot: Path, supplied_revision: str | None) -> tuple[dict, dict, object, str, str, dict]:
    if not snapshot.is_dir():
        raise NotADirectoryError(snapshot)
    missing = [name for name in SNAPSHOT_FILES if not (snapshot / name).is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete CLIP snapshot; missing: {', '.join(missing)}")

    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    vision = config.get("vision_config", {})
    if (config.get("model_type") != "clip"
            or config.get("_name_or_path") not in (None, "", MODEL_ID)
            or config.get("projection_dim") != 512
            or vision.get("patch_size") != 16
            or vision.get("image_size") != 224):
        raise ValueError("snapshot config is not the complete openai CLIP ViT-B/16 model")

    preproc = json.loads((snapshot / "preprocessor_config.json").read_text(encoding="utf-8"))
    transform = build_transform_b16(preproc)

    available_weights = [name for name in ("model.safetensors", "pytorch_model.bin")
                         if (snapshot / name).is_file()]
    if not available_weights:
        raise FileNotFoundError("complete B/16 weights missing (safetensors or pytorch_model.bin)")
    # Transformers prefers safetensors when both formats are present.
    weight_name = available_weights[0]
    file_hashes = {name: sha256_file(snapshot / name)
                   for name in (*SNAPSHOT_FILES, weight_name)}
    if file_hashes[weight_name] == B32_WEIGHT_SHA256:
        raise ValueError("B/32 checkpoint weights are forbidden in the B/16 extractor")

    snapshot_revision = snapshot.name if snapshot.parent.name == "snapshots" else None
    revision = supplied_revision or snapshot_revision
    if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("provide --model-revision as the 40-character checkpoint commit SHA")
    if snapshot_revision and supplied_revision and supplied_revision != snapshot_revision:
        raise ValueError("--model-revision disagrees with the Hugging Face snapshot directory")
    return config, preproc, transform, revision, weight_name, file_hashes


def validate_split(path: Path, split: str, records: list[dict],
                   manifest_sha256: str, weight_sha256: str) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    features = payload.get("features")
    expected_targets = torch.tensor([row["label"] for row in records], dtype=torch.long)
    targets = payload.get("targets")
    if (payload.get("manifest_sha256") != manifest_sha256
            or payload.get("model_weight_sha256") != weight_sha256
            or payload.get("split") != split
            or not isinstance(features, torch.Tensor)
            or features.dtype != torch.float16
            or tuple(features.shape) != (len(records), 512)
            or not bool(torch.isfinite(features).all())
            or not isinstance(targets, torch.Tensor)
            or targets.dtype != torch.long
            or not torch.equal(targets, expected_targets)):
        raise ValueError(f"existing {path} disagrees with the B/16 model or manifest")
    return features


def validate_prototypes(path: Path, names: list[str], label_to_wnid: list[str],
                        manifest_sha256: str, weight_sha256: str) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    prototypes = payload.get("text_prototypes")
    if (payload.get("manifest_sha256") != manifest_sha256
            or payload.get("model_weight_sha256") != weight_sha256
            or payload.get("class_names") != names
            or payload.get("label_to_wnid") != label_to_wnid
            or payload.get("prompt_template") != PROMPT
            or not isinstance(prototypes, torch.Tensor)
            or prototypes.dtype != torch.float32
            or tuple(prototypes.shape) != (NUM_CLASSES, 512)
            or not bool(torch.isfinite(prototypes).all())
            or not isinstance(payload.get("logit_scale"), float)
            or not 0 < payload["logit_scale"] < float("inf")):
        raise ValueError(f"existing {path} disagrees with the B/16 model or manifest")
    return prototypes


def record_hash(metadata: dict, key: str, name: str, actual: str) -> None:
    recorded = metadata[key].get(name)
    if recorded is not None and recorded != actual:
        raise ValueError(f"recorded {key}/{name} hash differs from the existing artifact")
    metadata[key][name] = actual


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True,
                        help="Local complete openai/clip-vit-base-patch16 snapshot")
    parser.add_argument("--model-revision", default=None,
                        help="40-character model commit SHA; required for non-HF snapshot paths")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", "--batch-size", dest="batch_size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--check-only", action="store_true",
                        help="Decode sample images without loading the model or writing artifacts")
    parser.add_argument("--check-samples", type=int, default=3)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0 or args.check_samples < 1:
        parser.error("batch size and check samples must be positive; workers must be nonnegative")

    snapshot = args.model_snapshot.resolve(strict=True)
    image_root = args.image_root.resolve(strict=True)
    if not image_root.is_dir():
        raise NotADirectoryError(image_root)
    manifest = load_manifest(args.manifest)
    _, preproc, transform, revision, weight_name, file_hashes = read_snapshot(
        snapshot, args.model_revision
    )
    if args.check_only:
        check_samples(manifest, image_root, transform, args.check_samples)
        return

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    categories = list(ResNet50_Weights.IMAGENET1K_V1.meta["categories"])
    if len(categories) != NUM_CLASSES:
        raise ValueError("torchvision ImageNet categories are not length 1000")
    names = [category.split(",", 1)[0].strip() for category in categories]
    if any(not name for name in names):
        raise ValueError("empty first synonym in torchvision category names")
    mapping = [{"label": label, "wnid": manifest["label_to_wnid"][label],
                "torchvision_category": categories[label], "prompt_name": names[label]}
               for label in range(NUM_CLASSES)]
    manifest_sha256 = manifest["manifest_sha256"]
    weight_sha256 = file_hashes[weight_name]
    static_metadata = {
        "schema_version": "rarevlm.imagenet_lt.features.v1",
        "model": MODEL_ID,
        "model_id": MODEL_ID,
        "snapshot": revision,
        "model_revision": revision,
        "model_snapshot": str(snapshot),
        "weight_file": weight_name,
        "weight_sha256": weight_sha256,
        "model_weight_sha256": weight_sha256,
        "model_config_sha256": file_hashes["config.json"],
        "preprocessor_config_sha256": file_hashes["preprocessor_config.json"],
        "tokenizer_config_sha256": file_hashes["tokenizer_config.json"],
        "tokenizer_sha256": file_hashes["tokenizer.json"],
        "vocab_sha256": file_hashes["vocab.json"],
        "merges_sha256": file_hashes["merges.txt"],
        "special_tokens_map_sha256": file_hashes["special_tokens_map.json"],
        "snapshot_file_sha256": file_hashes,
        "manifest_sha256": manifest_sha256,
        "manifest_file_sha256": sha256_file(args.manifest),
        "image_root": str(image_root),
        "split_source_sha256": manifest["split_sha256"],
        "split_records_sha256": {split: content_sha256(manifest["splits"][split])
                                 for split in SPLITS},
        "split_sizes": {split: len(manifest["splits"][split]) for split in SPLITS},
        "class_counts_sha256": content_sha256(manifest["class_counts"]),
        "class_names_sha256": content_sha256(names),
        "wnid_name_mapping_sha256": content_sha256(mapping),
        "label_to_wnid": manifest["label_to_wnid"],
        "class_names": names,
        "torchvision_categories": categories,
        "prompt_template": PROMPT,
        "prompt_name_rule": "first comma-separated torchvision category synonym, stripped",
        "image_preprocessing": preproc,
        "feature_dtype": "float16 stored, float32 used in adapter",
        "tta": False,
        "train_augmentation": False,
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if any(metadata.get(key) != value for key, value in static_metadata.items()):
            raise ValueError(f"existing {metadata_path} has different B/16 provenance")
        for key in ("split_feature_sha256", "feature_artifact_sha256"):
            if not isinstance(metadata.get(key), dict):
                raise ValueError(f"existing {metadata_path} has invalid {key}")
    else:
        if any((output_dir / name).exists() for name in
               ("train.pt", "val.pt", "test.pt", "text_prototypes.pt")):
            raise ValueError("output artifacts exist without metadata; choose a fresh B/16 directory")
        metadata = {**static_metadata, "split_feature_sha256": {},
                    "feature_artifact_sha256": {},
                    "text_prototype_sha256": None,
                    "text_prototype_artifact_sha256": None}

    prototype_path = output_dir / "text_prototypes.pt"
    if prototype_path.exists():
        prototypes = validate_prototypes(
            prototype_path, names, manifest["label_to_wnid"], manifest_sha256, weight_sha256
        )
        prototype_hash = tensor_sha256(prototypes)
        file_hash = sha256_file(prototype_path)
        if metadata["text_prototype_sha256"] not in (None, prototype_hash):
            raise ValueError("existing text prototype tensor hash disagrees with metadata")
        if metadata["text_prototype_artifact_sha256"] not in (None, file_hash):
            raise ValueError("existing text prototype file hash disagrees with metadata")
        metadata["text_prototype_sha256"] = prototype_hash
        metadata["text_prototype_artifact_sha256"] = file_hash
        print(f"reusing {prototype_path}", flush=True)
    elif metadata["text_prototype_sha256"] is not None:
        raise ValueError("metadata records a missing text prototype artifact")

    for split in SPLITS:
        path = output_dir / f"{split}.pt"
        if path.exists():
            features = validate_split(
                path, split, manifest["splits"][split], manifest_sha256, weight_sha256
            )
            record_hash(metadata, "split_feature_sha256", split, tensor_sha256(features))
            record_hash(metadata, "feature_artifact_sha256", split, sha256_file(path))
            print(f"reusing {path} ({len(features)} images)", flush=True)
        elif (split in metadata["split_feature_sha256"]
              or split in metadata["feature_artifact_sha256"]):
            raise ValueError(f"metadata records a missing {split} feature artifact")

    pending = (["text"] if not prototype_path.exists() else []) + [
        split for split in SPLITS if not (output_dir / f"{split}.pt").exists()
    ]
    if pending:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from transformers import CLIPModel, CLIPTokenizerFast

        model = CLIPModel.from_pretrained(
            str(snapshot), local_files_only=True,
            use_safetensors=weight_name == "model.safetensors",
        ).eval().to(device)
        patch_weight = model.vision_model.embeddings.patch_embedding.weight
        if (model.config.projection_dim != 512
                or model.config.vision_config.patch_size != 16
                or tuple(patch_weight.shape[-2:]) != (16, 16)):
            raise ValueError("loaded weights do not contain a complete CLIP ViT-B/16 visual tower")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        tokenizer = CLIPTokenizerFast.from_pretrained(str(snapshot), local_files_only=True)

        # Provenance is written before the first long encoding step.
        atomic_json_save(metadata, metadata_path)
        if "text" in pending:
            prototypes, logit_scale = encode_text(model, tokenizer, names, device)
            atomic_torch_save({
                "text_prototypes": prototypes,
                "logit_scale": logit_scale,
                "class_names": names,
                "label_to_wnid": manifest["label_to_wnid"],
                "model_weight_sha256": weight_sha256,
                "prompt_template": PROMPT,
                "manifest_sha256": manifest_sha256,
            }, prototype_path)
            metadata["text_prototype_sha256"] = tensor_sha256(prototypes)
            metadata["text_prototype_artifact_sha256"] = sha256_file(prototype_path)
            atomic_json_save(metadata, metadata_path)
            print(f"saved {prototype_path}", flush=True)
        for split in SPLITS:
            if split not in pending:
                continue
            records = manifest["splits"][split]
            dataset = ManifestImageDataset(records, image_root, transform)
            bank = encode_images(model, dataset, device, args.batch_size, args.workers, split)
            path = output_dir / f"{split}.pt"
            atomic_torch_save({
                "features": bank,
                "targets": torch.tensor([row["label"] for row in records], dtype=torch.long),
                "manifest_sha256": manifest_sha256,
                "model_weight_sha256": weight_sha256,
                "split": split,
            }, path)
            metadata["split_feature_sha256"][split] = tensor_sha256(bank)
            metadata["feature_artifact_sha256"][split] = sha256_file(path)
            atomic_json_save(metadata, metadata_path)
            print(f"saved {path}", flush=True)
    elif not metadata_path.exists() or json.loads(metadata_path.read_text(encoding="utf-8")) != metadata:
        atomic_json_save(metadata, metadata_path)


if __name__ == "__main__":
    main()
