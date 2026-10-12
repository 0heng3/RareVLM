"""Cache frozen CLIP features for an audited ImageNet-LT manifest.

Each split is written atomically and can be reused after an interrupted run.
The model receives only image pixels; labels are copied from the manifest into
the saved payload after encoding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from contextlib import nullcontext
from pathlib import Path, PurePosixPath

from project_paths import ROOT

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights
from torchvision.transforms import InterpolationMode

from extract_features import DEFAULT_SNAPSHOT, PROMPT, sha256_file


SPLITS = ("train", "val", "test")
NUM_CLASSES = 1000


def content_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = content_sha256({key: value for key, value in manifest.items()
                               if key != "manifest_sha256"})
    if manifest.get("manifest_sha256") != expected:
        raise ValueError("manifest content hash mismatch")
    if manifest.get("schema_version") != "rarevlm.imagenet_lt.v1":
        raise ValueError("unsupported ImageNet-LT manifest schema")
    if set(manifest.get("splits", {})) != set(SPLITS):
        raise ValueError("manifest must contain train, val, and test splits")
    label_to_wnid = manifest.get("label_to_wnid")
    if (not isinstance(label_to_wnid, list) or len(label_to_wnid) != NUM_CLASSES
            or len(set(label_to_wnid)) != NUM_CLASSES):
        raise ValueError("manifest needs 1000 distinct label_to_wnid entries")
    if not all(isinstance(wnid, str) and len(wnid) == 9 and wnid[0] == "n"
               and wnid[1:].isdigit() for wnid in label_to_wnid):
        raise ValueError("invalid WordNet ID in label_to_wnid")

    seen_paths: set[str] = set()
    for split in SPLITS:
        records = manifest["splits"][split]
        if not isinstance(records, list) or not records:
            raise ValueError(f"{split}: expected a nonempty record list")
        expected_top_dir = "val" if split == "test" else "train"
        for index, record in enumerate(records):
            if not isinstance(record, dict) or set(record) != {"path", "label"}:
                raise ValueError(f"{split}[{index}]: expected path and label")
            relative = record["path"]
            label = record["label"]
            if not isinstance(relative, str) or not isinstance(label, int) or isinstance(label, bool):
                raise ValueError(f"{split}[{index}]: invalid path or label type")
            parts = PurePosixPath(relative).parts
            if (len(parts) != 3 or parts[0] != expected_top_dir or
                    ".." in parts or "\\" in relative or PurePosixPath(relative).is_absolute()):
                raise ValueError(f"{split}[{index}]: invalid relative image path")
            if not 0 <= label < NUM_CLASSES or parts[1] != label_to_wnid[label]:
                raise ValueError(f"{split}[{index}]: path WordNet ID disagrees with label")
            if relative in seen_paths:
                raise ValueError(f"{split}[{index}]: duplicate image path")
            seen_paths.add(relative)
    train_counts = torch.bincount(
        torch.tensor([row["label"] for row in manifest["splits"]["train"]]),
        minlength=NUM_CLASSES,
    ).tolist()
    if manifest.get("class_counts") != train_counts:
        raise ValueError("manifest class_counts disagree with train records")
    return manifest


def build_transform(snapshot: Path):
    preproc = json.loads((snapshot / "preprocessor_config.json").read_text())
    if preproc["resample"] != 3 or preproc["crop_size"] != 224:
        raise ValueError("model preprocessing differs from the CIFAR CLIP pilot")
    transform = transforms.Compose([
        transforms.Resize(preproc["size"], interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(preproc["crop_size"]),
        transforms.ToTensor(),
        transforms.Normalize(preproc["image_mean"], preproc["image_std"]),
    ])
    return transform, preproc


class ManifestImageDataset(Dataset):
    def __init__(self, records: list[dict], image_root: Path, transform):
        self.records = records
        self.image_root = image_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> torch.Tensor:
        # Targets never enter the model's image encoder.
        image_path = self.image_root / self.records[index]["path"]
        with Image.open(image_path) as image:
            return self.transform(image.convert("RGB"))


def check_samples(manifest: dict, image_root: Path, transform, count: int) -> None:
    for split in SPLITS:
        dataset = ManifestImageDataset(manifest["splits"][split], image_root, transform)
        indices = sorted({0, len(dataset) // 2, len(dataset) - 1})[:count]
        for index in indices:
            image = dataset[index]
            if image.shape != (3, 224, 224) or not bool(torch.isfinite(image).all()):
                raise ValueError(f"{split}[{index}]: invalid transformed image")
        print(f"{split}: decoded {len(indices)} sample(s), total={len(dataset)}", flush=True)


def atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def atomic_json_save(payload: dict, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_existing_split(path: Path, split: str, records: list[dict], manifest_sha256: str) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected_targets = torch.tensor([record["label"] for record in records], dtype=torch.long)
    features = payload.get("features")
    targets = payload.get("targets")
    if (payload.get("manifest_sha256") != manifest_sha256 or payload.get("split") != split
            or not isinstance(features, torch.Tensor) or features.dtype != torch.float16
            or features.shape != (len(records), 512)
            or not bool(torch.isfinite(features).all())
            or not isinstance(targets, torch.Tensor) or targets.dtype != torch.long
            or not torch.equal(targets, expected_targets)):
        raise ValueError(f"existing {path} does not match this manifest")
    print(f"reusing {path} ({len(records)} images)", flush=True)


def validate_existing_prototypes(path: Path, names: list[str], label_to_wnid: list[str],
                                 model_weight_sha256: str, manifest_sha256: str) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    prototypes = payload.get("text_prototypes")
    if (payload.get("manifest_sha256") != manifest_sha256
            or payload.get("class_names") != names
            or payload.get("label_to_wnid") != label_to_wnid
            or payload.get("model_weight_sha256") != model_weight_sha256
            or payload.get("prompt_template") != PROMPT
            or not isinstance(prototypes, torch.Tensor)
            or prototypes.dtype != torch.float32
            or prototypes.shape != (NUM_CLASSES, 512)
            or not bool(torch.isfinite(prototypes).all())
            or not isinstance(payload.get("logit_scale"), float)
            or not 0 < payload["logit_scale"] < float("inf")):
        raise ValueError(f"existing {path} does not match this manifest")
    print(f"reusing {path}", flush=True)


def encode_text(model, tokenizer, names: list[str], device: torch.device) -> tuple[torch.Tensor, float]:
    prompts = [PROMPT.format(name) for name in names]
    batches = []
    with torch.inference_mode():
        for start in range(0, len(prompts), 128):
            tokens = tokenizer(prompts[start:start + 128], padding=True, truncation=True,
                               return_tensors="pt")
            tokens = {key: value.to(device) for key, value in tokens.items()}
            batches.append(F.normalize(model.get_text_features(**tokens).float(), dim=-1).cpu())
        logit_scale = float(model.logit_scale.float().exp().item())
    prototypes = torch.cat(batches, dim=0)
    if prototypes.shape != (NUM_CLASSES, 512) or not bool(torch.isfinite(prototypes).all()):
        raise ValueError("invalid text prototypes")
    return prototypes, logit_scale


def encode_images(model, dataset: Dataset, device: torch.device, batch_size: int,
                  workers: int, split: str) -> torch.Tensor:
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=device.type == "cuda", persistent_workers=workers > 0,
    )
    bank = torch.empty((len(dataset), 512), dtype=torch.float16)
    offset = 0
    with torch.inference_mode():
        for step, images in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            autocast = (torch.autocast(device_type="cuda", dtype=torch.float16)
                        if device.type == "cuda" else nullcontext())
            with autocast:
                embeddings = model.get_image_features(pixel_values=images)
            embeddings = F.normalize(embeddings.float(), dim=-1).cpu().half()
            bank[offset:offset + len(embeddings)] = embeddings
            offset += len(embeddings)
            if step % 50 == 0:
                print(f"{split}: encoded {offset}/{len(dataset)}", flush=True)
    if offset != len(dataset) or not bool(torch.isfinite(bank).all()):
        raise ValueError(f"{split}: incomplete or nonfinite feature bank")
    return bank


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", "--batch-size", dest="batch_size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--check-only", action="store_true",
                        help="Decode a few images per split without loading CLIP or writing output")
    parser.add_argument("--check-samples", type=int, default=3)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0 or args.check_samples < 1:
        parser.error("batch size and check samples must be positive; workers must be nonnegative")

    image_root = args.image_root.resolve(strict=True)
    if not image_root.is_dir():
        raise NotADirectoryError(image_root)
    manifest = load_manifest(args.manifest)
    transform, preproc = build_transform(args.model_snapshot)
    if args.check_only:
        check_samples(manifest, image_root, transform, args.check_samples)
        return

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_sha256 = manifest["manifest_sha256"]
    categories = list(ResNet50_Weights.IMAGENET1K_V1.meta["categories"])
    if len(categories) != NUM_CLASSES:
        raise ValueError("torchvision ImageNet categories are not length 1000")
    names = [category.split(",", 1)[0].strip() for category in categories]
    if any(not name for name in names):
        raise ValueError("empty first synonym in torchvision category names")
    mapping = [{"label": label, "wnid": manifest["label_to_wnid"][label],
                "torchvision_category": categories[label], "prompt_name": names[label]}
               for label in range(NUM_CLASSES)]
    weight_sha256 = sha256_file(args.model_snapshot / "pytorch_model.bin")
    metadata = {
        "schema_version": "rarevlm.imagenet_lt.features.v1",
        "model": "openai/clip-vit-base-patch32",
        "snapshot": args.model_snapshot.name,
        "weight_sha256": weight_sha256,
        "model_weight_sha256": weight_sha256,
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
    metadata_path = args.output_dir / "metadata.json"
    if metadata_path.exists() and json.loads(metadata_path.read_text(encoding="utf-8")) != metadata:
        raise ValueError(f"existing {metadata_path} has different provenance")

    prototype_path = args.output_dir / "text_prototypes.pt"
    if prototype_path.exists():
        validate_existing_prototypes(
            prototype_path, names, manifest["label_to_wnid"], weight_sha256,
            manifest_sha256,
        )
    for split in SPLITS:
        path = args.output_dir / f"{split}.pt"
        if path.exists():
            validate_existing_split(path, split, manifest["splits"][split], manifest_sha256)

    # Save provenance before any large encoding job, so an interrupted run
    # cannot later reuse its partial outputs with a different CLIP snapshot.
    if not metadata_path.exists():
        atomic_json_save(metadata, metadata_path)
        print(f"saved {metadata_path}", flush=True)

    pending = (["text"] if not prototype_path.exists() else []) + [
        split for split in SPLITS if not (args.output_dir / f"{split}.pt").exists()
    ]
    if pending:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from transformers import CLIPModel, CLIPTokenizerFast

        model = CLIPModel.from_pretrained(
            str(args.model_snapshot), local_files_only=True, use_safetensors=False
        ).eval().to(device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        if model.config.projection_dim != 512:
            raise ValueError("CLIP projection dimension differs from the CIFAR pilot")
        if "text" in pending:
            tokenizer = CLIPTokenizerFast.from_pretrained(
                str(args.model_snapshot), local_files_only=True
            )
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
            print(f"saved {prototype_path}", flush=True)
        for split in SPLITS:
            if split not in pending:
                continue
            records = manifest["splits"][split]
            dataset = ManifestImageDataset(records, image_root, transform)
            bank = encode_images(model, dataset, device, args.batch_size, args.workers, split)
            atomic_torch_save({
                "features": bank,
                "targets": torch.tensor([row["label"] for row in records], dtype=torch.long),
                "manifest_sha256": manifest_sha256,
                "split": split,
            }, args.output_dir / f"{split}.pt")
            print(f"saved {args.output_dir / f'{split}.pt'}", flush=True)
if __name__ == "__main__":
    main()
