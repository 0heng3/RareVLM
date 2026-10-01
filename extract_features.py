"""Extract one immutable CLIP feature bank for the CIFAR-100-LT pilot.

The long-tail manifests refer to original CIFAR indices, so one feature bank
serves every frequency-to-class permutation without re-encoding images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from project_paths import CLIP_B32, CIFAR100_ROOT

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR100
from torchvision.transforms import InterpolationMode



DEFAULT_SNAPSHOT = CLIP_B32
DEFAULT_DATA = CIFAR100_ROOT
PROMPT = "a photo of a {}."


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def extract_images(model, dataset, device, batch_size, workers):
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=True, persistent_workers=workers > 0,
    )
    bank = torch.empty((len(dataset), model.config.projection_dim), dtype=torch.float16)
    offset = 0
    with torch.inference_mode():
        for step, (images, _) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                embeddings = model.get_image_features(pixel_values=images)
            embeddings = F.normalize(embeddings.float(), dim=-1).cpu().half()
            bank[offset:offset + len(embeddings)] = embeddings
            offset += len(embeddings)
            if (step + 1) % 50 == 0:
                print(f"encoded {offset}/{len(dataset)}", flush=True)
    assert offset == len(dataset)
    assert torch.isfinite(bank).all()
    return bank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    # An incomplete feature bank must never be mistaken for a finished one.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import CLIPModel, CLIPTokenizerFast

    preproc = json.loads((args.model_snapshot / "preprocessor_config.json").read_text())
    assert preproc["resample"] == 3 and preproc["crop_size"] == 224
    transform = transforms.Compose([
        transforms.Resize(preproc["size"], interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(preproc["crop_size"]),
        transforms.ToTensor(),
        transforms.Normalize(preproc["image_mean"], preproc["image_std"]),
    ])
    train = CIFAR100(str(args.data_root), train=True, transform=transform, download=False)
    test = CIFAR100(str(args.data_root), train=False, transform=transform, download=False)
    assert len(train) == 50_000 and len(test) == 10_000
    assert train.classes == test.classes
    model = CLIPModel.from_pretrained(
        str(args.model_snapshot), local_files_only=True, use_safetensors=False
    ).eval().to(args.device)
    for param in model.parameters():
        param.requires_grad_(False)
    tokenizer = CLIPTokenizerFast.from_pretrained(str(args.model_snapshot), local_files_only=True)
    prompts = [PROMPT.format(name.replace("_", " ")) for name in train.classes]
    tokens = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
    tokens = {key: value.to(args.device) for key, value in tokens.items()}
    with torch.inference_mode():
        prototypes = F.normalize(model.get_text_features(**tokens).float(), dim=-1).cpu()
        logit_scale = float(model.logit_scale.float().exp().item())
    print(f"model ready: dim={prototypes.shape[1]}, scale={logit_scale:.3f}", flush=True)
    train_features = extract_images(model, train, args.device, args.batch_size, args.workers)
    test_features = extract_images(model, test, args.device, args.batch_size, args.workers)
    payload = {
        "train_features": train_features,
        "test_features": test_features,
        "train_targets": torch.tensor(train.targets, dtype=torch.long),
        "test_targets": torch.tensor(test.targets, dtype=torch.long),
        "text_prototypes": prototypes,
        "logit_scale": logit_scale,
        "class_names": train.classes,
        "metadata": {
            "model": "openai/clip-vit-base-patch32",
            "snapshot": args.model_snapshot.name,
            "weight_sha256": sha256_file(args.model_snapshot / "pytorch_model.bin"),
            "dataset_sha256": sha256_file(args.data_root / "cifar-100-python.tar.gz"),
            "prompt_template": PROMPT,
            "image_preprocessing": preproc,
            "feature_dtype": "float16 stored, float32 used in adapter",
            "tta": False,
            "train_augmentation": False,
        },
    }
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(args.output)
    print(f"saved {args.output} ({args.output.stat().st_size} bytes)", flush=True)


if __name__ == "__main__":
    main()
