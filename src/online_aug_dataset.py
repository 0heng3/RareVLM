"""Deterministic online ImageNet-LT train views for the frozen-CLIP P2 protocol.

The original position in the audited manifest is the sample index. A separate
torch Generator is seeded from (global seed, epoch, sample index), so CE and LA
receive the same crop and flip regardless of sampler order or DataLoader worker
count. Call ``set_epoch`` before starting each epoch's DataLoader iterator.
Validation and test data must keep their existing center-crop preprocessing.
"""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path, PurePosixPath
from typing import Iterable

from project_paths import ROOT

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


SIZE = 224
SCALE = (0.08, 1.0)  # torchvision RandomResizedCrop defaults
RATIO = (3.0 / 4.0, 4.0 / 3.0)  # torchvision RandomResizedCrop defaults
FLIP_PROBABILITY = 0.5
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
SCHEMA = "rarevlm.imagenet_lt.online_aug.v1"


def _sha256_json(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def sampler_order_sha256(indices: Iterable[int]) -> str:
    """Hash manifest indices in the order actually given to the DataLoader."""
    return _sha256_json([int(index) for index in indices])


class OnlineAugmentedTrainDataset(Dataset):
    """Return ``(CLIP-normalized view, label, original manifest index)``.

    ``manifest`` is the audited ImageNet-LT manifest dictionary. ``epoch`` is
    shared across worker processes, including persistent workers. The caller
    must change it only between DataLoader iterators, never during an epoch.
    """

    def __init__(self, manifest: dict, image_root: Path | str, seed: int):
        if manifest.get("schema_version") != "rarevlm.imagenet_lt.v1":
            raise ValueError("unsupported ImageNet-LT manifest schema")
        records = manifest.get("splits", {}).get("train")
        if not isinstance(records, list) or not records:
            raise ValueError("manifest requires a nonempty train split")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.records = records
        self.image_root = Path(image_root).resolve(strict=True)
        if not self.image_root.is_dir():
            raise NotADirectoryError(self.image_root)
        self.seed = seed
        self.manifest_sha256 = manifest.get("manifest_sha256")
        self._epoch = mp.Value("q", 0)

    def __len__(self) -> int:
        return len(self.records)

    @property
    def epoch(self) -> int:
        return self._epoch.value

    def set_epoch(self, epoch: int) -> None:
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        with self._epoch.get_lock():
            self._epoch.value = epoch

    @property
    def augmentation_config(self) -> dict:
        return {
            "schema_version": SCHEMA,
            "size": [SIZE, SIZE],
            "scale": list(SCALE),
            "ratio": list(RATIO),
            "interpolation": "bicubic",
            "antialias": True,
            "horizontal_flip_probability": FLIP_PROBABILITY,
            "image_mean": list(CLIP_MEAN),
            "image_std": list(CLIP_STD),
            "rng_policy": "SHA-256(global_seed,epoch,manifest_index) -> private torch.Generator",
            "worker_seed_policy": "augmentation ignores DataLoader worker RNG",
        }

    def _validate_index(self, index: int) -> int:
        index = int(index)
        if not 0 <= index < len(self.records):
            raise IndexError(index)
        return index

    def _path(self, index: int) -> Path:
        relative = self.records[index]["path"]
        parts = PurePosixPath(relative).parts
        if (len(parts) != 3 or parts[0] != "train" or ".." in parts
                or "\\" in relative or PurePosixPath(relative).is_absolute()):
            raise ValueError(f"unsafe train path at manifest index {index}: {relative!r}")
        return self.image_root / relative

    def _generator(self, seed: int, epoch: int, index: int) -> tuple[torch.Generator, int]:
        key = f"{SCHEMA}|{seed}|{epoch}|{index}".encode("ascii")
        private_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
        return torch.Generator().manual_seed(private_seed), private_seed

    def _params_for_size(self, index: int, epoch: int, height: int, width: int) -> dict:
        """Match RandomResizedCrop.get_params with a private Generator."""
        generator, private_seed = self._generator(self.seed, epoch, index)
        area = height * width
        # Preserve torchvision's float32 log-ratio and sampling order.
        log_ratio = torch.log(torch.tensor(RATIO))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(SCALE[0], SCALE[1], generator=generator).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1], generator=generator)
            ).item()
            crop_width = int(round(math.sqrt(target_area * aspect_ratio)))
            crop_height = int(round(math.sqrt(target_area / aspect_ratio)))
            if 0 < crop_width <= width and 0 < crop_height <= height:
                top = int(torch.randint(0, height - crop_height + 1, (1,), generator=generator).item())
                left = int(torch.randint(0, width - crop_width + 1, (1,), generator=generator).item())
                break
        else:
            in_ratio = width / height
            if in_ratio < min(RATIO):
                crop_width = width
                crop_height = int(round(crop_width / min(RATIO)))
            elif in_ratio > max(RATIO):
                crop_height = height
                crop_width = int(round(crop_height * max(RATIO)))
            else:
                crop_width, crop_height = width, height
            top = (height - crop_height) // 2
            left = (width - crop_width) // 2
        flip = bool(torch.rand(1, generator=generator).item() < FLIP_PROBABILITY)
        return {
            "seed": self.seed,
            "epoch": epoch,
            "sample_index": index,
            "private_generator_seed": private_seed,
            "source_height": height,
            "source_width": width,
            "top": top,
            "left": left,
            "crop_height": crop_height,
            "crop_width": crop_width,
            "flip": flip,
        }

    def augmentation_params(self, index: int, epoch: int | None = None) -> dict:
        """Return JSON-serializable crop/flip parameters without decoding pixels."""
        index = self._validate_index(index)
        epoch = self.epoch if epoch is None else int(epoch)
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        with Image.open(self._path(index)) as image:
            width, height = image.size
        return self._params_for_size(index, epoch, height, width)

    def params_sha256(self, index: int, epoch: int | None = None) -> str:
        return _sha256_json(self.augmentation_params(index, epoch))

    def stream_digest(self, epoch: int, sampled_indices: Iterable[int]) -> str:
        """Hash crop/flip parameters in sampler order for the given indices.

        Reading dimensions for every index has file-system cost; for the P2
        per-epoch preview, pass the first few indices from that epoch's order.
        """
        digest = hashlib.sha256()
        for index in sampled_indices:
            params = self.augmentation_params(int(index), epoch)
            digest.update(json.dumps(params, sort_keys=True, separators=(",", ":")).encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()

    def audit_manifest(self, epoch: int, sampled_indices: Iterable[int], preview_count: int = 8) -> dict:
        """Record the order hash and crop/flip hashes for the first samples."""
        if preview_count < 1:
            raise ValueError("preview_count must be positive")
        order = [self._validate_index(index) for index in sampled_indices]
        preview = order[:preview_count]
        return {
            "augmentation_config": self.augmentation_config,
            "manifest_sha256": self.manifest_sha256,
            "global_seed": self.seed,
            "epoch": int(epoch),
            "sampler_length": len(order),
            "sampler_order_sha256": sampler_order_sha256(order),
            "preview_indices": preview,
            "preview_parameter_sha256": [self.params_sha256(index, epoch) for index in preview],
            "preview_stream_sha256": self.stream_digest(epoch, preview),
        }

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        index = self._validate_index(index)
        epoch = self.epoch
        with Image.open(self._path(index)) as image:
            image = image.convert("RGB")
            width, height = image.size
            params = self._params_for_size(index, epoch, height, width)
            image = TF.resized_crop(
                image, params["top"], params["left"], params["crop_height"],
                params["crop_width"], [SIZE, SIZE], InterpolationMode.BICUBIC,
                antialias=True,
            )
            if params["flip"]:
                image = TF.hflip(image)
            image = TF.normalize(TF.to_tensor(image), CLIP_MEAN, CLIP_STD)
        return image, int(self.records[index]["label"]), index
