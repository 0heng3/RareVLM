"""Reproducible CIFAR-100-LT manifests for the RareVLM pilot.

Indices in ``train_indices`` and ``val_indices`` address the original CIFAR-100
*training* set. Indices in ``test_indices`` address the separate official test
set. The labels can be supplied directly from torchvision CIFAR100 ``targets``;
this module never downloads data or changes the original datasets.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Sequence


NUM_CLASSES = 100
TRAIN_PER_CLASS = 500
TEST_PER_CLASS = 100
DEFAULT_PERMUTATION_SEEDS = (0, 42, 200)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _class_indices(labels: Sequence[int], expected_per_class: int, name: str) -> list[list[int]]:
    indices: list[list[int]] = [[] for _ in range(NUM_CLASSES)]
    for index, raw_label in enumerate(labels):
        label = int(raw_label)
        if label < 0 or label >= NUM_CLASSES:
            raise ValueError(f"{name}[{index}] has invalid class ID {label}")
        indices[label].append(index)
    if len(labels) != NUM_CLASSES * expected_per_class:
        raise ValueError(f"{name} must contain {NUM_CLASSES * expected_per_class} images")
    if any(len(class_ids) != expected_per_class for class_ids in indices):
        raise ValueError(f"{name} must have exactly {expected_per_class} images per class")
    return indices


def _frequency_counts(head_count: int, imbalance_ratio: int) -> list[int]:
    if head_count <= 0 or imbalance_ratio <= 0 or head_count % imbalance_ratio:
        raise ValueError("head_count must be positive and divisible by imbalance_ratio")
    counts = [
        int(head_count * (1.0 / imbalance_ratio) ** (rank / (NUM_CLASSES - 1)))
        for rank in range(NUM_CLASSES)
    ]
    counts[0] = head_count
    counts[-1] = head_count // imbalance_ratio
    return counts


def validate_manifest(
    manifest: dict,
    train_labels: Sequence[int],
    test_labels: Sequence[int],
) -> None:
    """Raise ValueError if a manifest violates its saved split contract."""
    train_labels = [int(label) for label in train_labels]
    test_labels = [int(label) for label in test_labels]
    _class_indices(train_labels, TRAIN_PER_CLASS, "train_labels")
    _class_indices(test_labels, TEST_PER_CLASS, "test_labels")

    train = [int(index) for index in manifest["train_indices"]]
    val = [int(index) for index in manifest["val_indices"]]
    test = [int(index) for index in manifest["test_indices"]]
    pool_by_class = manifest["train_pool_indices_by_class"]
    counts = [int(count) for count in manifest["class_counts"]]
    rank_counts = [int(count) for count in manifest["frequency_rank_counts"]]
    rank_to_class = [int(label) for label in manifest["frequency_rank_to_class"]]
    val_per_class = int(manifest["val_per_class"])

    if len(pool_by_class) != NUM_CLASSES or len(counts) != NUM_CLASSES:
        raise ValueError("pool and class_counts must contain 100 classes")
    if sorted(rank_to_class) != list(range(NUM_CLASSES)) or len(rank_counts) != NUM_CLASSES:
        raise ValueError("frequency ranks must be a permutation of 100 classes")
    if any(counts[label] != rank_counts[rank] for rank, label in enumerate(rank_to_class)):
        raise ValueError("class_counts disagree with the frequency permutation")
    if len(train) != sum(counts) or len(set(train)) != len(train):
        raise ValueError("train_indices have an incorrect length or duplicates")
    if len(val) != NUM_CLASSES * val_per_class or len(set(val)) != len(val):
        raise ValueError("val_indices have an incorrect length or duplicates")
    if set(train) & set(val):
        raise ValueError("training and validation indices overlap")
    if any(index < 0 or index >= len(train_labels) for index in train + val):
        raise ValueError("training or validation index is outside CIFAR-100 train")
    if test != list(range(len(test_labels))):
        raise ValueError("test_indices must cover the untouched official test set")

    pool_flat: list[int] = []
    for label, raw_pool in enumerate(pool_by_class):
        pool = [int(index) for index in raw_pool]
        if len(pool) != TRAIN_PER_CLASS - val_per_class or len(set(pool)) != len(pool):
            raise ValueError(f"class {label} training pool has the wrong size or duplicates")
        if any(train_labels[index] != label for index in pool):
            raise ValueError(f"class {label} training pool contains another class")
        pool_flat.extend(pool)
    if len(set(pool_flat)) != len(pool_flat) or set(pool_flat) & set(val):
        raise ValueError("training pools overlap each other or validation")
    if set(pool_flat) | set(val) != set(range(len(train_labels))):
        raise ValueError("training pools and validation do not partition CIFAR-100 train")
    expected_train = sorted(
        index
        for label, pool in enumerate(pool_by_class)
        for index in pool[: counts[label]]
    )
    if train != expected_train or val != sorted(val):
        raise ValueError("selected training or validation indices are not canonical")
    if any(sum(train_labels[index] == label for index in val) != val_per_class for label in range(NUM_CLASSES)):
        raise ValueError("validation does not contain the required count per class")

    expected_groups = {
        "many": [label for label, count in enumerate(counts) if count > 100],
        "medium": [label for label, count in enumerate(counts) if 20 <= count <= 100],
        "few": [label for label, count in enumerate(counts) if count < 20],
    }
    if manifest["groups"] != expected_groups:
        raise ValueError("Many/Medium/Few groups disagree with training counts")
    if manifest["train_labels_sha256"] != _sha256(train_labels):
        raise ValueError("training labels hash mismatch")
    if manifest["test_labels_sha256"] != _sha256(test_labels):
        raise ValueError("test labels hash mismatch")
    if manifest["train_pool_sha256"] != _sha256(pool_by_class):
        raise ValueError("training pool hash mismatch")
    expected_hash = _sha256({key: value for key, value in manifest.items() if key != "manifest_sha256"})
    if manifest.get("manifest_sha256") != expected_hash:
        raise ValueError("manifest content hash mismatch")


def build_manifests(
    train_labels: Sequence[int],
    test_labels: Sequence[int],
    output_dir: str | Path,
    *,
    split_seed: int = 2026,
    permutation_seeds: Sequence[int] = DEFAULT_PERMUTATION_SEEDS,
    val_per_class: int = 50,
    head_count: int = 400,
    imbalance_ratio: int = 100,
) -> list[dict]:
    """Write and return three standalone, audited CIFAR-100-LT JSON manifests.

    Every permutation uses the same shuffled 450-image pool for each class.
    Its class-specific long-tail subset is a prefix of that shared pool, so
    changing the frequency assignment never changes the underlying split.
    """
    train_labels = [int(label) for label in train_labels]
    test_labels = [int(label) for label in test_labels]
    train_by_class = _class_indices(train_labels, TRAIN_PER_CLASS, "train_labels")
    _class_indices(test_labels, TEST_PER_CLASS, "test_labels")
    if val_per_class != 50 or head_count != 400 or imbalance_ratio != 100:
        raise ValueError("RareVLM pilot requires val_per_class=50, head_count=400, IR=100")
    seeds = [int(seed) for seed in permutation_seeds]
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("provide exactly three distinct permutation seeds")

    rng = random.Random(int(split_seed))
    pool_by_class: list[list[int]] = []
    val_indices: list[int] = []
    for original_indices in train_by_class:
        shuffled = original_indices.copy()
        rng.shuffle(shuffled)
        val_indices.extend(shuffled[:val_per_class])
        pool_by_class.append(shuffled[val_per_class:])
    val_indices.sort()

    rank_counts = _frequency_counts(head_count, imbalance_ratio)
    pool_hash = _sha256(pool_by_class)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifests: list[dict] = []
    for permutation_seed in seeds:
        rank_to_class = list(range(NUM_CLASSES))
        random.Random(permutation_seed).shuffle(rank_to_class)
        counts = [0] * NUM_CLASSES
        for rank, label in enumerate(rank_to_class):
            counts[label] = rank_counts[rank]
        train_indices = sorted(
            index for label, pool in enumerate(pool_by_class) for index in pool[: counts[label]]
        )
        groups = {
            "many": [label for label, count in enumerate(counts) if count > 100],
            "medium": [label for label, count in enumerate(counts) if 20 <= count <= 100],
            "few": [label for label, count in enumerate(counts) if count < 20],
        }
        manifest = {
            "schema_version": "rarevlm.cifar100_lt.v1",
            "dataset": "CIFAR-100",
            "index_sources": {
                "train_indices": "official_train",
                "val_indices": "official_train",
                "test_indices": "official_test",
            },
            "split_seed": int(split_seed),
            "permutation_seed": permutation_seed,
            "val_per_class": val_per_class,
            "train_pool_per_class": TRAIN_PER_CLASS - val_per_class,
            "head_count": head_count,
            "imbalance_ratio": imbalance_ratio,
            "frequency_rank_counts": rank_counts,
            "frequency_rank_to_class": rank_to_class,
            "class_counts": counts,
            "groups": groups,
            "train_pool_indices_by_class": pool_by_class,
            "train_indices": train_indices,
            "val_indices": val_indices,
            "test_indices": list(range(len(test_labels))),
            "train_labels_sha256": _sha256(train_labels),
            "test_labels_sha256": _sha256(test_labels),
            "train_pool_sha256": pool_hash,
        }
        manifest["manifest_sha256"] = _sha256(manifest)
        validate_manifest(manifest, train_labels, test_labels)
        destination = output_path / f"cifar100_lt_ir100_val50_split{split_seed}_perm{permutation_seed}.json"
        destination.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        manifests.append(manifest)
    return manifests
