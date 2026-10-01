"""Create an audited ImageNet-LT manifest from the three official list files.

List paths are relative to ``--image-root``. No images are copied or decoded.
The output keeps each list's original order for feature extraction and evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path, PurePosixPath


NUM_CLASSES = 1000
SPLITS = ("train", "val", "test")
EXPECTED_EVAL_COUNTS = {"val": 20, "test": 50}
WNID_PATTERN = re.compile(r"n\d{8}\Z")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def read_split(
    list_path: Path, image_root: Path, split: str, label_to_wnid: list[str | None]
) -> tuple[list[dict[str, str | int]], Counter[int]]:
    records: list[dict[str, str | int]] = []
    counts: Counter[int] = Counter()
    seen: set[str] = set()
    expected_directory = "val" if split == "test" else "train"

    with list_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            fields = line.split()
            if len(fields) != 2:
                raise ValueError(f"{list_path}:{line_number}: expected path and label")
            relative, raw_label = fields
            try:
                label = int(raw_label)
            except ValueError as exc:
                raise ValueError(f"{list_path}:{line_number}: invalid label {raw_label!r}") from exc
            if not 0 <= label < NUM_CLASSES:
                raise ValueError(f"{list_path}:{line_number}: label {label} outside 0..999")

            path = PurePosixPath(relative)
            if (path.is_absolute() or len(path.parts) != 3 or ".." in path.parts
                    or "\\" in relative or path.parts[0] != expected_directory):
                raise ValueError(f"{list_path}:{line_number}: invalid ImageNet path {relative!r}")
            wnid = path.parts[1]
            if not WNID_PATTERN.fullmatch(wnid):
                raise ValueError(f"{list_path}:{line_number}: invalid WordNet ID {wnid!r}")
            previous = label_to_wnid[label]
            if previous is not None and previous != wnid:
                raise ValueError(
                    f"{list_path}:{line_number}: label {label} maps to both {previous} and {wnid}"
                )
            label_to_wnid[label] = wnid
            if relative in seen:
                raise ValueError(f"{list_path}:{line_number}: duplicate path {relative!r}")
            seen.add(relative)
            if not (image_root / relative).is_file():
                raise FileNotFoundError(f"{list_path}:{line_number}: missing image {image_root / relative}")
            records.append({"path": relative, "label": label})
            counts[label] += 1

    if len(counts) != NUM_CLASSES:
        missing = sorted(set(range(NUM_CLASSES)) - counts.keys())
        raise ValueError(f"{list_path}: expected all 1000 classes; missing {missing[:20]}")
    expected = EXPECTED_EVAL_COUNTS.get(split)
    if expected is not None:
        incorrect = [(label, counts[label]) for label in range(NUM_CLASSES) if counts[label] != expected]
        if incorrect:
            raise ValueError(f"{list_path}: expected {expected} images per class; examples {incorrect[:10]}")
    return records, counts


def build_manifest(split_dir: Path, image_root: Path) -> dict:
    split_dir = split_dir.resolve(strict=True)
    image_root = image_root.resolve(strict=True)
    if not split_dir.is_dir() or not image_root.is_dir():
        raise NotADirectoryError("--split-dir and --image-root must be directories")

    label_to_wnid: list[str | None] = [None] * NUM_CLASSES
    splits = {}
    split_sha256 = {}
    counts_by_split = {}
    for split in SPLITS:
        list_path = split_dir / f"ImageNet_LT_{split}.txt"
        records, counts = read_split(list_path, image_root, split, label_to_wnid)
        splits[split] = records
        split_sha256[split] = file_sha256(list_path)
        counts_by_split[split] = counts

    path_sets = {name: {record["path"] for record in records} for name, records in splits.items()}
    for i, first in enumerate(SPLITS):
        for second in SPLITS[i + 1:]:
            overlap = path_sets[first] & path_sets[second]
            if overlap:
                raise ValueError(f"{first}/{second} paths overlap; example {next(iter(overlap))}")
    if any(wnid is None for wnid in label_to_wnid):
        raise ValueError("some labels have no WordNet ID")
    if len(set(label_to_wnid)) != NUM_CLASSES:
        raise ValueError("multiple labels map to the same WordNet ID")

    class_counts = [counts_by_split["train"][label] for label in range(NUM_CLASSES)]
    if min(class_counts) <= 0:
        raise ValueError("every training class must have at least one image")
    groups = {
        "many": [label for label, count in enumerate(class_counts) if count > 100],
        "medium": [label for label, count in enumerate(class_counts) if 20 <= count <= 100],
        "few": [label for label, count in enumerate(class_counts) if count < 20],
    }
    manifest = {
        "schema_version": "rarevlm.imagenet_lt.v1",
        "image_root": str(image_root),
        "split_dir": str(split_dir),
        "split_sha256": split_sha256,
        "splits": splits,
        "class_counts": class_counts,
        "groups": groups,
        "label_to_wnid": label_to_wnid,
    }
    manifest["manifest_sha256"] = content_sha256(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = build_manifest(args.split_dir, args.image_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
    temporary.replace(args.output)
    print(
        f"saved {args.output}: "
        + ", ".join(f"{name}={len(manifest['splits'][name])}" for name in SPLITS)
        + f"; groups={{{', '.join(f'{name}: {len(ids)}' for name, ids in manifest['groups'].items())}}}"
        + f"; sha256={manifest['manifest_sha256']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
