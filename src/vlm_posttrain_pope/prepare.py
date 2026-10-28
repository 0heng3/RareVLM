"""Prepare original POPE annotations and deduplicate one pinned mirror shard."""
from __future__ import annotations

from collections import Counter
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
import urllib.request

from vlm_probe.evaluate import private_path, read_json, sha_file, write_json

OFFICIAL_COMMIT = "08d957b917e5a378a2f99d35b6293c536a66298b"
MIRROR_REVISION = "4db1276663dfa5eb8ad16a52d24c31a09e470896"
MIRROR_REPOSITORY = "lmms-lab-encoder/POPE"
CATEGORIES = ("random", "popular", "adversarial")
PROMPT_SUFFIX = "\nAnswer yes or no."
ANNOTATIONS = {
    "random": ("ac25245170b975a5bdf9080b23fd431dfe6be458bc038259c1f4f09a6bef7994", 368166),
    "popular": ("72c1a8ad45d0c13514f5f22598261df41d3b533854d29682e924db50ed8aa753", 370190),
    "adversarial": ("420b3407db1fa9f1187a805dca41cb7b97fd91504e6c2179706188c107fb8ef8", 370401),
}
SCORER_SHA = "f0371dcec5f1bdbde017d1a02f5a9a04c53cb561a2874f3fb8f0718ce3d87036"
MIRROR_PATH = "data/test-00000-of-00003.parquet"
MIRROR_SHA = "38851606d0e9f473c69dd45e3b018286842070473f168f7fa1c82e4c5086f116"
MIRROR_BYTES = 85007167


def content_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def coco_id(value):
    match = re.search(r"COCO_(?:train|val)(?:2014|2017)_(\d{12})(?:\.jpg)?$", str(value))
    if not match:
        raise ValueError(f"unexpected original COCO image source: {value}")
    return match.group(1)


def download(url, path, expected_sha, expected_bytes):
    path = private_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.stat().st_size != expected_bytes or sha_file(path) != expected_sha:
            raise ValueError(f"existing pinned source bytes differ: {path}")
        return path
    temporary = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "RareVLM-fixed-POPE-audit"})
    digest, count = hashlib.sha256(), 0
    with urllib.request.urlopen(request, timeout=40) as response, temporary.open("wb") as stream:
        while True:
            chunk = response.read(2**20)
            if not chunk:
                break
            stream.write(chunk); digest.update(chunk); count += len(chunk)
            if count > expected_bytes:
                raise ValueError("pinned source exceeds declared byte count")
            if expected_bytes > 2**23 and count % (8 * 2**20) == 0:
                print("DOWNLOADED", path.name, count, "/", expected_bytes, flush=True)
        stream.flush(); os.fsync(stream.fileno())
    if count != expected_bytes or digest.hexdigest() != expected_sha:
        raise ValueError(f"pinned download size/LFS SHA differs: {path}")
    temporary.replace(path)
    return path


def lock_json(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"prepared POPE source lock changed: {path}")
    else:
        write_json(path, value)


def prepare_annotations(data_root):
    data_root = private_path(data_root)
    files, rows, category_images = {}, [], {}
    for category in CATEGORIES:
        name = f"coco_pope_{category}.json"
        expected_sha, expected_bytes = ANNOTATIONS[category]
        url = f"https://raw.githubusercontent.com/AoiDragon/POPE/{OFFICIAL_COMMIT}/output/coco/{name}"
        path = download(url, data_root / name, expected_sha, expected_bytes)
        records = [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]
        if len(records) != 3000 or len({r["question_id"] for r in records}) != 3000:
            raise ValueError("original category must contain 3000 unique questions")
        ids = [coco_id(r["image"]) for r in records]
        if len(set(ids)) != 500 or set(Counter(ids).values()) != {6}:
            raise ValueError("original category must contain six questions for each of 500 images")
        if Counter(r["label"] for r in records) != {"yes": 1500, "no": 1500}:
            raise ValueError("original category label balance differs")
        category_images[category] = set(ids)
        rows.extend({"question_id": f"pope_{category}_{r['question_id']}",
            "source_question_id": str(r["question_id"]), "image_id": f"COCO_val2014_{iid}",
            "question": r["text"], "label": r["label"], "category": category,
            "prompt_suffix": PROMPT_SUFFIX} for r, iid in zip(records, ids))
        files[name] = {"url": url, "sha256": expected_sha, "bytes": expected_bytes,
            "questions": 3000, "unique_images": 500}
    if any(ids != category_images[CATEGORIES[0]] for ids in category_images.values()):
        raise ValueError("original categories do not share the same 500 images")
    download(f"https://raw.githubusercontent.com/AoiDragon/POPE/{OFFICIAL_COMMIT}/evaluate.py",
        data_root / "official_evaluate.py", SCORER_SHA, 1617)
    files["official_evaluate.py"] = {"sha256": SCORER_SHA, "bytes": 1617}
    source_lock = {"official_repo": "AoiDragon/POPE", "official_commit": OFFICIAL_COMMIT,
        "mirror_dataset": MIRROR_REPOSITORY, "mirror_revision": MIRROR_REVISION, "files": files}
    lock_json(data_root / "SOURCE_LOCK.json", source_lock)
    ids = sorted(category_images[CATEGORIES[0]])
    id_lock = {"schema_version": "rarevlm.pope.source_ids.v1", "official_commit": OFFICIAL_COMMIT,
        "coco_ids": [int(i) for i in ids], "canonical_source_ids": [f"COCO_val2014_{i}" for i in ids],
        "unique_images": 500, "questions": 9000, "coco_image_ids": ids,
        "repository": "AoiDragon/POPE", "revision": OFFICIAL_COMMIT,
        "files_sha256": {name: value["sha256"] for name, value in files.items()}}
    lock_json(data_root / "COCO_SOURCE_IDS.json", id_lock)
    return rows, source_lock, id_lock


def extract_images(data_root, records, parquet_path):
    """Validate 3000 mirror rows against originals; keep the 500 unique JPEGs."""
    import pyarrow.parquet as pq
    from PIL import Image
    data_root, parquet_path = private_path(data_root), private_path(parquet_path)
    if parquet_path.stat().st_size != MIRROR_BYTES or sha_file(parquet_path) != MIRROR_SHA:
        raise ValueError("mirror shard does not match fixed LFS bytes")
    parquet = pq.ParquetFile(parquet_path)
    if parquet.metadata.num_rows != 3000:
        raise ValueError("expected one full 3000-row mirror category")
    lookup = {r["source_question_id"]: r for r in records if r["category"] == "adversarial"}
    images, occurrences, seen_questions, text_differences = {}, Counter(), set(), []
    for batch in parquet.iter_batches(batch_size=32):
        for row in batch.to_pylist():
            qid = str(row["question_id"])
            if qid in seen_questions:
                raise ValueError("duplicate mirror question")
            seen_questions.add(qid)
            original = lookup[qid]
            iid = f"COCO_val2014_{coco_id(row['image_source'])}"
            if (row["category"] != "adversarial" or row["answer"] != original["label"] or iid != original["image_id"]):
                raise ValueError("mirror category/question/answer/image differs from pinned original")
            if row["question"] != original["question"]:
                if row["question"].replace("imange", "image") != original["question"]:
                    raise ValueError("unexpected mirror question difference from original")
                text_differences.append({"source_question_id": qid, "mirror_question": row["question"],
                    "original_question_used_for_evaluation": original["question"]})
            raw = row["image"]["bytes"]
            digest = hashlib.sha256(raw).hexdigest()
            occurrences[iid] += 1
            if iid in images:
                if images[iid]["sha256"] != digest:
                    raise ValueError("same COCO source has inconsistent repeated image bytes")
                continue
            with Image.open(BytesIO(raw)) as image:
                rgb = image.convert("RGB")
                pixel_sha = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                width, height = rgb.size
            destination = data_root / "images" / f"{iid}.jpg"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if sha_file(destination) != digest:
                    raise ValueError("existing unique POPE image changed")
            else:
                destination.write_bytes(raw)
            images[iid] = {"path": str(destination.relative_to(data_root)), "sha256": digest,
                "pixel_sha256": pixel_sha, "width": width, "height": height,
                "coco_image_id": coco_id(iid)}
    expected_ids = {r["image_id"] for r in records}
    if (set(images) != expected_ids or len(images) != 500 or len(seen_questions) != 3000
        or set(occurrences.values()) != {6} or len(text_differences) != 57):
        raise ValueError("mirror image coverage/repetition differs from original 500-image set")
    audit = {"mirror_repository": MIRROR_REPOSITORY, "mirror_revision": MIRROR_REVISION,
        "mirror_file": MIRROR_PATH, "mirror_file_sha256": MIRROR_SHA, "mirror_file_bytes": MIRROR_BYTES,
        "parquet_schema": str(parquet.schema_arrow), "mirror_rows": 3000, "mirror_category": "adversarial",
        "unique_images": 500, "occurrences_per_image": 6,
        "crosschecked_original_fields": ["question_id", "image_source", "question", "answer", "category"],
        "question_text_differences": text_differences,
        "question_text_difference_policy": "57 pinned mirror imange->image spelling differences recorded; generation always uses the unmodified original GitHub JSON text",
        "repeated_image_bytes_consistent": True,
        "pixel_sha256_definition": "SHA256(str(RGB.size).encode()+RGB.tobytes())"}
    lock_json(data_root / "MIRROR_AUDIT.json", audit)
    return images, audit


def prepare(data_root, *, download_mirror=False, mirror_parquet=None):
    data_root = private_path(data_root)
    records, source, ids = prepare_annotations(data_root)
    if download_mirror and mirror_parquet is not None:
        raise ValueError("choose one mirror source")
    if download_mirror:
        mirror_parquet = download(f"https://huggingface.co/datasets/{MIRROR_REPOSITORY}/resolve/{MIRROR_REVISION}/{MIRROR_PATH}",
            data_root / "mirror" / Path(MIRROR_PATH).name, MIRROR_SHA, MIRROR_BYTES)
    if mirror_parquet is None:
        return {"status": "annotations_ready_images_not_downloaded", "questions": 9000,
            "unique_images": 500, "coco_source_ids": str(data_root / "COCO_SOURCE_IDS.json")}
    images, audit = extract_images(data_root, records, mirror_parquet)
    manifest = {"schema_version": "rarevlm.posttrain.pope.data.v1", "repository": "AoiDragon/POPE",
        "revision": OFFICIAL_COMMIT, "source": source, "mirror_audit": audit,
        "split": "original_COCO_external_diagnostic", "questions": 9000, "unique_images": 500,
        "categories": list(CATEGORIES), "images": images, "records": records,
        "prompt_suffix": PROMPT_SUFFIX,
        "scope": "frozen once-only external diagnostic; not checkpoint selection or task tuning",
        "pretraining_overlap": "unknown; no claim of unseen public benchmark"}
    manifest["manifest_sha256"] = content_sha(manifest)
    lock_json(data_root / "manifest.json", manifest)
    return {"status": "complete", "questions": 9000, "unique_images": 500,
        "manifest": str(data_root / "manifest.json"), "manifest_sha256": manifest["manifest_sha256"],
        "coco_source_ids": str(data_root / "COCO_SOURCE_IDS.json")}
