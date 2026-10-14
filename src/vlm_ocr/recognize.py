"""Image-only CPU OCR recognition over frozen PaddleOCR detection polygons.

This stage preserves raw recognition text and confidence for every valid crop.
It never reads question text, target, references, previous predictions or scores.
Question IDs are used only to declare the first 32-QA input-only API preflight.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time
from datetime import datetime, timezone

from vlm_region.detect import apply_headless_compatibility, private_path, sha_file, sha_object, write_json

PRIVATE = Path("/opt/data/private")
BASE = PRIVATE / "longtail_vlm/rarevlm_pilot/outputs"
DEFAULT_OUTPUT = BASE / "vlm_ocr_context_v1"
DEFAULT_MANIFEST = BASE / "vlm_deep_v2/prepared/manifest.json"
DEFAULT_DETECTIONS = BASE / "vlm_region_token_v1/detector/DETECTIONS.json"
MODEL_NAME = "en_PP-OCRv4_mobile_rec"
MODEL_URL = "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/en_PP-OCRv4_mobile_rec_infer.tar"
ARCHIVE_SHA256 = "42067abe43473ce9d68e3b4bf050765038062acfd810f7e1738af078c3198c8a"
MODEL_FILE_SHA256 = {
    "inference.json": "7a2302e09f0d501458db082cbcf21cdcb3a8d6dc9a143f73b17e297505f922f9",
    "inference.pdiparams": "75f64a1ffb70c56b7a25655963ca16f5bf3286202e3f52ac972bee05cdee2f56",
    "inference.yml": "5f6b19b6b09b6b6f7924de6c25076d0b3528b6f641e64d4f12e6e6207ec33200",
}
PARAMETERS = {"model_name": MODEL_NAME, "device": "cpu", "cpu_threads": 2,
              "enable_mkldnn": False, "input_shape": None}
EXPECTED_PACKAGES = {"paddleocr": "3.0.3", "paddlex": "3.0.3", "paddlepaddle": "3.0.0",
                     "opencv-contrib-python-headless": "4.10.0.84", "setuptools": "75.6.0"}
SOURCE_NAMES = (
    "paddleocr/_models/text_recognition.py", "paddleocr/_models/base.py",
    "paddlex/inference/models/text_recognition/predictor.py",
    "paddlex/inference/models/text_recognition/processors.py",
    "paddlex/inference/pipelines/components/common/crop_image_regions.py",
    "paddlex/inference/common/reader/image_reader.py",
    "paddlex/inference/utils/official_models.py", "paddlex/utils/deps.py",
)


def configure_environment(output: Path) -> dict:
    cache = output / "recognizer/cache"
    paths = {"PADDLE_PDX_CACHE_HOME": cache / "paddlex", "PADDLE_HOME": cache / "paddle",
             "PADDLE_DOWNLOAD_PATH": cache / "paddle_downloads",
             "PADDLE_EXTENSION_DIR": cache / "paddle_extensions", "XDG_CACHE_HOME": cache / "xdg",
             "MPLCONFIGDIR": cache / "matplotlib", "HF_HOME": cache / "huggingface", "TMPDIR": cache / "tmp"}
    for key, path in paths.items():
        private_path(path).mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                      OPENBLAS_NUM_THREADS="2", PADDLE_PDX_MODEL_SOURCE="BOS")
    old_fonts = BASE / "vlm_region_token_v1/detector/cache/paddlex/fonts"
    new_fonts = paths["PADDLE_PDX_CACHE_HOME"] / "fonts"
    if old_fonts.is_dir():
        new_fonts.mkdir(parents=True, exist_ok=True)
        for font in old_fonts.iterdir():
            if font.is_file() and not (new_fonts / font.name).exists():
                shutil.copyfile(font, new_fonts / font.name)
    return {key: str(path) for key, path in paths.items()}


def image_inputs(manifest_path: Path, detections_path: Path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    detections = json.loads(detections_path.read_text(encoding="utf-8"))
    dev = manifest["splits"]["dev"]
    image_ids = sorted({str(row["image_id"]) for row in dev})
    if len(dev) != 1617 or len(image_ids) != 1024:
        raise ValueError("Frozen internal-dev cardinality changed")
    if not detections["complete"] or len(detections["images"]) != len(image_ids):
        raise ValueError("Requires the completed original image-only detector cache")
    if detections["manifest_file_sha256"] != sha_file(manifest_path):
        raise ValueError("Detector/source manifest file binding differs")
    if detections["manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("Detector/source manifest content binding differs")
    source_images = {}
    for image_id in image_ids:
        source = manifest["images"][image_id]
        detected = detections["images"][image_id]
        source_images[image_id] = {"image_path": str(private_path(manifest_path.parent / source["path"])),
                                  "image_sha256": source["sha256"], "width": int(source["width"]),
                                  "height": int(source["height"])}
        for key in ("image_sha256", "width", "height"):
            if source_images[image_id][key] != detected[key]:
                raise ValueError(f"Detector/source image binding differs: {image_id}/{key}")
        if len(detected["polygons"]) != len(detected["scores"]):
            raise ValueError("Detector polygon/confidence counts differ")
    first_qa = sorted(dev, key=lambda row: int(row["question_id"]))[:32]
    preflight_ids = list(dict.fromkeys(str(row["image_id"]) for row in first_qa))
    metadata = {"manifest_file_sha256": sha_file(manifest_path), "manifest_sha256": manifest["manifest_sha256"],
                "detections_file_sha256": sha_file(detections_path),
                "detector_provenance_sha256": detections["provenance_sha256"],
                "image_input_sha256": sha_object(source_images), "qa_count": len(dev), "image_count": len(image_ids),
                "preflight_question_ids": [str(row["question_id"]) for row in first_qa],
                "preflight_image_ids": preflight_ids}
    return source_images, detections["images"], metadata


def build_provenance(model_dir: Path, cache_paths: dict, metadata: dict, config_path: Path | None) -> dict:
    packages = dict(sorted((dist.metadata["Name"].lower(), dist.version) for dist in importlib.metadata.distributions()))
    for name, version in EXPECTED_PACKAGES.items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"Expected {name}=={version}")
    if "opencv-contrib-python" in packages:
        raise ValueError("GUI OpenCV must not coexist with the fixed headless distribution")
    import cv2
    if cv2.__version__ != "4.10.0":
        raise ValueError("Actual cv2 version differs from the fixed distribution")
    archive = model_dir.parent / (MODEL_NAME + "_infer.tar")
    if sha_file(archive) != ARCHIVE_SHA256:
        raise ValueError("Official recognizer archive SHA differs")
    model_files = {}
    with tarfile.open(archive) as packed:
        for name, expected_sha in MODEL_FILE_SHA256.items():
            path = model_dir / name
            if sha_file(path) != expected_sha:
                raise ValueError(f"Official recognizer file SHA differs: {name}")
            handle = packed.extractfile(MODEL_NAME + "_infer/" + name)
            if handle is None or hashlib.sha256(handle.read()).hexdigest() != expected_sha:
                raise ValueError("Extracted recognizer file differs from its official archive")
            model_files[name] = {"path": str(path), "sha256": expected_sha, "bytes": path.stat().st_size}
    from vlm_region import detect
    source_locks = {"compatibility_helper": {"path": str(Path(detect.__file__).resolve()),
                                           "sha256": sha_file(Path(detect.__file__))}}
    for relative in SOURCE_NAMES:
        distribution = "paddleocr" if relative.startswith("paddleocr/") else "paddlex"
        path = Path(importlib.metadata.distribution(distribution).locate_file(relative))
        source_locks[relative] = {"path": str(path), "sha256": sha_file(path)}
    binary = Path(cv2.__file__).parent / "cv2.abi3.so"
    source_locks["cv2_binary"] = {"path": str(binary), "sha256": sha_file(binary)}
    config = {"path": str(config_path), "file_sha256": sha_file(config_path)} if config_path else None
    if config_path:
        config["content_sha256"] = sha_object(json.loads(config_path.read_text(encoding="utf-8")))
    return {"schema_version": "rarevlm.ocr_recognizer_provenance.v1", "model_name": MODEL_NAME,
            "official_model_url": MODEL_URL,
            "official_documentation": "https://github.com/PaddlePaddle/PaddleOCR/blob/v3.0.3/docs/version3.x/module_usage/text_recognition.en.md",
            "model_directory": str(model_dir), "model_archive": {"path": str(archive), "sha256": ARCHIVE_SHA256,
                                                                 "bytes": archive.stat().st_size},
            "model_files": model_files, "parameters": PARAMETERS, "recognition_batch_size": 8,
            "packages": packages, "python_version": sys.version, "source_locks": source_locks,
            "source_path": str(Path(__file__).resolve()), "source_sha256": sha_file(Path(__file__)),
            "config": config, "persistent_cache_paths": cache_paths, **metadata,
            "headless_compatibility": "Unmodified vlm_region.detect.apply_headless_compatibility; per-process dependency alias only",
            "input_contract": "Original image bytes and saved image-only detection polygons/confidence only; question IDs only declare preflight order",
            "crop_contract": "All finite clipped quadrilaterals with positive minAreaRect extents; official CropByPolys(quad), no expansion; BGR uint8 source and crop",
            "crop_sha256_contract": "SHA256 of C-contiguous uint8 BGR crop.tobytes(order=C), with crop_shape saved separately",
            "recognition_contract": "Official en_PP-OCRv4_mobile_rec inference.yml BGR, [3,48,320] dynamic-width preprocessing, CTC greedy raw rec_text and rec_score; no threshold/normalization",
            "timing_contract": "Per-image decode/hash/crop wall separately; predict API wall includes all batch8 inference/pre/post; no invented per-crop runtime; historical cached detector cost remains separate"}


def prepare_polygons(detected: dict, width: int, height: int):
    """Validate geometry only; keep original detector indices and all legal boxes."""
    import cv2
    import numpy as np
    valid, skipped = [], []
    for index, (polygon, confidence) in enumerate(zip(detected["polygons"], detected["scores"])):
        points = np.asarray(polygon, dtype=np.float32)
        if points.shape != (4, 2) or not np.isfinite(points).all():
            raise ValueError("Expected a finite 4x2 detector polygon")
        score = float(confidence)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid detector confidence")
        original = points.tolist()
        points[:, 0] = np.clip(points[:, 0], 0, width)
        points[:, 1] = np.clip(points[:, 1], 0, height)
        # Match the official cropper's int32 minAreaRect input and reject only
        # degeneracies; no confidence-based or text/question-dependent filter.
        _, extents, _ = cv2.minAreaRect(points.astype(np.int32))
        if min(extents) < 1 or abs(cv2.contourArea(points)) <= 0:
            skipped.append({"detector_index": index, "polygon": original,
                            "clipped_polygon": points.tolist(), "det_score": score,
                            "reason": "degenerate_clipped_polygon"})
            continue
        valid.append({"detector_index": index, "polygon": original,
                      "clipped_polygon": points.tolist(), "det_score": score})
    return valid, skipped


def recognize_image(image_id: str, image: dict, detected: dict, model, cropper) -> dict:
    import cv2
    import numpy as np
    started = time.perf_counter()
    path = Path(image["image_path"])
    if sha_file(path) != image["image_sha256"]:
        raise ValueError(f"Source image changed: {image_id}")
    pixels = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if pixels is None or pixels.shape != (image["height"], image["width"], 3) or pixels.dtype != np.uint8:
        raise ValueError("Original BGR source image decode contract failed")
    decode_seconds = time.perf_counter() - started
    before = time.perf_counter()
    valid, skipped = prepare_polygons(detected, image["width"], image["height"])
    crops = cropper(pixels, [row["clipped_polygon"] for row in valid]) if valid else []
    if len(crops) != len(valid):
        raise ValueError("Official cropper output count differs")
    crops = [np.ascontiguousarray(crop) for crop in crops]
    items = []
    for geometry, crop in zip(valid, crops):
        if crop.ndim != 3 or crop.shape[2] != 3 or crop.dtype != np.uint8 or min(crop.shape[:2]) < 1:
            raise ValueError("Official BGR crop dtype/dimension contract failed")
        items.append({**geometry, "crop_sha256": hashlib.sha256(crop.tobytes(order="C")).hexdigest(),
                      "crop_shape": list(crop.shape), "crop_channel_order": "BGR"})
    crop_seconds = time.perf_counter() - before
    before = time.perf_counter()
    outputs = list(model.predict(input=crops, batch_size=8)) if crops else []
    recognition_seconds = time.perf_counter() - before
    if len(outputs) != len(items):
        raise ValueError("Expected one recognition result per valid crop")
    for item, result in zip(items, outputs):
        text, score = result["rec_text"], float(result["rec_score"])
        if not isinstance(text, str) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid raw CTC recognition text/confidence")
        item.update(rec_text=text, rec_score=score)
    return {"image_id": image_id, **image, "items": items, "skipped_polygons": skipped,
            "detected_polygon_count": len(detected["polygons"]), "valid_crop_count": len(items),
            "decode_and_hash_seconds": decode_seconds, "crop_seconds": crop_seconds,
            "recognition_seconds": recognition_seconds, "total_seconds": time.perf_counter() - started}


def validate_cached_image(row: dict, image: dict, detected: dict, provenance_sha: str) -> None:
    for key, value in image.items():
        if row.get(key) != value:
            raise ValueError(f"Cached OCR/source binding differs: {key}")
    if row.get("provenance_sha256") != provenance_sha:
        raise ValueError("Cached OCR provenance changed")
    if row.get("detected_polygon_count") != len(detected["polygons"]):
        raise ValueError("Cached OCR detector count differs")
    if row.get("cache_key") != sha_object({"image_id": row["image_id"], "image_sha256": image["image_sha256"],
                                         "provenance_sha256": provenance_sha}):
        raise ValueError("Cached OCR key changed")
    if sha_file(Path(image["image_path"])) != image["image_sha256"]:
        raise ValueError("Cached OCR original image bytes changed")
    seen = []
    for item in row["items"]:
        index = int(item["detector_index"])
        if item["polygon"] != detected["polygons"][index] or item["det_score"] != detected["scores"][index]:
            raise ValueError("Cached OCR polygon/confidence binding differs")
        if not isinstance(item["rec_text"], str) or not math.isfinite(item["rec_score"]) or not 0 <= item["rec_score"] <= 1:
            raise ValueError("Cached OCR raw text/confidence contract failed")
        if len(item["crop_sha256"]) != 64 or item["crop_channel_order"] != "BGR":
            raise ValueError("Cached OCR crop hash/channel contract failed")
        seen.append(index)
    seen.extend(int(item["detector_index"]) for item in row["skipped_polygons"])
    if sorted(seen) != list(range(len(detected["polygons"]))):
        raise ValueError("Cached OCR lacks a detected polygon or duplicates one")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--detections", type=Path, default=DEFAULT_DETECTIONS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--config", type=Path, help="Frozen candidate config; required for full execution")
    parser.add_argument("--phase", choices=("preflight", "full"), default="preflight")
    args = parser.parse_args()
    if args.phase == "full" and args.config is None:
        raise ValueError("Full recognition requires the explicitly frozen experiment config")
    output = private_path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    manifest = private_path(args.manifest)
    detections = private_path(args.detections)
    model_dir = private_path(args.model_dir or output / "recognizer" / (MODEL_NAME + "_infer"))
    config = private_path(args.config) if args.config else None
    cache_paths = configure_environment(output)
    images, detected, metadata = image_inputs(manifest, detections)
    provenance = build_provenance(model_dir, cache_paths, metadata, config)
    prov_path = output / "recognizer/OCR_PROVENANCE.json"
    if prov_path.exists() and json.loads(prov_path.read_text(encoding="utf-8")) != provenance:
        raise ValueError("OCR provenance changed; refuse to silently mix cached results")
    write_json(prov_path, provenance)
    prov_sha = sha_file(prov_path)
    cache_dir = output / "recognizer/recognition_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    records = {}
    for image_id, image in images.items():
        cache = cache_dir / (image_id + ".json")
        if cache.exists():
            row = json.loads(cache.read_text(encoding="utf-8"))
            validate_cached_image(row, image, detected[image_id], prov_sha)
            records[image_id] = row
    preflight_ids = metadata["preflight_image_ids"]
    order = preflight_ids + ([image_id for image_id in images if image_id not in set(preflight_ids)] if args.phase == "full" else [])
    if args.phase == "full":
        precheck = output / "recognizer/OCR_PREFLIGHT.json"
        if not precheck.exists():
            raise ValueError("Full recognition requires a completed true CPU API preflight")
        contract = json.loads(precheck.read_text(encoding="utf-8"))
        if not contract.get("passed") or contract.get("provenance_sha256") != prov_sha:
            raise ValueError("CPU preflight/provenance binding differs")
    todo = [image_id for image_id in order if image_id not in records]
    model, cropper, load_seconds = None, None, None
    if todo:
        before = time.perf_counter()
        apply_headless_compatibility()
        from paddleocr import TextRecognition
        from paddlex.inference.pipelines.components import CropByPolys
        import cv2
        import paddle
        cv2.setNumThreads(2)
        if paddle.get_device() != "cpu":
            raise ValueError("CPU-only OCR unexpectedly activated a GPU")
        cropper = CropByPolys(det_box_type="quad")
        model = TextRecognition(model_dir=str(model_dir), **PARAMETERS)
        load_seconds = time.perf_counter() - before
    started = time.perf_counter()

    def snapshot():
        complete = len(records) == len(images)
        preflight_complete = all(image_id in records for image_id in preflight_ids)
        raw = {"schema_version": "rarevlm.ocr_raw.v1", "complete": complete,
               "expected_count": len(images), "completed_count": len(records), "phase": args.phase,
               "provenance_sha256": prov_sha, "source_sha256": provenance["source_sha256"],
               "manifest_file_sha256": metadata["manifest_file_sha256"], "manifest_sha256": metadata["manifest_sha256"],
               "detections_file_sha256": metadata["detections_file_sha256"],
               "image_input_sha256": metadata["image_input_sha256"], "images": records}
        raw_path = output / "recognizer/OCR_RAW.json"
        write_json(raw_path, raw)
        timing = {"decode_and_hash_seconds_saved": sum(row["decode_and_hash_seconds"] for row in records.values()),
                  "crop_seconds_saved": sum(row["crop_seconds"] for row in records.values()),
                  "recognition_predict_api_seconds_saved": sum(row["recognition_seconds"] for row in records.values()),
                  "image_total_seconds_saved": sum(row["total_seconds"] for row in records.values())}
        write_json(output / "recognizer/OCR_STATUS.json", {"updated_at_utc": datetime.now(timezone.utc).isoformat(),
                   "phase": args.phase, "complete": complete, "completed_count": len(records), "expected_count": len(images),
                   "preflight_complete": preflight_complete, "model_load_seconds_this_run": load_seconds,
                   "run_wall_seconds_without_model_load": time.perf_counter() - started, **timing})
        if preflight_complete:
            selected = [records[image_id] for image_id in preflight_ids]
            write_json(output / "recognizer/OCR_PREFLIGHT.json", {
                "schema_version": "rarevlm.ocr_preflight.v1", "passed": True, "provenance_sha256": prov_sha,
                "question_ids": metadata["preflight_question_ids"], "image_ids": preflight_ids,
                "questions": 32, "unique_images": len(selected), "valid_crops": sum(len(row["items"]) for row in selected),
                "scope": "True image-only official CPU crop/recognition API; finite raw CTC, BGR uint8 shape/hash and one-result-per-valid-crop; no task scoring",
                "raw_preflight_records_sha256": sha_object(selected)})
        if complete:
            write_json(output / "recognizer/OCR_READY.json", {
                "schema_version": "rarevlm.ocr_ready.v1", "ready": True, "complete": True,
                "images": len(records), "detected_polygons": sum(row["detected_polygon_count"] for row in records.values()),
                "recognized_crops": sum(len(row["items"]) for row in records.values()),
                "skipped_degenerate_polygons": sum(len(row["skipped_polygons"]) for row in records.values()),
                "empty_images": sum(not row["items"] for row in records.values()),
                "raw_path": str(raw_path), "raw_file_sha256": sha_file(raw_path),
                "provenance_path": str(prov_path), "provenance_sha256": prov_sha,
                "source_sha256": provenance["source_sha256"], "config": provenance["config"], **timing})

    snapshot()
    for index, image_id in enumerate(todo, start=1):
        row = recognize_image(image_id, images[image_id], detected[image_id], model, cropper)
        row["provenance_sha256"] = prov_sha
        row["cache_key"] = sha_object({"image_id": image_id, "image_sha256": row["image_sha256"],
                                       "provenance_sha256": prov_sha})
        validate_cached_image(row, images[image_id], detected[image_id], prov_sha)
        write_json(cache_dir / (image_id + ".json"), row)
        records[image_id] = row
        if index % 16 == 0:
            snapshot()
            print(json.dumps({"recognized_images": len(records), "expected": len(images), "phase": args.phase}), flush=True)
    snapshot()
    print(json.dumps({"recognized_images": len(records), "expected": len(images), "phase": args.phase,
                      "provenance_sha256": prov_sha}), flush=True)


if __name__ == "__main__":
    main()
