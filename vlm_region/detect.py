"""CPU-only, image-only PaddleOCR text-region detection with strict cache locks.

This module uses TextDetection, never text recognition. All original image
polygons are retained for a separately frozen selector; question/reference
contents are not forwarded to the detector.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import importlib.machinery
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import sys
import tarfile
import time
from datetime import datetime, timezone

PRIVATE = Path("/opt/data/private")
DEFAULT_MANIFEST = PRIVATE / "longtail_vlm/rarevlm_pilot/outputs/vlm_deep_v2/prepared/manifest.json"
DEFAULT_OUTPUT = PRIVATE / "longtail_vlm/rarevlm_pilot/outputs/vlm_region_token_v1"
MODEL_NAME = "PP-OCRv5_mobile_det"
MODEL_URL = "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_det_infer.tar"
DOC_URL = "https://www.paddleocr.ai/latest/en/version3.x/module_usage/text_detection.html"
PARAMETERS = {
    "model_name": MODEL_NAME,
    "device": "cpu",
    "cpu_threads": 2,
    "enable_mkldnn": False,
    "limit_side_len": 960,
    "limit_type": "max",
    "thresh": 0.3,
    "box_thresh": 0.6,
    "unclip_ratio": 1.5,
}
EXPECTED_PACKAGES = {"paddleocr": "3.0.3", "paddlex": "3.0.3", "paddlepaddle": "3.0.0", "opencv-contrib-python-headless": "4.10.0.84", "setuptools": "75.6.0"}


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha_object(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def private_path(path: Path) -> Path:
    path = path.resolve()
    if path != PRIVATE and PRIVATE not in path.parents:
        raise ValueError(f"Path must be persistent under {PRIVATE}: {path}")
    return path


def write_json(path: Path, value) -> None:
    private_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def configure_environment(output: Path) -> dict:
    """Set documented/cache-related paths before any Paddle/PaddleX import."""
    cache = output / "detector" / "cache"
    paths = {
        "PADDLE_PDX_CACHE_HOME": cache / "paddlex",
        "PADDLE_HOME": cache / "paddle",
        "PADDLE_DOWNLOAD_PATH": cache / "paddle_downloads",
        "PADDLE_EXTENSION_DIR": cache / "paddle_extensions",
        "XDG_CACHE_HOME": cache / "xdg",
        "MPLCONFIGDIR": cache / "matplotlib",
        "HF_HOME": cache / "huggingface",
        "TMPDIR": cache / "tmp",
    }
    for name, path in paths.items():
        private_path(path).mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    os.environ["PADDLE_PDX_MODEL_SOURCE"] = "BOS"
    return {key: str(value) for key, value in paths.items()}


def image_inputs(manifest_path: Path):
    """Reduce source manifest to image IDs, file locks and dimensions only."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dev = manifest["splits"]["dev"]
    ordered_qa = sorted(dev, key=lambda row: int(row["question_id"]))
    preflight_ids = list(dict.fromkeys(row["image_id"] for row in ordered_qa[:32]))
    image_ids = sorted({row["image_id"] for row in dev})
    validation_ids = (preflight_ids + [i for i in image_ids if i not in set(preflight_ids)])[:32]
    if len(dev) != 1617 or len(image_ids) != 1024:
        raise ValueError("Frozen internal-dev cardinality changed")
    reduced = {}
    for image_id in image_ids:
        image = manifest["images"][image_id]
        path = private_path(manifest_path.parent / image["path"])
        reduced[image_id] = {
            "image_path": str(path), "image_sha256": image["sha256"],
            "width": int(image["width"]), "height": int(image["height"]),
        }
    metadata = {
        "manifest_file_sha256": sha_file(manifest_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "qa_count": len(dev), "image_count": len(image_ids),
        "preflight_question_ids": [str(row["question_id"]) for row in ordered_qa[:32]],
        "preflight_image_ids": preflight_ids,
        "cpu_api_validation_image_ids": validation_ids,
        "image_input_sha256": sha_object(reduced),
    }
    return reduced, metadata


def build_provenance(manifest_path: Path, model_dir: Path, cache_paths: dict, metadata: dict) -> dict:
    packages = dict(sorted((dist.metadata["Name"].lower(), dist.version) for dist in importlib.metadata.distributions()))
    for name, expected in EXPECTED_PACKAGES.items():
        if importlib.metadata.version(name) != expected:
            raise ValueError(f"Expected {name}=={expected}, got {importlib.metadata.version(name)}")
    if "opencv-contrib-python" in packages:
        raise ValueError("GUI OpenCV must not coexist with the frozen headless package")
    import cv2
    if cv2.__version__ != "4.10.0":
        raise ValueError("Actual cv2 module version differs from pinned headless package")
    cv2_binary = Path(cv2.__file__).parent / "cv2.abi3.so"
    checker = Path(importlib.metadata.distribution("paddlex").locate_file("paddlex/utils/deps.py"))
    files = {}
    for name in ("inference.json", "inference.pdiparams", "inference.yml"):
        path = private_path(model_dir / name)
        if not path.is_file():
            raise ValueError(f"Missing official model file: {path}")
        files[name] = {"sha256": sha_file(path), "bytes": path.stat().st_size}
    archive = model_dir.parent / "PP-OCRv5_mobile_det_infer.tar"
    expected_tar = "50446e5d01ac2a73d5319c89513281f6578414c888c602f9af13f93feefffc58"
    if not archive.is_file() or sha_file(archive) != expected_tar:
        raise ValueError("Official detector archive lock differs")
    with tarfile.open(archive) as packed:
        for name, lock in files.items():
            handle = packed.extractfile(f"PP-OCRv5_mobile_det_infer/{name}")
            if handle is None or hashlib.sha256(handle.read()).hexdigest() != lock["sha256"]:
                raise ValueError(f"Extracted model file differs from official archive: {name}")
    reports = {}
    for name in ("INSTALL_REPORT.json", "INSTALL_PADDLEOCR_REPORT.json", "INSTALL_HEADLESS_REPORT.json", "INSTALL_SETUPTOOLS_REPORT.json"):
        path = model_dir.parent / name
        if not path.is_file():
            raise ValueError(f"Missing package byte/hash installation report: {path}")
        reports[name] = {"sha256": sha_file(path), "bytes": path.stat().st_size}
    return {
        "schema_version": "rarevlm.region_detector_provenance.v1",
        "detector": MODEL_NAME, "official_documentation": DOC_URL,
        "official_model_url": MODEL_URL, "model_directory": str(model_dir),
        "model_archive": {"sha256": expected_tar, "bytes": archive.stat().st_size},
        "model_files": files, "model_files_sha256": sha_object(files),
        "parameters": PARAMETERS, "packages": packages, "installation_reports": reports,
        "dependency_compatibility": "PaddleX 3.0.3 declares GUI opencv-contrib-python; this isolated CPU-only environment replaces it with the official same-version opencv-contrib-python-headless==4.10.0.84. The acknowledged pip-check metadata difference is saved; actual TextDetection API validation is required.",
        "headless_compatibility_shim": {
            "scope": "This detector process only; no package-file or distribution-metadata modifications",
            "alias": {"opencv-contrib-python": "opencv-contrib-python-headless==4.10.0.84"},
            "source": inspect.getsource(apply_headless_compatibility),
            "original_checker_path": str(checker), "original_checker_sha256": sha_file(checker),
            "original_checker_function": "get_dep_version (line 78); all other dependency and specifier checks retained",
            "lru_caches_cleared": ["is_dep_available", "is_extra_available"],
            "import_order_compatibility": "Temporary MetaPathFinder wraps only paddlex.utils.deps official loader; alias activates before downstream conditional cv2 imports; finder removed immediately after initial PaddleX import",
            "cv2_module_path": str(Path(cv2.__file__)), "cv2_module_sha256": sha_file(Path(cv2.__file__)),
            "cv2_binary_path": str(cv2_binary), "cv2_binary_sha256": sha_file(cv2_binary),
            "actual_cv2_version": cv2.__version__, "gui_distribution_installed": False,
            "pip_check_scope": "PaddleX metadata reports missing GUI distribution; same-version headless functional replacement explicitly acknowledged",
        },
        "runtime_cpu_threads": {"opencv": 2, "OMP_NUM_THREADS": 2, "MKL_NUM_THREADS": 2, "OPENBLAS_NUM_THREADS": 2},
        "python_version": sys.version, "source_path": str(Path(__file__).resolve()),
        "source_sha256": sha_file(Path(__file__)), "manifest_path": str(manifest_path),
        **metadata, "persistent_cache_paths": cache_paths,
        "input_contract": "Original RGB image file only; no question, target, reference or recognized text",
        "output_contract": "Original-pixel polygons and detection confidence; no text recognition",
        "timing_contract": "Per-image TextDetection.predict API wall time including decode/preprocessing/inference/postprocessing; first image can be cold; model load separately recorded",
    }


def apply_headless_compatibility():
    """Alias GUI dependency checks before PaddleX's conditional cv2 imports."""
    import cv2
    if importlib.metadata.version("opencv-contrib-python-headless") != "4.10.0.84" or cv2.__version__ != "4.10.0":
        raise ValueError("Headless compatibility alias requires exact pinned distribution/module versions")
    if "paddlex.utils.deps" in sys.modules:
        raise ValueError("PaddleX was imported before the compatibility hook")

    class AliasLoader(importlib.abc.Loader):
        def __init__(self, original):
            self.original = original

        def create_module(self, spec):
            return self.original.create_module(spec)

        def exec_module(self, module):
            self.original.exec_module(module)
            original = module.get_dep_version
            if original("opencv-contrib-python") is not None:
                raise ValueError("GUI distribution is unexpectedly installed")

            def compatible_get_dep_version(dep):
                if dep == "opencv-contrib-python":
                    return original("opencv-contrib-python-headless")
                return original(dep)

            module.get_dep_version = compatible_get_dep_version
            module.is_dep_available.cache_clear()
            module.is_extra_available.cache_clear()

    class DependencyFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != "paddlex.utils.deps":
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            expected = Path(importlib.metadata.distribution("paddlex").locate_file("paddlex/utils/deps.py"))
            if spec is None or spec.loader is None or Path(spec.origin).resolve() != expected.resolve():
                raise ValueError("Unexpected PaddleX dependency checker module")
            spec.loader = AliasLoader(spec.loader)
            return spec

    finder = DependencyFinder()
    sys.meta_path.insert(0, finder)
    try:
        import paddlex.utils.deps as deps
    finally:
        sys.meta_path.remove(finder)
    if not deps.is_dep_available("opencv-contrib-python", check_version=True):
        raise ValueError("Headless version does not satisfy the original OpenCV specifier")


def cv2_functional_check(images: dict, image_ids: list, output: Path):
    """Verify required cv2 APIs on original files; no outputs feed the detector."""
    import cv2
    cv2.setNumThreads(2)
    rows = []
    for image_id in image_ids:
        row = images[image_id]
        path = Path(row["image_path"])
        if sha_file(path) != row["image_sha256"]:
            raise ValueError("Source image changed during cv2 compatibility validation")
        pixels = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if pixels is None or pixels.shape != (row["height"], row["width"], 3):
            raise ValueError("cv2 imread dimensions differ from source lock")
        reduced = cv2.resize(pixels, (64, 64))
        gray = cv2.cvtColor(reduced, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            corners = cv2.boxPoints(cv2.minAreaRect(max(contours, key=cv2.contourArea)))
            if corners.shape != (4, 2):
                raise ValueError("cv2 minAreaRect/boxPoints API validation failed")
        else:
            raise ValueError("Expected at least one contour in cv2 functional check")
        rows.append({"image_id": image_id, "image_sha256": row["image_sha256"], "shape": list(pixels.shape),
                     "imread": True, "resize": True, "findContours": True, "minAreaRect": True})
    write_json(output / "detector" / "CV2_API_CHECK.json", {
        "cv2_version": cv2.__version__, "headless_distribution_version": "4.10.0.84",
        "source_sha256": sha_file(Path(__file__)), "unique_images": len(rows), "checks": rows,
        "scope": "API compatibility only; 64x64 resize/threshold contours are not detector inputs or experimental views",
    })


def normalize_output(result, width: int, height: int):
    polys = result["dt_polys"]
    scores = result["dt_scores"]
    if hasattr(polys, "tolist"):
        polys = polys.tolist()
    scores = [float(value) for value in scores]
    if len(polys) != len(scores):
        raise ValueError("Detector polygon/score count mismatch")
    normalized = []
    for poly, score in zip(polys, scores):
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid detection confidence")
        points = []
        if len(poly) != 4:
            raise ValueError("Expected four-point detection polygon")
        for x, y in poly:
            x, y = float(x), float(y)
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError("Invalid detection coordinate")
            if not 0 <= x <= width or not 0 <= y <= height:
                raise ValueError("Detection coordinate lies outside source image")
            points.append([x, y])
        normalized.append(points)
    return normalized, scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--phase", choices=("preflight", "full"), default="preflight")
    args = parser.parse_args()
    output = private_path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = private_path(args.manifest)
    model_dir = private_path(args.model_dir or output / "detector" / "PP-OCRv5_mobile_det_infer")
    cache_paths = configure_environment(output)
    images, metadata = image_inputs(manifest_path)
    provenance = build_provenance(manifest_path, model_dir, cache_paths, metadata)
    provenance_path = output / "detector" / "DETECTOR_PROVENANCE.json"
    if provenance_path.exists() and json.loads(provenance_path.read_text()) != provenance:
        raise ValueError("Detector provenance changed; refuse to reuse this output directory")
    write_json(provenance_path, provenance)
    provenance_sha = sha_file(provenance_path)
    cache_dir = output / "detector" / "detection_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    records = {}
    for image_id, image in images.items():
        cache_key = sha_object({"image_id": image_id, "image_sha256": image["image_sha256"], "provenance_sha256": provenance_sha})
        cache = cache_dir / f"{image_id}.json"
        if cache.exists():
            row = json.loads(cache.read_text())
            if row.get("cache_key") != cache_key or any(row.get(key) != value for key, value in image.items()):
                raise ValueError(f"Detection cache lock changed: {image_id}")
            normalize_output({"dt_polys": row["polygons"], "dt_scores": row["scores"]}, image["width"], image["height"])
            if sha_file(Path(image["image_path"])) != image["image_sha256"]:
                raise ValueError(f"Cached source image changed: {image_id}")
            records[image_id] = row
    order = metadata["cpu_api_validation_image_ids"]
    if args.phase == "full":
        order = order + [image_id for image_id in images if image_id not in set(order)]
    todo = [image_id for image_id in order if image_id not in records]
    load_seconds = None
    model = None
    if todo:
        started = time.perf_counter()
        apply_headless_compatibility()
        from paddleocr import TextDetection
        import paddle
        import cv2
        cv2.setNumThreads(2)
        cv2_functional_check(images, metadata["cpu_api_validation_image_ids"], output)
        if paddle.get_device() != "cpu":
            raise ValueError("CPU-only detector unexpectedly activated a GPU")
        model = TextDetection(model_dir=str(model_dir), **PARAMETERS)
        load_seconds = time.perf_counter() - started
    started = time.perf_counter()

    def snapshot():
        result = {
            "schema_version": "rarevlm.region_detections.v1", "complete": len(records) == len(images),
            "completed_count": len(records), "expected_count": len(images),
            "phase": args.phase, "preflight_complete": all(i in records for i in metadata["preflight_image_ids"]),
            "cpu_api_validation_complete": all(i in records for i in metadata["cpu_api_validation_image_ids"]),
            "provenance_sha256": provenance_sha, "source_sha256": provenance["source_sha256"],
            "manifest_sha256": metadata["manifest_sha256"],
            "manifest_file_sha256": metadata["manifest_file_sha256"],
            "image_input_sha256": metadata["image_input_sha256"],
            "images": records,
        }
        write_json(output / "detector" / "DETECTIONS.json", result)
        write_json(output / "detector" / "DETECTION_STATUS.json", {
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "complete": result["complete"], "phase": args.phase,
            "completed_count": len(records), "expected_count": len(images),
            "preflight_complete": result["preflight_complete"],
            "model_load_seconds_this_run": load_seconds,
            "api_detection_seconds_saved": sum(row["detection_seconds"] for row in records.values()),
            "run_wall_seconds_without_model_load": time.perf_counter() - started,
        })

    snapshot()
    for index, image_id in enumerate(todo, start=1):
        image = images[image_id]
        path = Path(image["image_path"])
        if sha_file(path) != image["image_sha256"]:
            raise ValueError(f"Source image hash changed: {image_id}")
        from PIL import Image
        with Image.open(path) as im:
            if im.size != (image["width"], image["height"]):
                raise ValueError(f"Image dimensions changed: {image_id}")
        before = time.perf_counter()
        outputs = list(model.predict(str(path), batch_size=1))
        elapsed = time.perf_counter() - before
        if len(outputs) != 1:
            raise ValueError("Expected one result per original image")
        polygons, scores = normalize_output(outputs[0], image["width"], image["height"])
        row = {"image_id": image_id, **image, "polygons": polygons, "scores": scores,
               "detection_seconds": elapsed, "provenance_sha256": provenance_sha,
               "cache_key": sha_object({"image_id": image_id, "image_sha256": image["image_sha256"], "provenance_sha256": provenance_sha})}
        write_json(cache_dir / f"{image_id}.json", row)
        records[image_id] = row
        if index % 16 == 0:
            snapshot()
            print(json.dumps({"detected": len(records), "expected": len(images), "phase": args.phase}), flush=True)
    snapshot()
    print(json.dumps({"detected": len(records), "expected": len(images), "phase": args.phase, "provenance_sha256": provenance_sha}), flush=True)


if __name__ == "__main__":
    main()
