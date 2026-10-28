"""Explicit shared CPU initialization for the four new visual-merger factors.

No base or historical adapter weights are read here.  Creation is an explicit
prepare action; model construction must load this artifact, never improvise it.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_key, "2")

SCHEMA = "rarevlm.visual_merger_kl.common_initialization.v1"
INITIAL_LANGUAGE_SHA256 = "6a44359406c53261b254c3a19393c63187b34af09baec539d6b235ca148c0fa7"
LANGUAGE_TENSORS, LANGUAGE_ELEMENTS = 144, 1843200
TRAINABLE_ELEMENTS = 1843200
TARGET_SPECS = (
    ("visual.merger.mlp.0", 96, 192, (96, 5120), (5120, 96)),
    ("visual.merger.mlp.2", 120, 240, (120, 5120), (2048, 120)),
)
METHOD = {"seed": 42, "method": "cpu_torch_generator_kaiming_uniform_a_sqrt5_B_zero",
          "dtype": "float32", "shared_across_order_seeds": True}


def visual_tensor_specs():
    return {f"base_model.model.{target}.lora_{factor}.default.weight": tuple(shape)
            for target, _, _, a, b in TARGET_SPECS for factor, shape in (("A", a), ("B", b))}


VISUAL_KEYS = tuple(visual_tensor_specs())


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _private(value):
    path = Path(value).resolve()
    _require(path.is_relative_to(Path("/opt/data/private")), "initializer path outside private workspace")
    return path


def _sha(file):
    digest = hashlib.sha256()
    with Path(file).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def initialization_paths(config):
    directory = _private(config["common_initialization_dir"])
    _require(directory == _private(config["output_root"]) / "common_initialization", "common initialization directory differs")
    return {"directory": directory, "tensors": directory / "initialization.safetensors", "manifest": directory / "MANIFEST.json"}


def validate_initialization_config(config):
    _require(config.get("common_initialization") == METHOD, "fixed CPU seed42/FP32/Kaiming/B0 initialization differs")
    expected = {"language_frozen_tensors": LANGUAGE_TENSORS, "language_frozen_elements": LANGUAGE_ELEMENTS,
        "new_trainable_tensors": 4, "new_trainable_elements": TRAINABLE_ELEMENTS,
        "total_adapter_tensors": LANGUAGE_TENSORS + 4,
        "total_adapter_elements": LANGUAGE_ELEMENTS + TRAINABLE_ELEMENTS,
        "active_adapter": "default", "always_active_composite": True,
        "targets": {target: {"r": rank, "lora_alpha": alpha, "A_shape": list(a), "B_shape": list(b)}
                    for target, rank, alpha, a, b in TARGET_SPECS}}
    _require(config.get("visual_adaptation") == expected, "visual rank/alpha/shape/148 inventory differs")
    _require(config.get("expected_trainable") == TRAINABLE_ELEMENTS, "trainable element budget differs")
    _require(config["teacher_generation"]["initial_runtime_master_sha256"] == INITIAL_LANGUAGE_SHA256,
             "common frozen language canonical identity differs")
    initialization_paths(config)
    return expected


def tensor_hash(named_tensors):
    """Same name/dtype/shape/raw-byte convention as frozen training tensor_hash."""
    import torch
    digest = hashlib.sha256()
    for name, tensor in named_tensors:
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def validate_visual_tensors(tensors):
    import torch
    specs = visual_tensor_specs()
    _require(set(tensors) == set(specs), "common initialization requires exactly the four declared runtime keys")
    for name, shape in specs.items():
        value = tensors[name]
        _require(isinstance(value, torch.Tensor) and value.device.type == "cpu" and value.dtype == torch.float32
                 and tuple(value.shape) == shape and bool(torch.isfinite(value).all()), "initializer tensor shape/dtype/finite/CPU mismatch: " + name)
        if ".lora_B." in name:
            _require(bool((value == 0).all()), "initial visual B must be exactly zero")
    _require(sum(value.numel() for value in tensors.values()) == TRAINABLE_ELEMENTS, "initial visual parameter count differs")
    return tensors


def _create_tensors():
    import torch
    generator = torch.Generator(device="cpu")
    generator.manual_seed(42)
    tensors = {}
    for name, shape in visual_tensor_specs().items():
        value = torch.empty(shape, dtype=torch.float32, device="cpu")
        if ".lora_A." in name:
            torch.nn.init.kaiming_uniform_(value, a=math.sqrt(5), generator=generator)
        else:
            value.zero_()
        tensors[name] = value
    return validate_visual_tensors(tensors)


def _tensor_identity(tensors):
    import torch
    names = tuple(visual_tensor_specs())
    return {"visual_fp32_tensor_sha256": tensor_hash((name, tensors[name]) for name in names),
        "visual_native_bf16_tensor_sha256": tensor_hash((name, tensors[name].to(torch.bfloat16)) for name in names),
        "visual_bf16_promoted_fp32_tensor_sha256": tensor_hash((name, tensors[name].to(torch.bfloat16).float()) for name in names),
        "per_tensor": [{"name": name, "shape": list(tensors[name].shape), "dtype": str(tensors[name].dtype),
                        "elements": tensors[name].numel(), "tensor_sha256": tensor_hash([(name, tensors[name])]),
                        "initial_zero_B": ".lora_B." in name} for name in names]}


def common_initializer_manifest(config, verify_file_bytes=True):
    """Return saved identity/telemetry; optional byte hash, never tensor loading."""
    import torch
    validate_initialization_config(config)
    paths = initialization_paths(config)
    _require(paths["manifest"].is_file(), "explicit common initialization prepare is required")
    manifest = json.loads(paths["manifest"].read_text())
    _require(manifest.get("schema_version") == SCHEMA and set(manifest) == {"schema_version", "identity", "identity_sha256", "telemetry"}, "common initialization manifest schema differs")
    identity = manifest["identity"]
    _require(manifest["identity_sha256"] == _content(identity), "initializer stable identity seal differs")
    expected = {"config_content_sha256": _content(config), "method": METHOD,
        "tensor_order": list(visual_tensor_specs()), "shapes": {name: list(shape) for name, shape in visual_tensor_specs().items()},
        "dtype": "torch.float32", "trainable_tensors": 4, "trainable_elements": TRAINABLE_ELEMENTS,
        "frozen_language_tensors": LANGUAGE_TENSORS, "frozen_language_elements": LANGUAGE_ELEMENTS,
        "total_adapter_tensors": LANGUAGE_TENSORS + 4, "total_adapter_elements": LANGUAGE_ELEMENTS + TRAINABLE_ELEMENTS,
        "frozen_language_canonical_fp32_sha256": INITIAL_LANGUAGE_SHA256,
        "initializer_source_file": str(Path(__file__).resolve()), "initializer_source_file_sha256": _sha(__file__),
        "torch_version": str(torch.__version__)}
    _require(all(identity.get(key) == value for key, value in expected.items()), "initializer config/source/inventory identity differs")
    _require(set(identity["files_sha256"]) == {"initialization.safetensors"}, "initializer file map differs")
    _require(paths["tensors"].is_file(), "initialization tensor artifact missing")
    if verify_file_bytes:
        _require(_sha(paths["tensors"]) == identity["files_sha256"]["initialization.safetensors"], "initializer tensor bytes changed")
    return manifest


def load_common_initialization(config, require_prepared=True):
    """Load exact saved four-factor artifact; this API never initializes anew."""
    _require(type(require_prepared) is bool, "require_prepared must be boolean")
    manifest = common_initializer_manifest(config, verify_file_bytes=True)
    from safetensors.torch import load_file
    tensors = validate_visual_tensors(load_file(str(initialization_paths(config)["tensors"]), device="cpu"))
    observed = _tensor_identity(tensors)
    _require(all(manifest["identity"].get(key) == value for key, value in observed.items()), "initialization saved tensor identity differs")
    return tensors, manifest


def prepare_common_initialization(config):
    """Explicit CPU prepare, shared by every order/replay seed; idempotent."""
    import torch
    validate_initialization_config(config)
    paths = initialization_paths(config)
    if paths["manifest"].exists():
        return load_common_initialization(config)[1]
    _require(not paths["directory"].exists() or not any(paths["directory"].iterdir()), "partial initializer exists; refuse overwrite")
    paths["directory"].parent.mkdir(parents=True, exist_ok=True)
    begun = time.monotonic()
    tensors = _create_tensors()
    from safetensors.torch import save_file
    with tempfile.TemporaryDirectory(prefix=".common-initialization-", dir=paths["directory"].parent) as temporary:
        stage = Path(temporary) / "artifact"
        stage.mkdir()
        save_file(tensors, str(stage / "initialization.safetensors"))
        identity = {"config_content_sha256": _content(config), "method": METHOD,
            "tensor_order": list(visual_tensor_specs()), "shapes": {name: list(shape) for name, shape in visual_tensor_specs().items()},
            "dtype": "torch.float32", "trainable_tensors": 4, "trainable_elements": TRAINABLE_ELEMENTS,
            "frozen_language_tensors": LANGUAGE_TENSORS, "frozen_language_elements": LANGUAGE_ELEMENTS,
            "total_adapter_tensors": LANGUAGE_TENSORS + 4, "total_adapter_elements": LANGUAGE_ELEMENTS + TRAINABLE_ELEMENTS,
            "frozen_language_canonical_fp32_sha256": INITIAL_LANGUAGE_SHA256,
            "initializer_source_file": str(Path(__file__).resolve()), "initializer_source_file_sha256": _sha(__file__),
            "torch_version": str(torch.__version__), "files_sha256": {"initialization.safetensors": _sha(stage / "initialization.safetensors")},
            **_tensor_identity(tensors)}
        manifest = {"schema_version": SCHEMA, "identity": identity, "identity_sha256": _content(identity),
            "telemetry": {"prepare_CPU_wall_seconds": time.monotonic() - begun, "model_weight_files_read": 0,
                          "model_forwards": 0, "GPU_operations": 0, "new_training_updates": 0}}
        (stage / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        os.rename(stage, paths["directory"])
    return load_common_initialization(config)[1]
