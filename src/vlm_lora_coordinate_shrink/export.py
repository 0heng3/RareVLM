"""Fixed CPU FP32 LoRA-coordinate update shrink; no model/PEFT merge is loaded."""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_name, '2')

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import shutil
import time
import uuid

import torch
from safetensors.torch import load_file, save_file

SCHEMA = 'rarevlm.lora_coordinate_shrink.export.v1'
REFERENCE_SCHEMA = 'rarevlm.lora_coordinate_shrink.producer_reference.v1'
INITIAL_SHA = '6a44359406c53261b254c3a19393c63187b34af09baec539d6b235ca148c0fa7'
SEED42_FINAL_SHA = 'd86e0ce25d30c57377e660a9c35cecdc03b3db2205338da81292a9239b13c485'
FORMULA = 'initial + (final - initial) * 0.5'
OPERATION_ORDER = ['raw_initial.to(bfloat16).to(float32)', 'delta = final_fp32 - initial_canonical_fp32',
    'half_fp32 = initial_canonical_fp32 + delta * 0.5']
EXPECTED_TENSORS, EXPECTED_ELEMENTS = 144, 1843200
_KEY = re.compile(r'^(base_model\.model\.(?:model\.language_model|model|language_model))\.layers\.(0|[1-9][0-9]*)\.self_attn\.(q_proj|v_proj)\.lora_([AB])(?:\.default)?\.weight$')


def private_path(value):
    path = Path(value).resolve()
    if Path('/opt/data/private') not in path.parents:
        raise ValueError('Export paths must remain below /opt/data/private.')
    return path


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def content_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    with Path(path).open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def _file_map(directory):
    directory = private_path(directory)
    files = {path.name: sha_file(path) for path in sorted(directory.iterdir()) if path.is_file()}
    if not {'adapter_config.json', 'adapter_model.safetensors'} <= set(files):
        raise ValueError('Single safetensors adapter and original adapter_config are required.')
    if any(name.endswith(('.bin', '.pt', '.safetensors')) and name != 'adapter_model.safetensors' for name in files):
        raise ValueError('Unsupported extra/sharded/pickle adapter tensors.')
    return files


def _factor_name(name):
    match = _KEY.fullmatch(name)
    if not match:
        raise ValueError(f'Unexpected/nonlanguage-qv LoRA key: {name}')
    prefix, layer, projection, factor = match.groups()
    canonical = f'{prefix}.layers.{int(layer)}.self_attn.{projection}.lora_{factor}.default.weight'
    order = (int(layer), {'q_proj': 0, 'v_proj': 1}[projection], {'A': 0, 'B': 1}[factor])
    return canonical, order, prefix, projection, factor


def _ordered(tensors):
    entries = []; seen = set()
    for raw_name, tensor in tensors.items():
        canonical, order, prefix, projection, factor = _factor_name(raw_name)
        if canonical in seen:
            raise ValueError('Duplicate canonical LoRA key after PEFT-name normalization.')
        seen.add(canonical); entries.append((order, canonical, raw_name, tensor, prefix, projection, factor))
    return sorted(entries, key=lambda value: value[0])


def tensor_sha256(named_tensors):
    """Original runtime name/dtype/shape/bytes hash; caller supplies traversal order."""
    digest = hashlib.sha256()
    for name, tensor in named_tensors:
        digest.update(name.encode()); digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def canonical_tensor_sha256(tensors):
    return tensor_sha256((canonical, tensor) for _, canonical, _, tensor, _, _, _ in _ordered(tensors))


def validate_adapter_config(config, model_root=None):
    fixed = {'peft_type': 'LORA', 'r': 8, 'lora_alpha': 16, 'bias': 'none', 'fan_in_fan_out': False,
        'use_dora': False, 'use_rslora': False, 'lora_bias': False, 'modules_to_save': None}
    if any(config.get(key, False if key == 'lora_bias' else None) != value for key, value in fixed.items()):
        raise ValueError('Original rank8/alpha16 plain-LoRA configuration required.')
    if (set(config.get('target_modules', [])) != {'q_proj', 'v_proj'}
            or config.get('rank_pattern', {}) or config.get('alpha_pattern', {})
            or config.get('layer_replication') is not None):
        raise ValueError('No extra modules, per-layer scales, ranks or layer replication allowed.')
    if model_root is not None and private_path(config['base_model_name_or_path']) != private_path(model_root):
        raise ValueError('Adapter base model path differs from the fixed original model.')
    return True


def validate_adapter_tensors(tensors):
    if len(tensors) != EXPECTED_TENSORS or sum(t.numel() for t in tensors.values()) != EXPECTED_ELEMENTS:
        raise ValueError('Exactly144 q/v A/B tensors /1843200 elements required.')
    entries = _ordered(tensors)
    expected_orders = {(layer, projection, factor) for layer in range(36) for projection in (0, 1) for factor in (0, 1)}
    if {entry[0] for entry in entries} != expected_orders or len({entry[4] for entry in entries}) != 1:
        raise ValueError('Complete36-layer language q/v A/B inventory required.')
    for _, _, name, tensor, _, projection, factor in entries:
        shape = (8, 2048) if factor == 'A' else ((2048, 8) if projection == 'q_proj' else (256, 8))
        if (tuple(tensor.shape) != shape or tensor.dtype != torch.float32 or tensor.device.type != 'cpu'
                or not bool(torch.isfinite(tensor).all())):
            raise ValueError(f'CPU finite FP32 factor shape/dtype differs: {name}')
    return entries


def coordinate_half(raw_initial, final):
    """Pure small-tensor-testable transform. Production inventory is checked separately."""
    if set(raw_initial) != set(final):
        raise ValueError('Initial/final tensor keys differ.')
    canonical, half = {}, {}
    for key in raw_initial:
        raw, target = raw_initial[key], final[key]
        if (raw.device.type != 'cpu' or target.device.type != 'cpu' or raw.dtype != torch.float32
                or target.dtype != torch.float32 or raw.shape != target.shape
                or not bool(torch.isfinite(raw).all()) or not bool(torch.isfinite(target).all())):
            raise ValueError('Matching finite CPU FP32 source tensors required.')
        theta0 = raw.to(torch.bfloat16).to(torch.float32)
        delta = target - theta0
        value = theta0 + delta * 0.5
        if not all(bool(torch.isfinite(t).all()) for t in (theta0, delta, value, value.to(torch.bfloat16))):
            raise ValueError('Fixed FP32 arithmetic or BF16 round became nonfinite; no precision fallback.')
        canonical[key] = theta0.contiguous(); half[key] = value.contiguous()
    return canonical, half


def _statistics(raw, canonical, final, half):
    rows = []; source_sq = 0.; half_sq = 0.; source_max = 0.; half_max = 0.
    for _, canonical_name, name, initial, _, _, _ in _ordered(canonical):
        delta = final[name] - initial; half_delta = half[name] - initial
        dsq = float(delta.double().square().sum()); hsq = float(half_delta.double().square().sum())
        source_sq += dsq; half_sq += hsq
        dmax, hmax = float(delta.abs().max()), float(half_delta.abs().max())
        source_max, half_max = max(source_max, dmax), max(half_max, hmax)
        rows.append({'raw_name': name, 'canonical_name': canonical_name, 'shape': list(initial.shape),
            'elements': initial.numel(), 'source_raw_to_canonical_changed_elements': int((raw[name] != initial).sum()),
            'source_changed_elements': int((final[name] != initial).sum()),
            'half_changed_elements': int((half[name] != initial).sum()),
            'native_bf16_half_changed_elements': int((half[name].to(torch.bfloat16) != initial.to(torch.bfloat16)).sum()),
            'source_update_l2': math.sqrt(dsq), 'half_update_l2': math.sqrt(hsq),
            'max_abs_source_update': dmax, 'max_abs_half_update': hmax,
            'initial_canonical_tensor_sha256': tensor_sha256([(canonical_name, initial)]),
            'source_final_tensor_sha256': tensor_sha256([(canonical_name, final[name])]),
            'exported_fp32_tensor_sha256': tensor_sha256([(canonical_name, half[name])]),
            'exported_bf16_tensor_sha256': tensor_sha256([(canonical_name, half[name].to(torch.bfloat16))])})
    source_l2, half_l2 = math.sqrt(source_sq), math.sqrt(half_sq)
    return rows, {'source_update_l2': source_l2, 'half_update_l2': half_l2,
        'half_over_source_l2_ratio': half_l2 / source_l2 if source_l2 else None,
        'max_abs_source_update': source_max, 'max_abs_half_update': half_max}


def _sources(config_file, seed, producer_result_path, producer_adapter_dir, output_dir, producer_reference):
    config_file = private_path(config_file); config = _read(config_file)
    spec = config.get('coordinate_shrink', {})
    fixed = {'alpha': .5, 'initial_master_sha256': INITIAL_SHA, 'seed42_producer_master_sha256': SEED42_FINAL_SHA,
        'formula': FORMULA, 'arithmetic': 'cpu_fp32', 'candidate_adapter_subdir': 'shrink/adapter'}
    if config.get('schema_version') != 'rarevlm.lora_coordinate_shrink.v1' or any(spec.get(k) != v for k, v in fixed.items()):
        raise ValueError('Fixed shrink protocol/configuration differs.')
    if type(seed) is not int or seed not in (0, 42, 200):
        raise ValueError('Only protocol seeds0/42/200 supported.')
    output_dir = private_path(output_dir); expected_dir = private_path(config['output_root']) / f'runs/lora_half32/seed{seed}/shrink'
    if output_dir != expected_dir:
        raise ValueError('Only the separate candidate shrink output directory is allowed.')
    if not isinstance(producer_reference, dict) or producer_reference.get('schema_version') != REFERENCE_SCHEMA:
        raise ValueError('Explicit independently bound producer reference required.')
    result_path, adapter_dir = private_path(producer_result_path), private_path(producer_adapter_dir)
    reference = producer_reference
    expected_final = reference.get('expected_final_master_sha256')
    if (reference.get('seed') != seed or reference.get('producer_result_file') != str(result_path)
            or reference.get('producer_adapter_dir') != str(adapter_dir)
            or not isinstance(expected_final, str) or not re.fullmatch('[0-9a-f]{64}', expected_final)
            or (seed == 42 and expected_final != SEED42_FINAL_SHA)
            or adapter_dir != result_path.parent / 'adapter'):
        raise ValueError('Producer expected signature/path/seed is missing or inconsistent.')
    source_hashes = {str(config_file): sha_file(config_file), str(result_path): sha_file(result_path)}
    if source_hashes[str(result_path)] != reference.get('producer_result_file_sha256'):
        raise ValueError('Producer result bytes differ from the independent reference.')
    bindings = reference.get('source_binding_files_sha256')
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError('Nonempty accepted producer metadata source chain required.')
    if seed == 42 and (len(config.get('producer_reference_files_sha256', {})) != 8
            or any(bindings.get(path) != value for path, value in config['producer_reference_files_sha256'].items())):
        raise ValueError('Producer metadata chain differs from explicitly configured accepted evidence.')
    for path, expected in bindings.items():
        path = private_path(path)
        if sha_file(path) != expected:
            raise ValueError(f'Producer metadata source binding changed: {path}')
        source_hashes[str(path)] = expected
    result = _read(result_path)
    final_fields = [result[k] for k in ('final_lora_tensor_sha256', 'final_master_sha256') if k in result]
    initial_fields = [result[k] for k in ('initial_adapter_tensor_sha256', 'initial_canonical_master_sha256') if k in result]
    if (result.get('status') != 'completed' or result.get('seed') != seed or result.get('completed_updates') != 181
            or not final_fields or any(value != expected_final for value in final_fields)
            or not initial_fields or any(value != INITIAL_SHA for value in initial_fields)):
        raise ValueError('Completed producer metadata does not bind expected initial/Final signatures.')
    if seed == 42:
        if result.get('schema_version') != 'rarevlm.generated_prefix_kl.train.v1' or result.get('condition') != 'teacher_kl32':
            raise ValueError('seed42 must reuse the accepted stage32 producer, not fabricate new training.')
    elif (result.get('schema_version') != 'rarevlm.lora_coordinate_shrink.producer_train.v1'
            or result.get('condition') != 'lora_half32' or result.get('artifact_role') != 'unshrunk_producer'):
        raise ValueError('Replication must use its actual independently bound unshrunk producer.')
    actual_files = _file_map(adapter_dir)
    if actual_files != reference.get('producer_adapter_files_sha256') or actual_files != result.get('adapter_files_sha256'):
        raise ValueError('Producer adapter raw file bytes differ from accepted result/reference.')
    initial_dir = private_path(config['initial_adapter']); initial_files = _file_map(initial_dir)
    if (reference.get('initial_canonical_master_sha256') != INITIAL_SHA
            or reference.get('initial_adapter_dir') != str(initial_dir)
            or reference.get('initial_adapter_files_sha256') != initial_files):
        raise ValueError('Independently bound initial adapter/config/raw file bytes differ.')
    for directory, files in ((initial_dir, initial_files), (adapter_dir, actual_files)):
        validate_adapter_config(_read(directory / 'adapter_config.json'), config['model_root'])
        source_hashes.update({str(directory / name): value for name, value in files.items()})
    protocol_path = private_path(config['study_protocol']); source_hashes[str(protocol_path)] = sha_file(protocol_path)
    return config, {'config_file': str(config_file), 'config_file_sha256': sha_file(config_file),
        'export_source_file_sha256': sha_file(__file__), 'study_protocol_file': str(protocol_path),
        'study_protocol_file_sha256': source_hashes[str(protocol_path)], 'producer_reference': reference,
        'source_initial_adapter_dir': str(initial_dir), 'source_initial_adapter_files_sha256': initial_files,
        'source_producer_result_file': str(result_path), 'source_producer_result_file_sha256': source_hashes[str(result_path)],
        'source_producer_adapter_dir': str(adapter_dir), 'source_producer_adapter_files_sha256': actual_files,
        'expected_source_final_fp32_tensor_sha256': expected_final, 'source_files_sha256': source_hashes,
        'packages': {name: importlib.metadata.version(name) for name in ('torch', 'safetensors')}}, initial_dir, adapter_dir


def _reconstruct(binding, initial_dir, final_dir):
    raw = load_file(str(initial_dir / 'adapter_model.safetensors'), device='cpu')
    final = load_file(str(final_dir / 'adapter_model.safetensors'), device='cpu')
    validate_adapter_tensors(raw); validate_adapter_tensors(final)
    if set(raw) != set(final):
        raise ValueError('Initial/Final PEFT raw key sets differ; no silent renaming/reordering.')
    canonical, half = coordinate_half(raw, final)
    if canonical_tensor_sha256(canonical) != INITIAL_SHA:
        raise ValueError('Canonical BF16-rounded initial signature is not the fixed6a44 anchor.')
    if canonical_tensor_sha256(final) != binding['expected_source_final_fp32_tensor_sha256']:
        raise ValueError('Actual Final FP32 tensors differ from independently expected producer signature.')
    validate_adapter_tensors(half)
    statistics, norms = _statistics(raw, canonical, final, half)
    summaries = {'raw_initial_fp32_tensor_sha256': canonical_tensor_sha256(raw),
        'initial_canonical_fp32_tensor_sha256': canonical_tensor_sha256(canonical),
        'source_final_fp32_tensor_sha256': canonical_tensor_sha256(final),
        'exported_fp32_tensor_sha256': canonical_tensor_sha256(half),
        'exported_bf16_tensor_sha256': canonical_tensor_sha256({k: t.to(torch.bfloat16) for k, t in half.items()}),
        'exported_bf16_round_fp32_tensor_sha256': canonical_tensor_sha256({k: t.to(torch.bfloat16).to(torch.float32) for k, t in half.items()}),
        'tensor_statistics': statistics, 'norms': norms}
    return half, summaries


def _unchanged(binding):
    if any(sha_file(path) != value for path, value in binding['source_files_sha256'].items()):
        raise ValueError('Source bytes changed during CPU export/verification.')


def verify_export(config_file, seed, output_dir, producer_reference=None):
    started = time.monotonic(); output_dir = private_path(output_dir)
    file = output_dir / 'EXPORT_MANIFEST.json'; manifest = _read(file)
    if manifest.get('manifest_sha256') != content_sha({k: v for k, v in manifest.items() if k != 'manifest_sha256'}):
        raise ValueError('Export manifest content seal changed.')
    reference = producer_reference if producer_reference is not None else manifest.get('producer_reference')
    _, binding, initial_dir, final_dir = _sources(config_file, seed, manifest['source_producer_result_file'],
        manifest['source_producer_adapter_dir'], output_dir, reference)
    fixed = {'schema_version': SCHEMA, 'status': 'completed_cpu_export', 'condition': 'lora_half32', 'seed': seed,
        'alpha': .5, 'formula': FORMULA, 'arithmetic': 'cpu_fp32', 'operation_order': OPERATION_ORDER,
        'tensor_count': EXPECTED_TENSORS, 'parameter_elements': EXPECTED_ELEMENTS, 'r': 8, 'lora_alpha': 16, 'scaling': 2,
        'new_training_updates': 0, 'new_GPU_operations': 0, 'PEFT_weighted_merge_called': False}
    if any(manifest.get(k) != value for k, value in {**fixed, **binding}.items()):
        raise ValueError('Export identity, formula, source or parameter budget differs.')
    for relative, value in manifest['export_files_sha256'].items():
        if relative not in ('adapter/adapter_config.json', 'adapter/adapter_model.safetensors') or sha_file(output_dir / relative) != value:
            raise ValueError('Export file bytes differ from immutable manifest.')
    if set(manifest['export_files_sha256']) != {'adapter/adapter_config.json', 'adapter/adapter_model.safetensors'}:
        raise ValueError('Both original config and exact candidate weights must be bound.')
    if (output_dir / 'adapter/adapter_config.json').read_bytes() != (final_dir / 'adapter_config.json').read_bytes():
        raise ValueError('Export must copy the original adapter_config bytes, including alpha16.')
    expected, summaries = _reconstruct(binding, initial_dir, final_dir)
    exported = load_file(str(output_dir / 'adapter/adapter_model.safetensors'), device='cpu')
    validate_adapter_tensors(exported)
    if set(exported) != set(expected) or any(not torch.equal(exported[k], expected[k]) for k in expected):
        raise ValueError('Export is not the exact prescribed three-step CPU FP32 transform.')
    if any(manifest.get(k) != value for k, value in summaries.items()):
        raise ValueError('Export tensor hashes, per-tensor diagnostics or norms differ.')
    _unchanged(binding)
    return {'schema_version': 'rarevlm.lora_coordinate_shrink.verified_export.v1', 'status': 'verified_cpu_export',
        'condition': 'lora_half32', 'seed': seed, 'export_manifest_file': str(file), 'export_manifest_file_sha256': sha_file(file),
        'adapter_dir': str(output_dir / 'adapter'), 'adapter_files_sha256': _file_map(output_dir / 'adapter'),
        'source_producer_result_file': binding['source_producer_result_file'],
        'source_producer_result_file_sha256': binding['source_producer_result_file_sha256'],
        'producer_reference': reference, **{k: summaries[k] for k in summaries if k.endswith('sha256')},
        'tensor_count': EXPECTED_TENSORS, 'parameter_elements': EXPECTED_ELEMENTS, 'alpha': .5,
        'r': 8, 'lora_alpha': 16, 'numeric_relationship_verified_by_torch_equal': True,
        'new_training_updates': 0, 'new_GPU_operations': 0, 'verify_CPU_wall_seconds': time.monotonic() - started}


def export_adapter(config_file, seed, producer_result_path, producer_adapter_dir, output_dir, producer_reference=None):
    started = time.monotonic(); torch.set_num_threads(2)
    _, binding, initial_dir, final_dir = _sources(config_file, seed, producer_result_path, producer_adapter_dir,
        output_dir, producer_reference)
    output_dir = private_path(output_dir); manifest_path = output_dir / 'EXPORT_MANIFEST.json'
    if manifest_path.exists():
        verify_export(config_file, seed, output_dir, producer_reference)
        return _read(manifest_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError('Nonempty incomplete/conflicting export directory; refusing silent overwrite.')
    expected, summaries = _reconstruct(binding, initial_dir, final_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.with_name('.' + output_dir.name + '.export-' + uuid.uuid4().hex)
    (staging / 'adapter').mkdir(parents=True)
    try:
        shutil.copyfile(final_dir / 'adapter_config.json', staging / 'adapter/adapter_config.json')
        save_file(expected, str(staging / 'adapter/adapter_model.safetensors'), metadata={'format': 'pt',
            'method': 'fixed_lora_coordinate_update_shrink', 'alpha': '0.5'})
        files = {name: sha_file(staging / name) for name in ('adapter/adapter_config.json', 'adapter/adapter_model.safetensors')}
        body = {'schema_version': SCHEMA, 'status': 'completed_cpu_export', 'condition': 'lora_half32', 'seed': seed,
            'alpha': .5, 'formula': FORMULA, 'arithmetic': 'cpu_fp32', 'operation_order': OPERATION_ORDER,
            'tensor_count': EXPECTED_TENSORS, 'parameter_elements': EXPECTED_ELEMENTS, 'r': 8, 'lora_alpha': 16, 'scaling': 2,
            'diagnostic_arithmetic': 'FP32 delta subtraction; FP64 per-tensor squared-sum; Python-float canonical-order aggregation; no FP64 transform intermediate',
            'exported_weight_dtype': 'torch.float32', 'evaluation_native_adapter_dtype': 'torch.bfloat16',
            **binding, **summaries, 'export_files_sha256': files, 'CPU_wall_seconds': time.monotonic() - started,
            'CPU_cost_scope': 'source validation, FP32 construction, diagnostics and adapter serialization; excludes later independent verify_export',
            'new_training_updates': 0, 'new_GPU_operations': 0, 'PEFT_weighted_merge_called': False,
            'dense_BA_midpoint_claim': False, 'norm_ratio_exact_half_claim': False}
        _unchanged(binding)
        _write(staging / 'EXPORT_MANIFEST.json', {**body, 'manifest_sha256': content_sha(body)})
        # Directory rename commits adapter+manifest together; no RESULT/COMPLETE split window.
        os.replace(staging, output_dir)
    except Exception as error:
        _write(staging / 'EXPORT_FAILURE.json', {'error': f'{type(error).__name__}: {error}',
            'new_training_updates': 0, 'new_GPU_operations': 0, 'source_files_modified': False})
        raise
    verify_export(config_file, seed, output_dir, producer_reference)
    return _read(manifest_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--seed', required=True, type=int, choices=(0, 42, 200))
    parser.add_argument('--producer-result', required=True, type=Path)
    parser.add_argument('--producer-adapter-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--producer-reference', required=True, type=Path)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args(); reference = _read(private_path(args.producer_reference))
    result = (verify_export(args.config, args.seed, args.output_dir, reference) if args.verify_only else
        export_adapter(args.config, args.seed, args.producer_result, args.producer_adapter_dir, args.output_dir, reference))
    print(json.dumps({k: v for k, v in result.items() if k not in ('tensor_statistics', 'source_files_sha256', 'producer_reference')},
        ensure_ascii=False))


if __name__ == '__main__':
    main()
