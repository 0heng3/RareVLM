"""Synthetic CPU-only tests; never read project initial/Final weight files."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file, load_file

from vlm_lora_coordinate_shrink import export as module


def test_exact_FP32_steps_round_initial_before_delta_without_mutating_sources():
    raw = {'x': torch.tensor([1.001, -.12345, .00331, 0.], dtype=torch.float32)}
    final = {'x': torch.tensor([1.02, -.14, .004, 0.], dtype=torch.float32)}
    before_raw, before_final = raw['x'].clone(), final['x'].clone()
    initial, half = module.coordinate_half(raw, final)
    expected_initial = raw['x'].to(torch.bfloat16).to(torch.float32)
    delta = final['x'] - expected_initial
    assert torch.equal(initial['x'], expected_initial)
    assert torch.equal(half['x'], expected_initial + delta * .5)
    assert not torch.equal(half['x'], raw['x'] + (final['x'] - raw['x']) * .5)
    assert half['x'].dtype == torch.float32 and half['x'].device.type == 'cpu'
    assert torch.equal(raw['x'], before_raw) and torch.equal(final['x'], before_final)


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_nonfinite_input_rejected(bad):
    with pytest.raises(ValueError, match='finite CPU FP32'):
        module.coordinate_half({'x': torch.tensor([bad])}, {'x': torch.tensor([1.])})


def test_dtype_keys_shapes_and_FP32_overflow_rejected():
    with pytest.raises(ValueError, match='keys differ'):
        module.coordinate_half({'x': torch.tensor([1.])}, {'y': torch.tensor([1.])})
    with pytest.raises(ValueError, match='finite CPU FP32'):
        module.coordinate_half({'x': torch.tensor([1.], dtype=torch.bfloat16)}, {'x': torch.tensor([1.])})
    with pytest.raises(ValueError, match='finite CPU FP32'):
        module.coordinate_half({'x': torch.tensor([1.])}, {'x': torch.tensor([[1.]])})
    largest = torch.finfo(torch.float32).max
    with pytest.raises(ValueError, match='nonfinite'):
        module.coordinate_half({'x': torch.tensor([largest])}, {'x': torch.tensor([-largest])})


def inventory():
    result = {}
    for layer in range(36):
        for projection in ('q_proj', 'v_proj'):
            for factor in ('A', 'B'):
                shape = (8, 2048) if factor == 'A' else ((2048, 8) if projection == 'q_proj' else (256, 8))
                key = f'base_model.model.model.layers.{layer}.self_attn.{projection}.lora_{factor}.weight'
                result[key] = torch.full(shape, .125 + layer / 4096, dtype=torch.float32)
    return result


def test_production_inventory_count_shape_and_canonical_order():
    values = inventory(); entries = module.validate_adapter_tensors(values)
    assert len(entries) == 144 and sum(t.numel() for t in values.values()) == 1843200
    names = [item[1] for item in entries]
    assert names[0].endswith('layers.0.self_attn.q_proj.lora_A.default.weight')
    assert names[8].endswith('layers.2.self_attn.q_proj.lora_A.default.weight')
    assert names[40].endswith('layers.10.self_attn.q_proj.lora_A.default.weight')
    missing = dict(values); missing.pop(next(iter(missing)))
    with pytest.raises(ValueError, match='Exactly144'):
        module.validate_adapter_tensors(missing)
    bad = dict(values); key = next(iter(bad)); bad[key] = bad[key].reshape(2048, 8)
    with pytest.raises(ValueError, match='shape/dtype'):
        module.validate_adapter_tensors(bad)


def test_visual_unknown_and_duplicate_canonical_keys_rejected():
    with pytest.raises(ValueError, match='nonlanguage-qv'):
        module._factor_name('base_model.model.visual.q_proj.lora_A.weight')
    key = 'base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight'
    with pytest.raises(ValueError, match='Duplicate canonical'):
        module._ordered({key: torch.zeros(1), key.replace('.lora_A.weight', '.lora_A.default.weight'): torch.zeros(1)})


@pytest.mark.parametrize('changed', [{'r': 16}, {'lora_alpha': 8}, {'use_dora': True}, {'rank_pattern': {'q_proj': 4}}])
def test_no_rank_scale_or_factorization_change(changed):
    cfg = adapter_config('/opt/data/private/synthetic_model'); cfg.update(changed)
    with pytest.raises(ValueError):
        module.validate_adapter_config(cfg)


def adapter_config(model):
    return {'peft_type': 'LORA', 'r': 8, 'lora_alpha': 16, 'bias': 'none', 'fan_in_fan_out': False,
        'use_dora': False, 'use_rslora': False, 'lora_bias': False, 'modules_to_save': None,
        'target_modules': ['v_proj', 'q_proj'], 'rank_pattern': {}, 'alpha_pattern': {},
        'layer_replication': None, 'base_model_name_or_path': str(model), 'inference_mode': True,
        'lora_dropout': .05}


@pytest.fixture
def synthetic_case(tmp_path, monkeypatch):
    # Full production inventory uses only ~7MiB, all deterministic synthetic values.
    raw = inventory(); canonical = {k: t.to(torch.bfloat16).float() for k, t in raw.items()}
    final = {k: t + .003 for k, t in canonical.items()}
    initial_sha, final_sha = module.canonical_tensor_sha256(canonical), module.canonical_tensor_sha256(final)
    monkeypatch.setattr(module, 'INITIAL_SHA', initial_sha)
    monkeypatch.setattr(module, 'SEED42_FINAL_SHA', final_sha)
    model, initial, producer = tmp_path / 'model', tmp_path / 'initial', tmp_path / 'producer'
    model.mkdir(); initial.mkdir(); (producer / 'adapter').mkdir(parents=True)
    config_bytes = json.dumps(adapter_config(model), indent=2).encode() + b'\n'
    for directory, tensors in ((initial, raw), (producer / 'adapter', final)):
        (directory / 'adapter_config.json').write_bytes(config_bytes)
        save_file(tensors, str(directory / 'adapter_model.safetensors'))
    result = {'schema_version': 'rarevlm.generated_prefix_kl.train.v1', 'condition': 'teacher_kl32', 'seed': 42,
        'status': 'completed', 'completed_updates': 181, 'initial_adapter_tensor_sha256': initial_sha,
        'final_lora_tensor_sha256': final_sha, 'adapter_files_sha256': module._file_map(producer / 'adapter')}
    result_path = producer / 'TRAIN_RESULT.json'; result_path.write_text(json.dumps(result))
    bindings = {}
    for i in range(7):
        proof = tmp_path / f'accepted_metadata_{i}.json'; proof.write_text(json.dumps({'synthetic': i}))
        bindings[str(proof)] = module.sha_file(proof)
    bindings[str(result_path)] = module.sha_file(result_path)
    protocol = tmp_path / 'protocol.md'; protocol.write_text('synthetic source contract')
    output_root = tmp_path / 'new'; output = output_root / 'runs/lora_half32/seed42/shrink'
    config = {'schema_version': 'rarevlm.lora_coordinate_shrink.v1', 'initial_adapter': str(initial),
        'model_root': str(model), 'output_root': str(output_root), 'study_protocol': str(protocol),
        'producer_reference_files_sha256': bindings,
        'coordinate_shrink': {'alpha': .5, 'initial_master_sha256': initial_sha,
            'seed42_producer_master_sha256': final_sha, 'formula': module.FORMULA,
            'arithmetic': 'cpu_fp32', 'candidate_adapter_subdir': 'shrink/adapter'}}
    config_path = tmp_path / 'config.json'; config_path.write_text(json.dumps(config))
    reference = {'schema_version': module.REFERENCE_SCHEMA, 'seed': 42,
        'producer_result_file': str(result_path), 'producer_result_file_sha256': module.sha_file(result_path),
        'producer_adapter_dir': str(producer / 'adapter'), 'producer_adapter_files_sha256': result['adapter_files_sha256'],
        'expected_final_master_sha256': final_sha, 'source_binding_files_sha256': bindings,
        'initial_canonical_master_sha256': initial_sha, 'initial_adapter_dir': str(initial),
        'initial_adapter_files_sha256': module._file_map(initial)}
    return config_path, result_path, producer / 'adapter', output, reference, raw, final


def test_synthetic_export_config_bytes_numeric_relation_idempotence_and_conflict(synthetic_case):
    cfg, result, producer, output, reference, raw, final = synthetic_case
    before = {str(p): module.sha_file(p) for directory in (producer, Path(json.loads(cfg.read_text())['initial_adapter'])) for p in directory.iterdir()}
    manifest = module.export_adapter(cfg, 42, result, producer, output, reference)
    assert manifest['alpha'] == .5 and manifest['r'] == 8 and manifest['lora_alpha'] == 16
    assert (output / 'adapter/adapter_config.json').read_bytes() == (producer / 'adapter_config.json').read_bytes()
    _, expected = module.coordinate_half(raw, final)
    actual = load_file(str(output / 'adapter/adapter_model.safetensors'))
    assert all(torch.equal(actual[k], expected[k]) for k in expected)
    assert all(module.sha_file(path) == sha for path, sha in before.items())
    saved = (output / 'EXPORT_MANIFEST.json').read_bytes()
    assert module.export_adapter(cfg, 42, result, producer, output, reference) == manifest
    assert (output / 'EXPORT_MANIFEST.json').read_bytes() == saved
    verified = module.verify_export(cfg, 42, output, reference)
    assert verified['numeric_relationship_verified_by_torch_equal'] is True
    assert verified['exported_bf16_tensor_sha256'] != verified['exported_bf16_round_fp32_tensor_sha256']
    wrong = deepcopy(reference); wrong['expected_final_master_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='signature/path/seed'):
        module.export_adapter(cfg, 42, result, producer, output, wrong)
    (output / 'adapter/adapter_config.json').write_bytes(b'{}')
    with pytest.raises(ValueError, match='file bytes'):
        module.verify_export(cfg, 42, output, reference)


def test_missing_independent_expected_signature_rejected_before_tensor_load(synthetic_case):
    cfg, result, producer, output, reference, _, _ = synthetic_case
    with pytest.raises(ValueError, match='independently bound producer reference'):
        module.export_adapter(cfg, 42, result, producer, output)
    missing = deepcopy(reference); missing.pop('expected_final_master_sha256')
    with pytest.raises(ValueError, match='signature/path/seed'):
        module.export_adapter(cfg, 42, result, producer, output, missing)
