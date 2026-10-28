"""CPU-only actual-ID/EOS/mask and source rejection contracts."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vlm_object_behavior_replay import data
from vlm_object_behavior_replay import rollout


@pytest.fixture
def config(tmp_path):
    model = tmp_path / 'model'; model.mkdir()
    (model / 'config.json').write_text(json.dumps({'vocab_size': 151936, 'image_token_id': 8}))
    return {'output_root': str(tmp_path / 'out'), 'teacher_reply_root': str(tmp_path / 'out/replies'),
        'model_root': str(model), 'max_sequence_length': 2048, 'visual_token_budget': 1024,
        'teacher_generation': {'shards': 2, 'batch_size': 4, 'max_new_tokens': 32, 'do_sample': False,
            'use_cache': True, 'repetition_penalty': 1.05, 'preflight_questions': 8, 'plan_questions_per_seed': 5792,
            'qa_per_shard': 2896, 'microbatches_per_shard': 724,
            'partition': 'contiguous_complete_microbatches',
            'actual_generated_ids_required': True, 'retokenize_teacher_answers': False,
            'eos_token_ids': [151645, 151643], 'pad_token_id': 151643,
            'chat_template_eos_token_id': 151645, 'post_chat_eos_template_ids': [198],
            'non_chat_eos_append_template': False, 'capped_append_eos_or_template': False,
            'keep_empty_content_when_real_eos': True, 'filter_or_resample_teacher_answers': False,
            'teacher_inference_parameter_precision': 'bf16_all_floating_including_lora',
            'teacher_inference_autocast': False,
            'initial_runtime_master_sha256': '6a44359406c53261b254c3a19393c63187b34af09baec539d6b235ca148c0fa7'}}


@pytest.fixture
def row(tmp_path):
    return {'question_id': 'q1', 'image_id': 'image1', 'question': 'read the sign',
        'target': 'GT_SENTINEL_MUST_NOT_ENTER_GENERATION', 'answers': ['secret'] * 10,
        'image_path': str(tmp_path / 'image.jpg'), 'prompt_suffix': '\nshort answer'}


def fake_sample(supervised=False, body=(31, 32)):
    prefix = [7, 8, 9]
    ids = prefix + (list(body) + [151645, 198] if supervised else [])
    result = {'input_ids': torch.tensor([ids]), 'attention_mask': torch.ones((1, len(ids)), dtype=torch.long),
        'image_grid_thw': torch.tensor([[1, 2, 2]]), 'pixel_values': torch.tensor([[.25, .5]] * 4)}
    if supervised:
        result['labels'] = result['input_ids'].clone(); result['labels'][:, :len(prefix)] = -100
    return result


def reply(config, row, raw=(44, 151645, 151643)):
    prompt = {**data.prompt_contract(fake_sample(), row, config), 'global_index': 0,
        'global_batch_index': 0, 'microbatch_input_sha256': 'actual-batch', 'padded_prompt_width': 3}
    return {'question_id': row['question_id'], 'image_id': row['image_id'], 'global_index': 0, 'seed': 42,
        'prompt_contract': prompt, **data.parse_generated_suffix(list(raw), config['teacher_generation'], 151936)}


@pytest.mark.parametrize('raw,target,roles,reason', [
    ([44, 45, 151645, 151643], [44, 45, 151645, 198], [1, 2, 3, 4], 'chat_eos'),
    ([44, 151643, 151643], [44, 151643], [1, 3], 'other_eos'),
    ([151645, 151643], [151645, 198], [3, 4], 'empty_content_chat_eos'),
    ([151643, 151643], [151643], [3], 'other_eos'),
    ([44] * 32, [44] * 32, [1] + [2] * 31, 'capped_without_eos'),
])
def test_actual_eos_pad_empty_and_cap_are_preserved(config, raw, target, roles, reason):
    result = data.parse_generated_suffix(raw, config['teacher_generation'], 151936)
    assert result['raw_generated_suffix_ids'] == raw
    assert result['training_target_ids'] == target
    assert result['training_answer_roles'] == roles
    assert result['termination_reason'] == reason


@pytest.mark.parametrize('raw', [[], [44] * 31, [44] * 33, [-1], [151936], [True], ['44'], [44, 151645, 45]])
def test_invalid_or_fabricated_termination_is_rejected(config, raw):
    with pytest.raises(ValueError):
        data.parse_generated_suffix(raw, config['teacher_generation'], 151936)


def test_content_newline_before_eos_is_not_a_template_role(config):
    parsed = data.parse_generated_suffix([44, 198, 151645], config['teacher_generation'], 151936)
    assert parsed['training_answer_roles'] == [1, 2, 3, 4]
    assert parsed['training_target_ids'] == [44, 198, 151645, 198]


def test_generation_allowlist_contains_no_GT_or_references(row):
    actual = data.generation_row(row)
    assert set(actual) == {'question_id', 'image_id', 'question', 'prompt_suffix'}
    assert row['target'] not in json.dumps(actual)
    assert 'answers' not in actual


def test_teacher_encoder_keeps_actual_ids_and_does_not_retokenize(config, row, monkeypatch):
    seen = []
    def encode(processor, actual_row, path, supervised):
        seen.append((actual_row, supervised)); assert not supervised
        assert 'target' not in actual_row and 'answers' not in actual_row
        return fake_sample()
    monkeypatch.setattr(data, 'encode', encode)
    processor = SimpleNamespace(tokenizer=SimpleNamespace())
    teacher = reply(config, row)
    sample = data.encode_training_sample(processor, row, config, source='replay', condition='teacher_ce32', teacher_replies={'q1': teacher})
    assert sample['input_ids'].tolist() == [[7, 8, 9, 44, 151645, 198]]
    assert sample['labels'].tolist() == [[-100, -100, -100, 44, 151645, 198]]
    assert sample['answer_roles'].tolist() == [[0, 0, 0, 1, 3, 4]]
    assert len(seen) == 1


@pytest.mark.parametrize('raw', [[151645], [151643], [44] * 32])
def test_teacher_encoder_does_not_filter_empty_or_capped(config, row, monkeypatch, raw):
    monkeypatch.setattr(data, 'encode', lambda *args: fake_sample())
    teacher = reply(config, row, raw)
    sample = data.encode_training_sample(SimpleNamespace(), row, config, source='replay', condition='teacher_ce32', teacher_replies={'q1': teacher})
    assert sample['labels'][sample['labels'] != -100].tolist() == teacher['training_target_ids']


def test_missing_teacher_never_falls_back_to_GT(config, row):
    with pytest.raises(ValueError, match='no GT fallback'):
        data.encode_training_sample(None, row, config, source='replay', condition='teacher_ce32', teacher_replies={})


@pytest.mark.parametrize('field,value', [('question', 'wrong prompt'), ('image_id', 'wrong image'), ('prompt_suffix', 'wrong suffix')])
def test_teacher_rejects_changed_prompt_identity(config, row, monkeypatch, field, value):
    monkeypatch.setattr(data, 'encode', lambda *args: fake_sample())
    teacher = reply(config, row)
    changed = {**row, field: value}
    with pytest.raises(ValueError, match='prompt input differs'):
        data.encode_training_sample(None, changed, config, source='replay', condition='teacher_ce32', teacher_replies={'q1': teacher})


def test_teacher_rejects_changed_target_ID_or_role(config, row, monkeypatch):
    monkeypatch.setattr(data, 'encode', lambda *args: fake_sample())
    teacher = reply(config, row); teacher['training_target_ids'][0] = 99
    with pytest.raises(ValueError, match='teacher IDs'):
        data.encode_training_sample(None, row, config, source='replay', condition='teacher_ce32', teacher_replies={'q1': teacher})


def test_original_GT_tensor_fields_and_legacy_hash_preserved(config, row, monkeypatch):
    original = fake_sample(True)
    monkeypatch.setattr(data, 'encode', lambda *args: {key: value.clone() for key, value in original.items()})
    sample = data.encode_training_sample(None, row, config, source='replay', condition='gt_ce32')
    for key, value in original.items():
        assert torch.equal(sample[key], value)
    assert sample['answer_roles'].tolist() == [[0, 0, 0, 1, 2, 3, 4]]
    assert data.sample_contract(sample, row, config)['input_sha256'] == data.batch_hash(original)


def test_left_padding_uses_training_sample_lengths_and_role_zero(config, row, monkeypatch):
    monkeypatch.setattr(data, 'encode', lambda processor, r, path, supervised: fake_sample(True, (31,) if r['question_id'] == 'q1' else (31, 32, 33)))
    first = data.encode_training_sample(None, row, config, source='object', condition='gt_ce32')
    second = data.encode_training_sample(None, {**row, 'question_id': 'q2'}, config, source='object', condition='gt_ce32')
    batch = data.collate_training([first, second], 151643)
    assert batch['labels'][0, :2].tolist() == [-100, -100]
    assert batch['answer_roles'][0, :2].tolist() == [0, 0]
    assert torch.equal(batch['labels'] != -100, batch['answer_roles'] != 0)
    assert batch['labels'][0, -3:].tolist() == [31, 151645, 198]


def test_seal_and_missing_teacher_manifest_reject(config, monkeypatch):
    value = data.sealed({'source': 'fixed'})
    changed = {**value, 'source': 'mutated'}
    with pytest.raises(ValueError):
        data.check_seal(changed)
    monkeypatch.setattr(data, 'load_prompt_inputs', lambda *args: ({}, {}, []))
    with pytest.raises(FileNotFoundError):
        data.load_teacher_replies(config, 42)


def test_missing_complete_shard_does_not_generate(config, monkeypatch):
    monkeypatch.setattr(rollout, 'load_prompt_inputs', lambda *args: ({}, {}, []))
    with pytest.raises(FileNotFoundError):
        rollout.verify_completed_shard(config, 42, 0)


def test_batch_rejects_wrong_phase_producer_or_input(config, row):
    teacher = reply(config, row)
    binding = {'seed': 42, 'producer_uuid': 'GPU-original', 'teacher_initial_master_sha256': 'fixed'}
    teacher.update(producer_uuid='GPU-original', teacher_initial_master_sha256='fixed', shard_binding_sha256=data.content_sha(binding))
    prompt = teacher['prompt_contract']
    value = data.sealed({'schema_version': 'rarevlm.object_behavior_replay.teacher_batch.v1',
        'phase': 'formal', 'binding_sha256': data.content_sha(binding), 'producer_uuid': 'GPU-original',
        'global_batch_index': 0, 'actual_microbatch_input_sha256': 'actual-batch', 'actual_padded_prompt_width': 3,
        'records': [teacher]}, 'batch_sha256')
    assert rollout.validate_batch(value, binding, [prompt], config, 151936) == value
    for key, bad in [('producer_uuid', 'GPU-other'), ('phase', 'preflight'), ('actual_microbatch_input_sha256', 'wrong')]:
        changed = {k: v for k, v in value.items() if k != 'batch_sha256'}; changed[key] = bad
        with pytest.raises(ValueError):
            rollout.validate_batch(data.sealed(changed, 'batch_sha256'), binding, [prompt], config, 151936)


def test_no_admission_prevents_any_CUDA_or_model_action(config, tmp_path, monkeypatch):
    import vlm_object_behavior_replay.contracts as contracts
    cfg = tmp_path / 'config.json'; cfg.write_text(json.dumps(config))
    monkeypatch.setattr(rollout, 'load_prompt_inputs', lambda *args: ({'plan_file_sha256': 'plan', 'prompt_inputs_file_sha256': 'prompts'}, {}, []))
    monkeypatch.setattr(rollout, 'visible_gpu_binding', lambda: {'uuid': 'GPU-test'})
    def deny(*args, **kwargs):
        raise ValueError('queue admission denied')
    monkeypatch.setattr(contracts, 'validate_gpu_admission', deny)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: pytest.fail('CUDA touched before admission'))
    monkeypatch.setattr(rollout, 'load_path', lambda *args: pytest.fail('model loaded without admission'))
    with pytest.raises(ValueError, match='admission denied'):
        rollout.run_teacher_shard(cfg, 42, 0, config['output_root'], tmp_path / 'admission.json')


@pytest.fixture
def plan_inputs(config, tmp_path, monkeypatch):
    config.update(epochs=1, object_effective_batch=16, replay_per_update=32, lr=5e-6, warmup_ratio=.1)
    for key in ['train_manifest', 'replay_manifest', 'study_protocol']:
        file = tmp_path / (key + '.json'); file.write_text('{}'); config[key] = str(file)
    objects = [{'question_id': f'o{i}', 'image_id': f'oi{i}'} for i in range(2888)]
    replays = [{'question_id': f'r{i}', 'image_id': f'ri{i}', 'target': 'not-a-selection-feature'} for i in range(32985)]
    monkeypatch.setattr(data, 'load_rows', lambda c: (objects, replays))
    return config, objects, replays


def test_plan_is_full_unique_SHA_prefix_and_independent_of_GT(plan_inputs):
    import hashlib
    config, objects, replays = plan_inputs
    plan = data.build_plan(config, 0)
    assert len(plan['blocks']) == 181
    assert sum(len(b['object_indices']) for b in plan['blocks']) == 2888
    actual = [i for b in plan['blocks'] for i in b['replay_indices']]
    expected = sorted(range(32985), key=lambda i: (hashlib.sha256(f'object-replay-v1|0|0|r{i}'.encode()).hexdigest(), f'r{i}'))[:5792]
    assert actual == expected and len(set(actual)) == 5792
    assert len(plan['schedule']['replay_microbatches_by_update'][0]) == 8
    for i, row in enumerate(replays):
        row['target'] = f'different-hidden-GT-{i}'
    assert data.build_plan(config, 0)['schedule'] == plan['schedule']


def test_seed42_plan_rejects_changed_historical_order(plan_inputs, tmp_path):
    config, _, _ = plan_inputs
    old = tmp_path / 'old/runs/wide32/seed42/train'; old.mkdir(parents=True)
    (old / 'SCHEDULE.json').write_text(json.dumps({'changed_order': True}))
    config['historical_coverage_study_root'] = str(tmp_path / 'old')
    with pytest.raises(ValueError, match='differs from completed frozen wide32'):
        data.build_plan(config, 42)


@pytest.mark.parametrize('field,value', [('retokenize_teacher_answers', True), ('actual_generated_ids_required', False), ('capped_append_eos_or_template', True)])
def test_generation_contract_rejects_unsafe_policy_changes(config, field, value):
    config['teacher_generation'][field] = value
    with pytest.raises(ValueError):
        data.generation_spec(config)


def test_native_BF16_canonical_identity_does_not_mutate_dtype_or_parameters(config):
    native = torch.nn.Parameter(torch.tensor([.125, .75], dtype=torch.bfloat16), requires_grad=False)
    base = torch.nn.Parameter(torch.tensor([1.], dtype=torch.bfloat16), requires_grad=False)
    name = 'model.layers.0.self_attn.q_proj.lora_A.default.weight'
    model = SimpleNamespace(named_parameters=lambda: [(name, native), ('base.weight', base)])
    config['expected_trainable'] = 2
    before = native.clone()
    identity = rollout.teacher_identity(model, config)
    assert identity['canonical_lora_fp32_representation_sha256'] == data.tensor_hash([(name, native.float())])
    assert identity['native_lora_sha256'] == data.tensor_hash([(name, native)])
    assert native.dtype == torch.bfloat16 and not native.requires_grad and torch.equal(native, before)
    native.data = native.data.float()
    with pytest.raises(ValueError, match='native BF16'):
        rollout.teacher_identity(model, config)


def test_effective_generation_uses_loader_policy_and_explicit_active_repetition(config):
    class FakePolicy(SimpleNamespace):
        def to_dict(self):
            return deepcopy(vars(self))
    native = FakePolicy(temperature=1e-6, top_p=None, top_k=None, repetition_penalty=1.05,
        bad_words_ids=[[999]], eos_token_id=[151645, 151643])
    loaded = deepcopy(native); loaded.temperature = 1.; loaded.top_p = 1.; loaded.top_k = 50
    loaded.loader_preserved_value = 'actual loader field'
    file = Path(config['model_root']) / 'generation_config.json'; file.write_text(json.dumps(native.to_dict()))
    effective, policy = rollout.generation_policy(config, native, loaded)
    assert native.temperature == 1e-6 and loaded.top_k == 50
    assert policy['loader_generation_config'] == loaded.to_dict()
    assert effective.to_dict()['bad_words_ids'] == [[999]]
    assert effective.loader_preserved_value == 'actual loader field'
    assert effective.repetition_penalty == policy['actual_generate_kwargs']['repetition_penalty'] == 1.05
    assert (effective.temperature, effective.top_p, effective.top_k) == (1., 1., 50)
    assert policy['native_generation_config_file_sha256'] == data.sha_file(file)


def test_saved_RESULT_finalizes_before_any_CUDA_or_model_action(config, tmp_path, monkeypatch):
    import vlm_object_behavior_replay.contracts as contracts
    cfg = tmp_path / 'config.json'; cfg.write_text(json.dumps(config))
    admission = tmp_path / 'admission.json'; admission.write_text('{}')
    prompts = {'plan_file_sha256': 'plan', 'prompt_inputs_file_sha256': 'prompts'}
    monkeypatch.setattr(rollout, 'load_prompt_inputs', lambda *args: (prompts, {}, []))
    monkeypatch.setattr(rollout, 'visible_gpu_binding', lambda: {'uuid': 'GPU-original'})
    monkeypatch.setattr(contracts, 'validate_gpu_admission', lambda *args, **kwargs: {})
    directory = data.reply_dir(config, 42) / 'shard0'; directory.mkdir(parents=True)
    data.atomic_json(directory / 'INPUT_LOCK.json', data.sealed({'producer_uuid': 'GPU-original',
        'config_file_sha256': data.sha_file(cfg), 'admission_file_sha256': data.sha_file(admission),
        **prompts}, 'input_lock_sha256'))
    data.atomic_json(directory / 'RESULT.json', {'original': 'immutable result'})
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: pytest.fail('CUDA touched for CPU finalization'))
    monkeypatch.setattr(rollout, 'load_path', lambda *args: pytest.fail('teacher reloaded for completed RESULT'))
    monkeypatch.setattr(rollout, 'commit_saved_result', lambda *args, **kwargs: {'status': 'completed', 'recovery': kwargs['recovery']})
    assert rollout.run_teacher_shard(cfg, 42, 0, config['output_root'], admission) == {'status': 'completed', 'recovery': True}


def test_CPU_finalization_preserves_RESULT_and_generation_attempt_accounting(config, monkeypatch):
    directory = data.reply_dir(config, 42) / 'shard0'; (directory / 'batches').mkdir(parents=True)
    for path in [directory / 'INPUT_LOCK.json', directory / 'PREFLIGHT.json',
            *[directory / f'batches/batch_{i:06d}.json' for i in range(724)]]:
        data.atomic_json(path, {'immutable': str(path)})
    rollout.event(directory, 'model_load_started')
    rollout.event(directory, 'generation_started', phase='formal', questions=4)
    rollout.event(directory, 'generation_completed', phase='formal', questions=4, generation_seconds=.5)
    accounting = rollout.execution_accounting(directory)
    result = {'schema_version': 'rarevlm.object_behavior_replay.teacher_shard_result.v1',
        'status': 'completed', **accounting}
    data.atomic_json(directory / 'RESULT.json', result)
    original = (directory / 'RESULT.json').read_bytes()
    def verifier(*args, _completion=None, **kwargs):
        complete = data.read_json(directory / 'COMPLETE.json') if _completion is None else _completion
        assert (directory / 'RESULT.json').read_bytes() == original
        assert all(data.sha_file(path) == digest for path, digest in complete['artifact_files_sha256'].items())
        assert rollout.execution_accounting(directory) == accounting
        return complete, []
    monkeypatch.setattr(rollout, 'verify_completed_shard', verifier)
    complete = rollout.commit_saved_result(config, 42, 0, recovery=True)
    assert complete['status'] == 'completed' and len(complete['artifact_files_sha256']) == 728
    assert (directory / 'RESULT.json').read_bytes() == original
    journal = [json.loads(line) for line in (directory / 'EXECUTION.jsonl').read_text().splitlines()]
    assert [r['event'] for r in journal][-2:] == ['finalization_CPU_started', 'finalization_CPU_completed']
    assert sum(r['event'] == 'model_load_started' for r in journal) == 1


def test_merge_interruption_after_MANIFEST_rechecks_shards_before_committing_missing_lock(config, monkeypatch):
    directory = data.reply_dir(config, 42); directory.mkdir(parents=True)
    data.atomic_json(directory / 'MANIFEST.json', {'original': 'must not silently overwrite'})
    monkeypatch.setattr(data, 'load_prompt_inputs', lambda *args: ({}, {}, []))
    monkeypatch.setattr(data, 'load_teacher_replies', lambda *args: pytest.fail('premature load requires missing lock'))
    def verify(*args, **kwargs):
        raise ValueError('continued complete-shard validation')
    monkeypatch.setattr(rollout, 'verify_completed_shard', verify)
    with pytest.raises(ValueError, match='continued complete-shard'):
        data.finalize_teacher_replies(config, 42)
