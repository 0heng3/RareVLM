"""CPU input provenance and actual-ID rejection contracts; no model or processor."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vlm_generated_prefix_kl import data, inputs, rollout


@pytest.fixture
def config(tmp_path):
    model = tmp_path / 'model'; model.mkdir()
    (model / 'config.json').write_text(json.dumps({'vocab_size': 151936, 'image_token_id': 8}))
    return {'output_root': str(tmp_path / 'new'), 'teacher_reply_root': str(tmp_path / 'new/teacher_replies'),
        'model_root': str(model), 'max_sequence_length': 2048, 'visual_token_budget': 1024}


@pytest.fixture
def row(tmp_path):
    return {'question_id': 'q0', 'image_id': 'i0', 'question': 'read it',
        'target': 'GT_MUST_NOT_BE_USED_FOR_REPLAY', 'answers': ['GT'] * 10,
        'image_path': str(tmp_path / 'image.jpg'), 'prompt_suffix': '\nshort answer'}


def prompt_sample():
    return {'input_ids': torch.tensor([[7, 8, 9]]), 'attention_mask': torch.ones((1, 3), dtype=torch.long),
        'pixel_values': torch.tensor([[.25, .5]] * 4), 'image_grid_thw': torch.tensor([[1, 2, 2]])}


def accepted_reply(config, row, target=(31, 151645, 198), roles=(1, 3, 4)):
    return {'question_id': row['question_id'], 'global_index': 0,
        'training_target_ids': list(target), 'training_answer_roles': list(roles),
        'prompt_contract': data.prompt_contract(prompt_sample(), row, config),
        'source_teacher_reply_row_sha256': 'accepted_original_full_producer_row_SHA'}


def test_ID_encoder_uses_actual_generated_target_without_GT_or_tokenizer(config, row, monkeypatch):
    calls = []
    def encode(processor, selected, path, supervised):
        calls.append((selected, supervised))
        assert 'target' not in selected and 'answers' not in selected and supervised is False
        return prompt_sample()
    monkeypatch.setattr(data, 'encode', encode)
    reply = accepted_reply(config, row)
    sample = data.encode_training_sample(None, row, config, source='replay', condition='teacher_kl32', teacher_replies={'q0': reply})
    assert len(calls) == 1
    assert sample['input_ids'].tolist() == [[7, 8, 9, 31, 151645, 198]]
    assert sample['labels'].tolist() == [[-100, -100, -100, 31, 151645, 198]]
    assert sample['answer_roles'].tolist() == [[0, 0, 0, 1, 3, 4]]


@pytest.mark.parametrize('field,value', [('question', 'changed question'), ('image_id', 'wrong image'), ('prompt_suffix', 'wrong prefix')])
def test_actual_prefix_identity_changes_are_rejected(config, row, monkeypatch, field, value):
    monkeypatch.setattr(data, 'encode', lambda *args: prompt_sample())
    reply = accepted_reply(config, row)
    with pytest.raises(ValueError, match='prompt input differs'):
        data.encode_training_sample(None, {**row, field: value}, config, source='replay', condition='teacher_kl32', teacher_replies={'q0': reply})


def test_actual_prompt_token_change_rejected(config, row, monkeypatch):
    reply = accepted_reply(config, row)
    def changed(*args):
        value = prompt_sample(); value['input_ids'][0, -1] = 10; return value
    monkeypatch.setattr(data, 'encode', changed)
    with pytest.raises(ValueError, match='prompt input differs'):
        data.encode_training_sample(None, row, config, source='replay', condition='teacher_kl32', teacher_replies={'q0': reply})


def test_missing_teacher_cannot_fallback_to_ground_truth(config, row):
    with pytest.raises(ValueError, match='no GT fallback'):
        data.encode_training_sample(None, row, config, source='replay', condition='teacher_kl32', teacher_replies={})
    with pytest.raises(ValueError, match='Only teacher_kl32'):
        data.encode_training_sample(None, row, config, source='replay', condition='teacher_ce32')


@pytest.mark.parametrize('bad_token', [True, -1, 151936, 1.5])
def test_illegal_saved_IDs_are_rejected_instead_of_silently_cast(config, row, monkeypatch, bad_token):
    monkeypatch.setattr(data, 'encode', lambda *args: prompt_sample())
    reply = accepted_reply(config, row); reply['training_target_ids'][0] = bad_token
    with pytest.raises(ValueError, match='target/role IDs are invalid'):
        data.encode_training_sample(None, row, config, source='replay', condition='teacher_kl32', teacher_replies={'q0': reply})


def test_runtime_group_keeps_accepted_original_row_digest_and_leftpad(config, row, monkeypatch):
    monkeypatch.setattr(data, 'encode', lambda *args: prompt_sample())
    second = {**row, 'question_id': 'q1'}
    replies = {'q0': accepted_reply(config, row),
        'q1': accepted_reply(config, second, (41, 42, 151645, 198), (1, 2, 3, 4))}
    batch, contracts, _ = inputs.encode_group(SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=151643)),
        [row, second], [0, 1], config, source='replay', condition='teacher_kl32', teacher_replies=replies)
    assert batch['labels'][0, 0] == -100 and batch['answer_roles'][0, 0] == 0
    assert torch.equal(batch['labels'] != -100, batch['answer_roles'] != 0)
    assert contracts[0]['teacher_reply_row_sha256'] == 'accepted_original_full_producer_row_SHA'
    assert contracts[0]['teacher_reply_row_sha256'] != data.content_sha(replies['q0'])
    assert all(c['target_source'] == 'fixed_initial_greedy_ids' for c in contracts)
    assert inputs.legacy_contracts(contracts)[0]['input_sha256'] == contracts[0]['input_sha256']


@pytest.fixture
def proofs(config, tmp_path):
    root = tmp_path / 'parent'; parent_config = tmp_path / 'parent_config.json'
    names = ['STUDY_INPUT_LOCK.json', 'recovery_v1/RECOVERY_INPUT_LOCK.json',
        'analysis/pilot_seed42/screen/ANALYSIS.json', 'independent_review/PILOT_PREDICTION_REVIEW_RECOVERY_V1.json',
        'independent_review/ROOT_PILOT_ACCEPTANCE_RECOVERY_V1.json', 'independent_review/TEACHER_COMPLETION_REVIEW.json',
        'independent_review/TRAINING_COMPLETION_REVIEW.json', 'teacher_replies/seed42/MANIFEST.json',
        'teacher_replies/seed42/TEACHER_REPLIES_LOCK.json', 'independent_review/TEACHER_TARGET_INDEX.json',
        'prepared/seed42/PLAN.json', 'prepared/seed42/TRAIN_INPUTS_teacher_ce32.json',
        'prepared/seed42/TRAIN_CONTRACTS_teacher_ce32.json']
    files = [parent_config, *[root / name for name in names]]
    for path in files:
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text('{}')
    config.update(parent_output_root=str(root), parent_config_file=str(parent_config),
        parent_proof_files_sha256={str(path): data.sha_file(path) for path in files},
        seed42_input_reference={'parent_condition': 'teacher_ce32', 'new_training_condition': 'teacher_kl32',
            'plan_file': str(root / 'prepared/seed42/PLAN.json'),
            'train_inputs_file': str(root / 'prepared/seed42/TRAIN_INPUTS_teacher_ce32.json'),
            'train_contracts_file': str(root / 'prepared/seed42/TRAIN_CONTRACTS_teacher_ce32.json'),
            'teacher_target_index_file': str(root / 'independent_review/TEACHER_TARGET_INDEX.json'),
            'teacher_reply_manifest_file': str(root / 'teacher_replies/seed42/MANIFEST.json'),
            'new_generation_questions': 0, 'new_CPU_full_input_encodings': 0})
    return config


def test_exact14_proof_byte_binding(proofs):
    assert len(data.validate_parent_proofs(proofs)) == 14
    path = Path(proofs['seed42_input_reference']['teacher_reply_manifest_file']); path.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='Parent proof bytes changed'):
        data.validate_parent_proofs(proofs)


def test_missing_parent_proof_or_wrong_new_identity_rejected(proofs):
    missing = deepcopy(proofs); missing['parent_proof_files_sha256'].pop(next(iter(missing['parent_proof_files_sha256'])))
    with pytest.raises(ValueError, match='All14'):
        data.validate_parent_proofs(missing)
    proofs['seed42_input_reference']['new_training_condition'] = 'teacher_ce32'
    with pytest.raises(ValueError, match='input layout/reference identity'):
        data.validate_parent_proofs(proofs)


def test_seed42_cannot_launch_new_teacher_or_new_generation_preparation(config):
    with pytest.raises(ValueError, match='seed42'):
        rollout.run_teacher_shard('unused.json', 42, 0, 'unused', 'unused')
    with pytest.raises(ValueError, match='seed42'):
        data.prepare_generation_inputs(config, 42)


def test_old_condition_cannot_label_new_output(config):
    with pytest.raises(ValueError, match='teacher_kl32'):
        inputs.paths(config, 42, 'teacher_ce32')
