"""CPU mathematical, input, data gate and exact-resume contracts for object SFT."""
from copy import deepcopy
from contextlib import nullcontext
import json
import math
import random
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn.functional as F

from vlm_object_train.data import OBJECT_SUFFIX, PROMPT_SUFFIX, check_rows, read_admission, verify_image_bindings
from vlm_object_train.loss import completion_logps, normalized_qa_loss, per_question_ce, sequence_logps
from vlm_object_train.schedule import actual_indices, build_schedule, coverage, replay_stream
from vlm_object_train import train as trainer


class FakeProcessor:
    tokenizer = SimpleNamespace(pad_token_id=0)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        prefix = 'user-image-question<assistant>'
        if add_generation_prompt:
            return prefix
        return prefix + messages[-1]['content'][0]['text'] + '<end>'

    def __call__(self, text, images, return_tensors):
        answer = text[0].split('<assistant>')[1]
        ids = [1, 2, 3, 4]
        if answer:
            ids += [{'yes<end>': 5, 'no<end>': 6, 'book<end>': 7}[answer], 8]
        return {'input_ids': torch.tensor([ids]), 'attention_mask': torch.ones(1, len(ids), dtype=torch.long),
                'pixel_values': torch.ones(1, 3), 'image_grid_thw': torch.tensor([[1, 2, 2]])}


class LoRAHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Parameter(torch.randn(9, 5), requires_grad=False)
        self.lora_weight = torch.nn.Parameter(torch.randn(9, 5) * .01)

    def forward(self, values):
        return F.linear(values, self.base + self.lora_weight)


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(9, 5)
        self.embedding.requires_grad_(False)
        self.head = LoRAHead()
        self.rope_deltas = None
        self.register_buffer('runtime_buffer', torch.zeros(1))

    def get_output_embeddings(self):
        return self.head

    def forward(self, input_ids, attention_mask=None, pixel_values=None, image_grid_thw=None, use_cache=False):
        self.rope_deltas = torch.tensor([input_ids.shape[1]])
        self.runtime_buffer.add_(1)
        return SimpleNamespace(logits=self.head(self.embedding(input_ids)))

    def save_pretrained(self, directory, safe_serialization=True):
        directory.mkdir()
        torch.save(self.state_dict(), directory / 'tiny_adapter.pt')


def rows(count, prefix='object'):
    return [{'question_id': f'{prefix}{index}', 'image_id': f'im{index // 2}',
             'question': f'question {index}', 'target': 'yes' if index % 2 else 'no',
             'prompt_suffix': OBJECT_SUFFIX, 'image_path': '/opt/data/private/no-read.jpg',
             'pixel_sha256': 'not-read'} for index in range(count)]


def config():
    return {'schema_version': trainer.SCHEMA, 'object_prompt_suffix': OBJECT_SUFFIX,
            'replay_prompt_suffix': PROMPT_SUFFIX, 'training_microbatch': 4, 'training_accumulation': 4,
            'object_effective_batch': 16,
            'replay_per_update': 8, 'replay_lambda': 1., 'expected_trainable': 1843200,
            'checkpoint_every_updates': 25, 'epochs': 1, 'lr': 5e-6, 'preflight_lr': 1e-4,
            'weight_decay': 0., 'warmup_ratio': .1, 'gradient_clip_norm': 1.,
            'gradient_checkpointing': True, 'visual_token_budget': 1024, 'max_sequence_length': 2048,
            **{name: '/opt/data/private/test-only' for name in ('train_manifest', 'replay_manifest',
                'reference_config', 'model_root', 'initial_adapter', 'data_admission', 'dataset_protocol')}}


def fake_encode(processor, row, settings):
    answer = 5 if row['target'] == 'yes' else 6
    ids = torch.tensor([[1, 2, 3, 4, answer, 8]])
    batch = {'input_ids': ids, 'attention_mask': torch.ones_like(ids),
             'labels': torch.tensor([[-100, -100, -100, -100, answer, 8]]),
             'pixel_values': torch.ones(1, 3), 'image_grid_thw': torch.tensor([[1, 2, 2]])}
    return batch, {'question_id': str(row['question_id']), 'image_id': str(row['image_id']),
                   'input_sha256': trainer.batch_hash(batch), 'input_tokens': 6,
                   'sequence_length': 6, 'visual_tokens': 1, 'assistant_tokens': 2, 'truncated': False}


def variable_fake_encode(processor, row, settings):
    """Different prompt/answer lengths catch token-weighting and padding errors."""
    index = int(''.join(character for character in row['question_id'] if character.isdigit()))
    prefix = [1, 2, 3] + [4] * (index % 3)
    answer = [5, 8] if row['target'] == 'yes' else [6, 7, 8]
    ids = torch.tensor([prefix + answer])
    batch = {'input_ids': ids, 'attention_mask': torch.ones_like(ids),
             'labels': torch.tensor([[-100] * len(prefix) + answer]),
             'pixel_values': torch.ones(1, 3), 'image_grid_thw': torch.tensor([[1, 2, 2]])}
    return batch, {'question_id': str(row['question_id']), 'image_id': str(row['image_id']),
                   'input_sha256': trainer.batch_hash(batch), 'input_tokens': len(prefix) + len(answer),
                   'sequence_length': len(prefix) + len(answer), 'visual_tokens': 1,
                   'assistant_tokens': len(answer), 'truncated': False}


def test_real_encoder_assistant_mask_includes_end_token_and_rejects_truncation(tmp_path):
    image = tmp_path / 'image.jpg'
    Image.new('RGB', (28, 28), (255, 0, 0)).save(image)
    row = {**rows(1)[0], 'image_path': str(image), 'target': 'yes'}
    batch, contract = trainer.encode_qa(FakeProcessor(), row, config())
    assert batch['labels'].tolist() == [[-100, -100, -100, -100, 5, 8]]
    assert contract['assistant_tokens'] == 2 and contract['visual_tokens'] == 1
    logits = torch.zeros(1, 6, 9, requires_grad=True)
    values, counts = completion_logps(logits, batch['labels'])
    assert counts.tolist() == [2]
    assert float(per_question_ce(values, counts)) == pytest.approx(math.log(9))
    per_question_ce(values, counts).sum().backward()
    assert not bool(logits.grad[:, :3].any())
    assert bool(logits.grad[:, 3:5].abs().sum() > 0)
    with pytest.raises(ValueError, match='truncation prohibited'):
        trainer.encode_qa(FakeProcessor(), row, {**config(), 'max_sequence_length': 5})


@pytest.mark.parametrize('condition', ['object_only', 'object_replay'])
def test_short_final_object_block_and_independent_replay_mean_gradient(condition):
    objects, replays = rows(18), rows(13, 'replay')
    blocks = build_schedule(objects, replays)
    obj, rep = actual_indices(blocks[-1], condition)
    assert len(obj) == 2 and len(rep) == (8 if condition == 'object_replay' else 0)
    parameter = torch.tensor(.3, requires_grad=True)
    for source, selected in (('object', obj), ('replay', rep)):
        for index in selected:
            ce = (parameter - index / 10) ** 2
            normalized_qa_loss(ce, source, len(obj), len(rep)).backward()
    expected = torch.tensor(.3, requires_grad=True)
    objective = ((expected - torch.tensor(obj) / 10) ** 2).mean()
    if rep:
        objective = objective + ((expected - torch.tensor(rep) / 10) ** 2).mean()
    objective.backward()
    assert torch.allclose(parameter.grad, expected.grad)


def test_same_object_sequence_and_input_only_complete_pool_replay():
    objects, replays = rows(37), rows(11, 'replay')
    schedule = build_schedule(objects, replays)
    assert [len(block['object_indices']) for block in schedule] == [16, 16, 5]
    seen_a, seen_b = [], []
    for block in schedule:
        seen_a.extend(actual_indices(block, 'object_only')[0])
        seen_b.extend(actual_indices(block, 'object_replay')[0])
    assert seen_a == seen_b and sorted(seen_a) == list(range(37))
    assert schedule == build_schedule(objects, replays)
    first = replay_stream(replays, 24, 42)
    assert set(first[:11]) == set(range(11)) and set(first[11:22]) == set(range(11))
    changed_answers = [{**row, 'target': 'different', 'dev_error': 100} for row in replays]
    assert replay_stream(changed_answers, 24, 42) == first
    assert coverage(replays, first[:8])['complete_pool_consumed'] is False
    assert coverage(replays, first[:8])['question_coverage_fraction'] == pytest.approx(8 / 11)
    assert coverage(replays, first)['unique_questions_consumed'] == 11
    # Input-only order is independent of original container ordering as well.
    reversed_rows = list(reversed(replays))
    assert [replays[i]['question_id'] for i in first] == [reversed_rows[i]['question_id'] for i in replay_stream(reversed_rows, 24, 42)]


@pytest.mark.parametrize('condition', ['object_only', 'object_replay'])
@pytest.mark.parametrize('microbatch', [1, 4])
def test_one_discarded_real_step_restores_master_rng_frozen_and_model_cache(tmp_path, monkeypatch, condition, microbatch):
    monkeypatch.setattr(trainer, 'encode_qa', fake_encode)
    torch.manual_seed(29)
    random.seed(29)
    np.random.seed(29)
    model = TinyLM()
    before = deepcopy(model.state_dict())
    rng_before = trainer.rng_state()
    objects, replays = rows(16), rows(8, 'replay')
    block = build_schedule(objects, replays)[0]
    settings = {**config(), 'training_microbatch': microbatch, 'training_accumulation': 16 // microbatch}
    result = trainer.preflight(model, [p for p in model.parameters() if p.requires_grad], FakeProcessor(),
        objects, replays, block, settings, condition, tmp_path, torch.device('cpu'))
    assert result['status'] == 'passed'
    assert result['discarded_engineering_optimizer_updates'] == 1
    assert result['formal_optimizer_updates'] == 0
    assert result['record']['assistant_logp_max_difference_after_temporary_update'] > 0
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())
    assert model.rope_deltas is None
    assert all(p.grad is None for p in model.parameters())
    after = trainer.rng_state()
    assert rng_before['python'] == after['python']
    assert np.array_equal(rng_before['numpy'][1], after['numpy'][1])
    assert torch.equal(rng_before['torch'], after['torch'])


@pytest.mark.parametrize('condition', ['object_only', 'object_replay'])
@pytest.mark.parametrize('microbatch', [1, 4])
def test_atomic_resume_restores_exact_adam_master_rng_and_question_cursor(tmp_path, monkeypatch, condition, microbatch):
    monkeypatch.setattr(trainer, 'encode_qa', variable_fake_encode)
    torch.manual_seed(11)
    initial = TinyLM()
    objects, replays = rows(24), rows(11, 'replay')
    blocks = build_schedule(objects, replays)
    assert len(blocks[-1]['object_indices']) == 8
    settings = {**config(), 'training_microbatch': microbatch, 'training_accumulation': 16 // microbatch}
    schedule_sha = trainer.content_sha(blocks)

    def execute(model, optimizer, stats, start, end):
        parameters = [p for p in model.parameters() if p.requires_grad]
        for step in range(start, end):
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group['lr'] = trainer.learning_rate(step, len(blocks), settings)
            trainer.backward_update(model, FakeProcessor(), objects, replays, blocks[step], condition,
                                    settings, torch.device('cpu'), stats)
            torch.nn.utils.clip_grad_norm_(parameters, 1.)
            optimizer.step()

    uninterrupted = deepcopy(initial)
    opt_a = torch.optim.AdamW([p for p in uninterrupted.parameters() if p.requires_grad], lr=settings['lr'], weight_decay=0.)
    stats_a = trainer.initial_stats()
    execute(uninterrupted, opt_a, stats_a, 0, 2)
    partial = deepcopy(initial)
    opt_b = torch.optim.AdamW([p for p in partial.parameters() if p.requires_grad], lr=settings['lr'], weight_decay=0.)
    stats_b = trainer.initial_stats()
    execute(partial, opt_b, stats_b, 0, 1)
    trainer.save_checkpoint(tmp_path, partial, opt_b, 1, stats_b, 'lineage', schedule_sha)
    expected_draw = torch.rand(4)
    random_draw = random.random()
    numpy_draw = np.random.random(4)
    resumed = deepcopy(initial)
    opt_c = torch.optim.AdamW([p for p in resumed.parameters() if p.requires_grad], lr=9.)
    torch.rand(30)
    random.random()
    np.random.random(30)
    completed, stats_c = trainer.resume_checkpoint(tmp_path, resumed, opt_c, 'lineage', schedule_sha, 2)
    assert completed == 1 and trainer.trainable_hash(partial) == trainer.trainable_hash(resumed)
    assert torch.equal(expected_draw, torch.rand(4))
    assert random_draw == random.random()
    assert np.array_equal(numpy_draw, np.random.random(4))
    trainer.validate_exposure_cursor(stats_c, blocks, completed, objects, replays, condition)
    execute(resumed, opt_c, stats_c, completed, 2)
    assert trainer.trainable_hash(uninterrupted) == trainer.trainable_hash(resumed)
    assert stats_a == stats_c
    trainer.validate_exposure_cursor(stats_c, blocks, 2, objects, replays, condition)
    with pytest.raises(ValueError, match='binding changed'):
        trainer.resume_checkpoint(tmp_path, resumed, opt_c, 'different-UUID-lineage', schedule_sha, 2)
    with pytest.raises(ValueError, match='binding changed'):
        trainer.resume_checkpoint(tmp_path, resumed, opt_c, 'lineage', 'changed-schedule', 2)


@pytest.mark.parametrize('condition', ['object_only', 'object_replay'])
@pytest.mark.parametrize('actual_objects', [16, 8, 5])
def test_real_microbatch4_gradient_matches_centralized_per_question_objective(monkeypatch, condition, actual_objects):
    monkeypatch.setattr(trainer, 'encode_qa', variable_fake_encode)
    torch.manual_seed(67)
    accumulated = TinyLM()
    centralized = deepcopy(accumulated)
    objects, replays = rows(actual_objects), rows(8, 'replay')
    block = build_schedule(objects, replays)[0]
    stats = trainer.initial_stats()
    trainer.backward_update(accumulated, FakeProcessor(), objects, replays, block, condition,
                            config(), torch.device('cpu'), stats)
    objective = 0.
    for source, data, indices in (('object', objects, block['object_indices']),
                                  ('replay', replays, actual_indices(block, condition)[1])):
        if not indices:
            continue
        # Independent concentrated objective: sum logp / answer count per QA,
        # then average questions, with no microbatch normalization helper.
        samples = [variable_fake_encode(None, data[index], config())[0] for index in indices]
        batch = trainer.collate(samples, 0)
        logps, counts = sequence_logps(centralized, batch)
        objective = objective + (-logps / counts).mean()
    objective.backward()
    assert torch.allclose(accumulated.head.lora_weight.grad, centralized.head.lora_weight.grad, atol=2e-7, rtol=2e-6)
    assert stats['object_microbatches'] == math.ceil(actual_objects / 4)
    assert stats['replay_microbatches'] == (2 if condition == 'object_replay' else 0)
    assert stats['object_exposures'] == actual_objects
    assert stats['replay_exposures'] == (8 if condition == 'object_replay' else 0)
    assert stats['object_padding_tokens'] > 0
    assert [record['microbatch_size'] for record in stats['object_input_contracts'][-(actual_objects % 4 or 4):]] == [actual_objects % 4 or 4] * (actual_objects % 4 or 4)


def test_replay_keeps_actual_object_microbatches_inputs_and_rng_identical(monkeypatch):
    monkeypatch.setattr(trainer, 'encode_qa', variable_fake_encode)
    torch.manual_seed(97)
    original = TinyLM()
    objects, replays = rows(24), rows(11, 'replay')
    blocks = build_schedule(objects, replays)
    states = {}
    for condition in ('object_only', 'object_replay'):
        model = deepcopy(original)
        stats = trainer.initial_stats()
        rng_before = trainer.rng_hash(trainer.rng_state())
        for block in blocks:
            trainer.backward_update(model, FakeProcessor(), objects, replays, block, condition,
                                    config(), torch.device('cpu'), stats)
        assert trainer.rng_hash(trainer.rng_state()) == rng_before
        states[condition] = stats
    assert states['object_only']['object_input_contracts'] == states['object_replay']['object_input_contracts']
    assert states['object_only']['object_microbatches'] == states['object_replay']['object_microbatches'] == 6
    assert states['object_replay']['replay_microbatches'] == 4
    assert states['object_replay']['replay_rng_unchanged_checks'] == 2


@pytest.mark.parametrize('allocated_gib,reserved_gib', [(65.01, 65.01), (40., 65.01)])
def test_memory_cap_failure_is_saved_after_exact_restore_without_real_cuda(tmp_path, monkeypatch, allocated_gib, reserved_gib):
    """Simulated CUDA metrics only; all tensors/forwards remain on the CPU."""
    monkeypatch.setattr(trainer, 'encode_qa', fake_encode)
    monkeypatch.setattr(trainer, 'to_device', lambda batch, device: batch)
    monkeypatch.setattr(trainer, 'autocast', lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda device: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda device: allocated_gib * 2**30)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda device: reserved_gib * 2**30)
    torch.manual_seed(101)
    model = TinyLM()
    initial = deepcopy(model.state_dict())
    rng = trainer.rng_hash(trainer.rng_state())
    objects, replays = rows(16), rows(8, 'replay')
    with pytest.raises(RuntimeError, match='preflight_memory_cap_exceeded'):
        trainer.preflight(model, [p for p in model.parameters() if p.requires_grad], FakeProcessor(),
            objects, replays, build_schedule(objects, replays)[0], config(), 'object_replay',
            tmp_path, torch.device('cuda', 0))
    record = json.loads((tmp_path / 'PREFLIGHT.json').read_text())
    assert record['status'] == 'failed' and record['memory_cap_passed'] is False
    assert record['memory_cap_gib'] == 65.
    assert record['discarded_engineering_optimizer_updates'] == 1
    assert record['initial_parameters_restored_exactly'] is True
    assert 'preflight_memory_cap_exceeded' in record['error']
    assert trainer.rng_hash(trainer.rng_state()) == rng
    assert all(torch.equal(tensor, model.state_dict()[key]) for key, tensor in initial.items())
    assert model.rope_deltas is None


@pytest.mark.parametrize('field,value', [('status', 'rejected'), ('train_manifest_sha256', 'wrong'), ('dataset_protocol_sha', 'wrong')])
def test_data_admission_rejects_failed_or_modified_evidence(tmp_path, field, value):
    protocol = tmp_path / 'protocol.json'
    protocol.write_text('{"frozen":true}')
    admission = tmp_path / 'admission.json'
    valid = {'status': 'admitted', 'train_manifest_sha256': 'train-sha', 'dataset_protocol_sha': trainer.sha_file(protocol)}
    admission.write_text(json.dumps(valid))
    paths = {'data_admission': str(admission), 'dataset_protocol': str(protocol)}
    assert read_admission(paths, 'train-sha')['dataset_protocol_sha'] == trainer.sha_file(protocol)
    admission.write_text(json.dumps({**valid, field: value}))
    with pytest.raises(ValueError, match='binding missing or changed'):
        read_admission(paths, 'train-sha')


@pytest.mark.parametrize('field,value', [('target', 'uncertain'), ('target', 'YES'), ('prompt_suffix', ''),
                                         ('split', 'dev'), ('split', 'final')])
def test_training_scope_rejects_non_yes_no_wrong_prompts_and_heldouts(field, value):
    with pytest.raises(ValueError):
        check_rows([{**rows(1)[0], field: value}], OBJECT_SUFFIX, object_questions=True)


def test_image_binding_rechecks_actual_rgb_and_config_is_fixed(tmp_path):
    image = tmp_path / 'input.png'
    Image.new('RGB', (28, 28), (0, 255, 0)).save(image)
    rgb = Image.open(image).convert('RGB')
    import hashlib
    pixels = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
    row = {**rows(1)[0], 'image_path': str(image), 'pixel_sha256': pixels}
    assert verify_image_bindings([row])[str(image)]['pixel_sha256'] == pixels
    Image.new('RGB', (28, 28), (0, 0, 255)).save(image)
    with pytest.raises(ValueError, match='RGB pixels changed'):
        verify_image_bindings([row])
    assert trainer.validate_config(config()) == config()
    assert trainer.validate_config({**config(), 'training_microbatch': 1, 'training_accumulation': 16})['training_microbatch'] == 1
    for field, value in [('epochs', 2), ('lr', 1e-4), ('replay_per_update', 16), ('training_accumulation', 8),
                         ('object_effective_batch', 8), ('training_microbatch', True),
                         ('preflight_memory_cap_gib', 66.),
                         ('visual_token_budget', 256), ('replay_lambda', .5), ('data_admission', '')]:
        with pytest.raises(ValueError):
            trainer.validate_config({**config(), field: value})
    with pytest.raises(ValueError, match='outside persistent'):
        trainer.validate_config({**config(), 'model_root': '/tmp/forbidden'})
