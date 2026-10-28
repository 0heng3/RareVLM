"""Meaningful CPU-only fixed-teacher KL mathematical and persistence contracts."""
from copy import deepcopy
from contextlib import nullcontext
import json
import math
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from vlm_object_kl import cache as caching, inputs, train as trainer
from vlm_object_kl.loss import assistant_logprobs, per_question_hard_ce, per_question_kl
from vlm_object_train import train as legacy
from vlm_object_train.schedule import build_schedule
from test_object_presence_training import FakeProcessor, TinyLM, config as oldconfig, rows, variable_fake_encode


def config():
    return {**oldconfig(), 'schema_version': trainer.SCHEMA, 'preflight_memory_cap_gib': 65.,
        'kl_temperature': 1., 'kl_direction': 'initial_to_actor', 'teacher_logprob_dtype': 'float32',
        'teacher_full_vocabulary': True, 'initial_kl_abs_tolerance': 1e-5,
        'teacher_cache_disk_reserve_bytes': 1073741824}


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setattr(inputs, 'encode_qa', variable_fake_encode)
    torch.manual_seed(91)
    random.seed(91)
    np.random.seed(91)
    model = TinyLM()
    model.config = SimpleNamespace(vocab_size=9)
    return model, rows(24), rows(11, 'replay'), config()


def make_cache(tmp_path, setup, count=2):
    model, objects, replays, settings = setup
    blocks = build_schedule(objects, replays)[:count]
    binding = {'initial_adapter_tensor_sha256': trainer.trainable_hash(model),
               'frozen_parameter_sha256_before': trainer.frozen_hash(model), 'config': settings,
               'training_microbatch_questions': 4, 'replay_questions_per_update': 8,
               'planned_updates': len(blocks), 'planned_replay_exposures': len(blocks) * 8,
               'replay_order_sha256': trainer.content_sha([[replays[i]['question_id'] for i in b['replay_indices']] for b in blocks]),
               'replay_microbatch_order_sha256': trainer.content_sha([[[replays[i]['question_id'] for i in b['replay_indices'][s:s+4]] for s in (0, 4)] for b in blocks]),
               'source': trainer.source_hashes(), 'gpu_uuid': 'CPU-contract-only'}
    directory = tmp_path / 'cache'
    manifest = caching.build_teacher_cache(model, FakeProcessor(), replays, blocks, settings,
        directory, binding, torch.device('cpu'))
    teacher = caching.TeacherCache(directory, binding, trainer.sha_file(directory / 'MANIFEST.json'))
    return blocks, teacher, manifest


def test_kl_direction_full_vocabulary_sum_and_teacher_detach():
    teacher_logits = torch.tensor([[2., -.5, 1.], [0., 1., -2.]], requires_grad=True)
    actor_logits = torch.tensor([[0., 1., -1.], [2., 0., 1.]], requires_grad=True)
    teacher = teacher_logits.log_softmax(-1)
    actor = actor_logits.log_softmax(-1)
    result = per_question_kl(actor, teacher, torch.tensor([1, 1]))
    expected = (teacher.detach().exp() * (teacher.detach() - actor)).sum(-1)
    reverse = (actor.exp() * (actor - teacher.detach())).sum(-1)
    assert torch.allclose(result, expected)
    assert not torch.allclose(result, reverse)
    assert not torch.allclose(result, expected / actor.shape[-1])
    result.mean().backward()
    assert actor_logits.grad.abs().sum() > 0
    assert teacher_logits.grad is None


def test_causal_shift_mask_prompt_padding_and_per_question_variable_length():
    torch.manual_seed(7)
    model = TinyLM()
    row_a = {'input_ids': torch.tensor([[1, 2, 3, 5, 8]]),
        'labels': torch.tensor([[-100, -100, -100, 5, 8]]),
        'attention_mask': torch.ones(1, 5, dtype=torch.long),
        'pixel_values': torch.ones(1, 3), 'image_grid_thw': torch.tensor([[1, 2, 2]])}
    row_b = {'input_ids': torch.tensor([[1, 2, 3, 4, 6, 7, 8]]),
        'labels': torch.tensor([[-100, -100, -100, -100, 6, 7, 8]]),
        'attention_mask': torch.ones(1, 7, dtype=torch.long),
        'pixel_values': torch.ones(1, 3), 'image_grid_thw': torch.tensor([[1, 2, 2]])}
    batch = legacy.collate([row_a, row_b], 0)
    logps, counts, targets = assistant_logprobs(model, batch)
    expected_logits = model.head(model.embedding(batch['input_ids']))[:, :-1].float()
    valid = batch['labels'][:, 1:] != -100
    expected = expected_logits[valid].log_softmax(-1)
    assert counts.tolist() == [2, 3]
    assert targets.tolist() == [5, 8, 6, 7, 8]
    assert torch.allclose(logps, expected)
    ce = per_question_hard_ce(logps, targets, counts)
    assert torch.allclose(ce, torch.stack([-expected[:2].gather(1, targets[:2, None]).mean(),
                                        -expected[2:].gather(1, targets[2:, None]).mean()]))
    actor = (logps.detach() + torch.randn_like(logps) * .1).log_softmax(-1)
    kl = per_question_kl(actor, logps, counts)
    tokens = (logps.detach().exp() * (logps.detach() - actor)).sum(-1)
    assert torch.allclose(kl, torch.stack([tokens[:2].mean(), tokens[2:].mean()]))
    assert kl.mean() != pytest.approx(float(tokens.mean()))


def test_cache_full_precision_restores_initial_rng_buffers_and_freezes_manifest(tmp_path, setup):
    model, *_ = setup
    state, rng = deepcopy(model.state_dict()), trainer.rng_hash(trainer.rng_state())
    blocks, teacher, manifest = make_cache(tmp_path, setup)
    assert manifest['status'] == 'frozen'
    assert manifest['teacher_forward_questions'] == 16
    assert manifest['full_vocabulary'] and manifest['logprob_dtype'] == 'float32'
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in state.items())
    assert model.rope_deltas is None and trainer.rng_hash(trainer.rng_state()) == rng
    before = trainer.file_hashes(teacher.directory)
    binding = json.loads((teacher.directory / 'INPUT_LOCK.json').read_text())
    again = caching.build_teacher_cache(model, FakeProcessor(), setup[2], blocks, setup[3],
        teacher.directory, binding, torch.device('cpu'))
    assert manifest == again and before == trainer.file_hashes(teacher.directory)
    first = next(iter(manifest['entries']))
    payload = torch.load(teacher.directory / f'{first}.pt', weights_only=False)
    assert payload['logprobs'].dtype == torch.float32
    assert payload['logprobs'].shape[1] == 9
    assert not payload['logprobs'].requires_grad


def test_partial_atomic_cache_resume_only_missing_batches(tmp_path, setup, monkeypatch):
    blocks, teacher, manifest = make_cache(tmp_path, setup)
    keys = list(manifest['entries'])
    (teacher.directory / 'MANIFEST.json').unlink()
    missing = teacher.directory / f'{keys[-1]}.pt'
    missing.unlink()
    preserved = {key: trainer.sha_file(teacher.directory / f'{key}.pt') for key in keys[:-1]}
    counter = {'calls': 0}
    original = caching.assistant_logprobs
    def tracked(*args, **kwargs):
        counter['calls'] += 1
        return original(*args, **kwargs)
    monkeypatch.setattr(caching, 'assistant_logprobs', tracked)
    binding = json.loads((teacher.directory / 'INPUT_LOCK.json').read_text())
    caching.build_teacher_cache(setup[0], FakeProcessor(), setup[2], blocks, setup[3],
        teacher.directory, binding, torch.device('cpu'))
    assert counter['calls'] == 1
    assert all(trainer.sha_file(teacher.directory / f'{key}.pt') == value for key, value in preserved.items())


@pytest.mark.parametrize('tamper', ['file', 'identity', 'dtype', 'tensor'])
def test_cache_rejects_changed_identity_precision_or_bytes(tmp_path, setup, tamper):
    blocks, teacher, manifest = make_cache(tmp_path, setup, count=1)
    if tamper == 'identity':
        with pytest.raises(ValueError, match='binding changed'):
            caching.TeacherCache(teacher.directory, {'different': 'initial'}, teacher.manifest_sha256)
        return
    key = next(iter(manifest['entries']))
    path = teacher.directory / f'{key}.pt'
    if tamper == 'file':
        with path.open('ab') as stream:
            stream.write(b'changed')
        with pytest.raises(ValueError, match='file changed'):
            teacher.verify()
        return
    payload = torch.load(path, weights_only=False)
    if tamper == 'dtype':
        payload['logprobs'] = payload['logprobs'].bfloat16()
        with pytest.raises(ValueError, match='FP32'):
            caching.validate_payload(payload, payload['identity'])
    else:
        payload['logprobs'][0, 0] += .00001
        with pytest.raises(ValueError, match='SHA changed'):
            caching.validate_payload(payload, payload['identity'])


def test_cache_rejects_padded_input_change_and_pool_order_change(tmp_path, setup):
    blocks, teacher, _ = make_cache(tmp_path, setup, count=1)
    batch, contracts, _ = inputs.encode_microbatch(FakeProcessor(), setup[2], blocks[0]['replay_indices'][:4], setup[3])
    modified = deepcopy(contracts)
    modified[0]['microbatch_input_sha256'] = 'changed-pixel-grid-padding-or-labels'
    with pytest.raises(ValueError, match='actual padded input'):
        teacher.load(0, 0, batch, modified, torch.device('cpu'))
    with pytest.raises(ValueError, match='actual padded input'):
        teacher.load(0, 0, batch, list(reversed(contracts)), torch.device('cpu'))


def test_existing_cache_cannot_be_reused_as_a_current_actor_teacher(tmp_path, setup):
    blocks, teacher, _ = make_cache(tmp_path, setup, count=1)
    binding = json.loads((teacher.directory / 'INPUT_LOCK.json').read_text())
    with torch.no_grad():
        setup[0].head.lora_weight.add_(.1)
    with pytest.raises(ValueError, match='exact fixed initial'):
        caching.build_teacher_cache(setup[0], FakeProcessor(), setup[2], blocks, setup[3],
            teacher.directory, binding, torch.device('cpu'))


@pytest.mark.parametrize('actual_objects', [16, 8, 5])
def test_micro4_object_and_replay_kl_gradient_matches_per_question_objective(tmp_path, setup, actual_objects):
    model, _, replays, settings = setup
    objects = rows(actual_objects)
    local = (model, objects, replays, settings)
    blocks, teacher, _ = make_cache(tmp_path, local, count=1)
    # Nonzero actor drift makes both the direction and KL backward observable.
    with torch.no_grad():
        model.head.lora_weight.add_(torch.randn_like(model.head.lora_weight) * .03)
    expected = deepcopy(model)
    stats = trainer.initial_stats()
    trainer.backward_update(model, FakeProcessor(), objects, replays, blocks[0], 'object_kl', settings,
        torch.device('cpu'), stats, teacher_cache=teacher, update=0)
    objective = 0.
    for source, data, indices in (('object', objects, blocks[0]['object_indices']),
                                  ('replay', replays, blocks[0]['replay_indices'])):
        for index in indices:
            batch, _ = variable_fake_encode(None, data[index], settings)
            logp, counts, targets = assistant_logprobs(expected, batch)
            if source == 'object':
                objective += -logp.gather(1, targets[:, None]).mean() / len(indices)
            else:
                # Teacher is the original model from setup's saved cache. Fetch
                # only matching QA rows from the exact padded cached microbatch.
                pos = indices.index(index)
                begin = pos // 4 * 4
                whole, contracts, _ = inputs.encode_microbatch(FakeProcessor(), data, indices[begin:begin+4], settings)
                cached, cached_counts = teacher.load(0, pos // 4, whole, contracts, torch.device('cpu'))
                offsets = torch.cat([torch.zeros(1, dtype=torch.int64), cached_counts.cumsum(0)]).tolist()
                teacher_rows = cached[offsets[pos % 4]:offsets[pos % 4 + 1]]
                objective += (teacher_rows.exp() * (teacher_rows - logp)).sum(-1).mean() / len(indices)
    objective.backward()
    assert torch.allclose(model.head.lora_weight.grad, expected.head.lora_weight.grad, atol=3e-7, rtol=3e-6)
    assert stats['object_exposures'] == actual_objects and stats['replay_exposures'] == 8
    assert stats['object_microbatches'] == math.ceil(actual_objects / 4)
    assert stats['replay_microbatches'] == 2 and stats['replay_kl_sum'] > 0


def test_preflight_restores_all_and_cache_remains_fixed(tmp_path, setup):
    blocks, teacher, _ = make_cache(tmp_path, setup, count=1)
    model, objects, replays, settings = setup
    initial, rng = deepcopy(model.state_dict()), trainer.rng_hash(trainer.rng_state())
    cache_hashes = trainer.file_hashes(teacher.directory)
    result = trainer.preflight(model, [p for p in model.parameters() if p.requires_grad], FakeProcessor(),
        objects, replays, blocks[0], settings, 'object_kl', tmp_path, torch.device('cpu'), teacher_cache=teacher)
    assert result['status'] == 'passed' and result['discarded_engineering_optimizer_updates'] == 1
    assert max(abs(v) for v in result['record']['initial_replay_kl_per_question']) <= 1e-5
    assert result['record']['after_temporary_update_mean_replay_kl'] > 0
    assert result['record']['teacher_has_grad'] is False
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in initial.items())
    assert model.rope_deltas is None and trainer.rng_hash(trainer.rng_state()) == rng
    assert all(p.grad is None for p in model.parameters())
    assert trainer.file_hashes(teacher.directory) == cache_hashes


@pytest.mark.parametrize('metric', ['allocated', 'reserved'])
def test_preflight_memory_cap_refuses_and_restores(tmp_path, setup, monkeypatch, metric):
    blocks, teacher, _ = make_cache(tmp_path, setup, count=1)
    model, objects, replays, settings = setup
    initial = deepcopy(model.state_dict())
    monkeypatch.setattr(trainer, 'to_device', lambda batch, device: batch)
    monkeypatch.setattr(trainer, 'autocast', lambda device: nullcontext())
    monkeypatch.setattr(caching.TeacherCache, 'load', lambda self, update, microbatch, batch, contracts, device:
        (torch.load(self.directory / f'update_{update:06d}_replay_{microbatch:03d}.pt', weights_only=False)['logprobs'],
         torch.tensor([item['assistant_tokens'] for item in contracts])))
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda device: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda device: (65.1 if metric == 'allocated' else 40.) * 2**30)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda device: (65.1 if metric == 'reserved' else 40.) * 2**30)
    with pytest.raises(RuntimeError, match='preflight_memory_cap_exceeded'):
        trainer.preflight(model, [p for p in model.parameters() if p.requires_grad], FakeProcessor(),
            objects, replays, blocks[0], settings, 'object_kl', tmp_path, torch.device('cuda', 0), teacher_cache=teacher)
    result = json.loads((tmp_path / 'PREFLIGHT.json').read_text())
    assert result['status'] == 'failed' and not result['memory_cap_passed']
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in initial.items())


def test_same_old_schedule_and_actual_encoded_object_replay_inputs(tmp_path, setup, monkeypatch):
    blocks, teacher, _ = make_cache(tmp_path, setup)
    model, objects, replays, settings = setup
    monkeypatch.setattr(legacy, 'encode_qa', variable_fake_encode)
    old = deepcopy(model)
    oldstats, newstats = legacy.initial_stats(), trainer.initial_stats()
    for step, block in enumerate(blocks):
        legacy.backward_update(old, FakeProcessor(), objects, replays, block, 'object_replay', settings,
            torch.device('cpu'), oldstats)
        trainer.backward_update(model, FakeProcessor(), objects, replays, block, 'object_kl', settings,
            torch.device('cpu'), newstats, teacher_cache=teacher, update=step)
    for source in ('object', 'replay'):
        assert oldstats[f'{source}_input_contracts'] == newstats[f'{source}_input_contracts']
    assert newstats['replay_rng_unchanged_checks'] == 2
    assert build_schedule(objects, replays) == blocks


def test_continuous_and_atomic_resume_adam_are_identical(tmp_path, setup):
    blocks, teacher, _ = make_cache(tmp_path, setup)
    initial, objects, replays, settings = setup
    def execute(model, optimizer, stats, begin, end):
        for step in range(begin, end):
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group['lr'] = trainer.learning_rate(step, len(blocks), settings)
            trainer.backward_update(model, FakeProcessor(), objects, replays, blocks[step], 'object_kl', settings,
                torch.device('cpu'), stats, teacher_cache=teacher, update=step)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
            optimizer.step()
    def optimizer(model):
        return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=settings['lr'], weight_decay=0.)
    full, partial = deepcopy(initial), deepcopy(initial)
    opt_a, opt_b = optimizer(full), optimizer(partial)
    stat_a, stat_b = trainer.initial_stats(), trainer.initial_stats()
    execute(full, opt_a, stat_a, 0, 2)
    execute(partial, opt_b, stat_b, 0, 1)
    trainer.save_checkpoint(tmp_path, partial, opt_b, 1, stat_b, 'locked-cache-source-UUID', 'schedule')
    resumed, opt_c = deepcopy(initial), None
    opt_c = optimizer(resumed)
    completed, stat_c = trainer.resume_checkpoint(tmp_path, resumed, opt_c, 'locked-cache-source-UUID', 'schedule', 2)
    trainer.validate_exposure_cursor(stat_c, blocks, completed, objects, replays, 'object_kl')
    execute(resumed, opt_c, stat_c, completed, 2)
    assert trainer.trainable_hash(full) == trainer.trainable_hash(resumed)
    assert stat_a == stat_c
    assert opt_a.state_dict()['param_groups'] == opt_c.state_dict()['param_groups']
    for key in opt_a.state_dict()['state']:
        for field, value in opt_a.state_dict()['state'][key].items():
            assert torch.equal(value, opt_c.state_dict()['state'][key][field])
    with pytest.raises(ValueError, match='binding changed'):
        trainer.resume_checkpoint(tmp_path, resumed, opt_c, 'changed-cache-UUID', 'schedule', 2)


@pytest.mark.parametrize('field,value', [('kl_temperature', 2.), ('kl_direction', 'actor_to_initial'),
    ('teacher_logprob_dtype', 'bfloat16'), ('teacher_full_vocabulary', False), ('training_microbatch', 1),
    ('initial_kl_abs_tolerance', .1), ('teacher_cache_disk_reserve_bytes', 0), ('preflight_memory_cap_gib', 66.)])
def test_fixed_contract_rejects_other_objectives_or_silent_runtime_changes(field, value):
    assert trainer.validate_config(config()) == config()
    with pytest.raises(ValueError):
        trainer.validate_config({**config(), field: value})
