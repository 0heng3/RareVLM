"""CPU checks for scientific contracts that can fail independently of a GPU run."""
from copy import deepcopy
import json
import math
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vlm_posttrain.loss import StaticReference, completion_logps, normalize_pair_loss, pair_objective, sequence_logps
from vlm_posttrain.train import content_sha, preflight, rng_state, update_blocks, validate_checkpoint_lineage


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 5)
        self.head = torch.nn.Linear(5, 8, bias=False)

    def get_output_embeddings(self):
        return self.head

    def forward(self, input_ids, attention_mask=None, pixel_values=None, image_grid_thw=None, use_cache=False):
        return SimpleNamespace(logits=self.head(self.embedding(input_ids)))


def tiny_pair():
    ids = torch.tensor([[0, 0, 5, 6, 1, 2], [0, 5, 6, 3, 4, 7]])
    labels = torch.full_like(ids, -100)
    labels[0, -2:] = ids[0, -2:]
    labels[1, -3:] = ids[1, -3:]
    return {'input_ids': ids, 'attention_mask': (ids != 0).long(), 'labels': labels,
            'pixel_values': torch.ones(2, 3), 'image_grid_thw': torch.tensor([[1, 2, 2], [1, 2, 2]])}


def test_sum_logp_masks_padding_and_preserves_length_effect():
    logits = torch.zeros(2, 6, 8)
    labels = tiny_pair()['labels']
    values, counts = completion_logps(logits, labels)
    assert counts.tolist() == [2, 3]
    assert values.tolist() == pytest.approx([-2 * math.log(8), -3 * math.log(8)])
    changed = logits.clone()
    changed[:, :2, :] = torch.arange(8) * 100.
    assert torch.equal(completion_logps(changed, labels)[0], values)
    assert float(pair_objective('sft', values[:1], counts[:1])) == pytest.approx(math.log(8))
    # Equal per-token probabilities still have distinct summed DPO logps.
    reference = values.clone()
    reference[0] -= 1
    expected = -math.log(1 / (1 + math.exp(-.1)))
    assert float(pair_objective('dpo', values, counts, reference)) == pytest.approx(expected)


def test_reference_survives_real_actor_update_and_return_value_mutation():
    torch.manual_seed(13)
    model = TinyLM()
    batch = tiny_pair()
    values, counts = sequence_logps(model, batch)
    cache = StaticReference([{'question_id': 'q', 'input_sha256': 'input',
                             'logps': values.detach().tolist(), 'counts': counts.tolist()}])
    original, _ = cache.lookup('q', 'input')
    initial_margin = float(values[0] - values[1])
    loss = pair_objective('dpo', values, counts, original)
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    loss.backward()
    optimizer.step()
    updated, _ = sequence_logps(model, batch)
    assert float(updated[0] - updated[1]) > initial_margin
    assert torch.equal(cache.lookup('q', 'input')[0], original)
    original.add_(500)
    assert not torch.equal(cache.lookup('q', 'input')[0], original)
    with pytest.raises(ValueError, match='input changed'):
        cache.lookup('q', 'other-input')


@pytest.mark.parametrize('condition', ['sft', 'dpo'])
def test_exposure_order_and_short_update_normalization(condition):
    blocks = update_blocks(18, 42, 2, 16)
    assert [len(b['indices']) for b in blocks] == [16, 2, 16, 2]
    for epoch in range(2):
        seen = [i for b in blocks if b['epoch'] == epoch for i in b['indices']]
        assert sorted(seen) == list(range(18))
    assert blocks == update_blocks(18, 42, 2, 16)
    # Compare actual per-pair backward accumulation with an independent vector
    # mean on a short final update. Dividing by 16 would fail this check.
    actual = torch.tensor(2., requires_grad=True)
    for index in blocks[1]['indices']:
        chosen = -(actual - index) ** 2
        if condition == 'sft':
            loss = pair_objective(condition, chosen[None], torch.tensor([2]))
        else:
            rejected = -(actual + index) ** 2
            loss = pair_objective(condition, torch.stack([chosen, rejected]), torch.tensor([2, 3]),
                                  torch.tensor([-1., -2.]))
        normalize_pair_loss(loss, len(blocks[1]['indices'])).backward()
    expected = torch.tensor(2., requires_grad=True)
    indices = torch.tensor(blocks[1]['indices'])
    if condition == 'sft':
        vector_mean = ((expected - indices) ** 2 / 2).mean()
    else:
        margins = -(expected - indices) ** 2 + (expected + indices) ** 2 - 1
        vector_mean = -torch.nn.functional.logsigmoid(.1 * margins).mean()
    vector_mean.backward()
    assert torch.allclose(actual.grad, expected.grad)


@pytest.mark.parametrize('changed_field', ['condition', 'seed', 'source', 'manifest', 'initial_adapter', 'hardware'])
def test_checkpoint_rejects_lineage_mixing(changed_field):
    lineage = {'condition': 'dpo', 'seed': 42, 'source': 'source-v1', 'manifest': 'human-v1',
               'initial_adapter': 'sft42', 'hardware': 'GPU-1'}
    saved = {'lineage_sha256': content_sha(lineage), 'completed_updates': 25, 'accumulated_pairs': 0}
    assert validate_checkpoint_lineage(saved, content_sha(lineage), 100) == 25
    changed = {**lineage, changed_field: 'changed'}
    with pytest.raises(ValueError, match='lineage differs'):
        validate_checkpoint_lineage(saved, content_sha(changed), 100)
    with pytest.raises(ValueError, match='optimizer boundary'):
        validate_checkpoint_lineage({**saved, 'accumulated_pairs': 1}, content_sha(lineage), 100)


@pytest.mark.parametrize('condition', ['sft', 'dpo'])
def test_real_preflight_restores_initial_tensors_and_rng(tmp_path, monkeypatch, condition):
    import vlm_posttrain.train as trainer
    torch.manual_seed(29)
    random.seed(29)
    np.random.seed(29)
    model = TinyLM()
    pair = tiny_pair()
    samples = [{key: value[i:i + 1].clone() for key, value in pair.items()} for i in range(2)]
    processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
    monkeypatch.setattr(trainer, 'encode_pair', lambda *args: (samples, pair, 'fixed-input'))
    with torch.no_grad():
        values, counts = sequence_logps(model, pair)
    cache = StaticReference([{'question_id': 'q', 'input_sha256': 'fixed-input',
                             'logps': values.tolist(), 'counts': counts.tolist()}])
    (tmp_path / 'reference_cache.jsonl').write_text('immutable initial reference\n')
    initial = deepcopy(model.state_dict())
    before_rng = rng_state()
    result = preflight(model, list(model.parameters()), processor, [{'question_id': 'q'}], [0],
                       {'lr': 5e-4, 'weight_decay': 0., 'beta': .1, 'gradient_clip_norm': 1.},
                       condition, cache, tmp_path, torch.device('cpu'))
    after_rng = rng_state()
    assert result['status'] == 'passed'
    assert result['discarded_engineering_optimizer_updates'] == 1
    assert result['formal_optimizer_updates'] == 0
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in initial.items())
    assert before_rng['python'] == after_rng['python']
    assert np.array_equal(before_rng['numpy'][1], after_rng['numpy'][1])
    assert torch.equal(before_rng['torch'], after_rng['torch'])
    assert all(parameter.grad is None for parameter in model.parameters())
    assert json.loads((tmp_path / 'PREFLIGHT.json').read_text())['initial_parameters_restored_exactly']
