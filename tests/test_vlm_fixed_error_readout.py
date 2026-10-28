"""CPU mathematics for prompt-only fixed-checkpoint readouts, without loaders."""

import inspect
from types import SimpleNamespace

import pytest
import torch

from vlm_fixed_error_readout.readout import classify_pair, forward_first_readout


YES, NO, YES_ALIAS, NO_ALIAS, VOCAB = 2, 3, 5, 7, 11


class TinyPromptActor(torch.nn.Module):
    def __init__(self, fail=False, dtype=torch.float32):
        super().__init__()
        self.embedding = torch.nn.Embedding(VOCAB, 6, dtype=dtype)
        self.head = torch.nn.Linear(6, VOCAB, bias=True, dtype=dtype)
        self.forward_calls = 0
        self.fail = fail
        self.seen_inputs = []
        self.eval()

    def get_output_embeddings(self):
        return self.head

    def forward(self, input_ids, attention_mask, use_cache=False):
        assert use_cache is False
        self.forward_calls += 1
        self.seen_inputs.append(input_ids.detach().clone())
        logits = self.head(self.embedding(input_ids))
        if self.fail:
            raise RuntimeError('deliberate failure after head')
        return SimpleNamespace(logits=logits)


def batch():
    return {'input_ids': torch.tensor([[0, 8, 9], [4, 6, 10]]),
            'attention_mask': torch.tensor([[0, 1, 1], [1, 1, 1]])}


def read(model, inputs=None, yes_ids=(YES, YES_ALIAS), no_ids=(NO, NO_ALIAS),
         *, slice_head=False):
    return forward_first_readout(model, batch() if inputs is None else inputs,
                                 yes_ids, no_ids, YES, NO, slice_head=slice_head)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('slice_head', [False, True])
def test_one_prompt_forward_matches_dense_last_position_and_fp32_reductions(dtype, slice_head):
    torch.manual_seed(11)
    model = TinyPromptActor(dtype=dtype)
    inputs = batch()
    with torch.no_grad():
        dense = model(**inputs, use_cache=False).logits[:, -1, :].float()
    calls_before = model.forward_calls
    actual_logits = []
    capture = model.head.register_forward_hook(
        lambda module, args, output: actual_logits.append(output.detach().float().clone()))
    try:
        rows = read(model, inputs, slice_head=slice_head)
    finally:
        capture.remove()
    assert model.forward_calls == calls_before + 1
    assert not model.head._forward_pre_hooks
    # BF16 GEMM rounding can depend on the flattened sequence dimension. Check
    # dtype-appropriate dense equivalence, then strictly check the actual FP32
    # reductions instead of falsely promising bit-identical BF16 head outputs.
    actual = actual_logits[0][:, -1, :]
    assert actual_logits[0].shape[1] == (1 if slice_head else inputs['input_ids'].shape[1])
    tolerance = (.01 if dtype == torch.bfloat16 else 2e-7) if slice_head else 0
    torch.testing.assert_close(actual, dense, rtol=0, atol=tolerance)
    for index, row in enumerate(rows):
        x = actual[index]
        yes_mass = torch.logsumexp(x[[YES, YES_ALIAS]], dim=0)
        no_mass = torch.logsumexp(x[[NO, NO_ALIAS]], dim=0)
        expected = {
            'canonical_margin': x[YES] - x[NO],
            'alias_margin': yes_mass - no_mass,
            'conditional_yes_probability': (yes_mass - no_mass).sigmoid(),
            'canonical_yes_log_probability': x[YES] - x.logsumexp(0),
            'canonical_no_log_probability': x[NO] - x.logsumexp(0),
            'yes_alias_log_probability': yes_mass - x.logsumexp(0),
            'no_alias_log_probability': no_mass - x.logsumexp(0),
            'top1_logit': x.max(),
        }
        for key, value in expected.items():
            assert row[key] == pytest.approx(float(value), abs=2e-7)
        assert row['top1_token_id'] == int(x.argmax())
    assert next(model.parameters()).dtype == dtype
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize('yes_ids,no_ids', [((YES, YES), (NO,)), ((YES,), (NO, NO)),
                                          ((YES, NO), (NO,)), ((), (NO,))])
def test_alias_ids_require_unique_nonempty_disjoint_sets(yes_ids, no_ids):
    model = TinyPromptActor()
    with pytest.raises(ValueError):
        read(model, yes_ids=yes_ids, no_ids=no_ids)
    assert model.forward_calls == 0


def test_alias_sets_must_include_canonical_ids_and_fit_vocabulary():
    model = TinyPromptActor()
    with pytest.raises(ValueError, match='Canonical'):
        read(model, yes_ids=(YES_ALIAS,))
    with pytest.raises(ValueError, match='outside'):
        read(model, yes_ids=(YES, VOCAB))
    assert not model.head._forward_pre_hooks


def test_common_logit_offset_does_not_change_readout_probabilities_or_margins():
    torch.manual_seed(12)
    model = TinyPromptActor()
    original = read(model)
    with torch.no_grad():
        model.head.bias.add_(32.0)
    shifted = read(model)
    for before, after in zip(original, shifted):
        for key in before:
            if key != 'top1_logit':
                assert after[key] == pytest.approx(before[key], abs=1e-5)
        assert after['top1_logit'] == pytest.approx(before['top1_logit'] + 32, abs=1e-5)


@pytest.mark.parametrize('key', ['labels', 'target', 'answers', 'reference'])
def test_reference_fields_never_reach_model(key):
    model = TinyPromptActor()
    inputs = {**batch(), key: torch.tensor([YES, NO])}
    with pytest.raises(ValueError, match='reference fields'):
        read(model, inputs)
    assert model.forward_calls == 0
    parameters = inspect.signature(forward_first_readout).parameters
    assert 'target' not in parameters and 'labels' not in parameters


def test_hook_removed_on_failure_without_removing_existing_hooks():
    model = TinyPromptActor(fail=True)
    existing_calls = []
    existing = model.head.register_forward_pre_hook(lambda module, args: existing_calls.append(1))
    try:
        with pytest.raises(RuntimeError, match='deliberate'):
            read(model)
        assert len(model.head._forward_pre_hooks) == 1
        assert existing_calls == [1]
    finally:
        existing.remove()
    assert not model.head._forward_pre_hooks


def test_prompt_position_is_not_an_appended_teacher_forced_answer_position():
    torch.manual_seed(13)
    model = TinyPromptActor()
    inputs = batch()
    rows = read(model, inputs)
    assert torch.equal(model.seen_inputs[-1], inputs['input_ids'])
    appended = {key: torch.cat([value, torch.tensor([[YES], [NO]])
                              if key == 'input_ids' else torch.ones(2, 1, dtype=value.dtype)], 1)
                for key, value in inputs.items()}
    with torch.no_grad():
        full = model(**appended).logits.float()
    for index, row in enumerate(rows):
        first = full[index, -2, YES] - full[index, -2, NO]
        teacher_forced_next = full[index, -1, YES] - full[index, -1, NO]
        assert row['canonical_margin'] == pytest.approx(float(first), abs=1e-7)
        assert abs(row['canonical_margin'] - float(teacher_forced_next)) > 1e-3


@pytest.mark.parametrize('mask', [[[1, 1, 0], [1, 1, 1]],
                                 [[1, 0, 1], [1, 1, 1]],
                                 [[0, 0, 0], [1, 1, 1]]])
def test_rejects_right_padding_holes_and_empty_prompts(mask):
    model = TinyPromptActor()
    with pytest.raises(ValueError, match='left padding'):
        read(model, {**batch(), 'attention_mask': torch.tensor(mask)})
    assert model.forward_calls == 0


def test_requires_eval_without_silently_changing_mode():
    model = TinyPromptActor().train()
    with pytest.raises(ValueError, match='eval'):
        read(model)
    assert model.training and model.forward_calls == 0


def test_ordering_and_zero_boundary_are_different_constraints():
    original = classify_pair(7.125, .25)
    shifted = classify_pair(9.125, 2.25)
    assert original['ordered'] and original['threshold_fail']
    assert original['gap'] == shifted['gap'] == 6.875
    assert shifted['center'] == original['center'] + 2
    assert classify_pair(2, -1)['both_binary_correct']
    assert classify_pair(0, -1)['boundary_tie']
    assert classify_pair(1, 1)['tie']
    assert not classify_pair(-1, 1)['ordered']
    with pytest.raises(ValueError, match='finite'):
        classify_pair(float('nan'), 1)
