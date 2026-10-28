"""Targeted CPU PEFT save/reload tests; no pretrained models or GPU required."""

import json
from unittest.mock import patch

import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

from vlm_runtime_metadata_v1.model import load, sync_training_dropout


def tiny_model(dropout=.05):
    config = LlamaConfig(vocab_size=16, hidden_size=8, intermediate_size=16,
                         num_hidden_layers=1, num_attention_heads=2,
                         num_key_value_heads=2, max_position_embeddings=16)
    base = LlamaForCausalLM(config)
    return get_peft_model(base, LoraConfig(r=2, lora_alpha=4,
                                          target_modules=['q_proj', 'v_proj'],
                                          lora_dropout=dropout,
                                          task_type='CAUSAL_LM'))


def test_actual_peft_zero_dropout_save_and_generic_reload(tmp_path):
    model = tiny_model()
    model.train()
    # Reproduce the old loader: runtime dropout is zero, config still .05.
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.
    state_before = {key: value.clone() for key, value in model.state_dict().items()}
    rng_before = torch.get_rng_state().clone()
    receipt = sync_training_dropout(model, 0.)
    assert receipt['config_before'] == {'default': .05}
    assert receipt['config_after'] == {'default': 0.}
    assert receipt['module_count_by_adapter'] == {'default': 2}
    assert all(row['before']['p'] == row['after']['p'] == 0.
               for row in receipt['modules'])
    assert torch.equal(torch.get_rng_state(), rng_before)
    for key, value in model.state_dict().items():
        assert torch.equal(value, state_before[key])
    model.save_pretrained(tmp_path)
    assert json.loads((tmp_path / 'adapter_config.json').read_text())['lora_dropout'] == 0.
    fresh_base = LlamaForCausalLM(model.get_base_model().config)
    reloaded = PeftModel.from_pretrained(fresh_base, tmp_path, is_trainable=True)
    assert reloaded.peft_config['default'].lora_dropout == 0.
    for name, module in reloaded.named_modules():
        if name.endswith('lora_dropout.default'):
            assert isinstance(module, torch.nn.Identity)
    for key, parameter in reloaded.named_parameters():
        if 'lora_' in key:
            assert torch.equal(parameter, dict(model.named_parameters())[key])


def test_all_adapters_identity_to_dropout_and_non_lora_unchanged():
    model = tiny_model(dropout=0.)
    model.add_adapter('other', LoraConfig(r=2, lora_alpha=4,
                                        target_modules=['q_proj'], lora_dropout=.4,
                                        task_type='CAUSAL_LM'))
    model.extra_non_lora_dropout = torch.nn.Dropout(.7)
    model.train()
    receipt = sync_training_dropout(model, .2)
    assert receipt['config_after'] == {'default': .2, 'other': .2}
    assert receipt['module_count_by_adapter'] == {'default': 2, 'other': 1}
    assert all(row['after']['p'] == .2 for row in receipt['modules'])
    assert model.extra_non_lora_dropout.p == .7


@pytest.mark.parametrize('value', [-.1, 1., float('nan'), float('inf'), '0', True])
def test_invalid_probability_rejected_before_any_load(value):
    with patch('vlm_fp32_adapter.model.load') as delegated:
        with pytest.raises(ValueError, match='finite real'):
            load('unused', 'unused', 'cpu', training=True, dropout=value)
        delegated.assert_not_called()


def test_eval_wrapper_preserves_historical_config_and_delegates_arguments():
    model = tiny_model()
    model.eval()
    with patch('vlm_fp32_adapter.model.load', return_value=(model, {'original': True})) as delegated:
        actual, binding = load('base', 'adapter', 'cpu', training=False,
                               dropout=0., gradient_checkpointing=False)
    delegated.assert_called_once_with('base', 'adapter', 'cpu', training=False,
                                      dropout=0., gradient_checkpointing=False)
    assert actual is model
    assert model.peft_config['default'].lora_dropout == .05
    assert binding['original'] is True
    assert binding['dropout_runtime_metadata']['configuration_mutated'] is False
    with pytest.raises(ValueError, match='training=True'):
        sync_training_dropout(model, 0.)


def test_training_wrapper_repairs_in_memory_configuration():
    model = tiny_model()
    model.train()
    with patch('vlm_fp32_adapter.model.load', return_value=(model, {'original': True})):
        actual, binding = load('base', 'adapter', 'cpu', training=True, dropout=0.)
    assert actual is model
    assert model.peft_config['default'].lora_dropout == 0.
    assert binding['dropout_runtime_metadata']['config_before'] == {'default': .05}
