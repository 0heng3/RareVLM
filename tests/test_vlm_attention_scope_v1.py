import torch
from peft import LoraConfig, get_peft_model
from peft.utils.save_and_load import get_peft_model_state_dict, set_peft_model_state_dict
from transformers import LlamaConfig, LlamaForCausalLM

from vlm_attention_scope_v1.model import apply_scope, expand_qv


def actor():
    torch.manual_seed(5)
    model = LlamaForCausalLM(LlamaConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, vocab_size=32, max_position_embeddings=32))
    return get_peft_model(model, LoraConfig(r=2, lora_alpha=4, lora_dropout=0.,
        target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))


def test_expansion_preserves_qv_and_new_ko_have_exact_zero_output():
    model = actor()
    original = get_peft_model_state_dict(model)
    expanded, config, receipt = expand_qv(original, model.peft_config['default'].to_dict())
    assert receipt['qv_values_preserved'] and receipt['new_B_all_zero']
    assert len(expanded) == 2 * len(original)
    for name, value in expanded.items():
        if '.k_proj.lora_B.' in name or '.o_proj.lora_B.' in name:
            assert not bool(value.ne(0).any())
    assert config['lora_dropout'] == 0


def test_actual_peft_forward_identical_then_only_declared_projections_train():
    source = actor().eval()
    # Make the retained q/v branch nonzero; parity must preserve learned values.
    with torch.no_grad():
        for name, p in source.named_parameters():
            if 'lora_B' in name:
                p.uniform_(-.01, .01)
    state = get_peft_model_state_dict(source)
    expanded, _, _ = expand_qv(state, source.peft_config['default'].to_dict())
    target = actor().eval()
    # Same deterministic base; recreate one adapter spanning all four projections.
    base = target.unload()
    target = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, lora_dropout=0.,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], task_type='CAUSAL_LM')).eval()
    result = set_peft_model_state_dict(target, expanded)
    assert not result.unexpected_keys
    ids = torch.tensor([[1, 4, 7, 3]])
    with torch.no_grad():
        torch.testing.assert_close(source(ids).logits, target(ids).logits, rtol=0, atol=0)
    qv = apply_scope(target, 'qv')
    assert set(qv['trainable_parameters_by_projection']) == {'q_proj', 'v_proj'}
    target.train()
    target(ids, labels=ids).loss.backward()
    assert all(p.grad is None for n, p in target.named_parameters() if '.k_proj.' in n or '.o_proj.' in n)
    target.zero_grad(set_to_none=True)
    full = apply_scope(target, 'qkvo')
    assert full['trainable_parameters'] > qv['trainable_parameters']
    target(ids, labels=ids).loss.backward()
    assert any(p.grad is not None and bool(p.grad.ne(0).any())
               for n, p in target.named_parameters() if ('.k_proj.' in n or '.o_proj.' in n) and 'lora_B' in n)


def test_initializer_seed_is_reproducible_and_does_not_touch_source():
    model = actor()
    state = get_peft_model_state_dict(model)
    before = {k: v.clone() for k, v in state.items()}
    a, _, _ = expand_qv(state, model.peft_config['default'].to_dict(), seed=42)
    b, _, _ = expand_qv(state, model.peft_config['default'].to_dict(), seed=42)
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert all(torch.equal(state[k], before[k]) for k in state)
