import torch

from vlm_fp32_readout_v1.model import install_fp32_readout


class TiedToy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(9, 5, dtype=torch.bfloat16)
        self.head = torch.nn.Linear(5, 9, bias=False, dtype=torch.bfloat16)
        self.head.weight = self.embedding.weight
        self.requires_grad_(False)

    def get_output_embeddings(self):return self.head
    def get_input_embeddings(self):return self.embedding
    def set_output_embeddings(self, value):self.head=value


def test_tied_input_embedding_is_preserved_and_same_frozen_values_are_promoted():
    model=TiedToy();original=model.embedding.weight.detach().clone()
    pointer=model.embedding.weight.data_ptr()
    report=install_fp32_readout(model)
    assert report['input_output_weights_were_tied']
    assert model.embedding.weight.data_ptr()==pointer
    assert model.embedding.weight.dtype==torch.bfloat16
    torch.testing.assert_close(model.embedding.weight,original,rtol=0,atol=0)
    torch.testing.assert_close(model.head.weight,original.float(),rtol=0,atol=0)
    assert model.head.weight.data_ptr()!=pointer and not model.head.weight.requires_grad


def test_fp32_head_keeps_gradients_to_hidden_states_even_inside_bf16_autocast():
    model=TiedToy();install_fp32_readout(model)
    hidden=torch.randn(2,3,5,dtype=torch.bfloat16,requires_grad=True)
    with torch.autocast('cpu',dtype=torch.bfloat16):
        logits=model.head(hidden)
    assert logits.dtype==torch.float32
    torch.testing.assert_close(logits,torch.nn.functional.linear(hidden.float(),model.head.weight),rtol=0,atol=0)
    logits.square().mean().backward()
    assert hidden.grad is not None and bool(hidden.grad.ne(0).any())
    assert model.head.weight.grad is None
