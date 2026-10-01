import pytest
import torch
from methods import PrototypeClassifier, ResidualAdapter


def test_identity_and_parameter_budget():
    adapter=ResidualAdapter(512,64);x=torch.randn(7,512)
    assert torch.equal(adapter(x),x)
    assert sum(p.numel() for p in adapter.parameters())==66240


def test_classifier_has_no_trainable_prototypes_or_label_argument():
    classifier=PrototypeClassifier(torch.randn(3,512),100.)
    assert list(classifier.parameters())==[]
    x=torch.randn(4,512);labels=torch.tensor([0,1,2,0])
    assert classifier(x).shape==(4,3)
    with pytest.raises(TypeError):classifier(x,labels)
    with pytest.raises(TypeError):classifier(x,targets=labels)


def test_gradient_update_is_confined_to_adapter():
    adapter=ResidualAdapter(512,64);classifier=PrototypeClassifier(torch.randn(3,512),10.)
    originals={k:v.clone() for k,v in classifier.state_dict().items()}
    before=adapter.fc2.weight.detach().clone()
    optimizer=torch.optim.AdamW(adapter.parameters(),lr=.001)
    loss=torch.nn.functional.cross_entropy(classifier(adapter(torch.randn(6,512))),torch.tensor([0,1,2,0,1,2]))
    loss.backward();optimizer.step()
    assert not torch.equal(before,adapter.fc2.weight)
    assert all(torch.equal(v,originals[k]) for k,v in classifier.state_dict().items())
