import numpy as np
import pytest
import torch
import torch.nn.functional as F

from clip_retrieval.train import multi_positive_ce
from clip_retrieval_set.loss import positive_set_ce


def test_set_probability_and_gradient():
    x=torch.tensor([[1.,2.,-.5,3.]],dtype=torch.float64,requires_grad=True)
    positive=torch.tensor([[True,True,False,False]])
    loss=positive_set_ce(x,positive)
    p=x.softmax(1)
    assert torch.allclose(loss,-p[positive].sum().log().reshape(1))
    grad=torch.autograd.grad(loss.sum(),x)[0]
    expected=p.detach()-x.detach().masked_fill(~positive,-torch.inf).softmax(1)
    assert torch.allclose(grad,expected,atol=1e-12)


def test_single_positive_equals_ce_and_multi_positive_differs():
    x=torch.tensor([[1.,2.,-3.],[2.,-.5,1.]],dtype=torch.float64)
    positive=F.one_hot(torch.tensor([1,0]),3).bool()
    assert torch.allclose(positive_set_ce(x,positive),multi_positive_ce(x,positive))
    positive[0,0]=True
    assert not torch.allclose(positive_set_ce(x,positive),multi_positive_ce(x,positive))
    assert torch.allclose(positive_set_ce(x+1000,positive),positive_set_ce(x,positive),atol=1e-12)


def test_zero_positive_rejected_all_positive_zero_finite_gradient():
    x=torch.tensor([[1.,2.,3.]],requires_grad=True)
    with pytest.raises(ValueError):positive_set_ce(x,torch.zeros_like(x,dtype=torch.bool))
    loss=positive_set_ce(x,torch.ones_like(x,dtype=torch.bool))
    assert float(loss)==0
    assert torch.equal(torch.autograd.grad(loss.sum(),x)[0],torch.zeros_like(x))
