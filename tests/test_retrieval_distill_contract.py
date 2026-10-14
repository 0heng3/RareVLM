import copy
import numpy as np
import torch
import torch.nn.functional as F
from clip_retrieval.train import DualAdapter,contrastive_loss
from clip_retrieval_distill.loss import teacher_kl,contrastive_distill_loss,distill_terms


def test_teacher_stopped_and_kl_gradient_matches_distribution_difference():
    s=torch.tensor([[1.,2.,-1.],[3.,0.,2.]],dtype=torch.float64,requires_grad=True)
    t=torch.tensor([[2.,1.,0.],[0.,1.,3.]],dtype=torch.float64,requires_grad=True)
    loss=teacher_kl(s,t).mean()
    grad_s,grad_t=torch.autograd.grad(loss,(s,t),allow_unused=True)
    assert grad_t is None
    assert float(loss)>0
    assert torch.allclose(grad_s,(s.softmax(1)-t.softmax(1)).detach()/2,atol=1e-12)


def inputs():
    torch.manual_seed(7)
    model=DualAdapter().double()
    with torch.no_grad():model.image.fc2.weight.normal_(0,.01);model.text.fc2.weight.normal_(0,.01)
    images=F.normalize(torch.randn(8,512,dtype=torch.float64),dim=-1)
    texts=F.normalize(torch.randn(40,512,dtype=torch.float64),dim=-1)
    return model,images,texts,np.array([0,1]),np.arange(2,8),np.arange(8),.07


def test_zero_weight_recovers_original_loss_and_all_parameter_gradients():
    model,*args=inputs();other=copy.deepcopy(model)
    a=contrastive_loss(model,*args);b=contrastive_distill_loss(other,*args,0.)
    assert torch.allclose(a,b,atol=1e-12)
    ga=torch.autograd.grad(a,tuple(model.parameters()));gb=torch.autograd.grad(b,tuple(other.parameters()))
    assert all(torch.allclose(x,y,atol=1e-10,rtol=1e-10) for x,y in zip(ga,gb))


def test_identity_student_zero_relation_loss():
    _,*args=inputs();model=DualAdapter().double()
    supervised,relation=distill_terms(model,*args)
    assert abs(float(relation))<1e-12
    assert torch.allclose(supervised,contrastive_loss(model,*args),atol=1e-12)
