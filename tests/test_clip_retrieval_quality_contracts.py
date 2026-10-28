"""Math and leakage-boundary contracts, not synthetic research outcomes."""
import numpy as np
import torch
from clip_retrieval.train import DualAdapter
from clip_retrieval_distill.loss import distill_terms
from clip_retrieval_quality.loss import selected_terms
from clip_retrieval_quality.selection import select_slots,reassign_slots


def test_reassignment_conserves_vocabulary_and_exact_rate():
    sha=[str(i//2) for i in range(12)];x=reassign_slots(sha,8);original=np.arange(60).reshape(12,5)
    assert np.all((x!=original).sum(1)==1)
    assert np.array_equal(np.sort(x.flatten()),np.arange(60))
    i,j=np.where(x!=original)
    assert all(sha[int(x[a,b])//5]!=sha[a] for a,b in zip(i,j))
    assert np.array_equal(x,reassign_slots(sha,8))


def test_random_selector_never_uses_similarity():
    a=np.arange(20).reshape(4,5);sha=['a','b','c','d']
    assert np.array_equal(select_slots(a,sha,'random',7),select_slots(-a,sha,'random',7))
    assert np.all(np.diff(select_slots(a,sha,'random',7),axis=1)>0)


def test_score_selector_has_fixed_budget_and_stable_ties():
    a=np.array([[.4,.2,.8,.2,.1],[1,1,1,1,1]])
    assert np.array_equal(select_slots(a,['a','b'],'score',7),[[0,1,2,3],[0,1,2,3]])


def test_five_caption_objective_matches_frozen_loss_and_gradients():
    torch.manual_seed(42);m=DualAdapter();im=torch.nn.functional.normalize(torch.randn(8,512),dim=-1)
    txt=torch.nn.functional.normalize(torch.randn(40,512),dim=-1);anchors=np.array([0,1]);pool=np.array([2,3]);groups=np.arange(8)
    a,b=distill_terms(m,im,txt,anchors,pool,groups,.07)
    c,d=selected_terms(m,im,txt.reshape(8,5,512),anchors,pool,groups,.07)
    assert torch.equal(a,c) and torch.equal(b,d)
    ga=torch.autograd.grad(a+b,tuple(m.parameters()));gb=torch.autograd.grad(c+d,tuple(m.parameters()))
    assert all(torch.equal(x,y) for x,y in zip(ga,gb))


def test_four_caption_teacher_detached_and_finite_gradients():
    torch.manual_seed(9);m=DualAdapter();im=torch.nn.functional.normalize(torch.randn(8,512),dim=-1)
    txt=torch.nn.functional.normalize(torch.randn(8,4,512),dim=-1)
    ce,kl=selected_terms(m,im,txt,np.array([0,1]),np.array([2,3]),np.arange(8),.07)
    assert abs(float(kl.detach()))<1e-6 and torch.isfinite(ce)
    grads=torch.autograd.grad(ce+kl,tuple(m.parameters()))
    assert all(torch.isfinite(g).all() for g in grads)
