"""Mathematical/relevance contracts,not retrieval experiment results."""
import numpy as np
import pytest
import torch
from clip_retrieval.train import multi_positive_ce,choose_negatives,DualAdapter
from clip_retrieval.evaluate import rank_metrics


def test_multi_positive_loss_uses_all_captions():
    logits=torch.zeros(2,4,requires_grad=True)
    positives=torch.tensor([[1,1,0,0],[0,0,1,1]],dtype=torch.bool)
    loss=multi_positive_ce(logits,positives).mean()
    assert abs(float(loss)-np.log(4))<1e-6
    loss.backward()
    assert torch.all(logits.grad[positives]<0) and torch.all(logits.grad[~positives]>0)
    with pytest.raises(ValueError):multi_positive_ce(logits,torch.zeros_like(positives))


def test_hit_any_recall_differs_from_fraction_of_positives():
    scores=torch.tensor([[3.,2.,1.]])
    positive=torch.tensor([[True,False,True]])
    values,_=rank_metrics(scores,positive)
    assert float(values['R@1'])==1.
    expected=(1+.5)/(1+1/np.log2(3))
    assert abs(float(values['nDCG@10'])-expected)<1e-6


def test_negative_pool_cannot_pick_excluded_ids():
    eligible=np.arange(10,90);scores=np.arange(100,dtype=float)
    result=choose_negatives('mixed_hard_negatives',eligible,scores,42,0)
    assert len(result)==64 and len(set(result))==64 and min(result)>=10 and max(result)<90
    assert set(result[:32])==set(range(58,90))
    with pytest.raises(ValueError):choose_negatives('uniform_negatives',np.arange(20),scores,0,0)


def test_adapter_starts_with_frozen_geometry_and_parameter_budget():
    model=DualAdapter();x=torch.nn.functional.normalize(torch.randn(7,512),dim=-1)
    assert sum(p.numel() for p in model.parameters())==132480
    torch.testing.assert_close(model.image_features(x),x,atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(model.text_features(x),x,atol=1e-6,rtol=1e-6)
