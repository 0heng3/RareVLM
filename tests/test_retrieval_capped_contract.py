import numpy as np
import pytest
from clip_retrieval.train import choose_negatives
from clip_retrieval_capped.sampler import sample_pool


def test_uncapped_and_inactive_cap_reproduce_original_sampler_with_ties():
    ids=np.arange(120);scores=np.ones(120);counts=np.zeros(120,dtype=np.int64)
    original=choose_negatives('mixed_hard_negatives',ids,scores,42,13)
    for method in ['mixed_distill','capped_distill']:
        pool,hard,random=sample_pool(method,ids,scores,42,13,counts)
        assert np.array_equal(pool,original)
        assert len(hard)==len(random)==32
        assert np.all(counts==0)  # caller owns count mutation/checkpoint state


def test_cap_excludes_only_hard_and_preserves_budget_eligibility_and_uniqueness():
    ids=np.arange(100);scores=-ids.astype(float);counts=np.zeros(100,dtype=np.int64)
    counts[:40]=8
    pool,hard,random=sample_pool('capped_distill',ids,scores,0,1,counts)
    assert len(pool)==len(np.unique(pool))==64
    assert np.array_equal(hard,np.arange(40,72))
    assert np.all(counts[hard]<8)
    assert set(pool)<=set(ids)
    assert any(counts[random]>=8)  # random entries are not capped


def test_no_available_hard_uniformly_fills_full_budget_and_tie_break_is_stable():
    ids=np.arange(80);counts=np.full(80,8,dtype=np.int64)
    pool,hard,random=sample_pool('capped_distill',ids,np.zeros(80),42,100,counts)
    assert len(hard)==0 and len(random)==64 and len(np.unique(pool))==64


def test_saved_counts_restore_the_next_pool_exactly():
    ids=np.arange(100);scores=-ids.astype(float);counts=np.zeros(100,dtype=np.int64)
    for step in range(10):
        _,hard,_=sample_pool('capped_distill',ids,scores,200,step,counts)
        counts[hard]+=1
    saved=counts.copy()
    a=sample_pool('capped_distill',ids,scores,200,10,counts)
    b=sample_pool('capped_distill',ids,scores,200,10,saved)
    assert all(np.array_equal(x,y) for x,y in zip(a,b))
    assert counts.max()<=8


def test_invalid_budget_fails_instead_of_using_smaller_gallery():
    with pytest.raises(ValueError):
        sample_pool('capped_distill',np.arange(10),np.zeros(10),0,0,np.zeros(10,dtype=int))
