import numpy as np
import pytest
from clip_retrieval_budget.route import budgets_from_margin,budget_keys,budget_ranking
from clip_retrieval_rerank.common import pair_indices


def test_three_modes_and_boundary_policy_do_not_require_targets():
    margin=np.array([.01,.05,.1,.2])
    assert np.array_equal(budgets_from_margin(margin,.1,.05),[50,20,20,0])
    assert np.array_equal(budgets_from_margin(margin,.1,-1),[20,20,20,0])
    assert np.array_equal(budgets_from_margin(margin,.1,2.1),[50,50,50,0])


def test_prefix_pair_keys_use_original_query_ids_and_do_not_score_tail_or_fast_rows():
    pool=np.tile(np.arange(50),(3,1));budget=np.array([0,20,50])
    keys=budget_keys({'T2I':pool},{'T2I':budget})
    expected=pair_indices(pool,'T2I').reshape(3,50)
    assert np.array_equal(keys,np.unique(np.r_[expected[1,:20],expected[2]]))
    assert len(keys)==70 and not np.isin(expected[1,20:],keys).any()


def test_prefix_reordering_preserves_tail_membership_and_fast_output():
    pool=np.tile(np.arange(50),(3,1));budget=np.array([0,20,50])
    keys=budget_keys({'T2I':pool},{'T2I':budget});score=(keys//5000).astype(float)
    rank=budget_ranking(pool,'T2I',budget,keys,score)
    assert np.array_equal(rank[0],pool[0])
    assert np.array_equal(rank[1,:20],np.arange(19,-1,-1))
    assert np.array_equal(rank[1,20:],pool[1,20:])
    assert np.array_equal(rank[2],np.arange(49,-1,-1))
    assert np.array_equal(np.sort(rank,1),np.sort(pool,1))
    with pytest.raises(ValueError):budget_ranking(pool,'T2I',budget,keys[:-1],score[:-1])


def test_more_full_queries_monotonically_expand_actual_pair_union_and_invalid_budget_fails():
    pool=np.tile(np.arange(50),(4,1));margin=np.array([.01,.02,.03,.2])
    a=budgets_from_margin(margin,.1,.015);b=budgets_from_margin(margin,.1,.04)
    ka=budget_keys({'T2I':pool},{'T2I':a});kb=budget_keys({'T2I':pool},{'T2I':b})
    assert np.isin(ka,kb).all()
    with pytest.raises(ValueError):budget_keys({'T2I':pool},{'T2I':np.array([10,20,50,0])})


def test_confirmation_values_cannot_change_budget_calibration_selection():
    from clip_retrieval_budget.calibrate import choose_cutoff
    cfg={'calibration_quantiles':[0,.5,1],'calibration_max_r1_loss_pp':.1,'calibration_max_ndcg_loss_pp':.5}
    mask=np.array([True,True,True,False,False,False]);margins=[np.array([.01,.02,.03,.04,.01,.2]) for _ in range(3)]
    metrics=[]
    for _ in range(3):
        metrics.append({k:{'T2I_R@1':np.array([0.,1.,1.,0.,0.,0.]) if k<50 else np.ones(6),
            'T2I_nDCG@10':np.array([0.,1.,1.,0.,0.,0.]) if k<50 else np.ones(6)} for k in [0,20,50]})
    before,scan=choose_cutoff(margins,metrics,mask,cfg,'T2I',.1)
    for m,conditions in zip(margins,metrics):
        m[~mask]=1.9
        for value in conditions.values():
            for key in value:value[key][~mask]=.123
    after,other=choose_cutoff(margins,metrics,mask,cfg,'T2I',.1)
    assert before==after and scan==other
