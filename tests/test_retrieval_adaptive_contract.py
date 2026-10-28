import numpy as np
import pytest
from clip_retrieval_adaptive.route import selected_keys,routed_ranking,route_mask,image_partition
from clip_retrieval_rerank.common import pair_indices


def test_selective_pair_keys_preserve_original_query_indices_and_deduplicate_directions():
    pools={'T2I':np.array([[0,1],[1,0],[0,1]]),'I2T':np.array([[0,2],[1,2]])}
    masks={'T2I':np.array([False,True,False]),'I2T':np.array([False,True])}
    keys=selected_keys(pools,masks)
    assert np.array_equal(keys,np.array([1,5001,5002]))
    assert len(keys)==3  # key5001 is shared by both selected directions


def test_never_route_requires_no_scores_and_keeps_complete_original_ranking():
    pool=np.array([[0,1],[1,0],[0,1]])
    out=routed_ranking(pool,'T2I',np.zeros(3,dtype=bool),np.array([],dtype=int),np.array([]))
    assert np.array_equal(out,pool)


def test_selected_rows_only_are_scored_missing_key_fails_and_ties_preserve_order():
    pool=np.array([[0,1],[1,0],[0,1]]);mask=np.array([False,True,False])
    keys=np.array([1,5001]);scores=np.array([2.,1.])
    out=routed_ranking(pool,'T2I',mask,keys,scores)
    assert np.array_equal(out,np.array([[0,1],[0,1],[0,1]]))
    assert np.array_equal(routed_ranking(pool,'T2I',mask,keys,np.ones(2)),pool)
    with pytest.raises(ValueError):routed_ranking(pool,'T2I',mask,np.array([1]),np.ones(1))


def test_threshold_changes_are_monotonic_and_do_not_accept_nonfinite_confidence():
    margin=np.array([0.,.01,.02,.5])
    assert not route_mask(margin,-1).any()
    assert route_mask(margin,2.1).all()
    assert np.all(route_mask(margin,.01)<=route_mask(margin,.02))
    with pytest.raises(ValueError):route_mask(np.array([np.nan]),.1)


def test_sha_partition_is_reproducible_and_all_caption_queries_follow_image_fold():
    rows=[{'image_sha256':str(i)} for i in range(1000)]
    a,b=image_partition(rows);c,d=image_partition(rows)
    assert a.sum()==b.sum()==500 and not (a&b).any()
    assert np.array_equal(a,c) and np.array_equal(b,d)
    assert np.repeat(a,5).sum()==2500
    with pytest.raises(ValueError):image_partition([rows[0]]*1000)


def test_confirmation_scores_and_margins_cannot_change_calibration_choice():
    from clip_retrieval_adaptive.calibrate import choose_direction
    cfg={'calibration_quantiles':[0,.5,1],'calibration_max_r1_loss_pp':.1,'calibration_max_ndcg_loss_pp':.5}
    qmask=np.array([True,True,True,False,False,False])
    margins=[np.array([.01,.02,.3,.001,.02,.03]) for _ in range(3)]
    clip=[{'T2I_R@1':np.array([0.,1.,1.,0.,0.,0.]),'T2I_nDCG@10':np.array([0.,1.,1.,0.,0.,0.])} for _ in range(3)]
    full=[{'T2I_R@1':np.ones(6),'T2I_nDCG@10':np.ones(6)} for _ in range(3)]
    choice,scan=choose_direction(margins,clip,full,qmask,cfg,'T2I')
    for m,c,f in zip(margins,clip,full):
        m[~qmask]=1.9
        for k in c:c[k][~qmask]=1.;f[k][~qmask]=0.
    changed,changed_scan=choose_direction(margins,clip,full,qmask,cfg,'T2I')
    assert changed==choice and changed_scan==scan
    assert choice['threshold']==.01
