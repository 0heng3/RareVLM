import numpy as np
import pytest
from clip_retrieval_rerank.common import pair_indices,retrieve_pair_scores,reorder,ranking_metrics,native


def test_two_directions_share_exact_pair_identifiers_and_missing_pairs_rejected():
    i2t=np.array([[7,8],[2,4]]);t2i=np.zeros((9,1),dtype=np.int64);t2i[2]=1;t2i[4]=1
    a=pair_indices(i2t,'I2T');b=pair_indices(t2i,'T2I')
    assert 7 in a and 7 in b and 5002 in a and 5002 in b
    union=np.unique(np.r_[a,b]);scores=union.astype(float)/10000
    assert np.array_equal(retrieve_pair_scores(a,union,scores),a/10000)
    with pytest.raises(ValueError):retrieve_pair_scores(np.array([9000000]),union,scores)


def test_reranking_has_stable_ties_and_never_changes_pool_or_recall50_bound():
    rng=np.random.default_rng(7)
    candidate=np.array([rng.choice(200,50,replace=False) for _ in range(20)])
    scores=rng.normal(size=candidate.shape);rank=reorder(candidate,scores)
    assert np.array_equal(np.sort(candidate,axis=1),np.sort(rank,axis=1))
    assert np.array_equal(reorder(candidate,np.zeros_like(scores)),candidate)
    for direction in ['I2T','T2I']:
        a=ranking_metrics(candidate,direction);b=ranking_metrics(rank,direction)
        assert np.array_equal(a[direction+'_R@50'],b[direction+'_R@50'])
        assert (b[direction+'_R@1']<=b[direction+'_R@50']).all()


def test_multicap_recall_is_hit_any_and_binary_ndcg_uses_five_positive_ideal():
    ranking=np.array([[9,0,1,2,3,4,*range(10,54)]])
    m=ranking_metrics(ranking,'I2T')
    assert m['I2T_R@1'][0]==0 and m['I2T_R@5'][0]==1
    discount=1/np.log2(np.arange(10)+2)
    assert np.isclose(m['I2T_nDCG@10'][0],discount[1:6].sum()/discount[:5].sum())


def test_numpy_bookkeeping_values_export_as_native_json_without_score_changes():
    import json
    value={'indices':[np.int64(2)],'score':np.float64(.75),'flag':np.bool_(True)}
    out=json.loads(json.dumps(native(value),allow_nan=False))
    assert out=={'indices':[2],'score':.75,'flag':True}
