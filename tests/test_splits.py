import copy
from collections import Counter
import pytest
from splits import build_manifests, validate_manifest


@pytest.fixture
def manifests(tmp_path):
    train=[c for c in range(100) for _ in range(500)]
    final=[c for c in range(100) for _ in range(100)]
    return train,final,build_manifests(train,final,tmp_path)


def test_split_disjointness_uses_dataset_namespaces(manifests):
    train,final,records=manifests
    for m in records:
        train_ids={('official_train',i) for i in m['train_indices']}
        val_ids={('official_train',i) for i in m['val_indices']}
        final_ids={('official_test',i) for i in m['test_indices']}
        assert not train_ids&val_ids and not train_ids&final_ids and not val_ids&final_ids
        assert set(Counter(train[i] for i in m['val_indices']).values())=={50}
        assert max(m['class_counts'])/min(m['class_counts'])==100
        assert m['test_indices']==list(range(len(final)))


def test_frequency_permutations_keep_validation_and_pool_fixed(manifests):
    _,_,records=manifests
    assert all(m['val_indices']==records[0]['val_indices'] for m in records)
    assert all(m['train_pool_sha256']==records[0]['train_pool_sha256'] for m in records)
    assert len({tuple(m['frequency_rank_to_class']) for m in records})==3


def test_overlapping_train_validation_is_rejected(manifests):
    train,final,records=manifests;m=copy.deepcopy(records[0])
    m['train_indices'][0]=m['val_indices'][0]
    with pytest.raises(ValueError):validate_manifest(m,train,final)
