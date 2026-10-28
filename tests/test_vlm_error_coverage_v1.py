import pytest
from vlm_error_coverage_v1.data import assigned_batches


def test_shard_merge_preserves_original_four_question_context():
    rows=[{'question_id':str(i),'fold':'training' if i<16 else 'validation'} for i in range(32)]
    left=assigned_batches(rows,0);right=assigned_batches(rows,1)
    merged=[row for pair in zip(left,right) for group in pair for row in group]
    assert merged==rows
    assert not ({q['question_id'] for g in left for q in g}&{q['question_id'] for g in right for q in g})
    assert all(len({q['fold'] for q in g})==1 for g in left+right)


def test_partial_or_cross_fold_batch_is_not_silently_repacked():
    with pytest.raises(ValueError):assigned_batches([{'fold':'training'}]*9,0)
    with pytest.raises(ValueError):assigned_batches([{'fold':'training'}]*3+[{'fold':'validation'}]*5,0)
