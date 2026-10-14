"""Protect task-specific aggregation and actual input parity."""
import pytest
from vlm_strong_confirm.evaluate import metric_percent,compare_input,DETAIL_FIELDS
from vlm_strong_confirm.analysis import paired_interval


def test_macro_is_not_qa_weighted_mean():
    rows=[{'question_id':str(i),'image_id':str(i),'subcategory':f'c{i}'} for i in range(18)]
    rows.append({'question_id':'18','image_id':'18','subcategory':'c0'})
    records=[{'question_id':str(i),'score':1. if i in [0,18] else 0.} for i in range(19)]
    assert abs(metric_percent(records,rows,'mmstar')-100/18)<1e-10
    with pytest.raises(ValueError):paired_interval(rows,[0.]*19,'mmstar',100,42,.025)
    interval=paired_interval(rows,[0.]*19,'textvqa',100,42,.025)
    assert interval['delta_pp']==0 and interval['one_sided_lower_pp']==0


def test_pixel_change_rejected_even_if_padding_differs():
    a={key:1 for key in DETAIL_FIELDS};a.update(input_truncated=False,references_used_for_generation=False,padded_input_tokens=100)
    b={**a,'padded_input_tokens':120};compare_input(a,b)
    with pytest.raises(ValueError):compare_input(a,{**b,'pixel_values_sha256':'different'})
