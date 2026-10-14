"""Protect sample coverage, input identity and interruption recovery."""
import json
from pathlib import Path
import pytest
from vlm_batching.schedule import make_batches,padding_summary
from vlm_batching.evaluate import validate_detail,cached_batches


def test_no_sample_loss_and_length_order():
    ids=['8','3','10','1','7'];lengths=dict(zip(ids,[20,10,10,30,5]))
    fifo=make_batches(ids,lengths,{'batch_size':2,'order':'manifest'})
    bucket=make_batches(ids,lengths,{'batch_size':2,'order':'input_tokens_ascending_numeric_qid_tiebreak'})
    assert fifo==[['8','3'],['10','1'],['7']]
    assert bucket==[['7','3'],['10','8'],['1']]
    assert padding_summary(bucket,lengths)['useful_input_tokens']==sum(lengths.values())
    assert padding_summary(bucket,lengths)['padding_slots']<padding_summary(fifo,lengths)['padding_slots']


def test_duplicate_or_unknown_policy_rejected():
    with pytest.raises(ValueError):make_batches(['1','1'],{'1':5},{'batch_size':2,'order':'manifest'})
    with pytest.raises(ValueError):make_batches(['1'],{'1':5},{'batch_size':2,'order':'answers'})


def detail():
    return {'question_id':'1','image_id':'image','image_grid_thw':[[1,2,2]],
        'input_tokens':5,'visual_tokens':1,'counted_image_tokens':1,'input_ids_sha256':'ids',
        'pixel_values_sha256':'pixels','pixel_values_shape':[4,1176],'image_sha256':'image_sha',
        'question_sha256':'question','input_truncated':False,'answers_or_target_used':False}


def test_pixel_or_label_leak_rejected():
    r=detail();changed={**r,'pixel_values_sha256':'different'}
    with pytest.raises(ValueError):validate_detail(changed,r)
    with pytest.raises(ValueError):validate_detail({**r,'answers_or_target_used':True},r)


def test_interrupted_final_batch_only_repaired(tmp_path):
    path=tmp_path/'batches.jsonl';path.write_bytes(b'{"interrupted":')
    batches,repair=cached_batches(path,[['1']],{}, {},{},False)
    assert batches==[] and repair['kept_bytes']==0 and path.read_bytes()==b''
    path.write_text('{"invalid complete batch":1}\n')
    with pytest.raises(KeyError):cached_batches(path,[['1']],{}, {},{},False)
