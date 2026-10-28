from copy import deepcopy
from types import SimpleNamespace
import json
from pathlib import Path
import pytest
import torch
from vlm_gt_anchored_kl import data,inputs,train

def sample(empty=False):
    ids=torch.tensor([[7,8,151645,198]] if empty else [[7,8,31,32,151645,198]])
    labels=ids.clone();labels[:,:2]=-100
    return {'input_ids':ids,'attention_mask':torch.ones_like(ids),'labels':labels,'pixel_values':torch.tensor([[.25,.5]]*4),'image_grid_thw':torch.tensor([[1,2,2]])}

@pytest.mark.parametrize('empty',[False,True])
def test_GT_encoder_same_targets_roles_both_conditions_no_generated_labels(monkeypatch,empty):
    calls=[]
    def enc(proc,row,path,supervised):
        calls.append((row['target'],supervised));return sample(empty)
    monkeypatch.setattr(data,'encode',enc)
    row={'target':'realGT','image_path':'unused'};cfg={'max_sequence_length':2048,'visual_token_budget':1024}
    results=[data.encode_training_sample(None,row,cfg,source='replay',condition=c,teacher_replies={'fake':'FORBIDDEN_GENERATED_IDS'}) for c in ('gt_kl32','gt_ce_kl32')]
    assert calls==[('realGT',True)]*2
    assert all(torch.equal(results[0][k],results[1][k]) for k in results[0])
    assert results[0]['answer_roles'].tolist()==([[0,0,3,4]] if empty else [[0,0,1,2,3,4]])

@pytest.mark.parametrize('mutation',['EOS','template','length','visual'])
def test_GT_encoder_rejects_actual_changed_contract(monkeypatch,mutation):
    value=sample();cfg={'max_sequence_length':2048,'visual_token_budget':1024}
    if mutation=='EOS':value['input_ids'][0,-2]=151643;value['labels'][0,-2]=151643
    elif mutation=='template':value['input_ids'][0,-1]=199;value['labels'][0,-1]=199
    elif mutation=='length':cfg['max_sequence_length']=4
    else:value['image_grid_thw']=torch.tensor([[1,128,128]])
    monkeypatch.setattr(data,'encode',lambda *a:value)
    with pytest.raises(ValueError):data.encode_training_sample(None,{'target':'GT','image_path':'unused'},cfg,source='replay',condition='gt_ce_kl32')

def test_runtime_group_groundtruth_full12_projection_and_leftpad(monkeypatch):
    monkeypatch.setattr(data,'encode',lambda p,r,*args:sample(r['question_id']=='q1'))
    cfg={'max_sequence_length':2048,'visual_token_budget':1024};rows=[{'question_id':f'q{i}','image_id':f'i{i}','target':'GT','image_path':'unused'} for i in range(2)]
    batch,contracts,_=inputs.encode_group(SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=151643)),rows,[0,1],cfg,source='replay',condition='gt_ce_kl32',teacher_replies={})
    assert batch['labels'].tolist()[1][:4]==[-100]*4
    assert torch.equal(batch['labels']!=-100,batch['answer_roles']>0)
    assert all(c['target_source']=='original_ground_truth' and c['teacher_reply_row_sha256'] is None for c in contracts)
    assert len(inputs.LEGACY_FIELDS)==12 and all(set(r)==set(inputs.LEGACY_FIELDS) for r in inputs.legacy_contracts(contracts))

def test_seed42_GT_reference_is_explicit_and_new_target_generation_absent():
    cfg=json.loads((Path(__file__).resolve().parents[1]/'configs/vlm_gt_anchored_kl_v1.json').read_text())
    assert cfg['seed42_input_reference']['parent_condition']=='gt_ce32' and cfg['seed42_input_reference']['new_generation_questions']==0
    assert data.load_teacher_replies(cfg,0)=={} and data.load_teacher_replies(cfg,200)=={}
    train.validate_config(cfg)
