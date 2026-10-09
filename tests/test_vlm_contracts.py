"""CPU contracts for the separate generative-VLM module; no model download."""
import copy
import pytest
import torch
from vlm_sft.common import answer_labels,consensus_score,normalize_answer,user_messages,validate_records


def test_assistant_only_loss_masks_the_entire_multimodal_prompt():
    prefix=torch.tensor([[1,2,151655,151655,3,4]])
    full=torch.tensor([[1,2,151655,151655,3,4,8,9,10]])
    labels,boundary=answer_labels(full,prefix)
    assert boundary==6 and torch.equal(labels[:,:6],torch.full_like(prefix,-100))
    assert torch.equal(labels[:,6:],full[:,6:])
    assert torch.equal(full,torch.tensor([[1,2,151655,151655,3,4,8,9,10]]))


def test_prefix_mismatch_and_missing_answer_are_rejected():
    with pytest.raises(ValueError):answer_labels(torch.tensor([[1,2,3]]),torch.tensor([[1,7]]))
    with pytest.raises(ValueError):answer_labels(torch.tensor([[1,2]]),torch.tensor([[1,2]]))


def test_generation_messages_have_no_reference_answer_interface():
    assert user_messages('What text is visible?')[0]['content'][0]=={'type':'image'}
    with pytest.raises(TypeError):user_messages('Question',answers=['secret target'])


@pytest.mark.parametrize('matches,expected',[(0,0.),(1,.3),(2,.6),(3,.9),(4,1.),(10,1.)])
def test_vqa_formula_uses_leave_one_annotator_out(matches,expected):
    answers=['yes']*matches+['no']*(10-matches)
    assert consensus_score('yes',answers)==pytest.approx(expected)


def test_declared_normalization_and_annotation_count():
    assert normalize_answer('The ONE.')=='1'
    assert normalize_answer('1,000')=='1000'
    with pytest.raises(ValueError):consensus_score('yes',['yes'])


def test_image_identity_and_decoded_content_do_not_cross_splits():
    row={'image_id':'image1','question':'Question?','answers':['answer']*10,'pixel_sha256':'pixel1'}
    other={**row,'image_id':'image2','pixel_sha256':'pixel2'}
    validate_records({'train':[row],'validation':[other]})
    with pytest.raises(ValueError):validate_records({'train':[row],'final':[row]})
    with pytest.raises(ValueError):validate_records({'train':[row],'final':[{**other,'pixel_sha256':'pixel1'}]})


def test_data_quality_filter_records_empty_supervision_without_model_scores(tmp_path):
    import json
    from vlm_sft.assets import DATASET_REVISION
    from vlm_sft.prepare_viewer import get_page
    rows=[]
    for i,answers in enumerate([['nokia']*10,['?']*10,['samsung']*10]):
        rows.append({'row_idx':i,'row':{'image_id':str(i),'question_id':i,'question':'Which brand?',
                     'answers':answers,'image':{'src':f'https://example.test/--/{DATASET_REVISION}/--/image.jpg'}},
                     'truncated_cells':[]})
    (tmp_path/'train_0_3.json').write_text(json.dumps({'partial':False,'rows':rows}))
    records,source=get_page('train',0,3,tmp_path)
    assert [r['question_id'] for r in records]==['0','2']
    assert source['excluded_unusable_answers']==[{'row':1,'question_id':'1','reason':'all normalized human answers empty'}]
