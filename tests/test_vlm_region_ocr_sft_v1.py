import pytest

from vlm_region_ocr_sft_v1.data import bounds, excludes_other_word_centers, region, schedule


POLICY={'max_characters':40,'min_resized_width':16,'min_resized_height':12,
        'max_horizontal_angle_degrees':15,'max_image_area_fraction':.5}


def test_points_define_xyxy_even_if_bbox_uses_another_convention():
    ann={'utf8_string':'Rare','points':[100,20,180,20,180,50,100,50],'bbox':[100,20,80,30]}
    value,reason=region(ann,400,200,800,400,POLICY)
    assert reason is None and value['box_xyxy_normalized_1000']==[250,100,450,250]
    assert value['box_xyxy_original_pixels']==[100,20,180,50]


@pytest.mark.parametrize('points',[[0,0,0,0,0,0,0,0],[-1,0,5,0,5,5,-1,5],
                                  [0,0,5,0,float('nan'),5,0,5]])
def test_invalid_original_geometry_is_not_silently_clipped(points):
    with pytest.raises(ValueError):bounds(points,100,100)


def test_unreadable_and_resized_tiny_text_are_excluded():
    ann={'utf8_string':'.','points':[10,10,40,10,40,30,10,30]}
    assert region(ann,100,100,100,100,POLICY)[0] is None
    ann['utf8_string']='word'
    assert region(ann,100,100,25,25,POLICY)[1]=='too_small_after_training_resize'


def test_box_does_not_include_the_center_of_another_word():
    assert not excludes_other_word_centers([0,0,30,30],[[10,10,20,20]])
    assert excludes_other_word_centers([0,0,30,30],[[31,0,40,10]])


def test_both_arms_have_same_common_qa_and_auxiliary_image_positions():
    c=[f'common{i}' for i in range(1600)];a=[f'control{i}' for i in range(1600)];b=[f'region{i}' for i in range(1600)]
    s=schedule(c,a,b)
    for role in s:
        flat=[q for update in s[role] for micro in update['micros'] for q in micro]
        assert len(flat)==3200 and len(set(flat))==3200 and len(s[role])==200
        for update in s[role]:
            assert len(update['micros'])==4 and all(len(m)==4 for m in update['micros'])
    assert s['control'][0]['micros'][0]==s['region'][0]['micros'][0]
