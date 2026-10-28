import pytest
from vlm_preference_review.audit import audit_gate, fixed_sample, wilson
from vlm_preference_review.repair import pair_reasons, repair_pair

def row(question='what brand is on the bottle?', answers=None):
    return {'question_id':'1','image_id':'x','question':question,'answers':answers or ['coca cola']*10}

def test_alias_is_not_negative():
    assert 'known_semantic_alias' in pair_reasons(row(), 'coca cola', 'coke')

def test_reverse_partial_span_is_screened():
    r=row('what does the shop sign say?', ['peace tea']*10)
    assert 'bidirectional_reference_span_overlap' in pair_reasons(r,'peace tea','peace')

def test_broad_question_is_not_literal():
    r=row('what is this a picture of?', ['broken meter']*10)
    assert 'outside_literal_reading_question_policy' in pair_reasons(r,'broken meter','parking meter')

def test_remine_uses_another_real_response():
    r=row();rec={'samples':[{'prediction':x} for x in ['coca cola','coke','pepsi','sprite']],
                 'rewards':[1.,0.,0.,0.], 'input_lock_sha256':'fixed'}
    pair=repair_pair(r,rec)
    assert pair['chosen_index']==0 and pair['rejected_index']==2
    assert pair['rejected']=='pepsi' and not pair['semantic_human_preference_verified']

def test_confirm_excludes_design_and_is_order_invariant():
    pairs=[{'question_id':str(i)} for i in range(100)]
    a=fixed_sample(pairs,'fixed:',64,exclude=['0','1'])
    assert a==fixed_sample(list(reversed(pairs)),'fixed:',64,exclude=['0','1'])
    assert not {'0','1'} & {x['question_id'] for x in a}

def test_uncertain_does_not_pass_gate():
    records=[{'question_id':str(i),'category':'clear_visual_preference' if i<50 else 'unresolved_visual_detail',
              'note':'actual observation'} for i in range(64)]
    plan={'sample_size':64,'selected_question_ids':[x['question_id'] for x in records],
          'clear_fraction_min':.9,'wilson95_lower_min':.8}
    result=audit_gate(records,plan)
    assert result['clear']==50 and not result['gate_passed']
    assert .66<result['wilson95'][0]<.67
    assert wilson(64,64)[0]>.94
    with pytest.raises(ValueError):audit_gate(records[:-1],plan)
