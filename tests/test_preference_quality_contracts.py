from vlm_preference_data.quality import quality_check

def test_wifi_alias_cannot_be_a_factual_negative():
    r={'answers':['wi-fi']*8+['the wi-fi menu']*2}
    assert not quality_check(r,'wi-fi','wifi')['candidate_admitted']

def test_long_span_with_correct_core_excluded_conservatively():
    r={'answers':['citizens']*10}
    assert not quality_check(r,'citizens','the shit out of citizens')['candidate_admitted']

def test_annotation_disagreement_excluded():
    r={'answers':['2014']*5+['2011']*5}
    assert not quality_check(r,'2014','2016')['candidate_admitted']

def test_clear_different_word_can_be_candidate_not_human_verified():
    r={'answers':['aaa']*9+['a']}
    x=quality_check(r,'aaa','typewriter')
    assert x['candidate_admitted'] and not x['semantic_human_preference_verified']
