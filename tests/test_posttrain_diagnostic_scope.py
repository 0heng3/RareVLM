import hashlib
import json
from pathlib import Path
from PIL import Image
import pytest
from vlm_posttrain.train import read_training_rows
from vlm_posttrain_study.branch import train_arguments
from vlm_posttrain_study.pipeline import publication_decision

def fixture_files(tmp_path, scope, gate, promotion):
    image=tmp_path/'im.png';Image.new('RGB',(2,2),'red').save(image)
    rgb=Image.open(image).convert('RGB');pixel=hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
    manifest=tmp_path/'train.json';manifest.write_text(json.dumps({'rows':[{'question_id':'q','image_id':'im','question':'What color?',
                              'chosen':'red','rejected':'blue','image_path':str(image),'pixel_sha256':pixel}]}))
    admission=tmp_path/'admission.json';admission.write_text(json.dumps({'status':'admitted','scope':scope,
        'train_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'factual_quality_gate_passed':gate,'automatic_promotion_allowed':promotion}))
    return {'train_manifest':str(manifest),'data_admission':str(admission),'prompt_suffix':'','study_scope':scope}

def test_noise_scope_preserves_failed_factual_gate_and_cannot_promote(tmp_path):
    cfg=fixture_files(tmp_path,'noisy_public_preference_diagnostic',False,False)
    rows,_=read_training_rows(cfg);assert len(rows)==1
    decision=publication_decision(json.loads(Path(cfg['data_admission']).read_text()),{'all_metrics_passed':True})
    assert not decision['automatic_promotion_allowed'] and not decision['factual_quality_gate_passed']
    with pytest.raises(ValueError,match='noise diagnostic'):
        read_training_rows({**cfg,'study_scope':'qualified_factual_preference'})

def test_qualified_route_cannot_silently_use_failed_data(tmp_path):
    cfg=fixture_files(tmp_path,'qualified_factual_preference',False,False)
    with pytest.raises(ValueError,match='quality gate'):read_training_rows(cfg)

def test_diagnostic_must_forbid_promotion(tmp_path):
    cfg=fixture_files(tmp_path,'noisy_public_preference_diagnostic',False,True)
    with pytest.raises(ValueError,match='prohibit promotion'):read_training_rows(cfg)

def test_branch_resume_keeps_same_scientific_arguments(tmp_path):
    cfg=tmp_path/'config.json'
    initial=train_arguments(cfg,tmp_path,'dpo',42)
    assert '--resume' not in initial
    checkpoint=tmp_path/'runs/dpo/seed42/train/LATEST.json';checkpoint.parent.mkdir(parents=True);checkpoint.write_text('{}')
    assert train_arguments(cfg,tmp_path,'dpo',42)==initial+['--resume']
