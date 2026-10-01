import json
import numpy as np
import pytest
import torch
from calibrate import run
from methods import PrototypeClassifier,ResidualAdapter
from run_imagenet_lt import sha256_file


def test_controls_keep_checkpoint_and_representation_fixed(calibration_case):
    c=calibration_case;before=sha256_file(c['checkpoint'])
    checkpoint=torch.load(c['checkpoint'],weights_only=True)
    adapter=ResidualAdapter(512,64).eval();adapter.load_state_dict(checkpoint['adapter_state_dict'])
    x=torch.load(c['features']/'val.pt',weights_only=True)['features'].float()
    with torch.inference_mode():representation=adapter(x).clone()
    rows=run(c['manifest'],c['features'],c['source'],c['output'],'cpu',32)
    assert sha256_file(c['checkpoint'])==before
    for row in rows.values():
        assert row['source_checkpoint_sha256']==before and row['representation_updated'] is False
    adapter.load_state_dict(torch.load(c['checkpoint'],weights_only=True)['adapter_state_dict'])
    with torch.inference_mode():assert torch.equal(adapter(x),representation)


def test_final_feature_tamper_is_rejected(calibration_case):
    c=calibration_case
    proto=torch.load(c['features']/'text_prototypes.pt',weights_only=True)
    classifier=PrototypeClassifier(proto['text_prototypes'],proto['logit_scale'])
    rows=run(c['manifest'],c['features'],c['source'],c['output'],'cpu',32)
    selected=rows['val_alpha']['selection']['alpha']
    payload=torch.load(c['features']/'test.pt',weights_only=True)
    payload['targets']=(payload['targets']+1)%3
    torch.save(payload,c['features']/'test.pt')
    manifest=json.loads(c['manifest'].read_text())
    # A final-label mutation must not silently enter calibration: provenance rejects it.
    with pytest.raises(ValueError,match='feature artifact changed'):
        run(c['manifest'],c['features'],c['source'],c['output'],'cpu',32)
    assert selected==max(rows['val_alpha']['selection']['curve'],key=lambda r:(r['val_oa'],-r['alpha']))['alpha']


def test_selected_alpha_is_invariant_to_valid_final_feature_changes(calibration_case):
    c=calibration_case
    first=run(c['manifest'],c['features'],c['source'],c['output'],'cpu',32)
    payload=torch.load(c['features']/'test.pt',weights_only=True)
    payload['features']=payload['features'].roll(2,dims=0)
    torch.save(payload,c['features']/'test.pt')
    source=json.loads(c['source'].read_text())
    source['bank_metadata']['feature_artifact_sha256']['test.pt']=sha256_file(c['features']/'test.pt')
    c['source'].write_text(json.dumps(source))
    second=run(c['manifest'],c['features'],c['source'],c['output'].with_name('alternate-controls'),'cpu',32)
    assert first['val_alpha']['selection']==second['val_alpha']['selection']
    assert first['val_alpha']['test']['oa']!=second['val_alpha']['test']['oa']


def test_checkpoint_tamper_is_rejected(calibration_case):
    c=calibration_case
    with c['checkpoint'].open('ab') as f:f.write(b'tamper')
    with pytest.raises(ValueError,match='source run provenance'):
        run(c['manifest'],c['features'],c['source'],c['output'],'cpu',32)


def test_effective_prior_uses_actual_train_predictions(calibration_case):
    c=calibration_case;run(c['manifest'],c['features'],c['source'],c['output'],'cpu',32)
    proto=torch.load(c['features']/'text_prototypes.pt',weights_only=True)
    x=torch.load(c['features']/'train.pt',weights_only=True)['features'].float()
    classifier=PrototypeClassifier(proto['text_prototypes'],proto['logit_scale'])
    with torch.inference_mode():expected=classifier(x).double().softmax(1).mean(0).numpy()
    actual=np.load(c['output']/'case_prior.npz')['effective_prior']
    # The classifier produces float32 logits: GEMM batch shape can change their
    # last bits even though the posterior accumulation itself is float64.
    assert np.allclose(actual,expected,rtol=1e-6,atol=1e-8)
    assert actual.sum()==pytest.approx(1.,abs=1e-12)
