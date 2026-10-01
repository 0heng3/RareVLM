import json
import os
from pathlib import Path

os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION','python')
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')

import pytest
import torch
import torch.nn.functional as F
from imagenet_lt_manifest import content_sha256
from methods import ResidualAdapter
from run_imagenet_lt import sha256_file


@pytest.fixture
def calibration_case(tmp_path):
    torch.set_num_threads(2);torch.manual_seed(2026)
    counts=[120,40,5];prototypes=F.normalize(torch.randn(3,512),dim=1)
    labels={'train':torch.repeat_interleave(torch.arange(3),torch.tensor(counts)),
            'val':torch.arange(3).repeat_interleave(2),'test':torch.arange(3).repeat_interleave(2)}
    manifest={'schema_version':'rarevlm.synthetic.test.v1','class_counts':counts,
              'groups':{'many':[0],'medium':[1],'few':[2]},
              'splits':{k:[{'path':f'synthetic/{k}/{i}','label':int(v)} for i,v in enumerate(y)] for k,y in labels.items()}}
    manifest['manifest_sha256']=content_sha256(manifest)
    path=tmp_path/'manifest.json';path.write_text(json.dumps(manifest))
    bank=tmp_path/'features';bank.mkdir()
    for split,y in labels.items():
        x=F.normalize(prototypes[y]+.2*torch.randn(len(y),512),dim=1)
        torch.save({'features':x.half(),'targets':y,'split':split,'manifest_sha256':manifest['manifest_sha256']},bank/f'{split}.pt')
    torch.save({'text_prototypes':prototypes,'logit_scale':10.,'manifest_sha256':manifest['manifest_sha256'],
                'model_weight_sha256':'synthetic-model'},bank/'text_prototypes.pt')
    metadata={'manifest_sha256':manifest['manifest_sha256'],'model_weight_sha256':'synthetic-model'}
    (bank/'metadata.json').write_text(json.dumps(metadata))
    adapter=ResidualAdapter(512,64)
    checkpoint=tmp_path/'case_adapter.pt'
    torch.save({'adapter_state_dict':adapter.state_dict(),'manifest_sha256':manifest['manifest_sha256']},checkpoint)
    source=tmp_path/'case.json'
    source.write_text(json.dumps({'run_id':'synthetic-case','method':'ce','train_seed':0,
         'manifest_sha256':manifest['manifest_sha256'],'checkpoint_sha256':sha256_file(checkpoint),
         'bank_metadata':{**metadata,'feature_artifact_sha256':{n:sha256_file(bank/n) for n in ('train.pt','val.pt','test.pt','text_prototypes.pt')}}}))
    return {'manifest':path,'features':bank,'source':source,'checkpoint':checkpoint,'output':tmp_path/'controls'}
