"""CPU-only synthetic execution check; these numbers are not research results."""
import argparse
import json
from pathlib import Path

from project_paths import ROOT
import torch
import torch.nn.functional as F
from calibrate import run as calibrate
from imagenet_lt_manifest import content_sha256
from methods import PrototypeClassifier,ResidualAdapter
from run_imagenet_lt import train_one,sha256_file


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=ROOT/'artifacts/smoke')
    args=p.parse_args();out=args.output_dir
    out.mkdir(parents=True,exist_ok=True);features=out/'features';features.mkdir(exist_ok=True)
    torch.set_num_threads(2);torch.manual_seed(2026)
    counts=[160,80,30,10,5,3];n_classes=len(counts)
    prototypes=F.normalize(torch.randn(n_classes,512),dim=1)
    classifier=PrototypeClassifier(prototypes,10.).eval()
    labels={'train':torch.repeat_interleave(torch.arange(n_classes),torch.tensor(counts)),
            'val':torch.arange(n_classes).repeat_interleave(2),'test':torch.arange(n_classes).repeat_interleave(2)}
    banks={k:(F.normalize(prototypes[y]+.2*torch.randn(len(y),512),dim=1),y) for k,y in labels.items()}
    m={'schema_version':'rarevlm.synthetic.smoke.v1','class_counts':counts,
       'splits':{k:[{'path':f'synthetic/{k}/{i}','label':int(y)} for i,y in enumerate(values)] for k,values in labels.items()},
       'groups':{'many':[0],'medium':[1,2],'few':[3,4,5]},'split_semantics':{'test':'synthetic smoke only'}}
    m['manifest_sha256']=content_sha256(m)
    manifest_path=out/'manifest.json';manifest_path.write_text(json.dumps(m)+'\n')
    for k,(x,y) in banks.items():
        torch.save({'features':x.half(),'targets':y,'split':k,'manifest_sha256':m['manifest_sha256']},features/f'{k}.pt')
        banks[k]=(x.half().float(),y)
    proto={'text_prototypes':prototypes,'logit_scale':10.,'manifest_sha256':m['manifest_sha256'],'model_weight_sha256':'synthetic'}
    torch.save(proto,features/'text_prototypes.pt')
    meta={'manifest_sha256':m['manifest_sha256'],'model_weight_sha256':'synthetic'}
    (features/'metadata.json').write_text(json.dumps(meta)+'\n')
    metadata={**meta,'feature_artifact_sha256':{name:sha256_file(features/name) for name in ('train.pt','val.pt','test.pt','text_prototypes.pt')}}
    adapter=ResidualAdapter(512,64)
    assert torch.equal(adapter(banks['val'][0]),banks['val'][0])
    assert sum(p.numel() for p in adapter.parameters())==66240
    original_prototypes=classifier.text_prototypes.clone()
    runs=out/'main';runs.mkdir(exist_ok=True)
    for method in ['ce','la']:
        r=train_one(seed=0,method=method,classifier=classifier,train_x=banks['train'][0],train_y=banks['train'][1],
                    val_x=banks['val'][0],val_y=banks['val'][1],test_x=banks['test'][0],test_y=banks['test'][1],
                    manifest=m,output_dir=runs,epochs=2,batch_size=32,lr=.001,weight_decay=.01,bank_metadata=metadata)
        assert len(r['history'])==2 and 0<=r['test']['oa']<=1
        controls=calibrate(manifest_path,features,runs/f'imagenet_lt_seed0_{method}.json',out/'controls','cpu',32)
        assert all(not x['representation_updated'] for x in controls.values())
        assert torch.equal(original_prototypes,classifier.text_prototypes)
    print('PASS: identity adapter, 66,240 parameters, CE/LA training, validation selection, frozen-prototype and fixed-checkpoint controls. Synthetic data only.')


if __name__=='__main__':main()
