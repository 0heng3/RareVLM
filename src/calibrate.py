"""Fixed-checkpoint empirical and train-only effective-prior controls.

Supports cached ImageNet-LT B/32/B/16, online-trained adapters evaluated on
center-crop caches, and iNaturalist banks. No adapter or prototype updates.
"""
import argparse
import json
import math
from pathlib import Path

import torch
from project_paths import ROOT
from imagenet_lt_manifest import content_sha256
from methods import PrototypeClassifier, ResidualAdapter
from run_imagenet_lt import classify_metrics, load_bank, save_json, sha256_file

ALPHAS=(0.,.25,.5,.75,1.,1.5,2.)


@torch.inference_mode()
def run(manifest_path, feature_dir, source_result, output_dir, device='cpu', batch_size=256):
    m=json.loads(manifest_path.read_text());digest=m['manifest_sha256']
    if content_sha256({k:v for k,v in m.items() if k!='manifest_sha256'})!=digest: raise ValueError('manifest hash mismatch')
    result=json.loads(source_result.read_text())
    checkpoint_path=source_result.with_name(source_result.stem+'_adapter.pt')
    source_hash=sha256_file(checkpoint_path)
    if result['manifest_sha256']!=digest or result['checkpoint_sha256']!=source_hash: raise ValueError('source run provenance mismatch')
    method=result['method'].split('_')[-1]
    if method not in ('ce','la'): raise ValueError('requires CE or LA adapter')
    metadata=json.loads((feature_dir/'metadata.json').read_text())
    proto=torch.load(feature_dir/'text_prototypes.pt',map_location='cpu',weights_only=True)
    if metadata['manifest_sha256']!=digest or proto['manifest_sha256']!=digest: raise ValueError('feature provenance mismatch')
    if metadata.get('model_weight_sha256')!=proto.get('model_weight_sha256'): raise ValueError('encoder weight mismatch')
    feature_hashes={n:sha256_file(feature_dir/n) for n in ('train.pt','val.pt','test.pt','text_prototypes.pt')}
    for name,value in result['bank_metadata'].get('feature_artifact_sha256',{}).items():
        if name in feature_hashes and feature_hashes[name]!=value: raise ValueError('source feature artifact changed')
    counts=torch.tensor(m['class_counts'],dtype=torch.float64,device=device)
    if bool((counts<=0).any()) or not torch.isfinite(counts).all(): raise ValueError('invalid train counts')
    n_classes=len(counts)
    if proto['text_prototypes'].shape!=(n_classes,512): raise ValueError('prototype shape mismatch')
    pi=counts/counts.sum();log_pi=pi.log().float()
    classifier=PrototypeClassifier(proto['text_prototypes'],proto['logit_scale']).to(device).eval()
    adapter=ResidualAdapter(512,64).to(device).eval()
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    if checkpoint['manifest_sha256']!=digest: raise ValueError('checkpoint manifest mismatch')
    adapter.load_state_dict(checkpoint['adapter_state_dict'])
    output_dir.mkdir(parents=True,exist_ok=True)
    x_val,y_val=load_bank(feature_dir/'val.pt','val',digest,m['splits']['val'],device)
    val_logits=classifier(adapter(x_val))
    curve=[{'alpha':a,'val_oa':float((val_logits-a*log_pi).argmax(1).eq(y_val).float().mean())} for a in ALPHAS]
    selected=max(curve,key=lambda x:(x['val_oa'],-x['alpha']))['alpha']
    selection={'alpha':selected,'curve':curve,'source_checkpoint_sha256':source_hash,'manifest_sha256':digest,'selection_split':'internal validation only'}
    selection_path=output_dir/(source_result.stem+'_validation_selection.json')
    if selection_path.exists() and json.loads(selection_path.read_text())!=selection: raise ValueError('existing validation selection differs')
    save_json(selection_path,selection)
    # Estimate q without reading final-evaluation features or labels.
    x_train,_=load_bank(feature_dir/'train.pt','train',digest,m['splits']['train'],'cpu')
    q=torch.zeros(n_classes,dtype=torch.float64,device=device)
    for start in range(0,len(x_train),batch_size):
        z=classifier(adapter(x_train[start:start+batch_size].to(device)))
        if method=='la': z=z+log_pi
        q+=z.double().softmax(1).sum(0)
    q/=len(x_train)
    if method=='la': q=q/pi;q/=q.sum()
    if not torch.isfinite(q).all() or (q<=0).any(): raise ValueError('invalid effective prior')
    import numpy as np
    np.savez_compressed(output_dir/(source_result.stem+'_prior.npz'),effective_prior=q.cpu().numpy(),empirical_prior=pi.cpu().numpy())
    x,y=load_bank(feature_dir/'test.pt','test',digest,m['splits']['test'],device)
    adapted=adapter(x);logits=classifier(adapted)
    offsets={'p2p_train_only':-q.log().float()-math.log(n_classes)}
    if method=='ce': offsets={'alpha1':-log_pi,'val_alpha':-selected*log_pi,**offsets}
    rows={}
    for name,offset in offsets.items():
        metrics,pred=classify_metrics(logits+offset,adapted,x,y,classifier.text_prototypes,m['groups'])
        pred_path=output_dir/(source_result.stem+'_'+name+'_pred.npz')
        np.savez_compressed(pred_path,predictions=pred,targets=y.cpu().numpy().astype(np.uint16))
        row={'method':method+'_'+name,'source_run':result['run_id'],'train_seed':result['train_seed'],
             'manifest_sha256':digest,'source_checkpoint_sha256':source_hash,'feature_artifact_sha256':feature_hashes,
             'representation_updated':False,'test':metrics,'prediction_sha256':sha256_file(pred_path),
             'selection':selection if name=='val_alpha' else {'strength':1.},
             'final_split_semantics':m.get('split_semantics',{}).get('test','official ImageNet-LT test')}
        save_json(output_dir/(source_result.stem+'_'+name+'.json'),row)
        rows[name]=row
        print(name,'OA',round(100*metrics['oa'],4),'Few',round(100*metrics['groups']['few']['accuracy'],4))
    if sha256_file(checkpoint_path)!=source_hash: raise ValueError('source checkpoint changed')
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--feature-dir',type=Path,required=True)
    p.add_argument('--source-result',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--batch-size',type=int,default=256)
    args=p.parse_args()
    if args.batch_size<1:p.error('batch size must be positive')
    torch.set_num_threads(4)
    run(args.manifest,args.feature_dir,args.source_result,args.output_dir,args.device,args.batch_size)


if __name__=='__main__':main()
