"""Recompute FP32 CLIP scores, verify the entire old Top50 and store margins."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from clip_retrieval.train import DualAdapter
from clip_retrieval_rerank.common import setup_cuda,write_json
from vlm_joint.evaluate import visible_gpu_binding
from vlm_probe.evaluate import lock_file,sha_file,now


def build(config_path,split):
    cfg=json.loads(config_path.read_text());root=Path(cfg['output_root']);old=Path(cfg['rerank_root'])
    if split=='test' and not json.loads((root/'analysis/dev/ANALYSIS.json').read_text())['gate']['passed']:raise ValueError('No test before joint dev gate')
    device=setup_cuda();d=root/'confidence'/split;d.mkdir(parents=True,exist_ok=True)
    assets=json.loads((root/'REUSED_ASSETS.json').read_text())
    identity={'config_sha256':sha_file(config_path),'assets_sha256':sha_file(root/'REUSED_ASSETS.json'),
        'feature_sha256':assets['features'][split]['features_sha256'],'hardware':visible_gpu_binding(),'source_sha256':sha_file(Path(__file__))}
    lock_file(d/'INPUT_LOCK.json',identity)
    if (d/'CONFIDENCE.json').exists():
        o=json.loads((d/'CONFIDENCE.json').read_text())
        if any(sha_file(d/k)!=v for k,v in o['files'].items()):raise ValueError('Confidence checkpoint changed')
        return o
    p=Path(cfg['feature_root'])/'features'/f'{split}.pt'
    if sha_file(p)!=assets['features'][split]['features_sha256']:raise ValueError('Gallery changed')
    feature=torch.load(p,map_location='cpu',weights_only=False)
    images=feature['image_features'].to(device);texts=feature['text_features'].to(device);records=[]
    started=time.monotonic()
    with torch.inference_mode():
        for seed in cfg['training_seeds']:
            checkpoint=Path(assets['checkpoints'][str(seed)]['source'])/'final.pt'
            if sha_file(checkpoint)!=assets['checkpoints'][str(seed)]['checkpoint_sha256']:raise ValueError('Bound model changed')
            model=DualAdapter().to(device);model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True));model.eval()
            torch.cuda.synchronize();stamp=time.monotonic();score=model.text_features(texts)@model.image_features(images).T
            outputs={}
            with np.load(old/'candidates'/split/f'seed{seed}.npz') as z:
                for direction,matrix in [('T2I',score),('I2T',score.T)]:
                    order=torch.argsort(matrix,dim=1,descending=True,stable=True)[:,:50]
                    expected=z[direction.lower()]
                    if not np.array_equal(order.cpu().numpy(),expected):raise ValueError('Full historical Top50 not reproduced')
                    top=matrix.gather(1,order[:,:2]);outputs[direction+'_margin']=(top[:,0]-top[:,1]).cpu().numpy()
                    outputs[direction+'_top1_cosine']=top[:,0].cpu().numpy();outputs[direction+'_top2_cosine']=top[:,1].cpu().numpy()
            torch.cuda.synchronize();records.append({'seed':seed,'full_top50_exact':True,'clip_score_sort_margin_seconds':time.monotonic()-stamp})
            np.savez(d/f'seed{seed}.npz',**outputs)
            del model,score
    result={'status':'completed','completed_at':now(),'split':split,'records':records,'total_stage_seconds':time.monotonic()-started,
        'files':{f'seed{s}.npz':sha_file(d/f'seed{s}.npz') for s in cfg['training_seeds']},
        'scope':'full-gallery FP32 margins,no target use;complete old Top50 exact'}
    write_json(d/'CONFIDENCE.json',result);print('CONFIDENCE_READY',split,flush=True);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--split',choices=['dev','test'],required=True)
    a=p.parse_args();build(a.config,a.split)
