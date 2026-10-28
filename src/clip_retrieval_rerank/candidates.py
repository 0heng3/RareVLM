"""Reproduce baseline ranks, then form the exact shared Top50 pair union."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from clip_retrieval.train import DualAdapter
from vlm_joint.evaluate import visible_gpu_binding
from vlm_probe.evaluate import sha_file,lock_file,now,package_versions
from .common import setup_cuda,pair_indices,write_json


def build(config_path,split):
    cfg=json.loads(config_path.read_text());root=Path(cfg['output_root']);old=Path(cfg['baseline_root'])
    if split=='test' and not json.loads((root/'analysis/dev/ANALYSIS.json').read_text())['gate']['passed']:
        raise ValueError('No test before dev gate')
    directory=root/'candidates'/split;directory.mkdir(parents=True,exist_ok=True)
    device=setup_cuda();feature_path=Path(cfg['feature_root'])/f'features/{split}.pt'
    marker=json.loads(feature_path.with_suffix('.json').read_text())
    if sha_file(feature_path)!=marker['features_sha256']:raise ValueError('Gallery cache changed')
    records=json.loads((root/'REUSED_BASELINES.json').read_text())
    identity={'config_sha256':sha_file(config_path),'feature_sha256':marker['features_sha256'],
        'checkpoints':records['checkpoints'],'hardware':visible_gpu_binding(),'source_sha256':sha_file(Path(__file__)),
        'packages':package_versions(),'tf32':False}
    lock_file(directory/'INPUT_LOCK.json',identity)
    if (directory/'CANDIDATES.json').exists():
        result=json.loads((directory/'CANDIDATES.json').read_text())
        for name,digest in result['files'].items():
            if sha_file(directory/name)!=digest:raise ValueError('Completed candidate pool changed')
        return result
    feature=torch.load(feature_path,map_location='cpu',weights_only=False)
    images=feature['image_features'].to(device);texts=feature['text_features'].to(device)
    if len(images)!=1000 or len(texts)!=5000:raise ValueError('Full gallery required')
    keys=[];records_out=[];started=time.monotonic()
    with torch.inference_mode():
        for seed in cfg['training_seeds']:
            checkpoint=old/f'runs/distill/seed{seed}/final.pt'
            if sha_file(checkpoint)!=records['checkpoints'][str(seed)]['checkpoint_sha256']:raise ValueError('Adapter changed')
            model=DualAdapter().to(device);model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True))
            model.eval();model.requires_grad_(False)
            torch.cuda.synchronize();begin=time.monotonic()
            image=model.image_features(images);text=model.text_features(texts);score=text@image.T
            t2i=torch.argsort(score,dim=1,descending=True,stable=True)[:,:cfg['candidate_k']].cpu().numpy()
            i2t=torch.argsort(score.T,dim=1,descending=True,stable=True)[:,:cfg['candidate_k']].cpu().numpy()
            torch.cuda.synchronize();seconds=time.monotonic()-begin
            with np.load(old/f'evaluation/distill_seed{seed}/{split}/per_query.npz') as original:
                if not np.array_equal(t2i[:,:10],original['t2i_top10']) or not np.array_equal(i2t[:,:10],original['i2t_top10']):
                    raise ValueError('Top10 does not exactly reproduce historical baseline')
            np.savez(directory/f'seed{seed}.npz',t2i=t2i,i2t=i2t)
            keys.extend([pair_indices(t2i,'T2I'),pair_indices(i2t,'I2T')])
            records_out.append({'seed':seed,'adapter_and_full_score_sort_seconds':seconds,'baseline_top10_exact':True})
            del model,image,text,score
    unique=np.unique(np.concatenate(keys));np.save(directory/'pair_keys.npy',unique)
    files={name:sha_file(directory/name) for name in ['pair_keys.npy',*[f'seed{s}.npz' for s in cfg['training_seeds']]]}
    result={'status':'completed','completed_at':now(),'split':split,'candidate_k':cfg['candidate_k'],
        'logical_pairs':sum(len(k) for k in keys),'unique_pairs':len(unique),'files':files,
        'baseline_reproduction':records_out,'candidate_stage_seconds':time.monotonic()-started,
        'scope':'exact Top10 old baseline; shared complete Top50 union; no target-based selection'}
    write_json(directory/'CANDIDATES.json',result);print('CANDIDATES_READY',split,len(unique),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--split',choices=['dev','test'],required=True)
    a=p.parse_args();build(a.config,a.split)
