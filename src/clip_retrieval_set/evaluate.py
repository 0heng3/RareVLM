"""Same full-gallery ranking contract, dev first and gated historical test."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch

from clip_retrieval.evaluate import rank_metrics
from clip_retrieval.train import DualAdapter
from vlm_joint.evaluate import visible_gpu_binding
from vlm_probe.evaluate import content_sha, lock_file, now, package_versions, sha_file, write_json


def evaluate(config_path,condition,seed,split):
    cfg=json.loads(config_path.read_text());root=Path(cfg['output_root'])
    if split not in ['dev','test'] or condition not in ['frozen',*cfg['conditions']]:raise ValueError('Unregistered evaluation')
    if split=='test':
        gate=json.loads((root/'analysis/dev/ANALYSIS.json').read_text())
        if not gate['gate']['passed']:raise ValueError('Dev gate failed; no new test evaluation')
    torch.cuda.set_device(0);torch.set_num_threads(2);device=torch.device('cuda',0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    checkpoint=None;model=None;key='frozen'
    if condition!='frozen':
        checkpoint=root/'runs'/condition/f'seed{seed}'/'final.pt'
        train_result=json.loads((checkpoint.parent/'TRAIN_RESULT.json').read_text())
        if train_result['checkpoint_sha256']!=sha_file(checkpoint):raise ValueError('Final checkpoint changed')
        model=DualAdapter().to(device);model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True))
        model.eval();model.requires_grad_(False);key=f'{condition}_seed{seed}'
    source=root/'features'/f'{split}.pt';marker=json.loads(source.with_suffix('.json').read_text())
    if marker['features_sha256']!=sha_file(source):raise ValueError('Frozen evaluation cache changed')
    feature=torch.load(source,map_location='cpu',weights_only=False)
    directory=root/'evaluation'/key/split;directory.mkdir(parents=True,exist_ok=True)
    lock={'config_sha256':sha_file(config_path),'feature_sha256':marker['features_sha256'],
        'checkpoint_sha256':sha_file(checkpoint) if checkpoint else None,'condition':condition,'seed':seed,'split':split,
        'hardware':visible_gpu_binding(),'packages':package_versions(),'tf32':False,'rank_ties':'stable supplied index'}
    lock_file(directory/'INPUT_LOCK.json',lock)
    if (directory/'EVALUATION.json').exists():
        previous=json.loads((directory/'EVALUATION.json').read_text())
        if previous['per_query_sha256']!=sha_file(directory/'per_query.npz'):raise ValueError('Completed rankings changed')
        return previous
    images=feature['image_features'].to(device);texts=feature['text_features'].to(device);n=len(images)
    if n!=1000 or len(texts)!=5000:raise ValueError('Complete 1000-image/5000-caption gallery required')
    with torch.inference_mode():
        if model is not None:images=model.image_features(images);texts=model.text_features(texts)
        torch.cuda.synchronize();start=time.monotonic();scores=texts@images.T
        target=torch.arange(n,device=device).repeat_interleave(5)
        positive=target[:,None]==torch.arange(n,device=device)[None,:]
        t2i,t2i_order=rank_metrics(scores,positive);i2t,i2t_order=rank_metrics(scores.T,positive.T)
        torch.cuda.synchronize();elapsed=time.monotonic()-start
    arrays={f'T2I_{name}':v.cpu().numpy() for name,v in t2i.items()}
    arrays.update({f'I2T_{name}':v.cpu().numpy() for name,v in i2t.items()})
    arrays.update(t2i_top10=t2i_order.cpu().numpy(),i2t_top10=i2t_order.cpu().numpy())
    np.savez(directory/'per_query.npz',**arrays)
    result={'status':'completed','completed_at':now(),'condition':condition,'seed':seed,'split':split,
        'images':n,'captions':5000,'metrics_percent':{name:100*float(np.mean(v)) for name,v in arrays.items() if 'top10' not in name},
        'per_query_sha256':sha_file(directory/'per_query.npz'),'input_lock_sha256':content_sha(lock),
        'offline_score_and_full_sort_seconds':elapsed,'pretraining_overlap':'unknown',
        'scope':'complete same gallery,hit-any Recall,stable ties; historical test exposed,not blind or online'}
    write_json(directory/'EVALUATION.json',result);print('EVALUATED',key,split,result['metrics_percent'],flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--condition',required=True);parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--split',choices=['dev','test'],required=True)
    args=parser.parse_args();evaluate(args.config,args.condition,args.seed,args.split)
