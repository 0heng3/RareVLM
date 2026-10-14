"""Complete retrieval rankings: upstream hit-any recall and binary nDCG."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from vlm_probe.evaluate import now,write_json,sha_file,content_sha,lock_file
from .data import DEFAULT_CONFIG
from .train import DualAdapter


def rank_metrics(scores,positive,k_values=(1,5,10)):
    if scores.shape!=positive.shape or not bool(positive.any(dim=1).all()):raise ValueError('relevance matrix invalid')
    order=torch.argsort(scores,dim=1,descending=True,stable=True)
    relevant=torch.gather(positive,1,order)
    output={f'R@{k}':relevant[:,:min(k,scores.shape[1])].any(dim=1).float() for k in k_values}
    k=min(10,scores.shape[1]);discount=1/torch.log2(torch.arange(k,device=scores.device,dtype=torch.float32)+2)
    dcg=(relevant[:,:k].float()*discount).sum(dim=1)
    counts=positive.sum(dim=1).clamp(max=k)
    ideal=discount.cumsum(0)[counts.long()-1]
    output['nDCG@10']=dcg/ideal
    return output,order[:,:k]


def evaluate(config_path,condition,seed):
    cfg=json.loads(config_path.read_text());root=Path(cfg['output_root']);torch.cuda.set_device(0);device=torch.device('cuda',0)
    if condition=='frozen':model=None;checkpoint=None;key='frozen'
    else:
        checkpoint=root/'runs'/condition/f'seed{seed}'/'final.pt'
        result=json.loads((checkpoint.parent/'TRAIN_RESULT.json').read_text())
        if result['checkpoint_sha256']!=sha_file(checkpoint):raise ValueError('fixed Final@10 checkpoint changed')
        model=DualAdapter().to(device);model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True));model.eval();model.requires_grad_(False)
        key=f'{condition}_seed{seed}'
    for split in ['dev','test']:
        source=root/'features'/f'{split}.pt';feature= torch.load(source,map_location='cpu',weights_only=False)
        marker=json.loads(source.with_suffix('.json').read_text())
        if marker['features_sha256']!=sha_file(source):raise ValueError('evaluation features changed')
        directory=root/'evaluation'/key/split;directory.mkdir(parents=True,exist_ok=True)
        lock={'config_sha256':sha_file(config_path),'feature_sha256':marker['features_sha256'],
            'checkpoint_sha256':sha_file(checkpoint) if checkpoint else None,'condition':condition,'seed':seed,
            'recall':'hit at least one annotation-positive in topK','split':split}
        lock_file(directory/'INPUT_LOCK.json',lock)
        if (directory/'EVALUATION.json').exists():continue
        images=feature['image_features'].to(device);texts=feature['text_features'].to(device);n=len(images)
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
            'images':n,'captions':n*5,'metrics_percent':{name:100*float(np.mean(v)) for name,v in arrays.items() if 'top10' not in name},
            'per_query_sha256':sha_file(directory/'per_query.npz'),'input_lock_sha256':content_sha(lock),
            'offline_score_and_full_sort_seconds':elapsed,'pretraining_overlap':'unknown',
            'scope':'complete gallery,stable ties by supplied index;not ANN or online service'}
        write_json(directory/'EVALUATION.json',result);print('RETRIEVAL_COMPLETE',key,split,result['metrics_percent'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--condition',choices=['frozen','uniform_negatives','mixed_hard_negatives'],required=True)
    parser.add_argument('--seed',type=int,default=0);args=parser.parse_args();evaluate(args.config,args.condition,args.seed)
