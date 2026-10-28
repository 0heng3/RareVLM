"""Native JSON, cached BLIP forward, and deterministic candidate metrics."""
import json
from pathlib import Path
import numpy as np
import torch
from vlm_probe.evaluate import write_json as original_write_json


def native(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,torch.Tensor):return value.detach().cpu().tolist()
    if isinstance(value,dict):return {k:native(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [native(v) for v in value]
    return value


def write_json(path,value):return original_write_json(path,native(value))


def setup_cuda():
    torch.cuda.set_device(0);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    return torch.device('cuda',0)


def load_matcher(cfg,device):
    from transformers import BlipForImageTextRetrieval
    model=BlipForImageTextRetrieval.from_pretrained(cfg['model_root'],local_files_only=True,
          torch_dtype=torch.bfloat16,use_safetensors=False).eval().to(device)
    model.requires_grad_(False)
    if model.config.vision_config.image_size!=384 or model.config.text_config.hidden_size!=768:
        raise ValueError('Unexpected pinned BLIP architecture')
    return model


def cached_logits(model,image_embeddings,input_ids,attention_mask):
    image_mask=torch.ones(image_embeddings.shape[:-1],dtype=torch.long,device=image_embeddings.device)
    text=model.text_encoder(input_ids=input_ids,attention_mask=attention_mask,
        encoder_hidden_states=image_embeddings,encoder_attention_mask=image_mask,return_dict=True)
    return model.itm_head(text.last_hidden_state[:,0,:]).float()


def pair_indices(candidates,direction):
    if direction=='I2T':
        return (np.arange(len(candidates))[:,None]*5000+candidates).reshape(-1)
    if direction=='T2I':
        return (candidates*5000+np.arange(len(candidates))[:,None]).reshape(-1)
    raise ValueError('Unknown retrieval direction')


def retrieve_pair_scores(keys,all_keys,all_scores):
    flat=np.asarray(keys).reshape(-1);where=np.searchsorted(all_keys,flat)
    if (where>=len(all_keys)).any() or not np.array_equal(all_keys[where],flat):
        raise ValueError('Missing candidate pair score')
    return all_scores[where].reshape(np.shape(keys))


def reorder(candidates,scores):
    if candidates.shape!=scores.shape or not np.isfinite(scores).all():raise ValueError('Invalid match scores')
    # Stable ties retain original CLIP rank inside its fixed candidate pool.
    return np.take_along_axis(candidates,np.argsort(-scores,axis=1,kind='stable'),axis=1)


def ranking_metrics(rankings,direction):
    n=len(rankings)
    relevant=rankings//5==np.arange(n)[:,None] if direction=='I2T' else rankings==(np.arange(n)//5)[:,None]
    metrics={f'{direction}_R@{k}':relevant[:,:k].any(1).astype(np.float64) for k in [1,5,10,50]}
    discount=1/np.log2(np.arange(10,dtype=np.float64)+2)
    ideal=discount[:5].sum() if direction=='I2T' else discount[0]
    metrics[direction+'_nDCG@10']=(relevant[:,:10]*discount).sum(1)/ideal
    return metrics
