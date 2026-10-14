"""Replace a dense block mask with equal-length segment batches, preserving RoPE.

This is a known attention execution transformation, not a new trained attention
mechanism. FP kernel dispatch can differ; real feature and QA checks are required.
"""
from __future__ import annotations
from collections import OrderedDict,defaultdict
import types
import numpy as np
import torch
from torch.nn import functional as F
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_rotary_pos_emb_vision

LAYOUT_CACHE=OrderedDict()
MAX_LAYOUTS=64


def layouts(boundaries,seq_length,device):
    bounds=tuple(int(v) for v in boundaries)
    if len(bounds)<2 or bounds[0]!=0 or bounds[-1]!=seq_length or any(b<a for a,b in zip(bounds,bounds[1:])):
        raise ValueError('cu_seqlens must partition the entire input')
    key=(bounds,str(device))
    if key in LAYOUT_CACHE:
        LAYOUT_CACHE.move_to_end(key);return LAYOUT_CACHE[key]
    groups=defaultdict(list)
    for a,b in zip(bounds,bounds[1:]):
        if b>a:groups[b-a].append((a,b))
    result=[]
    for length,segments in sorted(groups.items()):
        indices=np.concatenate([np.arange(a,b,dtype=np.int64) for a,b in segments])
        result.append((length,len(segments),torch.from_numpy(indices).to(device=device)))
    LAYOUT_CACHE[key]=result
    if len(LAYOUT_CACHE)>MAX_LAYOUTS:LAYOUT_CACHE.popitem(last=False)
    return result


def segmented_forward(self,hidden_states,cu_seqlens,rotary_pos_emb=None,position_embeddings=None):
    if self.training:raise ValueError('segmented study is frozen-model inference only')
    n=hidden_states.shape[0]
    q,k,v=self.qkv(hidden_states).reshape(n,3,self.num_heads,-1).permute(1,0,2,3).unbind(0)
    if position_embeddings is None:
        if rotary_pos_emb is None:raise ValueError('RoPE embeddings missing')
        embedding=torch.cat((rotary_pos_emb,rotary_pos_emb),dim=-1)
        cos,sin=embedding.cos(),embedding.sin()
    else:cos,sin=position_embeddings
    q,k=apply_rotary_pos_emb_vision(q,k,cos,sin)
    out=torch.empty_like(q)
    bounds=cu_seqlens.detach().cpu().tolist()
    for length,count,indices in layouts(bounds,n,q.device):
        shape=(count,length,self.num_heads,q.shape[-1])
        qs=q.index_select(0,indices).reshape(shape).transpose(1,2)
        ks=k.index_select(0,indices).reshape(shape).transpose(1,2)
        vs=v.index_select(0,indices).reshape(shape).transpose(1,2)
        result=F.scaled_dot_product_attention(qs,ks,vs,attn_mask=None,dropout_p=0.0,is_causal=False)
        restored=result.transpose(1,2).reshape(-1,self.num_heads,q.shape[-1])
        out.index_copy_(0,indices,restored)
    return self.proj(out.reshape(n,-1))


def attention_modules(model):
    modules={name:m for name,m in model.named_modules() if m.__class__.__name__=='Qwen2_5_VLVisionSdpaAttention'}
    if len(modules)!=32:raise ValueError(f'expected32 visual SDPA layers,got{len(modules)}')
    if any('.visual.' not in name for name in modules):raise ValueError('nonvisual attention selected')
    return modules


def set_segmented(model,enabled):
    LAYOUT_CACHE.clear()
    for module in attention_modules(model).values():
        if not hasattr(module,'_rarevlm_original_forward'):module._rarevlm_original_forward=module.forward
        module.forward=types.MethodType(segmented_forward,module) if enabled else module._rarevlm_original_forward
