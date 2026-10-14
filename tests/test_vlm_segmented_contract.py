"""Small mathematical checks; not experiment performance or real-QA evidence."""
import torch
import pytest
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionSdpaAttention
from vlm_segmented.attention import segmented_forward,layouts,LAYOUT_CACHE


def test_segmented_matches_dense_mixed_lengths_and_rope():
    torch.manual_seed(31)
    module=Qwen2_5_VLVisionSdpaAttention(32,4).eval()
    x=torch.randn(13,32);bounds=torch.tensor([0,3,10,13],dtype=torch.int32)
    theta=torch.randn(13,4);embedding=torch.cat([theta,theta],dim=-1)
    kwargs={'position_embeddings':(embedding.cos(),embedding.sin())}
    with torch.inference_mode():
        dense=module(x,bounds,**kwargs);segmented=segmented_forward(module,x,bounds,**kwargs)
    torch.testing.assert_close(segmented,dense,atol=2e-6,rtol=2e-5)


def test_segmented_preserves_block_isolation():
    torch.manual_seed(42);module=Qwen2_5_VLVisionSdpaAttention(32,4).eval()
    x=torch.randn(13,32);changed=x.clone();changed[3:10]+=10
    bounds=torch.tensor([0,3,10,13],dtype=torch.int32);rope=torch.zeros(13,4)
    with torch.inference_mode():
        a=segmented_forward(module,x,bounds,rotary_pos_emb=rope)
        b=segmented_forward(module,changed,bounds,rotary_pos_emb=rope)
    torch.testing.assert_close(a[:3],b[:3],atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(a[10:],b[10:],atol=1e-6,rtol=1e-6)


def test_partition_and_training_contract():
    with pytest.raises(ValueError):layouts([0,3,8],9,'cpu')
    with pytest.raises(ValueError):layouts([0,5,3,9],9,'cpu')
    module=Qwen2_5_VLVisionSdpaAttention(32,4).train()
    with pytest.raises(ValueError):segmented_forward(module,torch.randn(9,32),torch.tensor([0,9]),rotary_pos_emb=torch.zeros(9,4))
