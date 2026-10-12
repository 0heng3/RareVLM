import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vlm_deep.common import answer_loss,collate,consensus,training_target
from vlm_deep.evaluate import aggregate
from vlm_deep.scoring import anls
from vlm_deep.third_party.mcq_extract import extract_mcq_answer
from vlm_deep.train import schedule
from vlm_deep.train import BatchCollator


class MinimalLM(nn.Module):
    def __init__(self):
        super().__init__();self.embedding=nn.Embedding(32,7);self.head=nn.Linear(7,32)
    def get_output_embeddings(self):return self.head
    def forward(self,input_ids,attention_mask=None,use_cache=False):
        return SimpleNamespace(logits=self.head(self.embedding(input_ids)))


class ParallelWrapper(nn.Module):
    def __init__(self,module):super().__init__();self.module=module
    def forward(self,**kwargs):return self.module(**kwargs)


def sample(ids,labels):
    return {'input_ids':torch.tensor([ids]),'attention_mask':torch.ones(1,len(ids),dtype=torch.long),
            'labels':torch.tensor([labels]),'pixel_values':torch.zeros(2,3),'image_grid_thw':torch.tensor([[1,2,2]])}


def test_left_padding_preserves_answer_spans_and_visual_order():
    batch=collate([sample([4,5,6,7],[-100,-100,6,7]),
                   sample([8,9,10,11,12,13],[-100,-100,-100,11,12,13])],0)
    assert batch['input_ids'][0].tolist()==[0,0,4,5,6,7]
    assert batch['labels'][0].tolist()==[-100,-100,-100,-100,6,7]
    assert batch['attention_mask'][0].tolist()==[0,0,1,1,1,1]
    assert batch['pixel_values'].shape==(4,3)
    assert batch['image_grid_thw'].shape==(2,3)


def test_numpy_worker_transport_preserves_tensor_values_and_validity():
    items=[(sample([4,5,6,7],[-100,-100,6,7]),True,'q1'),
           (sample([8,9,10,11,12,13],[-100,-100,-100,11,12,13]),False,'q2')]
    standard,valid,ids=BatchCollator(0)(items)
    transported,v2,ids2=BatchCollator(0,numpy_transport=True)(items)
    for key,value in standard.items():torch.testing.assert_close(value,torch.from_numpy(transported[key]))
    torch.testing.assert_close(valid,torch.from_numpy(v2));assert ids==ids2


@pytest.mark.parametrize('wrapped',[False,True])
def test_answer_head_slicing_matches_full_loss_and_gradients(wrapped):
    torch.manual_seed(12);base=MinimalLM();model=ParallelWrapper(base) if wrapped else base
    b=collate([sample([4,5,6,7],[-100,-100,6,7]),
               sample([8,9,10,11,12,13],[-100,-100,-100,11,12,13])],0)
    inputs={k:v for k,v in b.items() if k in {'input_ids','attention_mask','labels'}}
    full,_,count=answer_loss(model,inputs,False);full.mean().backward()
    grads={n:p.grad.clone() for n,p in base.named_parameters()};base.zero_grad(set_to_none=True)
    optimized,_,count2=answer_loss(model,inputs,True);optimized.mean().backward()
    assert count==count2==5
    torch.testing.assert_close(full,optimized)
    for name,p in base.named_parameters():torch.testing.assert_close(p.grad,grads[name])
    assert len(base.head._forward_pre_hooks)==0


def test_temporary_head_hook_is_removed_after_forward_exception():
    model=MinimalLM()
    def fail(**kwargs):raise RuntimeError('intentional')
    model.forward=fail
    with pytest.raises(RuntimeError):
        answer_loss(model,{'input_ids':torch.tensor([[1,2]]),'labels':torch.tensor([[-100,2]])})
    assert not model.head._forward_pre_hooks


def test_full_schedule_neither_drops_nor_duplicates_real_questions():
    order,padded=schedule(17,42,4,4,1)
    assert sorted(order)==list(range(17))
    assert [x for x in padded if x>=0]==order
    assert len(padded)==32 and padded.count(-1)==15


def test_partial_batch_ddp_weighting_matches_global_question_mean():
    theta=torch.tensor(.7,requires_grad=True)
    x=torch.tensor([2.,3.,5.]);y=torch.tensor([1.,-1.,4.])
    expected=2*torch.mean(x*(theta.detach()*x-y))
    gradients=[]
    for rank in range(4):
        valid=rank<3
        idx=rank if valid else 0
        loss=(theta*x[idx]-y[idx])**2*(4/3 if valid else 0.)
        gradients.append(torch.autograd.grad(loss,theta)[0])
    torch.testing.assert_close(torch.stack(gradients).mean(),expected)


def test_raw_training_target_keeps_answer_letter_articles():
    assert training_target(['A']*10)=='A'
    assert training_target(['the']*10)=='the'
    assert training_target([' ']*10) is None


def test_standard_vqa_normalization_includes_contractions():
    assert consensus("don't",['dont']*10)==1.
    assert consensus('2',['two']*10)==1.
    assert math.isclose(consensus('red',['red']+['blue']*9),.3)


def test_pinned_anls_boundary_and_whitespace_semantics():
    assert anls('NEW YORK',['new york'])==1.
    assert anls('ab',['ac'])==.5
    assert anls('abc',['xyz'])==0.


def test_mcq_parser_rejects_words_and_ambiguous_choices():
    assert extract_mcq_answer('cat',choices=list('ABCD'))==''
    assert extract_mcq_answer('A or B',choices=list('ABCD'))==''
    assert extract_mcq_answer('Answer: C',choices=list('ABCD'))=='C'


def test_mmstar_primary_aggregation_is_macro_subcategory_mean():
    value,groups=aggregate([{'subcategory':'one','score':1.}]*3+
                           [{'subcategory':'two','score':0.}],'mmstar')
    assert value==50. and groups=={'one':100.,'two':0.}
