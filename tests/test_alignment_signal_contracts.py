import json
import math
import numpy as np
import torch
from vlm_alignment_signal.common import group_advantages,preference,native
from vlm_alignment_signal.loss import completion_logps,dpo_loss
from vlm_probe.evaluate import inference_row

def test_dpo_identity_and_gradient_signs():
    a=torch.tensor([-2.],requires_grad=True);b=torch.tensor([-5.],requires_grad=True)
    ra=a.detach().clone().requires_grad_();rb=b.detach().clone().requires_grad_()
    loss=dpo_loss(a,b,ra,rb,.1).mean();loss.backward()
    assert abs(float(loss)-math.log(2))<1e-6
    assert abs(float(a.grad)+.05)<1e-6 and abs(float(b.grad)-.05)<1e-6
    assert ra.grad is None and rb.grad is None

def test_dpo_prefers_improved_relative_margin():
    reference=torch.tensor([-2.]);rejected=torch.tensor([-5.])
    assert dpo_loss(torch.tensor([-1.]),rejected,reference,rejected)<dpo_loss(reference,rejected,reference,rejected)

def test_assistant_sum_against_independent_scalar_probabilities():
    torch.manual_seed(8);logits=torch.randn(2,9,12);labels=torch.full((2,9),-100)
    labels[0,-3:]=torch.tensor([2,3,4]);labels[1,-2:]=torch.tensor([5,6])
    lp,count=completion_logps(logits,labels)
    for row in range(2):
        scalar=sum(math.log(float(torch.softmax(logits[row,t-1],0)[int(labels[row,t])])) for t in range(1,9) if int(labels[row,t])!=-100)
        assert abs(float(lp[row])-scalar)<1e-5
    assert count.tolist()==[3,2]
    tail,_=completion_logps(logits[:,-4:,:],labels)
    assert torch.equal(lp,tail)

def test_constant_reward_has_zero_advantage():
    assert np.array_equal(group_advantages([1,1,1,1]),np.zeros(4))
    assert np.array_equal(group_advantages([0,0,0,0]),np.zeros(4))

def test_reward_advantage_shift_and_scale():
    r=np.array([0,0,1,1.]);a=group_advantages(r)
    assert abs(a.mean())<1e-12 and np.allclose(a,group_advantages(r+8),atol=1e-7)
    assert np.allclose(a,group_advantages(r*2),atol=1e-7)

def test_preferences_need_real_distinct_nonempty_samples():
    norm=lambda s:s.lower().strip()
    assert preference(['yes','YES','','yes'],[1,0,0,1],norm) is None
    assert preference(['yes','no','','maybe'],[1,0,0,.3],norm)=={'chosen_index':0,'rejected_index':1}

def test_generation_input_excludes_all_reference_fields():
    row={'question_id':'1','image_id':'a','question':'what?','target':'secret','answers':['secret']*10,'reward':1}
    x=inference_row(row)
    assert 'target' not in x and 'answers' not in x and 'reward' not in x and x['question']=='what?'

def test_reporting_handles_numpy_gate_types():
    assert json.loads(json.dumps(native({'gate':np.bool_(True),'count':np.int64(4)})))=={'gate':True,'count':4}
