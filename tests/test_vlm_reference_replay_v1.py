def test_reference_kl_direction_sequence_mean_mask_and_gradient():
    import torch
    import torch.nn.functional as F
    from vlm_reference_replay_v1.loss import action_logps,loss_values
    probabilities=torch.tensor([[[.4,.4,.2],[.2,.5,.3],[.1,.2,.7],[.3,.3,.4]],
                                [[.3,.3,.4],[.6,.2,.2],[.2,.2,.6],[.4,.2,.4]]])
    logits=probabilities.log().requires_grad_();labels=torch.tensor([[-100,0,1,2],[-100,-100,1,2]])
    teacher=[torch.tensor([[.8,.1,.1],[.2,.7,.1],[.1,.1,.8]]),torch.tensor([[.4,.4,.2],[.1,.1,.8]])]
    refs=[x.log() for x in teacher];ce,kl=loss_values(logits,labels,refs)
    expected_grad=torch.zeros_like(logits)
    for i,positions in enumerate(([0,1,2],[1,2])):
        y=labels[i,torch.tensor(positions)+1]
        assert torch.allclose(ce[i],F.cross_entropy(logits[i,positions],y),atol=1e-6)
        expected=(teacher[i]*(refs[i]-probabilities[i,positions].log())).sum(-1).mean()
        assert torch.allclose(kl[i],expected,atol=1e-6)
        expected_grad[i,positions]=(probabilities[i,positions]-teacher[i])/(2*len(positions))
    kl.mean().backward();assert torch.allclose(logits.grad,expected_grad,atol=1e-6)
    own,_=action_logps(logits.detach(),labels);_,zero=loss_values(logits.detach(),labels,own)
    assert torch.allclose(zero,torch.zeros(2),atol=1e-6)
