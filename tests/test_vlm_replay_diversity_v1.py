import torch
from vlm_replay_diversity_v1.common import source_coefficient


def test_duplicate_replay_preserves_source_loss_and_gradient_mass():
    beta=28.977663982973098
    x=torch.tensor([.2,-.3],requires_grad=True)
    hard=torch.stack([(x[0]-float(i))**2 for i in range(8)])
    replay=torch.stack([(x[1]+float(i)/8)**2 for i in range(8)])
    old=.5*hard.mean()+(.5+beta)*replay.mean()
    doubled=source_coefficient('hard',beta)*hard.sum()+source_coefficient('replay',beta)*replay.repeat(2).sum()
    go=torch.autograd.grad(old,x,retain_graph=True)[0];gn=torch.autograd.grad(doubled,x)[0]
    assert torch.allclose(old,doubled,atol=1e-6)
    assert torch.allclose(go,gn,atol=1e-6)
