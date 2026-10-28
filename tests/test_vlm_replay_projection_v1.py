import torch
from vlm_replay_projection_v1.common import project


def test_closest_feasible_displacement_with_conflicting_sources():
    # The unmodified Adam move hurts replay; only the violating component moves.
    d=torch.tensor([-2.,1.])
    got,info=project(d,{'hard':torch.tensor([1.,0.]),'replay':torch.tensor([0.,1.])})
    assert torch.allclose(got,torch.tensor([-2.,0.],dtype=torch.float64),atol=1e-12)
    assert info['active_constraints']==['replay']
    # Already compatible moves stay exactly unchanged; opposing source constraints
    # remain feasible at zero margin, including a retained orthogonal component.
    same,_=project(torch.tensor([-2.,-1.]),{'hard':torch.tensor([1.,0.]),'replay':torch.tensor([0.,1.])})
    assert torch.equal(same,torch.tensor([-2.,-1.],dtype=torch.float64))
    opposite,_=project(torch.tensor([2.,3.]),{'hard':torch.tensor([1.,0.]),'replay':torch.tensor([-1.,0.])})
    assert torch.allclose(opposite,torch.tensor([0.,3.],dtype=torch.float64),atol=1e-12)
