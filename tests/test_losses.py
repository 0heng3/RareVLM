import math
import pytest
import torch
from methods import build_loss


def test_la_sign_and_prior_are_the_frozen_training_definition():
    criterion=build_loss('la',[90,10],tau=1.)
    z=torch.zeros(1,2)
    assert criterion(z,torch.tensor([0])).item()==pytest.approx(-math.log(.9),abs=1e-6)
    assert criterion(z,torch.tensor([1])).item()==pytest.approx(-math.log(.1),abs=1e-6)


def test_zero_tau_reduces_to_ce():
    z=torch.tensor([[.2,1.3],[-.7,.4]]);y=torch.tensor([1,0])
    assert torch.equal(build_loss('la',[9,1],tau=0.)(z,y),build_loss('ce')(z,y))


@pytest.mark.parametrize('counts',[[0,1],[-1,2],[1,float('inf')]])
def test_invalid_priors_are_rejected(counts):
    with pytest.raises(ValueError):build_loss('la',counts,tau=1.)
