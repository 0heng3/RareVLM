"""Real CPU/Gloo DDP contract for the research runner's weighted objective.

This verifies distributed gradients against a centralized independent objective,
with unequal per-batch weight sums and both 9-real-QA and 1-real-QA tails.
It does not claim to verify GPU/model loading; the separate 33-QA GPU preflight
must additionally exercise first gradients, fixed weights, and checkpoint resume.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
import unittest

from vlm_weighted.train import local_block,schedule,weighted_objective


def _worker(rank,world,rendezvous,result_path):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=world)
    checked=[]
    # Larger reference pool fixes Z once. The 33/41-QA subsets intentionally do
    # not have mean weight one, resembling the GPU contract's full-train weights.
    frequency=torch.tensor([1,2,5,8,20,30,100]*14,dtype=torch.float64)
    raw=(20/frequency).pow(.25).clamp(1,2)
    fixed=raw/raw.mean()
    for n in (41,33):
        for condition in ('uniform','frequency'):
            torch.manual_seed(42)
            module=torch.nn.Linear(2,1,bias=False,dtype=torch.float64)
            with torch.no_grad():module.weight.copy_(torch.tensor([[.25,-.4]],dtype=torch.float64))
            ddp=DistributedDataParallel(module)
            reference=torch.nn.Linear(2,1,bias=False,dtype=torch.float64)
            reference.load_state_dict(module.state_dict())
            opt=torch.optim.SGD(ddp.parameters(),lr=.01)
            refopt=torch.optim.SGD(reference.parameters(),lr=.01)
            order,padded=schedule(n,42,world,8,1)
            assert sorted(order)==list(range(n))
            all_consumed=[]
            x=torch.stack((torch.linspace(-2,3,n),torch.arange(n).remainder(5)-2),dim=1).double()
            targets=(torch.arange(n).remainder(7).double()-3)/4
            weights=torch.ones(n,dtype=torch.float64) if condition=='uniform' else fixed[:n]
            if condition=='frequency':assert abs(float(weights.mean())-1)>1e-4
            for step in range(math.ceil(n/16)):
                local=local_block(padded,step,16,rank,8)
                idx=torch.tensor([i for i,_ in local],dtype=torch.long)
                valid=torch.tensor([v for _,v in local],dtype=torch.bool)
                local_weights=weights[idx]*valid
                local_loss=(ddp(x[idx]).squeeze(1)-targets[idx]).square()
                actual=min(16,n-step*16)
                opt.zero_grad();refopt.zero_grad()
                weighted_objective(local_loss,valid,local_weights,world,actual).backward()
                real=[i for i in padded[step*16:(step+1)*16] if i>=0]
                real_idx=torch.tensor(real,dtype=torch.long)
                # Independent oracle uses ordinary centralized autograd, no
                # DDP factor or runner helper. Denominator is real-QA count.
                reference_loss=((reference(x[real_idx]).squeeze(1)-targets[real_idx]).square()*weights[real_idx]).sum()/len(real)
                reference_loss.backward()
                torch.testing.assert_close(module.weight.grad,reference.weight.grad,rtol=1e-12,atol=1e-12)
                if condition=='frequency':
                    wrong=((reference(x[real_idx]).squeeze(1)-targets[real_idx]).square()*weights[real_idx]).sum()/weights[real_idx].sum()
                    wrong_grad=torch.autograd.grad(wrong,reference.weight)[0]
                    if step==0:assert not torch.allclose(wrong_grad,reference.weight.grad,rtol=1e-6,atol=1e-6)
                opt.step();refopt.step()
                torch.testing.assert_close(module.weight,reference.weight,rtol=1e-12,atol=1e-12)
                all_consumed.extend(real)
                checked.append({'n':n,'condition':condition,'step':step+1,
                    'actual_global_questions':actual,'local_real_questions':int(valid.sum()),
                    'local_sentinel_questions':int((~valid).sum()),
                    'actual_global_weight_sum':float(weights[real_idx].sum())})
            assert sorted(all_consumed)==list(range(n))
            assert checked[-1]['actual_global_questions']==(9 if n==41 else 1)
    if rank==0:
        Path(result_path).write_text(json.dumps({'passed':True,'world_size':world,
            'distributed_updates_checked_per_rank':len(checked),'cases':checked},indent=2)+'\n')
    dist.barrier();dist.destroy_process_group()


class TrainingContract(unittest.TestCase):
    def test_actual_ddp_gradients_and_sentinel_tails(self):
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory(prefix='cpu_ddp_') as tmp:
            result=Path(tmp)/'CPU_DDP_WEIGHT_CONTRACT.json'
            mp.spawn(_worker,args=(2,str(Path(tmp)/'rendezvous'),str(result)),nprocs=2,join=True)
            self.assertTrue(json.loads(result.read_text())['passed'])

    def test_invalid_weight_contract_fails(self):
        import torch
        losses=torch.tensor([2.,999.],requires_grad=True)
        valid=torch.tensor([True,False])
        for values in ([1.,1.],[0.,0.],[float('nan'),0.]):
            with self.assertRaises((ValueError,FloatingPointError)):
                weighted_objective(losses,valid,torch.tensor(values),2,1)
        objective=weighted_objective(losses,valid,torch.tensor([1.5,0.]),2,1)
        objective.backward()
        torch.testing.assert_close(losses.grad,torch.tensor([3.,0.]))


if __name__=='__main__':unittest.main()
