"""Sequence-logp sigmoid DPO; real precheck performs zero updates."""
import torch
import torch.nn.functional as F

def completion_logps(logits,labels):
    targets=labels[:,-logits.shape[1]:][:,1:];shifted=logits[:,:-1,:].float()
    valid=targets!=-100
    safe=targets.masked_fill(~valid,0)
    logp=F.log_softmax(shifted,dim=-1).gather(-1,safe.unsqueeze(-1)).squeeze(-1)
    count=valid.sum(1)
    if not bool((count>0).all()) or not bool((count==(labels!=-100).sum(1)).all()):raise ValueError('Lost supervised answer tokens')
    return (logp*valid).sum(1),count

def sequence_logps(model,batch):
    labels=batch['labels'];keep=int((labels!=-100).sum(1).max())+1
    hook=model.get_output_embeddings().register_forward_pre_hook(lambda m,args:(args[0][:,-keep:,:],))
    try:out=model(**{k:v for k,v in batch.items() if k!='labels'},use_cache=False)
    finally:hook.remove()
    return completion_logps(out.logits,labels)

def dpo_loss(chosen,rejected,reference_chosen,reference_rejected,beta=.1):
    if beta<=0:raise ValueError('Require positive beta')
    relative=(chosen-rejected)-(reference_chosen.detach()-reference_rejected.detach())
    return -F.logsigmoid(beta*relative)
