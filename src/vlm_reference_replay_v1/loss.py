import torch
import torch.nn.functional as F
from vlm_sampled_signal_v1.trajectory import readout_tail_length,completion_logps
from vlm_object_train.train import tensor_hash


def frame_hash(batch):
    return tensor_hash(sorted(batch.items()))


def forward_logits(model,batch):
    keep=readout_tail_length(batch['labels'])
    handle=model.get_output_embeddings().register_forward_pre_hook(lambda module,args:(args[0][:,-keep:,:],))
    try:return model(**{k:v for k,v in batch.items() if k!='labels'},use_cache=False).logits
    finally:handle.remove()


def action_logps(logits,labels):
    targets=labels[:,-logits.shape[1]:][:,1:];valid=targets.ne(-100)
    if not torch.equal(valid.sum(1),labels.ne(-100).sum(1)):raise ValueError('Lost supervised positions')
    logps=F.log_softmax(logits[:,:-1].float(),dim=-1)
    return [logps[i,valid[i]] for i in range(len(labels))], [targets[i,valid[i]] for i in range(len(labels))]


def loss_values(logits,labels,references=None):
    sums,counts=completion_logps(logits,labels);ce=-sums/counts
    if references is None:return ce,None
    current,_=action_logps(logits,labels)
    if len(current)!=len(references):raise ValueError('Reference QA count differs')
    values=[]
    for logp,reference in zip(current,references):
        if tuple(logp.shape)!=tuple(reference.shape) or reference.requires_grad:raise ValueError('Reference action frame/gradient differs')
        values.append(F.kl_div(logp,reference.to(device=logp.device,dtype=torch.float32),log_target=True,reduction='none').sum(-1).mean())
    return ce,torch.stack(values)
