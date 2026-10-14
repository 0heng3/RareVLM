"""Keep multi-positive CE; match the frozen teacher's same-pool distributions."""
import numpy as np
import torch
import torch.nn.functional as F
from clip_retrieval.train import multi_positive_ce


def teacher_kl(student_logits, teacher_logits):
    if student_logits.shape != teacher_logits.shape:
        raise ValueError('Teacher/student candidate pools differ')
    teacher_logp=F.log_softmax(teacher_logits.detach(),dim=1)
    return F.kl_div(F.log_softmax(student_logits,dim=1),teacher_logp,
                    reduction='none',log_target=True).sum(dim=1)


def distill_terms(model,images,texts,anchors,pool,groups,temperature):
    n=len(anchors);image_ids=np.concatenate([anchors,pool])
    ids=torch.as_tensor(image_ids,device=images.device)
    caps=(ids[:,None]*5+torch.arange(5,device=images.device)[None,:]).reshape(-1)
    zi=model.image_features(images[ids]);zt=model.text_features(texts[caps])
    group=groups[image_ids];caption_group=np.repeat(group,5)
    positive_i=torch.as_tensor(group[:n,None]==caption_group[None,:],device=images.device)
    positive_t=torch.as_tensor(caption_group[:n*5,None]==group[None,:],device=images.device)
    student_i=zi[:n]@zt.T/temperature;student_t=zt[:n*5]@zi.T/temperature
    supervised=(multi_positive_ce(student_i,positive_i).mean()+
        multi_positive_ce(student_t,positive_t).reshape(n,5).mean(1).mean())/2
    with torch.no_grad():
        teacher_i=F.normalize(images[ids],dim=-1)
        teacher_t=F.normalize(texts[caps],dim=-1)
        teacher_logits_i=teacher_i[:n]@teacher_t.T/temperature
        teacher_logits_t=teacher_t[:n*5]@teacher_i.T/temperature
    relation=(teacher_kl(student_i,teacher_logits_i).mean()+
        teacher_kl(student_t,teacher_logits_t).reshape(n,5).mean(1).mean())/2
    return supervised,relation


def contrastive_distill_loss(model,images,texts,anchors,pool,groups,temperature,weight):
    if weight<0:raise ValueError('Nonnegative distillation weight required')
    supervised,relation=distill_terms(model,images,texts,anchors,pool,groups,temperature)
    return supervised+weight*relation
