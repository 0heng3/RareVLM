"""Same symmetric meanCE+KL; caption count follows selected actual tensor."""
import numpy as np
import torch
import torch.nn.functional as F
from clip_retrieval.train import multi_positive_ce
from clip_retrieval_distill.loss import teacher_kl


def selected_terms(model,images,texts,anchors,pool,groups,temperature):
    n=len(anchors);k=texts.shape[1];image_ids=np.concatenate([anchors,pool])
    ids=torch.as_tensor(image_ids,device=images.device)
    raw_t=texts[ids].reshape(-1,512)
    zi=model.image_features(images[ids]);zt=model.text_features(raw_t)
    group=groups[image_ids];caption_group=np.repeat(group,k)
    positive_i=torch.as_tensor(group[:n,None]==caption_group[None,:],device=images.device)
    positive_t=torch.as_tensor(caption_group[:n*k,None]==group[None,:],device=images.device)
    student_i=zi[:n]@zt.T/temperature;student_t=zt[:n*k]@zi.T/temperature
    ce=(multi_positive_ce(student_i,positive_i).mean()+
        multi_positive_ce(student_t,positive_t).reshape(n,k).mean(1).mean())/2
    with torch.no_grad():
        ti=F.normalize(images[ids],dim=-1);tt=F.normalize(raw_t,dim=-1)
        logits_i=ti[:n]@tt.T/temperature;logits_t=tt[:n*k]@ti.T/temperature
    kl=(teacher_kl(student_i,logits_i).mean()+
        teacher_kl(student_t,logits_t).reshape(n,k).mean(1).mean())/2
    return ce,kl
