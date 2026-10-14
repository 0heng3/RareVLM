"""Positive-set NCE; keep the caption-to-image objective unchanged."""
import numpy as np
import torch

from clip_retrieval.train import multi_positive_ce


def positive_set_ce(logits, positive):
    if logits.shape != positive.shape or not bool(positive.any(dim=1).all()):
        raise ValueError("Each query needs at least one labeled positive")
    return torch.logsumexp(logits, dim=1) - torch.logsumexp(
        logits.masked_fill(~positive, -torch.inf), dim=1)


def contrastive_set_loss(model, images, texts, anchors, pool, groups, temperature):
    n = len(anchors)
    image_ids = np.concatenate([anchors, pool])
    image_indices = torch.as_tensor(image_ids, device=images.device)
    caption_indices = (image_indices[:, None] * 5 + torch.arange(5, device=images.device)[None, :]).reshape(-1)
    zi = model.image_features(images[image_indices])
    zt = model.text_features(texts[caption_indices])
    group = groups[image_ids]
    caption_group = np.repeat(group, 5)
    pos_image = torch.as_tensor(group[:n, None] == caption_group[None, :], device=images.device)
    pos_text = torch.as_tensor(caption_group[:n * 5, None] == group[None, :], device=images.device)
    image_loss = positive_set_ce(zi[:n] @ zt.T / temperature, pos_image).mean()
    text_loss = multi_positive_ce(zt[:n * 5] @ zi.T / temperature, pos_text).reshape(n, 5).mean(dim=1).mean()
    return (image_loss + text_loss) / 2
