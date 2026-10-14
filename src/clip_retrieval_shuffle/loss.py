"""Keep positive teacher targets; permute only negative target associations."""
import numpy as np
import torch
import torch.nn.functional as F
from clip_retrieval.train import multi_positive_ce


def negative_permutation(positive, generator):
    """Map destination columns to source columns; positives are fixed points."""
    if positive.ndim != 2 or positive.dtype != torch.bool:
        raise ValueError('Require a two-dimensional positive mask')
    rows, width = positive.shape
    columns = torch.arange(width, device=positive.device).expand(rows, -1)
    destinations = torch.argsort(positive.long() * width + columns, dim=1, stable=True)
    keys = torch.rand(positive.shape, device=positive.device, dtype=torch.float64, generator=generator)
    sources = torch.argsort(keys.masked_fill(positive, 2.), dim=1, stable=True)
    return columns.clone().scatter(1, destinations, sources)


def teacher_targets(teacher_logits, positive, generator, shuffle=True):
    if teacher_logits.shape != positive.shape:
        raise ValueError('Teacher target/mask shape differs')
    logp = F.log_softmax(teacher_logits.detach(), dim=1)
    if not shuffle:
        return logp, None
    mapping = negative_permutation(positive, generator)
    # Normalize once, THEN permute: do not introduce a second softmax reduction.
    return logp.gather(1, mapping), mapping


def target_kl(student_logits, teacher_logp):
    if student_logits.shape != teacher_logp.shape:
        raise ValueError('Teacher/student pools differ')
    return F.kl_div(F.log_softmax(student_logits, dim=1), teacher_logp.detach(),
                    reduction='none', log_target=True).sum(1)


def shuffle_terms(model, images, texts, anchors, pool, groups, temperature,
                  permutation_seed, step, shuffle=True):
    n = len(anchors)
    image_ids = np.concatenate([anchors, pool])
    ids = torch.as_tensor(image_ids, device=images.device)
    caps = (ids[:, None] * 5 + torch.arange(5, device=images.device)[None, :]).reshape(-1)
    zi = model.image_features(images[ids]); zt = model.text_features(texts[caps])
    group = groups[image_ids]; caption_group = np.repeat(group, 5)
    positive_i = torch.as_tensor(group[:n, None] == caption_group[None, :], device=images.device)
    positive_t = torch.as_tensor(caption_group[:n*5, None] == group[None, :], device=images.device)
    student_i = zi[:n] @ zt.T / temperature
    student_t = zt[:n*5] @ zi.T / temperature
    supervised = (multi_positive_ce(student_i, positive_i).mean() +
                  multi_positive_ce(student_t, positive_t).reshape(n, 5).mean(1).mean()) / 2
    generator = torch.Generator(device=images.device)
    generator.manual_seed(int(permutation_seed) * 1000003 + int(step) + 104729)
    with torch.no_grad():
        ti = F.normalize(images[ids], dim=-1); tt = F.normalize(texts[caps], dim=-1)
        teacher_i = ti[:n] @ tt.T / temperature
        teacher_t = tt[:n*5] @ ti.T / temperature
        target_i, mapping_i = teacher_targets(teacher_i, positive_i, generator, shuffle)
        target_t, mapping_t = teacher_targets(teacher_t, positive_t, generator, shuffle)
    relation = (target_kl(student_i, target_i).mean() +
                target_kl(student_t, target_t).reshape(n, 5).mean(1).mean()) / 2
    details = {'teacher_logits_i': teacher_i, 'teacher_logits_t': teacher_t,
               'student_logits_i': student_i.detach(), 'student_logits_t': student_t.detach(),
               'target_i': target_i, 'target_t': target_t,
               'positive_i': positive_i, 'positive_t': positive_t,
               'mapping_i': mapping_i, 'mapping_t': mapping_t,
               'image_ids': image_ids, 'caption_ids': caps.detach()}
    return supervised, relation, details


def contrastive_shuffle_loss(model, images, texts, anchors, pool, groups, temperature,
                             weight, permutation_seed=0, step=0, shuffle=True, return_details=False):
    if weight < 0:
        raise ValueError('Nonnegative weight required')
    supervised, relation, details = shuffle_terms(model, images, texts, anchors, pool, groups,
                                                 temperature, permutation_seed, step, shuffle)
    loss = supervised + weight * relation
    return (loss, details) if return_details else loss
