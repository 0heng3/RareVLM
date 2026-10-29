"""Per-QA CE plus temperature-scaled auxiliary targets on shared actual IDs."""
import torch
import torch.nn.functional as F


def aligned(logits, labels):
    targets = labels[:, -logits.shape[1]:][:, 1:]
    if not torch.equal(targets.ne(-100).sum(1), labels.ne(-100).sum(1)):
        raise ValueError('Readout lost supervised tokens')
    return logits[:, :-1].float(), targets


def objectives(logits, labels, valid_ids, caches, role, eos, temperature=2.):
    if role not in ('hard', 'uniform', 'kd') or temperature != 2.:
        raise ValueError('Unregistered auxiliary role or temperature')
    values, targets = aligned(logits, labels)
    ce = F.cross_entropy(values.transpose(1, 2), targets, ignore_index=-100, reduction='none')
    ce = ce.sum(1) / targets.ne(-100).sum(1)
    auxiliary = []
    for i, cache in enumerate(caches):
        if cache is None:
            auxiliary.append(values[i].sum() * 0)
            continue
        mask = targets[i].ne(-100) & targets[i].ne(eos)
        gold = targets[i, mask]
        if not torch.equal(gold.cpu(), cache['labels']):
            raise ValueError('Cached teacher GT-token frame differs')
        indices = torch.searchsorted(valid_ids, gold)
        if indices.max() >= len(valid_ids) or not torch.equal(valid_ids[indices], gold):
            raise ValueError('GT token outside the shared actual vocabulary')
        student = values[i, mask].index_select(-1, valid_ids)
        teacher = (cache['logits'].to(device=student.device, dtype=torch.float32) / temperature).softmax(-1).detach()
        if teacher.shape != student.shape:
            raise ValueError('Teacher/body/shared-vocabulary shape differs')
        if role == 'hard':
            target = F.one_hot(indices, len(valid_ids)).float()
        elif role == 'uniform':
            gt = teacher.gather(1, indices[:, None])
            target = ((1 - gt) / (len(valid_ids) - 1)).expand_as(teacher).clone()
            target.scatter_(1, indices[:, None], gt)
        else:
            target = teacher
        auxiliary.append(temperature**2 * F.kl_div((student / temperature).log_softmax(-1), target, reduction='batchmean'))
    return ce, torch.stack(auxiliary)
