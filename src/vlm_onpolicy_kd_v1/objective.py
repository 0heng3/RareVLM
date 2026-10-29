"""Raw-token completion frames and per-question forward KL, with no reward model."""
import torch
import torch.nn.functional as F
from vlm_full_teacher_kd_v1.loss import aligned


def attach_body(prefix, completion_ids, terminal_ids):
    body = []
    for token in completion_ids:
        if token in terminal_ids:
            break
        body.append(int(token))
    if not body:
        return None
    answer = torch.tensor([body], dtype=prefix['input_ids'].dtype)
    sample = {**prefix, 'input_ids': torch.cat((prefix['input_ids'], answer), 1),
        'attention_mask': torch.cat((prefix['attention_mask'], torch.ones_like(answer)), 1),
        'labels': torch.cat((torch.full_like(prefix['input_ids'], -100), answer), 1)}
    return sample


def per_qa_kl(student_logits, teacher_logits, labels, valid_ids, temperature=2.):
    if temperature != 2.:
        raise ValueError('Fixed temperature required')
    student, targets = aligned(student_logits, labels)
    teacher, teacher_targets = aligned(teacher_logits, labels)
    if not torch.equal(targets, teacher_targets):
        raise ValueError('Teacher/student raw-token frames differ')
    values = []
    for i in range(len(labels)):
        mask = targets[i].ne(-100)
        if not mask.any():
            raise ValueError('Empty bodies must be excluded before forward')
        s = student[i, mask].index_select(-1, valid_ids)
        t = teacher[i, mask].index_select(-1, valid_ids).detach()
        q = (t / temperature).softmax(-1)
        values.append(temperature**2 * F.kl_div((s / temperature).log_softmax(-1), q, reduction='batchmean'))
    return torch.stack(values)


def cpu_checks():
    import math
    # Padded head slots have overwhelming mass but must not enter the real vocabulary.
    valid = torch.tensor([0, 2]); labels = torch.tensor([[-100, 0]])
    s = torch.tensor([[[2 * math.log(.25), 1000., 2 * math.log(.75), 900.], [0., 0., 0., 0.]]], requires_grad=True)
    t = torch.tensor([[[2 * math.log(.75), 800., 2 * math.log(.25), 700., 600.], [0., 0., 0., 0., 0.]]], requires_grad=True)
    value = per_qa_kl(s, t, labels, valid)
    expected = 2 * math.log(3)
    if abs(value.item() - expected) > 1e-6:
        raise ValueError('Independent two-outcome KL differs')
    value.sum().backward()
    if not torch.allclose(s.grad[0, 0], torch.tensor([-1., 0., 1., 0.]), atol=1e-6) or t.grad is not None:
        raise ValueError('Analytic gradient/teacher detachment differs')
    prefix = {'input_ids': torch.tensor([[40, 41, 42]]), 'attention_mask': torch.ones(1, 3, dtype=torch.long)}
    raw = attach_body(prefix, [13, 17, 99, 25], {99, 100})
    if raw['input_ids'].tolist() != [[40, 41, 42, 13, 17]] or raw['labels'].tolist() != [[-100, -100, -100, 13, 17]]:
        raise ValueError('Raw-token body/prompt/termination frame differs')
    if attach_body(prefix, [100], {99, 100}) is not None:
        raise ValueError('Empty completion handling differs')
    # The shorter row is left padded; both questions have the same known KL.
    ss = s.detach()[:, :1].expand(2, 3, 4).clone().requires_grad_()
    tt = t.detach()[:, :1].expand(2, 3, 5).clone()
    ragged = torch.tensor([[-100, -100, 0], [-100, 0, 2]])
    ragged_value = per_qa_kl(ss, tt, ragged, valid)
    if not torch.allclose(ragged_value, torch.full((2,), expected), atol=1e-6):
        raise ValueError('Per-QA mean/left-padding alignment differs')
    return {'status': 'passed', 'checks': 4, 'closed_form_kl': value.item(),
        'analytic_student_gradient': s.grad[0, 0].tolist(), 'teacher_gradient_absent': True,
        'actual_token_vocabulary_only': True, 'raw_ids_not_retokenized': True,
        'empty_body_explicit': True, 'left_padded_per_qa_mean': True}
