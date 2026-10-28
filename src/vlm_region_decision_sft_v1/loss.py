"""Retain exact old forward shapes; add chosen decision CE only for frozen initial training errors."""
import torch
import torch.nn.functional as F
from vlm_alignment_signal.loss import completion_logps


def forward(model, batch, rows, decisions):
    labels = batch['labels']
    keep = int((labels != -100).sum(1).max()) + 1
    hook = model.get_output_embeddings().register_forward_pre_hook(lambda module, args: (args[0][:, -keep:, :],))
    try:
        output = model(**{k: v for k, v in batch.items() if k != 'labels'}, use_cache=False)
    finally:
        hook.remove()
    logp, counts = completion_logps(output.logits, labels)
    values = []
    active = []
    for i, row in enumerate(rows):
        decision = decisions.get(row['question_id'])
        if decision is None:
            values.append(logp[i] * 0.)
            active.append(False)
            continue
        answer_start = int((labels[i] != -100).nonzero()[0])
        position = answer_start + decision['divergence_index']
        if int(labels[i, position]) != decision['correct_token']:
            raise ValueError('Decision supervision token differs from frozen chosen label')
        tail_position = position - 1 - (labels.shape[1] - output.logits.shape[1])
        target = labels[i, position].reshape(1)
        values.append(F.cross_entropy(output.logits[i, tail_position].float().reshape(1, -1), target))
        active.append(True)
    return logp, counts, torch.stack(values), torch.tensor(active, device=logp.device)
