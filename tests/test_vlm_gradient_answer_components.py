import torch
import torch.nn.functional as F

from vlm_gradient_mechanism_v1.components import components_from_logits


def test_first_answer_uses_preceding_prediction_and_retains_all_targets():
    labels = torch.tensor([[-100, -100, 1, 2, 3], [-100, -100, -100, 2, 3]])
    logits = torch.arange(2 * 5 * 4, dtype=torch.float32).reshape(2, 5, 4) / 7
    result = components_from_logits(logits, labels)
    expected = F.cross_entropy(torch.stack([logits[0, 1], logits[1, 2]]),
                               torch.tensor([1, 2]), reduction='none')
    torch.testing.assert_close(result['first_answer_ce'], expected)
    assert result['counts'].tolist() == [3, 2]
    tail = components_from_logits(logits[:, -4:], labels)
    torch.testing.assert_close(tail['qa_ce'], result['qa_ce'])
    torch.testing.assert_close(tail['first_answer_ce'], result['first_answer_ce'])


def test_three_token_qa_gradient_decomposes_without_changing_loss_scale():
    labels = torch.tensor([[-100, -100, 1, 2, 3]])
    logits = (torch.arange(20, dtype=torch.float32).reshape(1, 5, 4) / 9).requires_grad_()
    result = components_from_logits(logits, labels)
    qa = result['qa_ce'].mean()
    answer = result['first_answer_ce'].mean()
    ending = F.cross_entropy(logits[0, 2:4], torch.tensor([2, 3]))
    g_qa = torch.autograd.grad(qa, logits, retain_graph=True)[0]
    g_parts = torch.autograd.grad((answer + 2 * ending) / 3, logits, retain_graph=True)[0]
    torch.testing.assert_close(g_qa, g_parts)
    weighted = qa + (2 / 3) * answer
    g_extra = torch.autograd.grad(weighted, logits, retain_graph=True)[0] - g_qa
    g_answer = torch.autograd.grad(answer, logits)[0]
    torch.testing.assert_close(g_extra, (2 / 3) * g_answer)
