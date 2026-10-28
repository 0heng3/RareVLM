def test_content_eos_diagnostic_matches_independent_crossentropy_with_left_mask():
    import torch
    import torch.nn.functional as F
    from vlm_fit_positive_control_v1.study import token_statistics
    # Different left padding and answer lengths; last position predicts no action.
    labels = torch.tensor([[-100, -100, 1, 2, 4], [-100, -100, -100, 3, 4]])
    logits = torch.tensor([[[0., 0, 0, 0, 0], [0, 3., 1, 2, -1], [0, 0, 2., 3, 1], [0, 1, 2, 0, 4.], [0, 0, 0, 0, 0]],
                           [[0., 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 5., 2], [0, 1, 0, 2, 4.], [0, 0, 0, 0, 0]]])
    result = token_statistics(logits, labels, 4)
    for index, selected in enumerate(([1, 2, 3], [2, 3])):
        y = labels[index, torch.tensor(selected) + 1]
        oracle = F.cross_entropy(logits[index, selected], y, reduction='none')
        assert abs(result[index]['content_ce'] - float(oracle[:-1].mean())) < 1e-6
        assert abs(result[index]['eos_ce'] - float(oracle[-1])) < 1e-6
    assert result[0]['target_ranks'] == [1, 2, 1]
    assert result[0]['target_minus_best_rival'] == [1., -1., 2.]
    assert not result[0]['content_target_argmax']
    assert result[1]['all_target_argmax']
