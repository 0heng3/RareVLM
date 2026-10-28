"""Closest Adam displacement satisfying a few local loss-descent constraints.

This projects the displacement AFTER Adam preconditioning, not the raw source
gradients. It guarantees only linearized constraints, never finite-step BF16
loss decrease or independent model capability. Callers must measure the latter.
"""

from itertools import combinations


def project_displacement(delta, gradients, minimum_descent_cosine=.05):
    import torch

    delta = delta.detach().cpu().double()
    if (delta.ndim != 1 or not gradients or len(gradients) > 4
            or not 0 <= minimum_descent_cosine < 1
            or not bool(torch.isfinite(delta).all())):
        raise ValueError('Need a finite displacement and one to four local constraints')
    names, vectors = list(gradients), []
    for name in names:
        g = gradients[name].detach().cpu().double()
        if g.shape != delta.shape or not bool(torch.isfinite(g).all()) or not float(g.norm()) > 0:
            raise ValueError('Constraint gradient must match finite nonzero parameter coordinates')
        vectors.append(g / g.norm())
    matrix = torch.stack(vectors)
    gram = matrix @ matrix.T
    proposed = matrix @ delta
    bound = torch.full_like(proposed, -minimum_descent_cosine * float(delta.norm()))
    tolerance = max(1e-12, float(delta.norm()) * 1e-9)
    best = None
    for size in range(len(names)+1):
        for active in combinations(range(len(names)), size):
            if not active:
                multipliers = torch.empty(0, dtype=torch.float64)
                residual = proposed - bound
                distance = 0.
            else:
                index = list(active)
                sub = gram[index][:, index]
                multipliers = torch.linalg.lstsq(sub, proposed[index]-bound[index], driver='gelsd').solution
                if bool((multipliers < -tolerance).any()):
                    continue
                residual = proposed - gram[:, index] @ multipliers - bound
                distance = float(multipliers @ sub @ multipliers)
            if bool((residual > tolerance).any()):
                continue
            if best is None or distance < best[0]:
                best = (distance, list(active), multipliers)
    if best is None:
        raise ValueError('No feasible local descent constraints; do not apply a fabricated step')
    _, active, multipliers = best
    projected = delta - matrix[active].T @ multipliers if active else delta.clone()
    return projected, {'constraint_names': names, 'active_constraints': [names[i] for i in active],
        'minimum_descent_cosine': minimum_descent_cosine,
        'proposed_norm': float(delta.norm()), 'projected_norm': float(projected.norm()),
        'correction_norm': float((projected-delta).norm()),
        'proposed_normalized_projections': dict(zip(names, proposed.tolist())),
        'projected_normalized_projections': dict(zip(names, (matrix @ projected).tolist())),
        'normalized_upper_bound': float(bound[0]),
        'guarantee': 'local linearized constraints only; finite-step losses must be measured'}
