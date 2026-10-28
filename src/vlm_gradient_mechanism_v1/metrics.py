"""Read-only FP64 mathematics for a fixed-checkpoint gradient diagnostic.

All snapshots describe the same ordered, trainable FP32 parameter coordinates.
They contain CPU copies, never views of live ``parameter.grad`` storage.  This
module performs no forward/backward pass, gradient assignment, or update.

If g_s = grad(mean_QA L_s), the historical per-update CE gradient is
sum_s (n_s / 16) g_s.  For a three-token human answer, if g_answer denotes its
first-token NLL gradient and g_format the mean of its two ending-token NLLs,
g_QA = (g_answer + 2*g_format)/3.  Coefficients must not be normalized again.

For the hypothetical unpreconditioned step theta' = theta - eta*g_objective,
the first-order change in L_s is -eta * dot(g_s, g_objective).  This local SGD
direction is not the historical Adam update, a finite-step guarantee, or a
causal explanation for an entire training trajectory.
"""

from dataclasses import dataclass
import math
from collections.abc import Iterable, Mapping

import torch


@dataclass(frozen=True)
class GradientSnapshot:
    parameter_names: tuple[str, ...]
    parameter_shapes: tuple[tuple[int, ...], ...]
    values: torch.Tensor
    missing_gradients: tuple[str, ...] = ()

    @property
    def numel(self) -> int:
        return self.values.numel()


def _validate(snapshot: GradientSnapshot) -> None:
    values = snapshot.values
    if values.device.type != 'cpu' or values.dtype != torch.float64 or values.ndim != 1:
        raise ValueError('Snapshot values must be a one-dimensional CPU FP64 tensor')
    if len(snapshot.parameter_names) != len(snapshot.parameter_shapes):
        raise ValueError('Parameter names and shapes have different lengths')
    if len(set(snapshot.parameter_names)) != len(snapshot.parameter_names):
        raise ValueError('Parameter names must be unique')
    expected = sum(math.prod(shape) for shape in snapshot.parameter_shapes)
    if not snapshot.parameter_names or values.numel() != expected:
        raise ValueError('Empty parameters or incompatible flattened parameter shapes')
    if not bool(torch.isfinite(values).all()):
        raise ValueError('Gradient snapshot must be finite')


def _compatible(left: GradientSnapshot, right: GradientSnapshot) -> None:
    _validate(left)
    _validate(right)
    if (left.parameter_names != right.parameter_names or
            left.parameter_shapes != right.parameter_shapes):
        raise ValueError('Gradient parameter coordinates/order do not match')


def capture_gradients(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    gradients: Iterable[torch.Tensor | None] | None = None,
) -> GradientSnapshot:
    """Copy real trainable parameter gradients to CPU FP64 without mutation.

    Frozen parameters are excluded.  Optional ``gradients`` follows the order of
    the remaining trainable parameters, as returned by ``torch.autograd.grad``.
    Otherwise current ``parameter.grad`` is copied. Missing gradients become
    explicit zeros and their names are reported; they are never silently dropped.
    """
    params = [(name, param) for name, param in named_parameters if param.requires_grad]
    if not params or len({name for name, _ in params}) != len(params):
        raise ValueError('Need nonempty, uniquely named trainable parameters')
    grads = [param.grad for _, param in params] if gradients is None else list(gradients)
    if len(grads) != len(params):
        raise ValueError('Gradients must match the trainable parameter count')
    parts, missing = [], []
    for (name, param), grad in zip(params, grads):
        if param.dtype != torch.float32:
            raise ValueError(f'Trainable parameter {name} must be FP32')
        if grad is None:
            missing.append(name)
            parts.append(torch.zeros(param.numel(), dtype=torch.float64, device='cpu'))
            continue
        if grad.dtype != torch.float32 or grad.shape != param.shape:
            raise ValueError(f'Gradient {name} must match its FP32 parameter shape')
        if grad.layout != torch.strided:
            raise ValueError(f'Gradient {name} must be dense')
        # clone is intentional even when the input already resides on CPU.
        part = grad.detach().to(device='cpu', dtype=torch.float64).reshape(-1).clone()
        if not bool(torch.isfinite(part).all()):
            raise ValueError(f'Gradient {name} must be finite')
        parts.append(part)
    snapshot = GradientSnapshot(
        tuple(name for name, _ in params),
        tuple(tuple(param.shape) for _, param in params),
        torch.cat(parts), tuple(missing),
    )
    _validate(snapshot)
    return snapshot


def qa_mean_weights(qa_counts: Mapping[str, int], total_qa: int) -> dict[str, float]:
    """Convert per-group QA means into a declared per-update mean objective.

    A proper subset is permitted, so weights may sum below one.  This preserves
    1/16 rather than accidentally rescaling a subset to a complete update.
    """
    if isinstance(total_qa, bool) or not isinstance(total_qa, int) or total_qa <= 0:
        raise ValueError('total_qa must be a positive integer')
    if not qa_counts or any(isinstance(n, bool) or not isinstance(n, int) or n < 0
                            for n in qa_counts.values()):
        raise ValueError('QA counts must be nonnegative integers')
    if sum(qa_counts.values()) > total_qa:
        raise ValueError('Group QA counts exceed the declared update size')
    return {name: count / total_qa for name, count in qa_counts.items()}


def _weights(groups: Mapping[str, GradientSnapshot], weights: Mapping[str, float]) -> dict[str, float]:
    if not groups or set(groups) != set(weights):
        raise ValueError('Weights must specify exactly the gradient groups')
    coefficients = {name: float(weights[name]) for name in groups}
    if any(not math.isfinite(value) or value < 0 for value in coefficients.values()):
        raise ValueError('Objective weights must be finite and nonnegative')
    first = next(iter(groups.values()))
    for snapshot in groups.values():
        _compatible(first, snapshot)
    return coefficients


def compose_gradients(
    groups: Mapping[str, GradientSnapshot], weights: Mapping[str, float],
) -> GradientSnapshot:
    """Return sum_s weights[s]*g_s, without normalizing or changing inputs."""
    coefficients = _weights(groups, weights)
    first = next(iter(groups.values()))
    result = torch.zeros_like(first.values)
    for name, snapshot in groups.items():
        result.add_(snapshot.values, alpha=coefficients[name])
    combined = GradientSnapshot(first.parameter_names, first.parameter_shapes, result)
    _validate(combined)
    return combined


def compare_gradients(left: GradientSnapshot, right: GradientSnapshot) -> dict:
    """FP64 dot/cosine; an undefined zero-norm cosine is JSON ``null``.

    A negative dot implies local damage to the *left* objective when descending
    the right gradient with positive learning rate. Zero is stationary to first
    order. Norm alone is never classified as conflict.
    """
    _compatible(left, right)
    left_norm = float(torch.linalg.vector_norm(left.values))
    right_norm = float(torch.linalg.vector_norm(right.values))
    dot = float(torch.dot(left.values, right.values))
    if not all(math.isfinite(x) for x in (left_norm, right_norm, dot)):
        raise ValueError('Gradient norm/dot overflowed FP64')
    cosine = None if left_norm == 0 or right_norm == 0 else dot / left_norm / right_norm
    if cosine is not None:
        cosine = min(1.0, max(-1.0, cosine))
    return {
        'left_norm': left_norm,
        'right_norm': right_norm,
        'dot': dot,
        'cosine': cosine,
        'negative_dot_conflict': dot < 0,
        'first_order_left_loss_change_for_right_descent_per_unit_lr': -dot,
        'direction_scope': 'local_unpreconditioned_sgd_not_historical_adam',
    }


def summarize_gradients(
    groups: Mapping[str, GradientSnapshot], weights: Mapping[str, float],
) -> dict:
    """Compare raw groups and their actual weighted objective direction.

    ``weighted_source_dots`` sum to each group's ``dot_with_objective``; this
    permits distinguishing an adverse replay contribution from a net adverse
    objective, rather than equating pairwise conflict with net harm.
    """
    coefficients = _weights(groups, weights)
    objective = compose_gradients(groups, coefficients)
    pairwise = {
        name: {other: compare_gradients(left, right) for other, right in groups.items()}
        for name, left in groups.items()
    }
    per_group = {}
    for name, snapshot in groups.items():
        combined = compare_gradients(snapshot, objective)
        per_group[name] = {
            'norm': combined['left_norm'],
            'objective_coefficient': coefficients[name],
            'weighted_norm': coefficients[name] * combined['left_norm'],
            'missing_gradients': list(snapshot.missing_gradients),
            'dot_with_objective': combined['dot'],
            'cosine_with_objective': combined['cosine'],
            'first_order_loss_change_per_unit_lr':
                combined['first_order_left_loss_change_for_right_descent_per_unit_lr'],
            'negative_dot_conflict_with_objective': combined['negative_dot_conflict'],
            'weighted_source_dots': {
                other: coefficients[other] * pairwise[name][other]['dot'] for other in groups
            },
        }
    return {
        'parameter_count': objective.numel,
        'parameter_tensor_count': len(objective.parameter_names),
        'accumulation_dtype': 'float64',
        'accumulation_device': 'cpu',
        'weights': coefficients,
        'weight_sum': sum(coefficients.values()),
        'objective_norm': float(torch.linalg.vector_norm(objective.values)),
        'pairwise': pairwise,
        'groups': per_group,
        'interpretation': 'Fixed-checkpoint local SGD directions; not historical Adam updates, '
                          'whole-training causality, or capability improvement.',
    }
