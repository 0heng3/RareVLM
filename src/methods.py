"""Small, source-aligned methods for a frozen-feature CLIP long-tail pilot.

The adapter starts as the identity map.  The classifier stores fixed, normalized
text prototypes and never accepts targets during inference.  TailSpecStaticLoss
keeps GALE's frequency scale, target margin, and inverse-frequency weighting;
its epoch boost and center penalty are deliberately absent.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _validated_counts(class_counts: Sequence[int] | Tensor) -> Tensor:
    counts = torch.as_tensor(class_counts, dtype=torch.float32).detach().clone()
    if counts.ndim != 1 or counts.numel() < 2:
        raise ValueError("class_counts must be a one-dimensional vector with at least two classes")
    if not torch.isfinite(counts).all() or (counts <= 0).any():
        raise ValueError("class_counts must contain finite, positive training counts")
    return counts


def _check_logits_targets(logits: Tensor, targets: Tensor, num_classes: int) -> None:
    if logits.ndim != 2 or logits.shape[1] != num_classes:
        raise ValueError(f"logits must have shape [batch, {num_classes}]")
    if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
        raise ValueError("targets must have shape [batch]")


class ResidualAdapter(nn.Module):
    """LN/GELU bottleneck adapter initialized to leave the feature unchanged."""

    def __init__(self, dim: int, bottleneck: int) -> None:
        super().__init__()
        if dim <= 0 or bottleneck <= 0:
            raise ValueError("dim and bottleneck must be positive")
        self.fc1 = nn.Linear(dim, bottleneck)
        self.norm = nn.LayerNorm(bottleneck)
        self.fc2 = nn.Linear(bottleneck, dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, features: Tensor) -> Tensor:
        return features + self.fc2(F.gelu(self.norm(self.fc1(features))))


class PrototypeClassifier(nn.Module):
    """Cosine classifier with frozen text prototypes and a fixed logit scale."""

    def __init__(self, text_prototypes: Tensor, logit_scale: float) -> None:
        super().__init__()
        if not math.isfinite(logit_scale) or logit_scale <= 0:
            raise ValueError("logit_scale must be finite and positive")
        prototypes = torch.as_tensor(text_prototypes, dtype=torch.float32).detach().clone()
        if prototypes.ndim != 2 or prototypes.shape[0] < 2:
            raise ValueError("text_prototypes must have shape [classes >= 2, dim]")
        if not torch.isfinite(prototypes).all() or (prototypes.norm(dim=1) <= 0).any():
            raise ValueError("text_prototypes must contain finite, nonzero vectors")
        self.register_buffer("text_prototypes", F.normalize(prototypes, dim=1))
        self.register_buffer("logit_scale", torch.tensor(float(logit_scale), dtype=torch.float32))

    def forward(self, image_features: Tensor) -> Tensor:
        if image_features.ndim != 2 or image_features.shape[1] != self.text_prototypes.shape[1]:
            raise ValueError("image_features must have shape [batch, prototype_dim]")
        image_unit = F.normalize(image_features.float(), dim=1)
        return self.logit_scale * (image_unit @ self.text_prototypes.T)


class CELoss(nn.Module):
    """Ordinary cross-entropy on the unchanged cosine classifier logits."""

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        return F.cross_entropy(logits, targets)


class LogitAdjustmentLoss(nn.Module):
    """Training-time CE(logits + tau * log(training class prior), targets)."""

    def __init__(self, class_counts: Sequence[int] | Tensor, tau: float) -> None:
        super().__init__()
        if not math.isfinite(tau) or tau < 0:
            raise ValueError("tau must be finite and nonnegative")
        counts = _validated_counts(class_counts)
        self.register_buffer("log_prior", torch.log(counts / counts.sum()))
        self.tau = float(tau)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        _check_logits_targets(logits, targets, self.log_prior.numel())
        return F.cross_entropy(logits.float() + self.tau * self.log_prior, targets)


class TailSpecStaticLoss(nn.Module):
    """Minimal static part of GALE ``losses/tailspec.py``.

    Source-equivalent settings are ``max_margin=0.1``, ``max_logit_scale=70``,
    ``temperature=0.1``, ``inverse_weight_power=1``,
    ``class_weight_gain=1.6*3.25``, and ``class_weight_cap=2``.  All settings
    are required so a VLM experiment can control or ablate each effect.
    """

    def __init__(
        self,
        class_counts: Sequence[int] | Tensor,
        *,
        max_margin: float,
        max_logit_scale: float,
        temperature: float,
        inverse_weight_power: float,
        class_weight_gain: float,
        class_weight_cap: float,
    ) -> None:
        super().__init__()
        values = (
            max_margin,
            max_logit_scale,
            temperature,
            inverse_weight_power,
            class_weight_gain,
            class_weight_cap,
        )
        if not all(math.isfinite(v) for v in values):
            raise ValueError("TailSpec-static hyperparameters must be finite")
        if max_margin < 0 or max_logit_scale < 1 or temperature <= 0:
            raise ValueError("margin must be nonnegative, scale >= 1, and temperature positive")
        if inverse_weight_power < 0 or class_weight_gain <= 0 or class_weight_cap <= 0:
            raise ValueError("weight power must be nonnegative; gain and cap must be positive")

        counts = _validated_counts(class_counts)
        count_range = counts.max() - counts.min()
        if count_range > 0:
            freq_norm = (counts - counts.min()) / count_range
        else:
            freq_norm = torch.zeros_like(counts)
        margins = float(max_margin) * (1.0 - freq_norm)

        log_counts = torch.log1p(counts)
        log_counts = log_counts - log_counts.min()
        scale_range = log_counts.max()
        if scale_range > 0:
            log_counts = log_counts / scale_range
        scales = 1.0 + log_counts * (float(max_logit_scale) - 1.0)

        inverse = counts.pow(-float(inverse_weight_power))
        class_weights = inverse / inverse.max()
        class_weights = (class_weights * float(class_weight_gain)).clamp(max=float(class_weight_cap))

        self.register_buffer("margins", margins)
        self.register_buffer("scales", scales)
        self.register_buffer("class_weights", class_weights)
        self.temperature = float(temperature)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        _check_logits_targets(logits, targets, self.scales.numel())
        scaled = logits.float() * self.scales
        row = torch.arange(targets.numel(), device=targets.device)
        scaled = scaled.clone()
        scaled[row, targets] -= self.margins[targets]
        scaled = scaled / self.temperature
        # GALE uses reduction='none' followed by an unweighted batch mean.
        return F.cross_entropy(scaled, targets, weight=self.class_weights, reduction="none").mean()


def build_loss(
    name: str,
    class_counts: Sequence[int] | Tensor | None = None,
    **hyperparameters: float,
) -> nn.Module:
    """Build CE, LA, or TailSpec-static; each non-CE setting is explicit."""

    key = name.lower().replace("-", "_")
    if key == "ce":
        if hyperparameters:
            raise TypeError("CE does not take hyperparameters")
        return CELoss()
    if class_counts is None:
        raise ValueError("class_counts are required for long-tail losses")
    if key in {"la", "logit_adjustment"}:
        return LogitAdjustmentLoss(class_counts, **hyperparameters)
    if key in {"tailspec_static", "ts"}:
        return TailSpecStaticLoss(class_counts, **hyperparameters)
    raise ValueError(f"unknown loss: {name}")
