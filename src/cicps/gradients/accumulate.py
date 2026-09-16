"""Online gradient inner products - never store a raw gradient vector.

A ResNet-18 gradient is ~11.2 M floats (45 MB). Persisting one per
``(checkpoint, transform, image)`` would run to hundreds of gigabytes and would
buy nothing: every quantity Experiment 1A reports is a function of three scalars
per scope,

.. math::

    \\langle g_x, g_T \\rangle, \\qquad \\lVert g_x \\rVert^2, \\qquad \\lVert g_T \\rVert^2

from which the cosine and the norm ratio follow. This module accumulates exactly
those, per scope, in float64 - gradients are float32, and a naive float32 sum
over 11 M terms loses precision that the cosine would then inherit.

A gradient's own per-parameter squared norms are computed once and cached on the
vector, because the clean gradient is compared against every transformation of
its image: recomputing them per comparison would repeat the most expensive
reduction seven times over.

Only a bounded number of gradient vectors is ever resident: the clean one for the
image being processed, the transformed one currently being compared against it,
and one buffered clean gradient for the cross-image null control.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

from .scopes import ScopeSet

__all__ = ["PairStatistics", "GradientVector", "capture_gradients", "compare", "scope_norms"]

_ZERO_TOLERANCE = 1e-30
"""Squared norm below which a gradient counts as numerically zero."""


class GradientVector:
    """One captured gradient, held as per-parameter tensors with cached norms."""

    __slots__ = ("tensors", "loss", "_squares")

    def __init__(self, tensors: tuple[torch.Tensor, ...], loss: float) -> None:
        self.tensors = tensors
        """Detached clones, aligned with :attr:`ScopeSet.parameter_names`."""
        self.loss = float(loss)
        """The scalar loss this gradient was taken of."""
        self._squares: tuple[float, ...] | None = None

    def __len__(self) -> int:
        return len(self.tensors)

    @property
    def squares(self) -> tuple[float, ...]:
        """Per-parameter ``sum(g * g)``, computed once on first use."""
        if self._squares is None:
            self._squares = _reduce(self.tensors, self.tensors)
        return self._squares


class PairStatistics:
    """Scope-wise comparison of two gradients.

    Entry ``k`` of each tuple corresponds to ``scopes.names[k]``.
    """

    __slots__ = ("cosine", "norm_ratio", "norm_left", "norm_right", "degenerate")

    def __init__(self, cosine, norm_ratio, norm_left, norm_right, degenerate) -> None:
        self.cosine = cosine
        self.norm_ratio = norm_ratio
        self.norm_left = norm_left
        self.norm_right = norm_right
        self.degenerate = degenerate
        """True when either gradient was numerically zero in the primary scope, so
        the cosine there is undefined and is reported as NaN rather than invented."""


def capture_gradients(
    parameters: Sequence[torch.nn.Parameter], loss: torch.Tensor
) -> GradientVector:
    """Clone the current ``.grad`` of every parameter into a detached vector.

    The caller is responsible for having zeroed the gradients and run
    ``loss.backward()``. A parameter whose gradient is ``None`` (no path to the
    loss) contributes an explicit zero rather than being skipped, so the ordering
    stays aligned with the scope indices.
    """
    tensors = tuple(
        torch.zeros_like(parameter) if parameter.grad is None
        else parameter.grad.detach().clone()
        for parameter in parameters
    )
    return GradientVector(tensors=tensors, loss=float(loss.detach()))


def scope_norms(gradient: GradientVector, scopes: ScopeSet) -> tuple[float, ...]:
    """Euclidean norm of ``gradient`` within each scope."""
    return tuple(
        math.sqrt(math.fsum(gradient.squares[index] for index in scope.indices))
        for scope in scopes.scopes
    )


def compare(left: GradientVector, right: GradientVector, scopes: ScopeSet) -> PairStatistics:
    """Cosine similarity and norm ratio of two gradients, scope by scope.

    All reductions accumulate in float64, and scope totals are summed with
    :func:`math.fsum` so the result does not depend on parameter ordering. The
    cosine is formed as ``dot / (|left| * |right|)`` - two square roots of
    separately accumulated sums rather than one square root of their product -
    which keeps the ``identity`` control arm at 1.0 to within a couple of ulps.
    """
    if not (len(left) == len(right) == len(scopes.parameter_names)):
        raise ValueError(
            f"Gradient vectors have {len(left)} and {len(right)} tensors but the scope set "
            f"describes {len(scopes.parameter_names)} parameters."
        )

    dots = _reduce(left.tensors, right.tensors)
    left_squares = left.squares
    right_squares = right.squares

    cosines: list[float] = []
    ratios: list[float] = []
    left_norms: list[float] = []
    right_norms: list[float] = []
    degenerate = False

    for scope in scopes.scopes:
        dot = math.fsum(dots[index] for index in scope.indices)
        left_square = math.fsum(left_squares[index] for index in scope.indices)
        right_square = math.fsum(right_squares[index] for index in scope.indices)
        left_norm = math.sqrt(left_square)
        right_norm = math.sqrt(right_square)

        if left_square <= _ZERO_TOLERANCE or right_square <= _ZERO_TOLERANCE:
            # A zero gradient has no direction. Reporting NaN and flagging it is
            # the only honest option; a 0 or a 1 here would silently corrupt
            # every downstream average.
            cosine = float("nan")
            ratio = float("nan")
            if scope.name == scopes.primary:
                degenerate = True
        else:
            cosine = dot / (left_norm * right_norm)
            ratio = right_norm / left_norm

        cosines.append(cosine)
        ratios.append(ratio)
        left_norms.append(left_norm)
        right_norms.append(right_norm)

    return PairStatistics(
        cosine=tuple(cosines),
        norm_ratio=tuple(ratios),
        norm_left=tuple(left_norms),
        norm_right=tuple(right_norms),
        degenerate=degenerate,
    )


def _reduce(
    left: Sequence[torch.Tensor], right: Sequence[torch.Tensor]
) -> tuple[float, ...]:
    """Per-parameter ``sum(left * right)`` accumulated in float64.

    One device synchronisation for the whole vector rather than one per tensor.
    """
    partials = torch.stack([
        torch.sum(a * b, dtype=torch.float64) for a, b in zip(left, right)
    ])
    return tuple(float(value) for value in partials.cpu())
