"""Audit metrics.

The **primary** metric of Experiment 0A is the change in the pairwise
discriminative margin,

.. math::

    \\Delta M = M_{yc}(T(x)) - M_{yc}(x),

and it is computed explicitly by the audit engine - not through this registry.
It is deliberately not pluggable: replacing it would change the experiment.

Everything registered here is **supporting** evidence: feature consistency,
prediction consistency, and correctness. A future metric is added by writing a
function that maps a :class:`MetricContext` to named columns and decorating it
with :func:`register_metric`; the engine then emits those columns automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from .competitor import OriginalState

__all__ = [
    "MetricContext",
    "register_metric",
    "available_metrics",
    "compute_supporting_metrics",
    "cosine_similarity_rows",
]

MetricFn = Callable[["MetricContext"], Mapping[str, np.ndarray]]

_REGISTRY: dict[str, MetricFn] = {}


@dataclass(frozen=True)
class MetricContext:
    """Everything a supporting metric may look at, for one transformation."""

    transform_name: str
    original: OriginalState
    """Frozen original-image state (labels, competitor, margin, features)."""

    augmented_probs: np.ndarray
    """(N, C) probabilities for T(x)."""

    augmented_features: np.ndarray
    """(N, D) penultimate representations f(T(x))."""

    augmented_predictions: np.ndarray
    """(N,) argmax_k p(k|T(x))."""

    augmented_true_prob: np.ndarray
    """(N,) p_y(T(x))."""

    augmented_competitor_prob: np.ndarray
    """(N,) p_c(T(x)), for the FROZEN competitor c(x)."""

    augmented_margin: np.ndarray
    """(N,) M_yc(T(x))."""

    delta_margin: np.ndarray
    """(N,) the primary metric, M_yc(T(x)) - M_yc(x)."""


def register_metric(name: str) -> Callable[[MetricFn], MetricFn]:
    """Register a supporting metric under ``name``."""

    def decorator(function: MetricFn) -> MetricFn:
        if name in _REGISTRY:
            raise ValueError(f"Metric '{name}' is already registered.")
        _REGISTRY[name] = function
        return function

    return decorator


def available_metrics() -> list[str]:
    """Sorted names of the registered supporting metrics."""
    return sorted(_REGISTRY)


def compute_supporting_metrics(context: MetricContext) -> dict[str, np.ndarray]:
    """Run every registered supporting metric and merge their columns."""
    columns: dict[str, np.ndarray] = {}
    for name, function in sorted(_REGISTRY.items()):
        produced = function(context)
        overlap = set(produced) & set(columns)
        if overlap:
            raise ValueError(
                f"Metric '{name}' produces column(s) {sorted(overlap)} that another metric "
                "already produced. Column names must be unique."
            )
        columns.update({key: np.asarray(value) for key, value in produced.items()})
    return columns


# ---------------------------------------------------------------------------
# Built-in supporting metrics
# ---------------------------------------------------------------------------


@register_metric("feature_consistency")
def feature_consistency(context: MetricContext) -> dict[str, np.ndarray]:
    """Cosine similarity ``Cf = cos(z, zT)`` between penultimate features.

    Supporting evidence only. It says whether the representation moved, not
    whether the *discriminative evidence for y against c(x)* survived - which is
    what ``delta_margin`` measures.
    """
    return {
        "feature_cosine": cosine_similarity_rows(
            context.original.features, context.augmented_features
        )
    }


@register_metric("prediction_consistency")
def prediction_consistency(context: MetricContext) -> dict[str, np.ndarray]:
    """``Cp = 1[argmax p(x) == argmax p(T(x))]``."""
    return {
        "prediction_consistent": context.augmented_predictions == context.original.predictions
    }


@register_metric("correctness")
def correctness(context: MetricContext) -> dict[str, np.ndarray]:
    """Top-1 correctness before and after the transformation."""
    return {
        "correct_original": context.original.correct,
        "correct_augmented": context.augmented_predictions == context.original.true_labels,
    }


def cosine_similarity_rows(
    left: np.ndarray, right: np.ndarray, *, epsilon: float = 1e-12
) -> np.ndarray:
    """Row-wise cosine similarity between two (N, D) matrices."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError(
            f"Feature matrices must have identical shapes, got {left.shape} and {right.shape}."
        )
    numerator = (left * right).sum(axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return numerator / np.maximum(denominator, epsilon)
