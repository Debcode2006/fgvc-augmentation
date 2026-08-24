"""Frozen hardest-competitor selection and the pairwise discriminative margin.

Definition (Experiment 0A, not negotiable)
------------------------------------------
For a validation image ``x`` with true label ``y``, the hardest competing class
is taken from the ORIGINAL image only:

.. math::

    c(x) = \\arg\\max_{k \\neq y} p(k \\mid x)

``c(x)`` is then **frozen**: every transformation ``T`` of that image is scored
against the same pair ``(y, c(x))``. If the strongest incorrect class of
``T(x)`` becomes some other class, that does *not* change ``c(x)``. Re-selecting
the competitor per transformation would silently redefine the quantity being
measured and would make ``delta_margin`` incomparable across transformations.

The pairwise discriminative margin is

.. math::

    M_{yc}(x) = p_y(x) - p_c(x)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["OriginalState", "compute_original_state", "pairwise_margin"]


@dataclass(frozen=True)
class OriginalState:
    """Per-image quantities derived from the ORIGINAL image, frozen thereafter."""

    image_ids: np.ndarray
    """(N,) official CUB image ids."""

    true_labels: np.ndarray
    """(N,) true labels y (contiguous 0..C-1)."""

    competitor_labels: np.ndarray
    """(N,) frozen hardest competitors c(x)."""

    true_prob: np.ndarray
    """(N,) p_y(x)."""

    competitor_prob: np.ndarray
    """(N,) p_c(x)."""

    margin: np.ndarray
    """(N,) M_yc(x) = p_y(x) - p_c(x)."""

    predictions: np.ndarray
    """(N,) argmax_k p(k|x)."""

    correct: np.ndarray
    """(N,) bool, argmax p(x) == y."""

    features: np.ndarray
    """(N, D) penultimate representation z = f(x)."""

    def __len__(self) -> int:
        return int(self.image_ids.shape[0])

    @property
    def accuracy(self) -> float:
        return float(self.correct.mean()) if len(self) else float("nan")


def compute_original_state(
    image_ids: np.ndarray,
    true_labels: np.ndarray,
    probs: np.ndarray,
    features: np.ndarray,
) -> OriginalState:
    """Select the frozen competitor and compute the original margin.

    ``probs`` is (N, C); the true class is masked out before the arg-max so the
    competitor is genuinely the strongest *other* class - whether or not the
    model's top-1 prediction is correct.
    """
    if probs.ndim != 2:
        raise ValueError(f"probs must be a 2-D (N, C) array, got shape {probs.shape}.")
    num_images, num_classes = probs.shape
    if num_classes < 2:
        raise ValueError(
            f"A hardest competitor requires at least 2 classes, got {num_classes}."
        )
    if true_labels.shape[0] != num_images:
        raise ValueError(
            f"true_labels has {true_labels.shape[0]} rows but probs has {num_images}."
        )

    rows = np.arange(num_images)
    masked = probs.copy()
    masked[rows, true_labels] = -np.inf  # exclude y from the arg-max
    competitor_labels = masked.argmax(axis=1).astype(np.int64)

    true_prob = probs[rows, true_labels].astype(np.float64)
    competitor_prob = probs[rows, competitor_labels].astype(np.float64)
    predictions = probs.argmax(axis=1).astype(np.int64)

    return OriginalState(
        image_ids=image_ids.astype(np.int64),
        true_labels=true_labels.astype(np.int64),
        competitor_labels=competitor_labels,
        true_prob=true_prob,
        competitor_prob=competitor_prob,
        margin=pairwise_margin(true_prob, competitor_prob),
        predictions=predictions,
        correct=(predictions == true_labels),
        features=features,
    )


def pairwise_margin(true_prob: np.ndarray, competitor_prob: np.ndarray) -> np.ndarray:
    """Return ``M = p_y - p_c``, the pairwise discriminative margin."""
    return np.asarray(true_prob, dtype=np.float64) - np.asarray(competitor_prob, dtype=np.float64)
