"""Experiment 1A dataframe schemas.

Two tables are written.

``gradient_records``
    One row per ``(checkpoint stage, transformation, realisation, image)``. It
    carries the gradient comparison at every measured scope plus the zeroth-order
    quantities (losses, margins, ``delta_margin``) needed to relate the learning
    signal back to Experiment 0A on the same image.

``clean_reference``
    One row per ``(checkpoint stage, image)``. The clean pass is computed once
    per image and shared by every transformation, exactly as Experiment 0A
    computes the original pass once; this table is where it is persisted, along
    with the cross-image null control.

Raw gradient vectors are never written. Every reported quantity is a function of
the three float64 scalars per scope that :mod:`cicps.gradients.accumulate`
computes online.
"""

from __future__ import annotations

import pandas as pd

__all__ = [
    "RECORD_COLUMNS",
    "REQUIRED_RECORD_COLUMNS",
    "CLEAN_COLUMNS",
    "scope_column",
    "order_records",
    "validate_records",
]

REQUIRED_RECORD_COLUMNS: tuple[str, ...] = (
    "checkpoint",
    "transform",
    "realisation",
    "image_id",
    "true_class",
    "hard_competitor",
    "clean_loss",
    "transformed_loss",
    "clean_margin",
    "transformed_margin",
    "delta_margin",
    "grad_cosine",
    "grad_norm_ratio",
    "clean_grad_norm",
    "transformed_grad_norm",
    "feature_cosine",
    "prediction_consistent",
    "correct_clean",
    "correct_transformed",
)
"""Columns Experiment 1A mandates. The ``grad_*`` four are the primary-scope
measurement; per-scope columns are added alongside them."""

RECORD_COLUMNS: tuple[str, ...] = (
    # stage identity
    "checkpoint",
    "checkpoint_kind",
    "checkpoint_epoch",
    "checkpoint_policy",
    # sample identity
    "image_id",
    "image_path",
    "true_class",
    "true_label",
    "true_class_name",
    "hard_competitor",
    "hard_competitor_label",
    # transformation
    "transform",
    "transform_type",
    "realisation",
    "transform_params",
    # zeroth-order quantities
    "clean_loss",
    "transformed_loss",
    "delta_loss",
    "clean_true_prob",
    "transformed_true_prob",
    "clean_competitor_prob",
    "transformed_competitor_prob",
    "clean_margin",
    "transformed_margin",
    "delta_margin",
    # primary gradient measurement (primary scope)
    "grad_cosine",
    "grad_norm_ratio",
    "clean_grad_norm",
    "transformed_grad_norm",
    "grad_degenerate",
    # supporting metrics
    "feature_cosine",
    "prediction_consistent",
    "correct_clean",
    "correct_transformed",
)
"""Canonical column order; per-scope ``cosine_<scope>`` / ``norm_ratio_<scope>``
columns and any joined Experiment 0A columns are appended after these."""

CLEAN_COLUMNS: tuple[str, ...] = (
    "checkpoint",
    "checkpoint_kind",
    "checkpoint_epoch",
    "image_id",
    "true_class",
    "true_label",
    "hard_competitor",
    "hard_competitor_label",
    "clean_loss",
    "clean_true_prob",
    "clean_competitor_prob",
    "clean_margin",
    "clean_prediction",
    "correct_clean",
    "clean_grad_norm",
    "null_cosine",
    "null_partner_image_id",
    "null_partner_same_class",
)
"""Per-image clean reference, including the cross-image null control."""


def scope_column(prefix: str, scope: str) -> str:
    """Name of a per-scope column, e.g. ``cosine_head``."""
    return f"{prefix}_{scope}"


def validate_records(frame: pd.DataFrame) -> None:
    """Raise when the gradient dataframe is missing a mandated column."""
    missing = [column for column in REQUIRED_RECORD_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Experiment 1A gradient dataframe is missing required column(s): "
            f"{', '.join(missing)}."
        )


def order_records(frame: pd.DataFrame) -> pd.DataFrame:
    """Return ``frame`` with the canonical columns first, extras appended."""
    known = [column for column in RECORD_COLUMNS if column in frame.columns]
    extras = [column for column in frame.columns if column not in RECORD_COLUMNS]
    return frame[known + extras]
