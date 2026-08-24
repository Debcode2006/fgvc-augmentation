"""Per-transformation summary statistics.

These are **descriptive** statistics only. Experiment 0A implements no
hypothesis test, so nothing here licenses a claim of statistical significance,
and a higher mean ``delta_margin`` does not make a transformation "better" - the
audit measures the phenomenon, it does not rank augmentations.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from ..config import Config
from ..logging_utils import get_logger

__all__ = ["summarize_by_transform", "filter_records"]

logger = get_logger(__name__)


def filter_records(records: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Apply the configured analysis filter to the audit records."""
    if not bool(config.get("analysis.filter_correct_original_only")):
        return records
    filtered = records[records["correct_original"].astype(bool)]
    logger.info(
        "analysis.filter_correct_original_only is on: kept %d of %d row(s) "
        "(images the baseline already classified correctly).",
        len(filtered), len(records),
    )
    return filtered


def summarize_by_transform(
    records: pd.DataFrame,
    quantiles: Sequence[float],
    *,
    transform_order: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Summarise ``delta_margin`` and the supporting metrics per transformation.

    Returns one row per transformation with sample counts, the ``delta_margin``
    location/spread/quantiles, the fraction of samples whose separation shrank,
    mean feature cosine, prediction-consistency rate, and top-1 accuracy before
    and after the transformation.
    """
    rows: list[dict[str, float | str | int]] = []
    names = list(transform_order) if transform_order else list(
        records["transform"].drop_duplicates()
    )

    for name in names:
        group = records[records["transform"] == name]
        if group.empty:
            logger.warning("No audit rows for transformation '%s'; skipping in the summary.", name)
            continue
        delta = group["delta_margin"].to_numpy(dtype=np.float64)
        summary: dict[str, float | str | int] = {
            "transform": name,
            "n_samples": int(len(group)),
            "delta_margin_mean": float(delta.mean()),
            "delta_margin_median": float(np.median(delta)),
            "delta_margin_std": float(delta.std(ddof=1)) if len(delta) > 1 else float("nan"),
            "delta_margin_min": float(delta.min()),
            "delta_margin_max": float(delta.max()),
            "delta_margin_mean_abs": float(np.abs(delta).mean()),
            "frac_delta_negative": float((delta < 0).mean()),
            "frac_delta_positive": float((delta > 0).mean()),
            "original_margin_mean": float(group["original_margin"].mean()),
            "augmented_margin_mean": float(group["augmented_margin"].mean()),
            "feature_cosine_mean": float(group["feature_cosine"].mean()),
            "feature_cosine_median": float(group["feature_cosine"].median()),
            "prediction_consistency_rate": float(group["prediction_consistent"].astype(bool).mean()),
            "accuracy_original": float(group["correct_original"].astype(bool).mean()),
            "accuracy_augmented": float(group["correct_augmented"].astype(bool).mean()),
        }
        summary["accuracy_delta"] = summary["accuracy_augmented"] - summary["accuracy_original"]
        for quantile in quantiles:
            key = f"delta_margin_q{_format_quantile(quantile)}"
            summary[key] = float(np.quantile(delta, float(quantile)))
        rows.append(summary)

    frame = pd.DataFrame(rows)
    logger.info("Built summary table for %d transformation(s).", len(frame))
    return frame


def _format_quantile(quantile: float) -> str:
    """Render a quantile as a compact column suffix (0.05 -> ``05``)."""
    text = f"{float(quantile):.4f}".rstrip("0").rstrip(".")
    return text.split(".", 1)[1] if "." in text else text
