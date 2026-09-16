"""Experiment 1A analysis: summaries, relationships and the 0A / 0B joins.

Three deliberate restraints, inherited from Experiments 0A and 0B.

**Descriptive first.** Per-``(checkpoint, transformation)`` statistics are
reported with a bootstrap confidence interval on the mean, because the paired
per-image sample is large (hundreds of images). Comparisons *across*
transformations, by contrast, have only five or six points, so they are reported
as coefficients with ``n`` attached and no significance test - exactly as
Experiment 0B does.

**No composite score.** Gradient cosine and gradient norm ratio are kept
separate. Collapsing them into one "compatibility" number would invent a
weighting nobody has justified.

**Cosine is not usefulness.** A high ``C_g`` means the transformed sample points
the update where the clean sample already pointed it - which is as consistent
with "redundant" as with "safe". The analysis reports the relationships; it does
not rank augmentations.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from ..config import Config, ConfigError
from ..logging_utils import get_logger
from ..seeding import derive_seed

__all__ = [
    "describe_association",
    "summarize_gradients",
    "summarize_null_control",
    "relate_to_delta_margin",
    "join_with_0a",
    "join_with_0b",
    "bootstrap_mean_ci",
]

logger = get_logger(__name__)

EXPLORATORY_NOTE = "exploratory / descriptive only; no significance test"


def bootstrap_mean_ci(
    values: np.ndarray, *, resamples: int, confidence: float, seed: int
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean.

    Resampling is over images, which is the unit of independence here: the rows
    of one transformation arm are one measurement per image.
    """
    finite = values[np.isfinite(values)]
    if finite.size < 2 or resamples < 1:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(finite, size=(int(resamples), finite.size), replace=True).mean(axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def summarize_gradients(
    records: pd.DataFrame,
    *,
    quantiles: Sequence[float],
    scopes: Sequence[str],
    checkpoint_order: Sequence[str],
    transform_order: Sequence[str],
    bootstrap_resamples: int,
    confidence: float,
    seed: int,
) -> pd.DataFrame:
    """One row per ``(checkpoint, transformation)`` with the primary metrics."""
    rows: list[dict] = []
    for checkpoint in checkpoint_order:
        stage_rows = records[records["checkpoint"] == checkpoint]
        if stage_rows.empty:
            continue
        for transform in transform_order:
            group = stage_rows[stage_rows["transform"] == transform]
            if group.empty:
                continue
            cosine = group["grad_cosine"].to_numpy(dtype=np.float64)
            ratio = group["grad_norm_ratio"].to_numpy(dtype=np.float64)
            low, high = bootstrap_mean_ci(
                cosine, resamples=bootstrap_resamples, confidence=confidence,
                # BLAKE2b, never Python's per-process-salted hash(): the interval
                # must reproduce across runs and machines.
                seed=derive_seed(seed, "bootstrap", checkpoint, transform) % (2**31),
            )
            summary: dict = {
                "checkpoint": checkpoint,
                "transform": transform,
                "n_samples": int(len(group)),
                "n_images": int(group["image_id"].nunique()),
                "n_degenerate": int(group["grad_degenerate"].astype(bool).sum()),
                "grad_cosine_mean": _nanmean(cosine),
                "grad_cosine_median": _nanmedian(cosine),
                "grad_cosine_std": _nanstd(cosine),
                "grad_cosine_min": _nanmin(cosine),
                "grad_cosine_max": _nanmax(cosine),
                "grad_cosine_ci_low": low,
                "grad_cosine_ci_high": high,
                "frac_cosine_negative": _fraction(cosine < 0),
                "grad_norm_ratio_mean": _nanmean(ratio),
                "grad_norm_ratio_median": _nanmedian(ratio),
                "grad_norm_ratio_std": _nanstd(ratio),
                "frac_norm_ratio_above_one": _fraction(ratio > 1.0),
                "clean_grad_norm_mean": _nanmean(
                    group["clean_grad_norm"].to_numpy(dtype=np.float64)),
                "transformed_grad_norm_mean": _nanmean(
                    group["transformed_grad_norm"].to_numpy(dtype=np.float64)),
                "clean_loss_mean": float(group["clean_loss"].mean()),
                "transformed_loss_mean": float(group["transformed_loss"].mean()),
                "delta_loss_mean": float(group["delta_loss"].mean()),
                "clean_margin_mean": float(group["clean_margin"].mean()),
                "transformed_margin_mean": float(group["transformed_margin"].mean()),
                "delta_margin_mean": float(group["delta_margin"].mean()),
                "delta_margin_median": float(group["delta_margin"].median()),
                "frac_delta_margin_negative": _fraction(
                    group["delta_margin"].to_numpy(dtype=np.float64) < 0),
                "feature_cosine_mean": float(group["feature_cosine"].mean()),
                "prediction_consistency_rate": float(
                    group["prediction_consistent"].astype(bool).mean()),
                "accuracy_clean": float(group["correct_clean"].astype(bool).mean()),
                "accuracy_transformed": float(group["correct_transformed"].astype(bool).mean()),
            }
            summary["accuracy_delta"] = (
                summary["accuracy_transformed"] - summary["accuracy_clean"]
            )
            for quantile in quantiles:
                key = _quantile_suffix(quantile)
                summary[f"grad_cosine_q{key}"] = _nanquantile(cosine, quantile)
                summary[f"grad_norm_ratio_q{key}"] = _nanquantile(ratio, quantile)
            for scope in scopes:
                column = f"cosine_{scope}"
                if column in group.columns:
                    summary[f"grad_cosine_mean_{scope}"] = _nanmean(
                        group[column].to_numpy(dtype=np.float64))
                column = f"norm_ratio_{scope}"
                if column in group.columns:
                    summary[f"grad_norm_ratio_mean_{scope}"] = _nanmean(
                        group[column].to_numpy(dtype=np.float64))
            rows.append(summary)

    frame = pd.DataFrame(rows)
    logger.info("Built gradient summary for %d (checkpoint x transformation) cell(s).", len(frame))
    return frame


def summarize_null_control(
    clean: pd.DataFrame, *, checkpoint_order: Sequence[str],
    bootstrap_resamples: int, confidence: float, seed: int,
) -> pd.DataFrame:
    """Per-checkpoint summary of the cross-image null control.

    This is the scale reference for every cosine in the experiment: it is what
    two *unrelated* learning signals score at this checkpoint and this scope.
    Without it a value like ``C_g = 0.2`` cannot be called high or low.
    """
    rows: list[dict] = []
    for checkpoint in checkpoint_order:
        group = clean[clean["checkpoint"] == checkpoint]
        if group.empty or "null_cosine" not in group.columns:
            continue
        values = group["null_cosine"].to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        low, high = bootstrap_mean_ci(
            finite, resamples=bootstrap_resamples, confidence=confidence,
            seed=derive_seed(seed, "bootstrap_null", checkpoint) % (2**31),
        )
        same_class = group["null_partner_same_class"].astype(bool).to_numpy()
        rows.append({
            "checkpoint": checkpoint,
            "n_pairs": int(finite.size),
            "null_cosine_mean": float(finite.mean()),
            "null_cosine_median": float(np.median(finite)),
            "null_cosine_std": float(finite.std(ddof=1)) if finite.size > 1 else float("nan"),
            "null_cosine_ci_low": low,
            "null_cosine_ci_high": high,
            "null_cosine_q05": float(np.quantile(finite, 0.05)),
            "null_cosine_q95": float(np.quantile(finite, 0.95)),
            "frac_null_negative": float((finite < 0).mean()),
            "n_same_class_pairs": int(same_class[np.isfinite(values)].sum()),
            "clean_grad_norm_mean": float(group["clean_grad_norm"].mean()),
            "clean_loss_mean": float(group["clean_loss"].mean()),
            "accuracy_clean": float(group["correct_clean"].astype(bool).mean()),
        })
    return pd.DataFrame(rows)


def relate_to_delta_margin(
    records: pd.DataFrame,
    *,
    checkpoint_order: Sequence[str],
    transform_order: Sequence[str],
    control_transform: str,
) -> pd.DataFrame:
    """Within-transformation association between the gradient and margin views.

    Computed *across images* within one ``(checkpoint, transformation)`` cell, so
    ``n`` is the number of images - hundreds, not five. This is the analysis that
    answers the non-redundancy question: if ``C_g`` were a repackaging of
    ``delta_margin``, these coefficients would be close to |1|.

    The control arm is excluded: its cosine is 1.0 and its ``delta_margin`` 0.0
    by construction, so a correlation over it is undefined.
    """
    rows: list[dict] = []
    for checkpoint in checkpoint_order:
        stage_rows = records[records["checkpoint"] == checkpoint]
        if stage_rows.empty:
            continue
        for transform in list(transform_order) + ["__pooled__"]:
            if transform == control_transform:
                continue
            group = (
                stage_rows[stage_rows["transform"] != control_transform]
                if transform == "__pooled__"
                else stage_rows[stage_rows["transform"] == transform]
            )
            if len(group) < 3:
                continue
            for target, label in (
                ("delta_margin", "grad_cosine_vs_delta_margin"),
                ("feature_cosine", "grad_cosine_vs_feature_cosine"),
            ):
                rows.append(_association_row(
                    checkpoint, transform, label,
                    group["grad_cosine"].to_numpy(dtype=np.float64),
                    group[target].to_numpy(dtype=np.float64),
                ))
            rows.append(_association_row(
                checkpoint, transform, "grad_norm_ratio_vs_delta_margin",
                group["grad_norm_ratio"].to_numpy(dtype=np.float64),
                group["delta_margin"].to_numpy(dtype=np.float64),
            ))
    frame = pd.DataFrame(rows)
    logger.info("Built %d within-transformation association row(s).", len(frame))
    return frame


def join_with_0a(
    records: pd.DataFrame, config: Config, anchor_checkpoint: str
) -> pd.DataFrame:
    """Join Experiment 0A's per-image ``delta_margin`` onto the anchor stage's rows.

    The join is on ``(image_id, transform)`` and is only valid for the stage whose
    checkpoint *is* the model Experiment 0A audited, and only for realisation 0,
    where the transformed image is bit-identical to 0A's. The 0A table is opened
    read-only; Experiment 1A never writes into ``outputs/experiment_0a/``.
    """
    path = config.path("analysis.reference_0a_records")
    if not path.is_file():
        raise ConfigError(
            f"Experiment 0A audit dataframe not found at {path}. Run "
            "`python -m cicps audit --config config/experiment_0a.yaml` first, or set "
            "analysis.reference_0a_records to its location."
        )
    reference = pd.read_csv(path)
    columns = ["image_id", "transform", "delta_margin", "original_margin",
               "augmented_margin", "feature_cosine", "hard_competitor"]
    missing = [column for column in columns if column not in reference.columns]
    if missing:
        raise ConfigError(f"Experiment 0A dataframe at {path} lacks column(s): {missing}.")
    reference = reference[columns].rename(columns={
        "delta_margin": "delta_margin_0a",
        "original_margin": "original_margin_0a",
        "augmented_margin": "augmented_margin_0a",
        "feature_cosine": "feature_cosine_0a",
        "hard_competitor": "hard_competitor_0a",
    })

    anchor = records[
        (records["checkpoint"] == anchor_checkpoint) & (records["realisation"] == 0)
    ].copy()
    if anchor.empty:
        raise ConfigError(
            f"No realisation-0 rows for anchor checkpoint '{anchor_checkpoint}'. Set "
            "analysis.anchor_checkpoint to a declared stage measured in this run."
        )
    joined = anchor.merge(reference, on=["image_id", "transform"], how="inner")
    logger.info(
        "Joined Experiment 0A per-image delta_margin onto %d of %d anchor row(s) (stage '%s').",
        len(joined), len(anchor), anchor_checkpoint,
    )
    if joined.empty:
        raise ConfigError(
            "The Experiment 0A join produced no rows. The 1A measurement and the 0A audit must "
            "cover the same validation images and transformation names."
        )
    return joined


def join_with_0b(
    summary: pd.DataFrame, config: Config, checkpoint: str
) -> pd.DataFrame:
    """Put each transformation's gradient statistics next to its 0B downstream result.

    Five transformation points. The coefficients this supports are descriptive
    and carry ``n``; nothing here is a significance test, and a correspondence
    across five transformations cannot separate transformation identity from
    anything else those transformations share.
    """
    path = config.path("analysis.reference_0b_runs")
    if not path.is_file():
        raise ConfigError(
            f"Experiment 0B run table not found at {path}. Run "
            "`python -m cicps train --config config/experiment_0b.yaml` first, or set "
            "analysis.reference_0b_runs to its location."
        )
    runs = pd.read_csv(path)
    baseline_name = str(config.get("analysis.reference_0b_baseline"))
    baseline = runs[runs["policy"] == baseline_name]
    if baseline.empty:
        raise ConfigError(
            f"Experiment 0B run table has no policy '{baseline_name}' to use as the reference "
            f"arm. Policies present: {', '.join(runs['policy'].astype(str))}."
        )
    baseline_top1 = float(baseline["best_val_top1"].iloc[0])

    downstream = runs[runs["audit_transform"].notna() & (runs["audit_transform"] != "")].copy()
    downstream = downstream[["policy", "audit_transform", "best_val_top1", "best_val_top5",
                             "best_val_loss", "best_epoch"]]
    downstream = downstream.rename(columns={
        "policy": "policy_0b",
        "audit_transform": "transform",
        "best_val_top1": "val_top1_0b",
        "best_val_top5": "val_top5_0b",
        "best_val_loss": "val_loss_0b",
        "best_epoch": "best_epoch_0b",
    })
    downstream["val_top1_delta_0b"] = downstream["val_top1_0b"] - baseline_top1
    downstream["baseline_val_top1_0b"] = baseline_top1

    cell = summary[summary["checkpoint"] == checkpoint]
    if cell.empty:
        raise ConfigError(
            f"No summary rows for checkpoint '{checkpoint}'; cannot join against Experiment 0B."
        )
    joined = cell.merge(downstream, on="transform", how="inner")
    logger.info(
        "Joined %d transformation(s) against Experiment 0B downstream results "
        "(baseline val top-1 %.4f).", len(joined), baseline_top1,
    )
    return joined


def describe_association(
    frame: pd.DataFrame, pairs: Sequence[tuple[str, str]], *, methods: Sequence[str],
    context: str,
) -> pd.DataFrame:
    """Correlation coefficients over a small number of transformation points."""
    rows: list[dict] = []
    for left, right in pairs:
        if left not in frame.columns or right not in frame.columns:
            continue
        x = frame[left].to_numpy(dtype=np.float64)
        y = frame[right].to_numpy(dtype=np.float64)
        mask = np.isfinite(x) & np.isfinite(y)
        for method in methods:
            rows.append({
                "context": context,
                "x": left,
                "y": right,
                "method": method,
                "coefficient": _coefficient(x[mask], y[mask], method),
                "n_points": int(mask.sum()),
                "note": EXPLORATORY_NOTE,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _association_row(checkpoint: str, transform: str, label: str,
                     x: np.ndarray, y: np.ndarray) -> dict:
    mask = np.isfinite(x) & np.isfinite(y)
    return {
        "checkpoint": checkpoint,
        "transform": "pooled (non-control)" if transform == "__pooled__" else transform,
        "relationship": label,
        "pearson": _coefficient(x[mask], y[mask], "pearson"),
        "spearman": _coefficient(x[mask], y[mask], "spearman"),
        "n_samples": int(mask.sum()),
        "note": EXPLORATORY_NOTE,
    }


def _coefficient(x: np.ndarray, y: np.ndarray, method: str) -> float:
    if x.size < 3:
        return float("nan")
    if method == "spearman":
        x, y = _rank(x), _rank(y)
    elif method != "pearson":
        raise ConfigError(f"Unknown correlation method '{method}'. Supported: pearson, spearman.")
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rank(values: np.ndarray) -> np.ndarray:
    order = values.argsort()
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    # Average ties so the Spearman coefficient is the usual one.
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if counts.max() > 1:
        sums = np.zeros(unique.size, dtype=np.float64)
        np.add.at(sums, inverse, ranks)
        ranks = (sums / counts)[inverse]
    return ranks


def _fraction(mask: np.ndarray) -> float:
    mask = np.asarray(mask)
    return float(mask.mean()) if mask.size else float("nan")


def _nanmean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def _nanmedian(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else float("nan")


def _nanstd(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.std(ddof=1)) if finite.size > 1 else float("nan")


def _nanmin(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.min()) if finite.size else float("nan")


def _nanmax(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.max()) if finite.size else float("nan")


def _nanquantile(values: np.ndarray, quantile: float) -> float:
    finite = values[np.isfinite(values)]
    return float(np.quantile(finite, float(quantile))) if finite.size else float("nan")


def _quantile_suffix(quantile: float) -> str:
    text = f"{float(quantile):.4f}".rstrip("0").rstrip(".")
    return text.split(".", 1)[1] if "." in text else text
