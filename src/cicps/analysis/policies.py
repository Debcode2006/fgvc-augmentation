"""Experiment 0B: policy summaries and the Experiment 0A / 0B join.

Two tables are produced.

``policy_summary``
    One row per trained policy: the validation outcome plus its change relative
    to the ``baseline`` arm. This is the downstream evidence.

``joined_0a_0b_summary``
    The same rows joined against Experiment 0A's per-transformation summary, so
    that a transformation's frozen-model margin damage sits next to the
    downstream result of training with it. This is the table the experiment
    exists to produce.

The join is *analysis only*. No 0A quantity influences which policies exist,
what strengths they use, or how long they train - that is fixed in the
configuration before any 0B model is trained. Everything here is descriptive:
with a handful of transformations, no significance claim is available, and the
optional correlations are reported as exploratory numbers with their sample
count attached.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ..config import Config, ConfigError
from ..logging_utils import get_logger

__all__ = [
    "load_policy_runs",
    "load_reference_0a",
    "build_policy_summary",
    "join_with_0a",
    "describe_association",
    "load_policy_histories",
]

logger = get_logger(__name__)

_REFERENCE_COLUMNS: tuple[str, ...] = (
    "delta_margin_mean",
    "delta_margin_median",
    "delta_margin_std",
    "delta_margin_mean_abs",
    "frac_delta_negative",
    "frac_delta_positive",
    "feature_cosine_mean",
    "prediction_consistency_rate",
    "accuracy_original",
    "accuracy_augmented",
    "accuracy_delta",
)
"""Experiment 0A columns carried into the joined table, when present."""

_SUMMARY_METRICS: tuple[str, ...] = (
    "best_val_top1",
    "best_val_top5",
    "best_val_loss",
    "final_val_top1",
    "final_val_top5",
    "best_train_top1",
)
"""0B metrics for which a baseline-relative delta is computed."""


def load_policy_runs(config: Config) -> pd.DataFrame:
    """Read ``train_runs.csv`` written by the policy training stage."""
    path = config.path("policies.output.runs_path")
    if not path.is_file():
        raise ConfigError(
            f"Policy run table not found: {path}. Run "
            "`python -m cicps train --config <0B config>` first."
        )
    runs = pd.read_csv(path)
    if runs.empty:
        raise ConfigError(f"Policy run table {path} contains no rows.")
    missing = [column for column in ("policy", "best_val_top1", "best_val_top5")
               if column not in runs.columns]
    if missing:
        raise ConfigError(f"Policy run table {path} is missing column(s): {missing}.")
    duplicated = runs["policy"][runs["policy"].duplicated()].tolist()
    if duplicated:
        raise ConfigError(f"Policy run table {path} repeats policy name(s): {duplicated}.")
    logger.info("Loaded %d policy run row(s) from %s", len(runs), path)
    return runs


def load_reference_0a(config: Config) -> pd.DataFrame:
    """Read Experiment 0A's per-transformation summary (read-only)."""
    path = config.path("analysis.reference_0a.summary_path")
    if not path.is_file():
        raise ConfigError(
            f"Experiment 0A summary not found: {path}. Run Experiment 0A's analyze stage, "
            "or point analysis.reference_0a.summary_path at an existing summary. "
            "Experiment 0B never modifies this file."
        )
    reference = pd.read_csv(path)
    if "transform" not in reference.columns:
        raise ConfigError(f"Experiment 0A summary {path} has no 'transform' column.")
    logger.info(
        "Loaded Experiment 0A summary from %s (%d transformation(s), read-only).",
        path, len(reference),
    )
    return reference


def build_policy_summary(runs: pd.DataFrame, baseline_name: str) -> pd.DataFrame:
    """Add baseline-relative deltas and order the table with the baseline first."""
    if baseline_name not in set(runs["policy"]):
        raise ConfigError(
            f"The reference policy '{baseline_name}' has no row in the policy run table. "
            "Train it before running the analysis: its result is what every other arm is "
            "measured against."
        )
    summary = runs.copy()
    baseline = summary.loc[summary["policy"] == baseline_name].iloc[0]

    for metric in _SUMMARY_METRICS:
        if metric in summary.columns:
            summary[f"delta_{metric}_vs_baseline"] = (
                summary[metric].astype(float) - float(baseline[metric])
            )

    summary["is_baseline_policy"] = summary["policy"] == baseline_name
    summary = pd.concat([
        summary[summary["is_baseline_policy"]],
        summary[~summary["is_baseline_policy"]],
    ], ignore_index=True)
    logger.info("Built policy summary for %d policy(ies).", len(summary))
    return summary


def join_with_0a(
    summary: pd.DataFrame, reference: pd.DataFrame, baseline_name: str
) -> pd.DataFrame:
    """Join each non-baseline policy to its Experiment 0A transformation row.

    The link comes from ``policies.list[].audit_transform`` in the configuration,
    declared alongside the policy, so the correspondence is explicit rather than
    inferred from a policy name.
    """
    arms = summary[summary["policy"] != baseline_name].copy()
    arms["audit_transform"] = arms["audit_transform"].fillna("").astype(str)
    unlinked = arms.loc[arms["audit_transform"] == "", "policy"].tolist()
    if unlinked:
        raise ConfigError(
            f"Policy(ies) {unlinked} carry no 'audit_transform', so they cannot be joined "
            "against Experiment 0A."
        )

    columns = ["transform"] + [column for column in _REFERENCE_COLUMNS
                               if column in reference.columns]
    joined = arms.merge(
        reference[columns].rename(columns={"transform": "audit_transform"}),
        on="audit_transform", how="left", validate="many_to_one",
    )
    missing = joined.loc[joined["delta_margin_mean"].isna(), "audit_transform"].tolist()
    if missing:
        raise ConfigError(
            f"Experiment 0A has no summary row for transformation(s) {sorted(set(missing))}. "
            "Check that the 0A audit and this 0B configuration name them identically."
        )

    front = [column for column in
             ("audit_transform", "policy", "delta_margin_mean", "delta_margin_median",
              "frac_delta_negative", "delta_margin_mean_abs", "feature_cosine_mean",
              "prediction_consistency_rate", "best_val_top1",
              "delta_best_val_top1_vs_baseline", "best_val_top5",
              "delta_best_val_top5_vs_baseline")
             if column in joined.columns]
    ordered = joined[front + [c for c in joined.columns if c not in front]]
    logger.info("Joined %d policy arm(s) to Experiment 0A transformations.", len(ordered))
    return ordered


def describe_association(
    joined: pd.DataFrame, x_column: str, y_columns: Sequence[str], methods: Sequence[str]
) -> pd.DataFrame:
    """Exploratory association between a 0A metric and 0B outcomes.

    Returned purely as description. With a handful of transformations these
    coefficients are unstable by construction: the sample count is reported on
    every row precisely so that no reader mistakes one for evidence of
    significance. No p-value is computed and no hypothesis test is run.
    """
    rows: list[dict[str, object]] = []
    x = joined[x_column].to_numpy(dtype=np.float64)
    for y_column in y_columns:
        if y_column not in joined.columns:
            logger.warning("No column '%s' to correlate against; skipping.", y_column)
            continue
        y = joined[y_column].to_numpy(dtype=np.float64)
        for method in methods:
            rows.append({
                "x": x_column,
                "y": y_column,
                "method": str(method).lower(),
                "n_points": int(len(x)),
                "coefficient": _coefficient(x, y, str(method).lower()),
                "interpretation": "exploratory / descriptive only; no significance test",
            })
    return pd.DataFrame(rows)


def load_policy_histories(
    summary: pd.DataFrame, config: Config
) -> dict[str, pd.DataFrame]:
    """Load each policy's per-epoch history CSV, keyed by policy name."""
    directory = config.path("policies.output.history_dir")
    histories: dict[str, pd.DataFrame] = {}
    for policy in summary["policy"]:
        path = Path(directory) / f"{policy}.csv"
        if not path.is_file():
            raise ConfigError(
                f"Training history not found for policy '{policy}': {path}. Re-run "
                f"`python -m cicps train --config <0B config> --policy {policy}`."
            )
        histories[str(policy)] = pd.read_csv(path)
    logger.info("Loaded per-epoch history for %d policy(ies).", len(histories))
    return histories


def _coefficient(x: np.ndarray, y: np.ndarray, method: str) -> float:
    """Pearson or Spearman coefficient, computed without a SciPy dependency."""
    if len(x) < 2:
        return float("nan")
    if method == "spearman":
        x, y = _rank(x), _rank(y)
    elif method != "pearson":
        raise ConfigError(
            f"Unknown correlation method '{method}'. Supported: pearson, spearman."
        )
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rank(values: np.ndarray) -> np.ndarray:
    """Average ranks, so ties do not bias the Spearman coefficient."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    for value in np.unique(values):
        mask = values == value
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    return ranks
