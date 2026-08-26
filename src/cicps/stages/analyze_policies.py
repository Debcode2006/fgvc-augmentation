"""Stage: Experiment 0B analysis - policy outcomes joined against Experiment 0A.

Reads the policy run table produced by the training stage, joins it to
Experiment 0A's per-transformation summary (read-only; 0A's files are never
written by this stage), and renders the figures.

The headline question is *not* "which policy scored highest". It is whether the
transformation-level margin damage 0A measured on a frozen model corresponds to
anything downstream when the same transformation is used to train. The stage
therefore refuses to pick a winner: it emits the joined table, the exploratory
association numbers with their sample count, and figures that carry their own
caveats.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..analysis.policies import (
    build_policy_summary,
    describe_association,
    join_with_0a,
    load_policy_histories,
    load_policy_runs,
    load_reference_0a,
)
from ..analysis.policy_plots import PolicyPlotContext, render_policy_plots
from ..audit.records import table_path, write_table
from ..config import Config
from ..logging_utils import get_logger, setup_logging

__all__ = ["run_analyze_policies", "PolicyAnalysisPaths"]

logger = get_logger(__name__)


@dataclass(frozen=True)
class PolicyAnalysisPaths:
    """Artefacts written by the Experiment 0B analysis stage."""

    summary: Path
    joined: Path
    association: Path | None
    plots: list[Path]


def run_analyze_policies(config: Config) -> PolicyAnalysisPaths:
    """Summarise the policy sweep, join it to Experiment 0A, and plot."""
    setup_logging(config, stage="analyze")
    logger.info("Configuration: %s", config.source)
    logger.info("Experiment: %s", config.get("experiment.name"))

    baseline_name = str(config.get("policies.baseline_name"))
    runs = load_policy_runs(config)
    _warn_on_inconsistent_runs(runs)

    summary = build_policy_summary(runs, baseline_name)
    reference = load_reference_0a(config)
    joined = join_with_0a(summary, reference, baseline_name)

    fmt = str(config.get("analysis.output.format"))
    output_dir = config.path("analysis.output.dir")
    summary_path = table_path(output_dir, str(config.get("analysis.output.summary_name")), fmt)
    joined_path = table_path(output_dir, str(config.get("analysis.output.joined_name")), fmt)
    write_table(summary, summary_path, fmt=fmt)
    write_table(joined, joined_path, fmt=fmt)

    association_path = _write_association(config, joined, output_dir, fmt)

    histories = load_policy_histories(summary, config)
    plots = render_policy_plots(
        config, PolicyPlotContext(summary, joined, histories, baseline_name)
    )

    _log_summary(summary, baseline_name)
    _log_joined(joined)
    logger.info(
        "Analysis complete. These are descriptive statistics over %d transformation(s). "
        "Experiment 0B tests whether the 0A phenomenon has downstream relevance; an "
        "association between mean delta_margin and validation accuracy is NOT evidence "
        "that margin damage causes the downstream change.",
        len(joined),
    )
    return PolicyAnalysisPaths(
        summary=summary_path, joined=joined_path, association=association_path, plots=plots
    )


def _write_association(
    config: Config, joined: pd.DataFrame, output_dir: Path, fmt: str
) -> Path | None:
    """Optionally emit the exploratory 0A/0B correlation table."""
    if not bool(config.get("analysis.association.enabled")):
        logger.info("analysis.association.enabled is false; no correlation was computed.")
        return None

    association = describe_association(
        joined,
        x_column=str(config.get("analysis.association.x")),
        y_columns=list(config.get("analysis.association.y")),
        methods=list(config.get("analysis.association.methods")),
    )
    path = table_path(output_dir, str(config.get("analysis.output.association_name")), fmt)
    write_table(association, path, fmt=fmt)
    for row in association.itertuples():
        logger.info(
            "  EXPLORATORY %s(%s, %s) = %.4f over n=%d transformation(s) - descriptive "
            "only, no significance test was run.",
            row.method, row.x, row.y, row.coefficient, row.n_points,
        )
    return path


def _warn_on_inconsistent_runs(runs: pd.DataFrame) -> None:
    """Surface any way in which the arms were *not* a controlled comparison."""
    controlled = (
        "split_fingerprint", "num_train_images", "num_val_images", "epochs", "batch_size",
        "optimizer", "learning_rate", "weight_decay", "scheduler", "architecture",
        "pretrained_weights", "application_probability",
    )
    for column in controlled:
        if column not in runs.columns:
            continue
        distinct = runs[column].astype(str).drop_duplicates().tolist()
        if len(distinct) > 1:
            logger.warning(
                "Policies do NOT share the same '%s' (%s). Experiment 0B assumes every arm "
                "is identical except for the augmentation policy; this comparison is "
                "confounded until the arms are retrained under one configuration.",
                column, distinct,
            )


def _log_summary(summary: pd.DataFrame, baseline_name: str) -> None:
    columns = [column for column in
               ("policy", "audit_transform", "seed", "best_epoch", "best_val_top1",
                "best_val_top5", "best_val_loss", "delta_best_val_top1_vs_baseline")
               if column in summary.columns]
    logger.info("Policy summary (reference arm: %s):", baseline_name)
    with pd.option_context("display.width", 220, "display.max_columns", None):
        for line in summary[columns].to_string(index=False, float_format="%.5f").splitlines():
            logger.info("  %s", line)


def _log_joined(joined: pd.DataFrame) -> None:
    columns = [column for column in
               ("audit_transform", "delta_margin_mean", "delta_margin_median",
                "frac_delta_negative", "best_val_top1", "delta_best_val_top1_vs_baseline")
               if column in joined.columns]
    logger.info("Experiment 0A evidence joined to Experiment 0B outcome:")
    with pd.option_context("display.width", 220, "display.max_columns", None):
        for line in joined[columns].to_string(index=False, float_format="%.5f").splitlines():
            logger.info("  %s", line)
