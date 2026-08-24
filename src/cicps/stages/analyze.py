"""Stage: descriptive analysis and plots over the audit dataframe."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..analysis.plots import plot_delta_margin_distributions, resolve_transform_order
from ..analysis.summary import filter_records, summarize_by_transform
from ..audit.records import table_path, validate_schema, write_table
from ..config import Config, ConfigError
from ..logging_utils import get_logger, setup_logging
from ..transforms.base import build_transform_suite

__all__ = ["run_analyze", "AnalysisPaths"]

logger = get_logger(__name__)


@dataclass(frozen=True)
class AnalysisPaths:
    """Artefacts written by the analysis stage."""

    summary: Path
    plots: list[Path]


def run_analyze(config: Config, *, records_path: Path | None = None) -> AnalysisPaths:
    """Summarise the audit records and render the delta-margin distributions."""
    setup_logging(config, stage="analyze")
    logger.info("Configuration: %s", config.source)

    path = records_path or table_path(
        config.path("audit.output.dir"),
        str(config.get("audit.output.records_name")),
        str(config.get("audit.output.format")),
    )
    records = _read_table(path)
    validate_schema(records)
    logger.info("Loaded %d audit row(s) from %s", len(records), path)

    records = filter_records(records, config)
    if records.empty:
        raise RuntimeError("No audit rows remain after filtering; nothing to analyse.")

    config_order = [transform.name for transform in build_transform_suite(config)]
    summary = summarize_by_transform(
        records,
        quantiles=list(config.get("analysis.quantiles")),
        transform_order=[name for name in config_order
                         if name in set(records["transform"].unique())],
    )
    _log_summary(summary)

    summary_path = table_path(
        config.path("analysis.output.dir"),
        str(config.get("analysis.output.summary_name")),
        str(config.get("analysis.output.format")),
    )
    write_table(summary, summary_path, fmt=str(config.get("analysis.output.format")))

    order = resolve_transform_order(records, config, config_order)
    plots = plot_delta_margin_distributions(records, config, order)

    logger.info(
        "Analysis complete. These are descriptive statistics only: no significance test was "
        "run, and a higher mean delta_margin does not by itself make a transformation better."
    )
    return AnalysisPaths(summary=summary_path, plots=plots)


def _read_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise ConfigError(
            f"Audit dataframe not found: {path}. Run the `audit` stage first."
        )
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    raise ConfigError(f"Unsupported audit dataframe format '{path.suffix}'.")


def _log_summary(summary: pd.DataFrame) -> None:
    """Log the headline columns of the summary table."""
    columns = [
        "transform", "n_samples", "delta_margin_mean", "delta_margin_median",
        "delta_margin_std", "frac_delta_negative", "feature_cosine_mean",
        "prediction_consistency_rate", "accuracy_original", "accuracy_augmented",
    ]
    present = [column for column in columns if column in summary.columns]
    with pd.option_context("display.width", 200, "display.max_columns", None):
        for line in summary[present].to_string(index=False, float_format="%.5f").splitlines():
            logger.info("  %s", line)
