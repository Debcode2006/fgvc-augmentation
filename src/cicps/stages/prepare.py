"""Stage: dataset validation and internal split preparation."""

from __future__ import annotations

from ..config import Config
from ..data.cub import load_cub_metadata
from ..data.splits import resolve_split
from ..data.validation import ValidationReport, validate_dataset, validate_split
from ..logging_utils import get_logger, setup_logging

__all__ = ["run_prepare"]

logger = get_logger(__name__)


def run_prepare(config: Config, *, overwrite: bool = False) -> ValidationReport:
    """Validate CUB, build/load the internal split, and validate the split.

    Raises when any check fails, so a broken dataset or a leaking split stops the
    pipeline before a single GPU-hour is spent.
    """
    setup_logging(config, stage="prepare")
    logger.info("Configuration: %s", config.source)
    logger.info("Experiment: %s", config.get("experiment.name"))

    metadata = load_cub_metadata(config)

    logger.info("Dataset validation:")
    report = validate_dataset(metadata, config)
    report.log()
    if not report.ok:
        raise RuntimeError(
            f"Dataset validation failed with {len(report.errors)} error(s); see the log above."
        )

    split = resolve_split(metadata, config, overwrite=overwrite)

    logger.info("Split validation:")
    split_report = validate_split(split, metadata, config)
    split_report.log()
    report.merge(split_report)
    if not report.ok:
        raise RuntimeError(
            f"Split validation failed with {len(split_report.errors)} error(s); see the log above."
        )

    logger.info(
        "Dataset and split are valid. Official test portion (%d images) was not used.",
        len(metadata.official_test),
    )
    return report
