"""Audit dataframe schema and serialisation.

One logical row per ``validation image x transformation``.

Class identifier convention
---------------------------
Every class column exists in three forms so that no downstream consumer has to
guess an offset:

``*_class``       official CUB class id, 1..200 (as in ``classes.txt``)
``*_label``       contiguous model index, 0..199 (``label = class_id - 1``)
``*_class_name``  species name, e.g. ``001.Black_footed_Albatross``
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..config import ConfigError
from ..logging_utils import get_logger

__all__ = [
    "AUDIT_COLUMNS",
    "REQUIRED_COLUMNS",
    "encode_params",
    "order_columns",
    "table_path",
    "validate_schema",
    "write_table",
]

logger = get_logger(__name__)

REQUIRED_COLUMNS: tuple[str, ...] = (
    "image_id",
    "true_class",
    "hard_competitor",
    "original_true_prob",
    "original_competitor_prob",
    "original_margin",
    "transform",
    "augmented_true_prob",
    "augmented_competitor_prob",
    "augmented_margin",
    "delta_margin",
    "feature_cosine",
    "prediction_consistent",
    "correct_original",
    "correct_augmented",
)
"""The columns Experiment 0A mandates. Additional columns may be added; none of
these may be removed."""

AUDIT_COLUMNS: tuple[str, ...] = (
    # identity of the sample
    "image_id",
    "image_path",
    "true_class",
    "true_label",
    "true_class_name",
    # frozen competitor, selected from the ORIGINAL image only
    "hard_competitor",
    "hard_competitor_label",
    "hard_competitor_name",
    # original-image quantities
    "original_true_prob",
    "original_competitor_prob",
    "original_margin",
    "original_prediction",
    # transformation
    "transform",
    "transform_type",
    "transform_params",
    # transformed-image quantities
    "augmented_true_prob",
    "augmented_competitor_prob",
    "augmented_margin",
    "augmented_prediction",
    # primary metric
    "delta_margin",
    # supporting metrics
    "feature_cosine",
    "prediction_consistent",
    "correct_original",
    "correct_augmented",
)
"""Canonical column order of the audit dataframe."""


def validate_schema(frame: pd.DataFrame) -> None:
    """Raise when the dataframe is missing a mandated column."""
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Audit dataframe is missing required column(s): {', '.join(missing)}."
        )


def order_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return ``frame`` with canonical columns first, extras appended after."""
    known = [column for column in AUDIT_COLUMNS if column in frame.columns]
    extras = [column for column in frame.columns if column not in AUDIT_COLUMNS]
    return frame[known + extras]


def encode_params(params: dict) -> str:
    """Serialise resolved transformation parameters into a compact JSON string.

    Stored per row so that any sample's exact frozen transformation can be
    inspected - or reconstructed - without re-running the audit.
    """
    return json.dumps(
        {key: (round(value, 6) if isinstance(value, float) else value)
         for key, value in sorted(params.items())},
        separators=(",", ":"),
    )


def write_table(frame: pd.DataFrame, path: Path, *, fmt: str, float_precision: int = 8) -> Path:
    """Write ``frame`` as CSV or Parquet, returning the path actually written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = fmt.lower()
    if fmt == "csv":
        frame.to_csv(path, index=False, float_format=f"%.{int(float_precision)}g")
    elif fmt == "parquet":
        try:
            frame.to_parquet(path, index=False)
        except ImportError as error:
            raise ConfigError(
                "Writing Parquet requires pyarrow. Install it, or set the output format "
                "to 'csv' in the configuration."
            ) from error
    else:
        raise ConfigError(f"Unsupported output format '{fmt}'. Use 'csv' or 'parquet'.")
    logger.info("Wrote %d row(s) x %d column(s) to %s", len(frame), frame.shape[1], path)
    return path


def table_path(directory: Path, stem: str, fmt: str) -> Path:
    """Compose an output path from a directory, filename stem and format."""
    suffix = {"csv": ".csv", "parquet": ".parquet"}.get(fmt.lower())
    if suffix is None:
        raise ConfigError(f"Unsupported output format '{fmt}'. Use 'csv' or 'parquet'.")
    return directory / f"{stem}{suffix}"
