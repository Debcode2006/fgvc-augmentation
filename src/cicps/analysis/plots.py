"""Distribution plots for the primary metric.

The first and most important visualisation of Experiment 0A is the distribution
of ``delta_margin`` for each transformation. Plot kinds, ordering, size and
output format are configuration driven; no plot here draws a conclusion about
which transformation is preferable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # headless: figures are written to disk, never shown
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..config import Config, ConfigError  # noqa: E402
from ..logging_utils import get_logger  # noqa: E402

__all__ = ["plot_delta_margin_distributions", "resolve_transform_order"]

logger = get_logger(__name__)

_SUPPORTED_KINDS = {"violin", "box", "hist", "ecdf"}


def resolve_transform_order(
    records: pd.DataFrame, config: Config, config_order: Sequence[str]
) -> list[str]:
    """Order transformations for the axes, per ``analysis.plots.order``."""
    mode = str(config.get("analysis.plots.order")).lower()
    present = list(records["transform"].drop_duplicates())
    if mode == "config":
        ordered = [name for name in config_order if name in present]
        return ordered + [name for name in present if name not in ordered]
    if mode == "median":
        medians = records.groupby("transform")["delta_margin"].median().sort_values()
        return list(medians.index)
    raise ConfigError(
        f"Unknown analysis.plots.order '{mode}'. Use 'config' or 'median'."
    )


def plot_delta_margin_distributions(
    records: pd.DataFrame, config: Config, transform_order: Sequence[str]
) -> list[Path]:
    """Render every configured plot kind; return the written file paths."""
    kinds = [str(kind).lower() for kind in config.get("analysis.plots.kinds")]
    unknown = set(kinds) - _SUPPORTED_KINDS
    if unknown:
        raise ConfigError(
            f"Unknown analysis.plots.kinds entries: {sorted(unknown)}. "
            f"Supported: {sorted(_SUPPORTED_KINDS)}."
        )

    output_dir = config.path("analysis.plots.dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    file_format = str(config.get("analysis.plots.file_format"))
    dpi = int(config.get("analysis.plots.dpi"))
    figsize = tuple(float(value) for value in config.get("analysis.plots.figsize"))
    limit = config.get("analysis.plots.delta_margin_limit", None)

    grouped = [
        records.loc[records["transform"] == name, "delta_margin"].to_numpy(dtype=np.float64)
        for name in transform_order
    ]
    written: list[Path] = []
    renderers = {"violin": _violin, "box": _box, "hist": _hist, "ecdf": _ecdf}

    for kind in kinds:
        figure, axes = plt.subplots(figsize=figsize)
        renderers[kind](axes, grouped, list(transform_order), config)
        _finalise(axes, kind, limit)
        path = output_dir / f"delta_margin_{kind}.{file_format}"
        figure.tight_layout()
        figure.savefig(path, dpi=dpi)
        plt.close(figure)
        written.append(path)
        logger.info("Wrote plot %s", path)

    return written


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def _violin(axes, grouped, names, config: Config) -> None:
    parts = axes.violinplot(grouped, showextrema=False, showmedians=True, widths=0.85)
    for body in parts["bodies"]:
        body.set_alpha(0.6)
    axes.set_xticks(range(1, len(names) + 1))
    axes.set_xticklabels(names, rotation=25, ha="right")
    axes.set_ylabel(r"$\Delta M = M_{yc}(T(x)) - M_{yc}(x)$")
    axes.set_title("Distribution of the change in discriminative margin")


def _box(axes, grouped, names, config: Config) -> None:
    axes.boxplot(grouped, showfliers=True, whis=(5, 95), tick_labels=names)
    axes.set_xticklabels(names, rotation=25, ha="right")
    axes.set_ylabel(r"$\Delta M$")
    axes.set_title(r"$\Delta M$ per transformation (box: IQR, whiskers: 5th-95th pct)")


def _hist(axes, grouped, names, config: Config) -> None:
    bins = int(config.get("analysis.plots.hist_bins"))
    stacked = np.concatenate(grouped) if grouped else np.array([0.0])
    edges = np.linspace(float(stacked.min()), float(stacked.max()), bins + 1)
    for values, name in zip(grouped, names):
        axes.hist(values, bins=edges, histtype="step", linewidth=1.6, label=name, density=True)
    axes.set_xlabel(r"$\Delta M$")
    axes.set_ylabel("density")
    axes.legend(fontsize="small", ncol=2)
    axes.set_title(r"$\Delta M$ density per transformation")


def _ecdf(axes, grouped, names, config: Config) -> None:
    for values, name in zip(grouped, names):
        ordered = np.sort(values)
        axes.step(ordered, np.arange(1, len(ordered) + 1) / len(ordered), where="post", label=name)
    axes.set_xlabel(r"$\Delta M$")
    axes.set_ylabel("empirical CDF")
    axes.legend(fontsize="small", ncol=2)
    axes.set_title(r"Empirical CDF of $\Delta M$")


def _finalise(axes, kind: str, limit: float | None) -> None:
    """Shared axis dressing: a zero reference line and optional symmetric limits."""
    if kind in {"violin", "box"}:
        axes.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
        if limit is not None:
            axes.set_ylim(-float(limit), float(limit))
    else:
        axes.axvline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
        if limit is not None:
            axes.set_xlim(-float(limit), float(limit))
    axes.grid(alpha=0.25, linestyle=":")
