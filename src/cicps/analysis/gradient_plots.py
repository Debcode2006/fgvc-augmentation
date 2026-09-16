"""Figures for Experiment 1A.

Every figure answers one question, and none of them ranks augmentations.

``grad_cosine_distribution``   How does the learning signal differ by transformation?
``grad_norm_ratio_distribution`` Is the transformed signal stronger or weaker?
``grad_cosine_vs_delta_margin`` Is the gradient view redundant with the margin view?
``grad_cosine_vs_feature_cosine`` Is it redundant with generic representation drift?
``grad_cosine_by_checkpoint``  Does compatibility change as the model learns?
``grad_cosine_vs_downstream``  Does it correspond to Experiment 0B's outcome?
``scope_heatmap``              Where in the network does the disagreement live?

The distribution figures draw the two reference levels that make a cosine
readable: ``1.0`` (the ``identity`` control - the same image, so the same
gradient) and the cross-image null (two unrelated images at the same checkpoint).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # headless: figures are written to disk, never shown
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..config import Config  # noqa: E402
from ..logging_utils import get_logger  # noqa: E402

__all__ = ["render_plots"]

logger = get_logger(__name__)

_CAPTION = (
    "Descriptive only. Gradient alignment is evidence about learning-signal "
    "compatibility, not proof of augmentation quality."
)


def render_plots(
    records: pd.DataFrame,
    summary: pd.DataFrame,
    null_summary: pd.DataFrame,
    joined_0a: pd.DataFrame | None,
    joined_0b: pd.DataFrame | None,
    config: Config,
    *,
    transform_order: Sequence[str],
    checkpoint_order: Sequence[str],
    control_transform: str,
    anchor_checkpoint: str,
    scopes: Sequence[str],
) -> list[Path]:
    """Render every configured figure; return the written paths."""
    kinds = [str(kind).lower() for kind in config.get("analysis.plots.kinds")]
    output_dir = config.path("analysis.plots.dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    figsize = tuple(float(value) for value in config.get("analysis.plots.figsize"))
    dpi = int(config.get("analysis.plots.dpi"))
    suffix = str(config.get("analysis.plots.file_format"))

    arms = [name for name in transform_order if name != control_transform]
    anchor_null = _null_level(null_summary, anchor_checkpoint)
    written: list[Path] = []

    renderers = {
        "grad_cosine_distribution": lambda axes: _distribution(
            axes, records, anchor_checkpoint, arms, "grad_cosine",
            "gradient cosine  $C_g$", anchor_null, reference_one=True),
        "grad_norm_ratio_distribution": lambda axes: _distribution(
            axes, records, anchor_checkpoint, arms, "grad_norm_ratio",
            "gradient norm ratio  $R_g$", None, reference_one=True),
        "grad_cosine_vs_delta_margin": lambda axes: _scatter_per_sample(
            axes, joined_0a, arms, "delta_margin_0a", "grad_cosine",
            "Experiment 0A  $\\Delta M$ (same image, same draw)", "gradient cosine  $C_g$"),
        "grad_cosine_vs_feature_cosine": lambda axes: _scatter_per_sample(
            axes, records[records["checkpoint"] == anchor_checkpoint], arms,
            "feature_cosine", "grad_cosine",
            "feature cosine  $C_f$", "gradient cosine  $C_g$"),
        "grad_cosine_by_checkpoint": lambda axes: _by_checkpoint(
            axes, summary, null_summary, checkpoint_order, arms),
        "grad_cosine_vs_downstream": lambda axes: _downstream(axes, joined_0b),
        "grad_norm_ratio_by_checkpoint": lambda axes: _by_checkpoint(
            axes, summary, None, checkpoint_order, arms,
            column="grad_norm_ratio_mean", label="mean gradient norm ratio  $R_g$"),
        "scope_heatmap": lambda axes: _scope_heatmap(
            axes, summary, anchor_checkpoint, arms, scopes),
    }

    for kind in kinds:
        renderer = renderers.get(kind)
        if renderer is None:
            logger.warning(
                "Unknown analysis.plots.kinds entry '%s'; supported: %s.",
                kind, ", ".join(sorted(renderers)),
            )
            continue
        figure, axes = plt.subplots(figsize=figsize)
        try:
            drawn = renderer(axes)
        except _NoData as error:
            plt.close(figure)
            logger.warning("Skipping plot '%s': %s", kind, error)
            continue
        if drawn is False:
            plt.close(figure)
            continue
        figure.text(0.01, 0.01, _CAPTION, fontsize=7, style="italic", alpha=0.75)
        figure.tight_layout(rect=(0, 0.035, 1, 1))
        path = output_dir / f"{kind}.{suffix}"
        figure.savefig(path, dpi=dpi)
        plt.close(figure)
        written.append(path)
        logger.info("Wrote plot %s", path)

    return written


class _NoData(RuntimeError):
    """Raised when a figure has nothing to draw, so it is skipped with a warning."""


def _null_level(null_summary: pd.DataFrame, checkpoint: str) -> float | None:
    if null_summary is None or null_summary.empty:
        return None
    row = null_summary[null_summary["checkpoint"] == checkpoint]
    return None if row.empty else float(row["null_cosine_mean"].iloc[0])


def _distribution(axes, records, checkpoint, arms, column, label, null_level,
                  *, reference_one: bool):
    stage = records[records["checkpoint"] == checkpoint]
    grouped = [
        stage.loc[stage["transform"] == name, column].dropna().to_numpy()
        for name in arms
    ]
    grouped = [values for values in grouped if values.size]
    if not grouped:
        raise _NoData(f"no finite '{column}' values at checkpoint '{checkpoint}'")

    positions = np.arange(1, len(grouped) + 1)
    parts = axes.violinplot(grouped, positions=positions, showmedians=True, widths=0.85)
    for body in parts["bodies"]:
        body.set_alpha(0.55)
    if reference_one:
        axes.axhline(1.0, linestyle="--", linewidth=1.0, color="0.35",
                     label="identity control ($C_g = 1$)" if column == "grad_cosine"
                     else "no change ($R_g = 1$)")
    if null_level is not None:
        axes.axhline(null_level, linestyle=":", linewidth=1.4, color="crimson",
                     label=f"cross-image null ({null_level:+.3f})")
        axes.axhline(0.0, linestyle="-", linewidth=0.6, color="0.7")
    axes.set_xticks(positions)
    axes.set_xticklabels(arms[:len(grouped)], rotation=20, ha="right")
    axes.set_ylabel(label)
    axes.set_title(f"{label} by transformation - checkpoint '{checkpoint}'")
    axes.legend(fontsize=8, loc="best")
    axes.grid(axis="y", alpha=0.25)


def _scatter_per_sample(axes, frame, arms, x_column, y_column, x_label, y_label):
    if frame is None or frame.empty:
        raise _NoData("the required joined table is empty")
    if x_column not in frame.columns:
        raise _NoData(f"column '{x_column}' is absent")
    drawn = 0
    for name in arms:
        group = frame[frame["transform"] == name]
        if group.empty:
            continue
        axes.scatter(group[x_column], group[y_column], s=6, alpha=0.35, label=name)
        drawn += 1
    if not drawn:
        raise _NoData("no transformation arm had rows")
    axes.axhline(0.0, linestyle="-", linewidth=0.6, color="0.7")
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    axes.set_title(f"{y_label} vs {x_label}  (n = {len(frame)} paired samples)")
    axes.legend(fontsize=8, markerscale=2.0)
    axes.grid(alpha=0.25)


def _by_checkpoint(axes, summary, null_summary, checkpoint_order, arms,
                   column="grad_cosine_mean", label="mean gradient cosine  $C_g$"):
    present = [name for name in checkpoint_order
               if not summary[summary["checkpoint"] == name].empty]
    if len(present) < 2:
        raise _NoData("fewer than two checkpoint stages were measured")
    positions = np.arange(len(present))
    for name in arms:
        values, xs = [], []
        for index, checkpoint in enumerate(present):
            cell = summary[(summary["checkpoint"] == checkpoint) & (summary["transform"] == name)]
            if cell.empty:
                continue
            values.append(float(cell[column].iloc[0]))
            xs.append(index)
        if values:
            axes.plot(xs, values, marker="o", label=name)
    if null_summary is not None and not null_summary.empty:
        nulls, xs = [], []
        for index, checkpoint in enumerate(present):
            level = _null_level(null_summary, checkpoint)
            if level is not None:
                nulls.append(level)
                xs.append(index)
        if nulls:
            axes.plot(xs, nulls, linestyle=":", marker="x", color="crimson",
                      label="cross-image null")
    axes.set_xticks(positions)
    axes.set_xticklabels(present, rotation=15, ha="right")
    axes.set_ylabel(label)
    axes.set_title(f"{label} across checkpoint stages")
    axes.legend(fontsize=8)
    axes.grid(alpha=0.25)


def _downstream(axes, joined):
    if joined is None or joined.empty:
        raise _NoData("the Experiment 0B join is empty")
    x = joined["grad_cosine_mean"].to_numpy(dtype=float)
    y = joined["val_top1_delta_0b"].to_numpy(dtype=float)
    axes.axhline(0.0, linestyle="--", linewidth=1.0, color="0.4")
    axes.scatter(x, y, s=70, zorder=3)
    for xi, yi, name in zip(x, y, joined["transform"]):
        axes.annotate(name, (xi, yi), textcoords="offset points", xytext=(6, 5), fontsize=8)
    axes.set_xlabel("mean gradient cosine  $C_g$  (frozen checkpoint)")
    axes.set_ylabel("Experiment 0B  $\\Delta$ val top-1 vs baseline")
    axes.set_title(
        f"Gradient compatibility vs downstream outcome  (n = {len(joined)} transformations)"
    )
    axes.grid(alpha=0.25)


def _scope_heatmap(axes, summary, checkpoint, arms, scopes):
    columns = [f"grad_cosine_mean_{scope}" for scope in scopes]
    cell = summary[summary["checkpoint"] == checkpoint]
    available = [(scope, column) for scope, column in zip(scopes, columns)
                 if column in cell.columns]
    if not available or cell.empty:
        raise _NoData("no per-scope columns were produced")

    rows = [name for name in arms if not cell[cell["transform"] == name].empty]
    matrix = np.array([
        [float(cell.loc[cell["transform"] == name, column].iloc[0])
         for _, column in available]
        for name in rows
    ])
    image = axes.imshow(matrix, aspect="auto", cmap="viridis")
    axes.set_xticks(np.arange(len(available)))
    axes.set_xticklabels([scope for scope, _ in available], rotation=25, ha="right")
    axes.set_yticks(np.arange(len(rows)))
    axes.set_yticklabels(rows)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axes.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                      fontsize=7, color="w")
    axes.figure.colorbar(image, ax=axes, label="mean $C_g$")
    axes.set_title(f"Gradient cosine by parameter scope - checkpoint '{checkpoint}'")
