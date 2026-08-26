"""Experiment 0B plots.

Six figures, all configuration driven through ``analysis.plots``:

``validation_top1_by_policy`` / ``validation_top5_by_policy``
    Downstream outcome per policy, with the baseline arm drawn as an explicit
    reference line rather than as just another bar.

``validation_curves`` / ``training_curves``
    Whether differences between policies appear immediately or emerge late.

``delta_margin_vs_accuracy_delta``
    **The plot the experiment exists for**: Experiment 0A's mean ΔM on the x
    axis against the change in 0B validation top-1 relative to baseline on the y
    axis, one point per transformation.

``delta_margin_vs_accuracy``
    The same relation against absolute validation top-1.

No renderer encodes an expected outcome. The optional trend line is a
least-squares fit drawn as a visual aid over a handful of points; it is not
evidence, and the plots say so in their own captions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import matplotlib

matplotlib.use("Agg")  # headless: figures are written to disk, never shown
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..config import Config, ConfigError  # noqa: E402
from ..logging_utils import get_logger  # noqa: E402

__all__ = ["PolicyPlotContext", "render_policy_plots", "SUPPORTED_PLOT_KINDS"]

logger = get_logger(__name__)

_BASELINE_COLOUR = "#444444"
_ARM_COLOUR = "#4C72B0"
_TREND_COLOUR = "#C44E52"


class PolicyPlotContext:
    """Everything the Experiment 0B renderers draw from."""

    def __init__(
        self,
        summary: pd.DataFrame,
        joined: pd.DataFrame,
        histories: dict[str, pd.DataFrame],
        baseline_name: str,
    ) -> None:
        self.summary = summary
        self.joined = joined
        self.histories = histories
        self.baseline_name = baseline_name

    @property
    def policy_order(self) -> list[str]:
        """Baseline first, then the arms in configuration order."""
        return [str(name) for name in self.summary["policy"]]

    def baseline_value(self, column: str) -> float:
        row = self.summary.loc[self.summary["policy"] == self.baseline_name]
        return float(row.iloc[0][column])


def render_policy_plots(config: Config, context: PolicyPlotContext) -> list[Path]:
    """Render every configured plot kind; return the written file paths."""
    kinds = [str(kind).lower() for kind in config.get("analysis.plots.kinds")]
    unknown = set(kinds) - set(SUPPORTED_PLOT_KINDS)
    if unknown:
        raise ConfigError(
            f"Unknown analysis.plots.kinds entries: {sorted(unknown)}. "
            f"Supported: {sorted(SUPPORTED_PLOT_KINDS)}."
        )

    output_dir = config.path("analysis.plots.dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    file_format = str(config.get("analysis.plots.file_format"))
    dpi = int(config.get("analysis.plots.dpi"))
    figsize = tuple(float(value) for value in config.get("analysis.plots.figsize"))

    written: list[Path] = []
    for kind in kinds:
        figure = plt.figure(figsize=figsize)
        SUPPORTED_PLOT_KINDS[kind](figure, context, config)
        path = output_dir / f"{kind}.{file_format}"
        figure.tight_layout()
        figure.savefig(path, dpi=dpi)
        plt.close(figure)
        written.append(path)
        logger.info("Wrote plot %s", path)
    return written


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def _bar_by_policy(figure, context: PolicyPlotContext, config: Config, *,
                   column: str, label: str) -> None:
    axes = figure.add_subplot(111)
    names = context.policy_order
    values = [float(context.summary.loc[context.summary["policy"] == name, column].iloc[0])
              for name in names]
    colours = [_BASELINE_COLOUR if name == context.baseline_name else _ARM_COLOUR
               for name in names]

    positions = np.arange(len(names))
    axes.bar(positions, values, color=colours, alpha=0.85, width=0.65)
    reference = context.baseline_value(column)
    axes.axhline(reference, color=_BASELINE_COLOUR, linestyle="--", linewidth=1.2,
                 label=f"{context.baseline_name} = {reference:.4f}")

    for position, value in zip(positions, values):
        axes.annotate(f"{value:.4f}", (position, value), ha="center", va="bottom",
                      fontsize="small")

    axes.set_xticks(positions)
    axes.set_xticklabels(names, rotation=20, ha="right")
    axes.set_ylabel(label)
    axes.set_title(f"Experiment 0B: {label} by training augmentation policy")
    axes.legend(fontsize="small")
    _dress(axes)
    _pad_ylim(axes, values)


def _validation_top1_by_policy(figure, context, config) -> None:
    _bar_by_policy(figure, context, config,
                   column="best_val_top1", label="best internal-validation top-1")


def _validation_top5_by_policy(figure, context, config) -> None:
    _bar_by_policy(figure, context, config,
                   column="best_val_top5", label="best internal-validation top-5")


def _validation_curves(figure, context: PolicyPlotContext, config: Config) -> None:
    axes = figure.add_subplot(111)
    for name in context.policy_order:
        history = context.histories[name]
        axes.plot(
            history["epoch"] + 1, history["val_top1"],
            label=name, linewidth=2.0 if name == context.baseline_name else 1.4,
            color=_BASELINE_COLOUR if name == context.baseline_name else None,
            linestyle="--" if name == context.baseline_name else "-",
        )
    axes.set_xlabel("epoch")
    axes.set_ylabel("internal-validation top-1")
    axes.set_title("Experiment 0B: validation top-1 per epoch, by policy")
    axes.legend(fontsize="small", ncol=2)
    _dress(axes)


def _training_curves(figure, context: PolicyPlotContext, config: Config) -> None:
    train_axes = figure.add_subplot(121)
    loss_axes = figure.add_subplot(122)
    for name in context.policy_order:
        history = context.histories[name]
        style = {
            "label": name,
            "linewidth": 2.0 if name == context.baseline_name else 1.4,
            "color": _BASELINE_COLOUR if name == context.baseline_name else None,
            "linestyle": "--" if name == context.baseline_name else "-",
        }
        train_axes.plot(history["epoch"] + 1, history["train_top1"], **style)
        loss_axes.plot(history["epoch"] + 1, history["val_loss"], **style)

    train_axes.set_xlabel("epoch")
    train_axes.set_ylabel("training top-1")
    train_axes.set_title("Training top-1 by policy")
    loss_axes.set_xlabel("epoch")
    loss_axes.set_ylabel("validation loss")
    loss_axes.set_title("Validation loss by policy")
    loss_axes.legend(fontsize="small", ncol=2)
    _dress(train_axes)
    _dress(loss_axes)


def _scatter_0a_vs_0b(figure, context: PolicyPlotContext, config: Config, *,
                      y_column: str, y_label: str, zero_line: bool,
                      reference: float | None) -> None:
    axes = figure.add_subplot(111)
    joined = context.joined
    x = joined["delta_margin_mean"].to_numpy(dtype=np.float64)
    y = joined[y_column].to_numpy(dtype=np.float64)

    axes.scatter(x, y, s=90, color=_ARM_COLOUR, zorder=3, edgecolor="white", linewidth=1.2)
    for xi, yi, name in zip(x, y, joined["audit_transform"]):
        axes.annotate(str(name), (xi, yi), textcoords="offset points", xytext=(8, 6),
                      fontsize="small")

    axes.axvline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    if zero_line:
        axes.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    if reference is not None:
        axes.axhline(reference, color=_BASELINE_COLOUR, linestyle="--", linewidth=1.2,
                     label=f"{context.baseline_name} = {reference:.4f}")
        axes.legend(fontsize="small", loc="best")

    if bool(config.get("analysis.plots.trend_line")) and len(x) >= 2 and np.std(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        grid = np.linspace(float(x.min()), float(x.max()), 50)
        axes.plot(grid, slope * grid + intercept, color=_TREND_COLOUR, linewidth=1.2,
                  linestyle=":", alpha=0.9, zorder=2)

    axes.set_xlabel(r"Experiment 0A: mean $\Delta M$ (frozen-model audit)")
    axes.set_ylabel(f"Experiment 0B: {y_label}")
    axes.set_title(
        r"0A margin damage vs. 0B downstream outcome ($n$ = "
        f"{len(x)} transformations)"
    )
    axes.text(
        0.5, -0.16,
        "Descriptive only. Association is not causation, and any trend line over "
        f"{len(x)} points is a visual aid, not evidence.",
        transform=axes.transAxes, ha="center", va="top", fontsize="small", alpha=0.75,
    )
    _dress(axes)


def _delta_margin_vs_accuracy_delta(figure, context, config) -> None:
    _scatter_0a_vs_0b(
        figure, context, config,
        y_column="delta_best_val_top1_vs_baseline",
        y_label="change in validation top-1 vs. baseline policy",
        zero_line=True, reference=None,
    )


def _delta_margin_vs_accuracy(figure, context, config) -> None:
    _scatter_0a_vs_0b(
        figure, context, config,
        y_column="best_val_top1",
        y_label="best validation top-1",
        zero_line=False, reference=context.baseline_value("best_val_top1"),
    )


SUPPORTED_PLOT_KINDS: dict[str, Callable] = {
    "validation_top1_by_policy": _validation_top1_by_policy,
    "validation_top5_by_policy": _validation_top5_by_policy,
    "validation_curves": _validation_curves,
    "training_curves": _training_curves,
    "delta_margin_vs_accuracy_delta": _delta_margin_vs_accuracy_delta,
    "delta_margin_vs_accuracy": _delta_margin_vs_accuracy,
}
"""Plot kind -> renderer. Adding a plot means adding one entry and one config name."""


def _dress(axes) -> None:
    axes.grid(alpha=0.25, linestyle=":")


def _pad_ylim(axes, values: Sequence[float]) -> None:
    """Zoom the y axis onto the range the policies actually span.

    Accuracy differences between arms are small next to the absolute accuracy,
    and a bar chart anchored at zero would hide them entirely. The axis is
    therefore explicitly *not* zero-anchored, which is worth knowing when reading
    the figure.
    """
    low, high = float(min(values)), float(max(values))
    margin = max((high - low) * 0.35, 0.01)
    axes.set_ylim(max(0.0, low - margin), min(1.0, high + margin))
