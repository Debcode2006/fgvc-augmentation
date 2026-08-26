"""Descriptive analysis and plots.

Experiment 0A: per-transformation statistics over the audit dataframe.
Experiment 0B: per-policy statistics joined against Experiment 0A's summary.
"""

from .plots import plot_delta_margin_distributions
from .policies import build_policy_summary, join_with_0a
from .policy_plots import render_policy_plots
from .summary import summarize_by_transform

__all__ = [
    "build_policy_summary",
    "join_with_0a",
    "plot_delta_margin_distributions",
    "render_policy_plots",
    "summarize_by_transform",
]
