"""Experiment 0A transformation audit.

Pipeline:

    validation image x
      -> original prediction / features (frozen baseline)
      -> frozen hardest competitor c(x) = argmax_{k != y} p(k|x)
      -> each deterministic transformation T
      -> transformed prediction / features
      -> margin metrics: M(x), M(T(x)), delta_margin
      -> one record per (image, transformation)
"""

from .competitor import OriginalState, compute_original_state
from .engine import AuditEngine, AuditResult
from .inference import InferenceResult, run_inference
from .metrics import MetricContext, available_metrics, register_metric
from .records import AUDIT_COLUMNS, write_table

__all__ = [
    "AuditEngine",
    "AuditResult",
    "InferenceResult",
    "MetricContext",
    "OriginalState",
    "AUDIT_COLUMNS",
    "available_metrics",
    "compute_original_state",
    "register_metric",
    "run_inference",
    "write_table",
]
