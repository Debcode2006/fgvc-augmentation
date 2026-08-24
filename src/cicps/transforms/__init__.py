"""Deterministic transformations audited by Experiment 0A.

Adding a transformation to the audit takes two steps and no engine changes:

1. Subclass :class:`~cicps.transforms.base.DeterministicTransform` and decorate
   it with ``@register_transform("my_type")``.
2. Append ``{name, type: my_type, params: {...}}`` to ``transformations`` in the
   YAML configuration.
"""

from . import builtin  # noqa: F401  (registers the built-in transformations)
from .base import (
    DeterministicTransform,
    available_transform_types,
    build_transform,
    build_transform_suite,
    register_transform,
)
from .pipeline import (
    CanonicalView,
    EvalPipeline,
    Normalizer,
    TrainPipeline,
    build_canonical_view,
    build_eval_transform,
    build_normalizer,
    build_train_transform,
    resolve_interpolation,
)

__all__ = [
    "DeterministicTransform",
    "available_transform_types",
    "build_transform",
    "build_transform_suite",
    "register_transform",
    "CanonicalView",
    "EvalPipeline",
    "Normalizer",
    "TrainPipeline",
    "build_canonical_view",
    "build_eval_transform",
    "build_normalizer",
    "build_train_transform",
    "resolve_interpolation",
]
