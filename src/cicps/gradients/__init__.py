"""Experiment 1A - gradient-compatibility measurement.

Where :mod:`cicps.audit` asks what a transformation does to a frozen model's
*output* (the pairwise margin against a frozen hardest competitor), this package
asks what it does to the *learning signal*: the gradient of the loss with respect
to the parameters at the same frozen checkpoint.

The modules are deliberately generic so that Experiment 1B - representation and
learning-signal evolution through training - can reuse them unchanged:

``scopes``       named subsets of the trainable parameters, declared in YAML
``accumulate``   online float64 inner products; no raw gradient is ever stored
``checkpoints``  checkpoint stage resolution, including a reconstructed epoch 0
``datasets``     paired clean/transformed views with Experiment 0A's frozen draws
``records``      dataframe schemas
``engine``       the measurement itself
"""

from .accumulate import GradientVector, PairStatistics, capture_gradients, compare
from .checkpoints import CheckpointStage, build_stages, load_stage_model, state_fingerprint
from .datasets import PairedViewDataset, collate_paired, realisation_seed
from .engine import GradientEngine, GradientResult
from .scopes import ParameterScope, ScopeSet, build_scopes

__all__ = [
    "GradientVector",
    "PairStatistics",
    "capture_gradients",
    "compare",
    "CheckpointStage",
    "build_stages",
    "load_stage_model",
    "state_fingerprint",
    "PairedViewDataset",
    "collate_paired",
    "realisation_seed",
    "GradientEngine",
    "GradientResult",
    "ParameterScope",
    "ScopeSet",
    "build_scopes",
]
