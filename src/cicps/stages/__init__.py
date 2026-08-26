"""Executable pipeline stages.

Each stage is independently runnable and reads all of its settings from the YAML
configuration:

``prepare``  validate the dataset and materialise the internal train/val split
``train``    train the baseline model and write checkpoints
``audit``    run the transformation audit against a frozen checkpoint
``analyze``  summarise the audit dataframe and render the plots
``verify``   check that the audited transformations really are deterministic

Experiment 0B reuses ``prepare`` unchanged and swaps the training and analysis
implementations for policy-aware ones. Which pair runs is decided by the
configuration - a config that declares a ``policies`` section is a policy sweep -
so the command line stays identical between the two experiments:

``train_policies``    train one model per declared augmentation policy
``analyze_policies``  summarise the sweep and join it against Experiment 0A
"""

from .analyze import run_analyze
from .analyze_policies import run_analyze_policies
from .audit import run_audit
from .prepare import run_prepare
from .train import run_train
from .train_policies import run_train_policies
from .verify import run_verify_transforms

__all__ = [
    "run_analyze",
    "run_analyze_policies",
    "run_audit",
    "run_prepare",
    "run_train",
    "run_train_policies",
    "run_verify_transforms",
]
