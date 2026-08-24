"""Executable pipeline stages.

Each stage is independently runnable and reads all of its settings from the YAML
configuration:

``prepare``  validate the dataset and materialise the internal train/val split
``train``    train the baseline model and write checkpoints
``audit``    run the transformation audit against a frozen checkpoint
``analyze``  summarise the audit dataframe and render the plots
``verify``   check that the audited transformations really are deterministic
"""

from .analyze import run_analyze
from .audit import run_audit
from .prepare import run_prepare
from .train import run_train
from .verify import run_verify_transforms

__all__ = ["run_analyze", "run_audit", "run_prepare", "run_train", "run_verify_transforms"]
