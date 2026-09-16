"""Checkpoint stage resolution for Experiment 1A.

A *stage* is one point in model learning at which the gradient measurement is
repeated. Stages are declared in YAML; two kinds exist.

``file``
    A checkpoint written by Experiment 0A or 0B. It is loaded read-only through
    the existing :mod:`cicps.models.checkpoint` loader, and Experiment 1A never
    writes into ``outputs/experiment_0a/`` or ``outputs/experiment_0b/``.

``init``
    Epoch 0 - the ImageNet-pretrained backbone with a freshly drawn head. No
    such checkpoint was ever written, because the trainer only persists ``best``
    and ``last``. It is *reconstructed* by installing the master seed and
    building the model, which is exactly what Experiment 0B does before training
    each arm (``stages/train_policies.py``), and which was verified to produce
    bit-identical weights on repeated builds. Its state is hashed and recorded so
    the reconstruction is checkable after the fact.

Unlike :func:`cicps.models.factory.load_model_for_audit`, the model returned here
keeps ``requires_grad=True``: Experiment 1A differentiates the loss with respect
to the parameters. The parameters are nonetheless *fixed* - no optimizer is ever
constructed and no ``step()`` is ever called - and :func:`state_fingerprint` lets
the engine prove the weights did not move.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from ..config import Config, ConfigError
from ..logging_utils import get_logger
from ..models.checkpoint import load_checkpoint
from ..models.factory import FeatureClassifier, build_model
from ..seeding import seed_run

__all__ = ["CheckpointStage", "build_stages", "load_stage_model", "state_fingerprint"]

logger = get_logger(__name__)

_KIND_FILE = "file"
_KIND_INIT = "init"
_KINDS = (_KIND_FILE, _KIND_INIT)


@dataclass(frozen=True)
class CheckpointStage:
    """One point in training at which gradients are measured."""

    name: str
    """Stable identifier; appears in every 1A output row."""

    kind: str
    """``file`` (load a written checkpoint) or ``init`` (reconstruct epoch 0)."""

    description: str

    path: Path | None = None
    """Checkpoint path, for ``kind: file``."""

    transforms: tuple[str, ...] | None = None
    """Restrict this stage to a subset of the transformation suite.

    ``None`` means "every declared transformation". The adaptation probe uses it
    to measure a policy's own transformation (plus the control arm) on the model
    that was trained with it, without paying for the other five arms.
    """

    policy: str | None = None
    """Experiment 0B policy this checkpoint came from, when applicable."""

    metrics: dict[str, float] = field(default_factory=dict)
    """Validation metrics recorded in the checkpoint (populated on load)."""

    epoch: int = -1
    """Epoch the checkpoint was written at (-1 for a reconstructed init)."""

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "description": self.description,
            "path": None if self.path is None else str(self.path),
            "policy": self.policy,
            "transforms": None if self.transforms is None else list(self.transforms),
            "epoch": self.epoch,
            "metrics": dict(self.metrics),
        }


def build_stages(config: Config, available_transforms: set[str]) -> list[CheckpointStage]:
    """Build the checkpoint stages declared under ``gradients.checkpoints``."""
    stages: list[CheckpointStage] = []
    seen: set[str] = set()

    for spec in config.sections("gradients.checkpoints"):
        name = str(spec.get("name")).strip()
        if not name:
            raise ConfigError("Every entry of 'gradients.checkpoints' needs a non-empty 'name'.")
        if name in seen:
            raise ConfigError(f"Duplicate checkpoint stage name '{name}'.")
        seen.add(name)

        kind = str(spec.get("kind")).strip().lower()
        if kind not in _KINDS:
            raise ConfigError(
                f"Checkpoint stage '{name}': kind must be one of {_KINDS}, got '{kind}'."
            )

        path: Path | None = None
        if kind == _KIND_FILE:
            raw = spec.get("path", None)
            if raw is None:
                raise ConfigError(f"Checkpoint stage '{name}' (kind 'file') requires a 'path'.")
            path = config.resolve(str(raw))
            if not path.is_file():
                raise ConfigError(
                    f"Checkpoint stage '{name}': no checkpoint at {path}. Experiment 1A reuses "
                    "checkpoints written by Experiments 0A and 0B; run those first, or remove "
                    "this stage from the configuration."
                )

        transforms = spec.get("transforms", None)
        if transforms is not None:
            if isinstance(transforms, str) or not isinstance(transforms, (list, tuple)):
                raise ConfigError(
                    f"Checkpoint stage '{name}': 'transforms' must be a list of transformation "
                    f"names, got {transforms!r}."
                )
            unknown = [t for t in transforms if str(t) not in available_transforms]
            if unknown:
                raise ConfigError(
                    f"Checkpoint stage '{name}' names transformation(s) {unknown} that are not "
                    f"declared under 'transformations'. Declared: "
                    f"{', '.join(sorted(available_transforms))}."
                )
            transforms = tuple(str(t) for t in transforms)

        stages.append(
            CheckpointStage(
                name=name,
                kind=kind,
                description=str(spec.get("description", "")),
                path=path,
                transforms=transforms,
                policy=(None if spec.get("policy", None) is None else str(spec.get("policy"))),
            )
        )

    if not stages:
        raise ConfigError(
            "Configuration key 'gradients.checkpoints' must declare at least one stage."
        )
    return stages


def load_stage_model(
    stage: CheckpointStage, config: Config, device: torch.device
) -> tuple[FeatureClassifier, CheckpointStage]:
    """Materialise ``stage``'s model, frozen in value but differentiable.

    Returns the model and a copy of the stage enriched with the epoch and the
    metrics read from the checkpoint. The model is put in ``eval()`` mode - see
    :mod:`cicps.gradients.engine` for why that is the scientifically correct
    choice for a per-sample gradient - and every parameter keeps
    ``requires_grad=True`` so the loss can be differentiated with respect to it.
    """
    if stage.kind == _KIND_INIT:
        master_seed = int(config.get("seed.value"))
        logger.info(
            "Stage '%s': reconstructing the epoch-0 initialisation under master seed %d "
            "(the same two-step seeding Experiment 0B used before training each arm).",
            stage.name, master_seed,
        )
        seed_run(master_seed)
        model = build_model(config)
        enriched = stage
    else:
        checkpoint = load_checkpoint(stage.path, map_location="cpu")
        model = build_model(_without_pretrained(config))
        model.load_state_dict(checkpoint.model_state, strict=True)
        enriched = CheckpointStage(
            name=stage.name,
            kind=stage.kind,
            description=stage.description,
            path=stage.path,
            transforms=stage.transforms,
            policy=stage.policy,
            metrics=dict(checkpoint.metrics),
            epoch=int(checkpoint.epoch),
        )

    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return model, enriched


def state_fingerprint(model: torch.nn.Module) -> str:
    """BLAKE2b digest over every parameter *and* buffer of ``model``.

    Buffers are included on purpose: BatchNorm running statistics are not
    parameters, and an accidental ``train()`` mode would mutate them without
    touching a single weight. Comparing this digest before and after a stage is
    what proves Experiment 1A left the checkpoint untouched.
    """
    digest = hashlib.blake2b(digest_size=16)
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _without_pretrained(config: Config) -> Config:
    """Disable the pretrained download when the weights are about to be replaced."""
    data = config.as_dict()
    data.setdefault("model", {})["pretrained"] = False
    return Config(data, root=config.root, source=config.source)
