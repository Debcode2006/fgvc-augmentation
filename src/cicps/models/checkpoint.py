"""Checkpoint saving and loading.

A checkpoint carries everything the audit stage needs to reconstruct the exact
frozen model without re-reading the training configuration from elsewhere:
weights, the configuration the model was trained under, the master seed, the
split fingerprint, and the validation metric that selected it.

Training and auditing are therefore separable stages - the audit can be re-run
any number of times against a checkpoint trained once.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from ..config import Config, ConfigError
from ..logging_utils import get_logger

__all__ = ["Checkpoint", "save_checkpoint", "load_checkpoint", "CHECKPOINT_FORMAT_VERSION"]

logger = get_logger(__name__)

CHECKPOINT_FORMAT_VERSION = 1


@dataclass
class Checkpoint:
    """A trained baseline together with the metadata needed to reproduce it."""

    model_state: dict[str, torch.Tensor]
    config: dict[str, Any]
    epoch: int
    seed: int
    metrics: dict[str, float]
    metadata: dict[str, Any]
    optimizer_state: dict[str, Any] | None = None
    scheduler_state: dict[str, Any] | None = None
    format_version: int = CHECKPOINT_FORMAT_VERSION

    def to_payload(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "model_state": self.model_state,
            "optimizer_state": self.optimizer_state,
            "scheduler_state": self.scheduler_state,
            "config": self.config,
            "epoch": self.epoch,
            "seed": self.seed,
            "metrics": self.metrics,
            "metadata": self.metadata,
        }


def save_checkpoint(checkpoint: Checkpoint, path: Path) -> Path:
    """Write a checkpoint atomically (temp file then replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.metadata.setdefault(
        "saved_at", datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint.to_payload(), temporary)
    temporary.replace(path)
    logger.info(
        "Saved checkpoint to %s (epoch %d, %s).",
        path, checkpoint.epoch,
        ", ".join(f"{key}={value:.4f}" for key, value in sorted(checkpoint.metrics.items())),
    )
    return path


def load_checkpoint(path: Path, *, map_location: str | torch.device = "cpu") -> Checkpoint:
    """Load a checkpoint previously written by :func:`save_checkpoint`."""
    if not path.is_file():
        raise ConfigError(
            f"Checkpoint not found: {path}. Train the baseline first "
            "(`python -m cicps train --config ...`), or point audit.checkpoint elsewhere."
        )
    # weights_only=False: the payload intentionally carries the configuration
    # mapping and metadata, not just tensors. Only load checkpoints you produced.
    payload = torch.load(path, map_location=map_location, weights_only=False)

    version = int(payload.get("format_version", 0))
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ConfigError(
            f"Checkpoint {path} has format_version {version}, expected "
            f"{CHECKPOINT_FORMAT_VERSION}. Retrain the baseline."
        )
    checkpoint = Checkpoint(
        model_state=payload["model_state"],
        optimizer_state=payload.get("optimizer_state"),
        scheduler_state=payload.get("scheduler_state"),
        config=payload.get("config", {}),
        epoch=int(payload.get("epoch", -1)),
        seed=int(payload.get("seed", -1)),
        metrics={str(key): float(value) for key, value in payload.get("metrics", {}).items()},
        metadata=dict(payload.get("metadata", {})),
        format_version=version,
    )
    logger.info(
        "Loaded checkpoint %s (epoch %d, seed %d, metrics: %s).",
        path, checkpoint.epoch, checkpoint.seed,
        ", ".join(f"{key}={value:.4f}" for key, value in sorted(checkpoint.metrics.items()))
        or "none",
    )
    return checkpoint


def warn_on_config_drift(checkpoint: Checkpoint, config: Config, keys: tuple[str, ...]) -> None:
    """Log a warning when the audit configuration disagrees with the checkpoint.

    Auditing a model under preprocessing it was never trained with silently
    changes what ``delta_margin`` means, so the drift is surfaced explicitly.
    """
    trained_under = Config(checkpoint.config, root=config.root) if checkpoint.config else None
    if trained_under is None:
        logger.warning("Checkpoint carries no configuration; cannot check for drift.")
        return
    for key in keys:
        current = config.get(key, None)
        previous = trained_under.get(key, None)
        if previous is not None and current != previous:
            logger.warning(
                "Configuration drift: '%s' is %r now but was %r when the checkpoint was trained.",
                key, current, previous,
            )
