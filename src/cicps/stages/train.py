"""Stage: baseline training."""

from __future__ import annotations

from pathlib import Path

from ..config import Config
from ..data.cub import load_cub_metadata
from ..data.splits import resolve_split
from ..data.validation import validate_split
from ..device import resolve_device
from ..logging_utils import get_logger, setup_logging
from ..seeding import seed_everything
from ..training.trainer import BaselineTrainer

__all__ = ["run_train"]

logger = get_logger(__name__)


def run_train(config: Config) -> Path:
    """Train the single baseline model; return the best checkpoint path."""
    setup_logging(config, stage="train")
    logger.info("Configuration: %s", config.source)
    seed = seed_everything(config)
    device = resolve_device(config)

    metadata = load_cub_metadata(config)
    split = resolve_split(metadata, config)

    report = validate_split(split, metadata, config)
    if not report.ok:
        report.log()
        raise RuntimeError("Refusing to train: the internal split failed validation.")
    logger.info(
        "Internal split: %d train / %d val images from the official train portion "
        "(official test set locked, %d images untouched).",
        split.num_train, split.num_val, len(metadata.official_test),
    )

    trainer = BaselineTrainer(config, metadata, split, device, seed)
    best_path = trainer.fit()
    logger.info("Training complete. Best checkpoint: %s", best_path)
    return best_path
