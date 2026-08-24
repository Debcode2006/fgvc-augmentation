"""Batched inference over a frozen model.

Every pass runs under ``torch.no_grad()`` with the model in ``eval()`` mode, in
batches whose size comes from the configuration. Probabilities are always
computed in float32 - they are the measurement instrument of Experiment 0A and
small margins must not be quantised away by half precision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..config import Config
from ..logging_utils import get_logger
from ..models.factory import FeatureClassifier
from ..seeding import worker_init_fn

__all__ = ["InferenceResult", "run_inference", "build_audit_loader"]

logger = get_logger(__name__)


@dataclass(frozen=True)
class InferenceResult:
    """Model outputs for a set of images, row-aligned with ``image_ids``."""

    image_ids: np.ndarray
    """(N,) official CUB image ids."""

    labels: np.ndarray
    """(N,) contiguous true labels (0..C-1)."""

    probs: np.ndarray
    """(N, C) float32 softmax probabilities."""

    features: np.ndarray
    """(N, D) float32 penultimate representations."""

    def __len__(self) -> int:
        return int(self.image_ids.shape[0])

    @property
    def predictions(self) -> np.ndarray:
        """(N,) arg-max predicted labels."""
        return self.probs.argmax(axis=1)

    def aligned_to(self, image_ids: np.ndarray) -> "InferenceResult":
        """Reorder rows to match ``image_ids`` exactly.

        Loaders are built with ``shuffle=False`` over the same id list, so this
        is normally a no-op; it is applied defensively so that a future change
        to loading order can never silently misalign an original with its
        transformed counterpart.
        """
        if np.array_equal(self.image_ids, image_ids):
            return self
        if set(self.image_ids.tolist()) != set(image_ids.tolist()):
            raise ValueError(
                "Cannot align inference results: the image id sets differ "
                f"({len(self)} vs {len(image_ids)} rows)."
            )
        position = {int(image_id): index for index, image_id in enumerate(self.image_ids)}
        order = np.array([position[int(image_id)] for image_id in image_ids], dtype=np.int64)
        return InferenceResult(
            image_ids=self.image_ids[order],
            labels=self.labels[order],
            probs=self.probs[order],
            features=self.features[order],
        )


def build_audit_loader(dataset: Dataset, config: Config, device: torch.device) -> DataLoader:
    """Build a deterministic, non-shuffled loader for the audit."""
    num_workers = int(config.get("audit.num_workers"))
    kwargs = {
        "batch_size": int(config.get("audit.batch_size")),
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": bool(config.get("audit.pin_memory")) and device.type == "cuda",
        "drop_last": False,
    }
    if num_workers > 0 and bool(config.get("seed.seed_dataloader_workers")):
        kwargs["worker_init_fn"] = worker_init_fn
    return DataLoader(dataset, **kwargs)


@torch.no_grad()
def run_inference(
    model: FeatureClassifier,
    loader: DataLoader,
    device: torch.device,
    *,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    progress: bool = False,
    description: str = "inference",
) -> InferenceResult:
    """Run the frozen model over ``loader`` and collect probabilities + features."""
    model.eval()

    batches = _maybe_progress(loader, enabled=progress, description=description)
    image_ids: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    probs: list[np.ndarray] = []
    features: list[np.ndarray] = []

    for batch in batches:
        images = batch["image"].to(device, non_blocking=True)
        with torch.amp.autocast(device.type, dtype=amp_dtype,
                                enabled=amp_enabled and device.type == "cuda"):
            logits, representation = model(images, return_features=True)

        # float32 softmax: margins are the measurement, not an intermediate.
        batch_probs = torch.softmax(logits.float(), dim=1)
        probs.append(batch_probs.cpu().numpy().astype(np.float32))
        features.append(representation.float().cpu().numpy().astype(np.float32))
        image_ids.append(batch["image_id"].numpy().astype(np.int64))
        labels.append(batch["label"].numpy().astype(np.int64))

    return InferenceResult(
        image_ids=np.concatenate(image_ids),
        labels=np.concatenate(labels),
        probs=np.concatenate(probs),
        features=np.concatenate(features),
    )


def _maybe_progress(loader: DataLoader, *, enabled: bool, description: str):
    """Wrap ``loader`` in a tqdm bar when enabled and tqdm is importable."""
    if not enabled:
        return loader
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover - tqdm is an optional convenience
        logger.debug("tqdm is not installed; progress bars disabled.")
        return loader
    return tqdm(loader, desc=description, leave=False)
