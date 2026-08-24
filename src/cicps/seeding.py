"""Seed management and determinism control.

All randomness in Experiment 0A descends from one master seed declared in the
YAML configuration. Two distinct notions of determinism are handled here:

*Global* determinism (this module's :func:`seed_everything`) makes training and
data loading reproducible.

*Per-sample* determinism (:func:`derive_seed`) gives every (image, transform)
pair its own fixed parameter draw. That is what stops the transformation audit
from degenerating into stochastic augmentation sampling: re-running the audit
re-derives byte-identical transformation parameters for every sample.
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
from typing import Any

import numpy as np
import torch

from .config import Config

__all__ = ["seed_everything", "derive_seed", "worker_init_fn", "dataloader_generator"]

logger = logging.getLogger(__name__)

_UINT64 = 1 << 64
_INT32 = 1 << 31


def seed_everything(config: Config) -> int:
    """Seed Python, NumPy, torch and CUDA from the configuration.

    Returns the master seed so callers can record it in checkpoints and logs.
    """
    seed = int(config.get("seed.value"))

    random.seed(seed)
    np.random.seed(seed % _INT32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = bool(config.get("seed.cudnn_deterministic"))
    torch.backends.cudnn.benchmark = bool(config.get("seed.cudnn_benchmark"))

    if bool(config.get("seed.deterministic_algorithms")):
        # Required by several deterministic CUDA kernels; must be set before use.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)

    logger.info(
        "Seeded RNGs with master seed %d (cudnn.deterministic=%s, cudnn.benchmark=%s, "
        "deterministic_algorithms=%s)",
        seed,
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
        bool(config.get("seed.deterministic_algorithms")),
    )
    return seed


def derive_seed(*parts: Any) -> int:
    """Derive a stable 64-bit seed from arbitrary hashable parts.

    Uses BLAKE2b over the string form of ``parts`` rather than :func:`hash`,
    because Python's built-in hash is salted per process and would silently make
    the audit irreproducible across runs.

    >>> derive_seed(42, 7, "rotation") == derive_seed(42, 7, "rotation")
    True
    """
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") % _UINT64


def worker_init_fn(worker_id: int) -> None:
    """Seed a DataLoader worker deterministically from the parent seed.

    ``torch.initial_seed()`` inside a worker is already derived from the parent
    generator, so re-seeding Python/NumPy from it keeps every library's stream
    reproducible without collapsing workers onto the same stream.
    """
    base_seed = torch.initial_seed() % _UINT64
    random.seed(base_seed + worker_id)
    np.random.seed((base_seed + worker_id) % _INT32)


def dataloader_generator(seed: int) -> torch.Generator:
    """Return a CPU generator for DataLoader shuffling."""
    generator = torch.Generator()
    generator.manual_seed(int(seed) % _UINT64)
    return generator
