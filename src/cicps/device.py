"""Device resolution."""

from __future__ import annotations

import torch

from .config import Config, ConfigError
from .logging_utils import get_logger

__all__ = ["resolve_device"]

logger = get_logger(__name__)


def resolve_device(config: Config) -> torch.device:
    """Resolve ``runtime.device`` into a concrete :class:`torch.device`.

    ``auto`` picks CUDA when available and falls back to CPU. An explicit CUDA
    request on a machine without CUDA raises when ``runtime.require_device`` is
    set, rather than silently training on CPU for hours.
    """
    requested = str(config.get("runtime.device")).strip().lower()
    require = bool(config.get("runtime.require_device"))
    cuda_available = torch.cuda.is_available()

    if requested == "auto":
        device = torch.device("cuda" if cuda_available else "cpu")
    elif requested.startswith("cuda"):
        if not cuda_available:
            message = (
                f"runtime.device='{requested}' was requested but torch.cuda.is_available() "
                "is False."
            )
            if require:
                raise ConfigError(message + " Set runtime.device: cpu or fix the CUDA install.")
            logger.warning("%s Falling back to CPU.", message)
            device = torch.device("cpu")
        else:
            device = torch.device(requested)
    else:
        device = torch.device(requested)

    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        logger.info(
            "Device: %s (%s, %.1f GB)",
            device, properties.name, properties.total_memory / 1024**3,
        )
    else:
        logger.info("Device: %s", device)
    return device
