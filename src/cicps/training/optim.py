"""Optimizer, scheduler and loss factories, all configuration driven."""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from ..config import Config, ConfigError

__all__ = ["build_optimizer", "build_scheduler", "build_loss", "build_amp_dtype"]

_AMP_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def build_optimizer(parameters: Iterable[nn.Parameter], config: Config) -> Optimizer:
    """Build the optimizer from ``train.optimizer``."""
    section = config.section("train.optimizer")
    name = str(section.get("name")).lower()
    lr = float(section.get("lr"))
    weight_decay = float(section.get("weight_decay"))

    if name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            betas=tuple(float(beta) for beta in section.get("betas")),
            eps=float(section.get("eps")),
        )
    if name == "adam":
        return torch.optim.Adam(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            betas=tuple(float(beta) for beta in section.get("betas")),
            eps=float(section.get("eps")),
        )
    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            momentum=float(section.get("momentum", 0.9)),
            nesterov=bool(section.get("nesterov", True)),
        )
    raise ConfigError(
        f"Unknown train.optimizer.name '{name}'. Supported: adamw, adam, sgd."
    )


def build_scheduler(
    optimizer: Optimizer, config: Config, *, epochs: int
) -> LRScheduler | None:
    """Build a per-epoch learning-rate scheduler from ``train.scheduler``."""
    section = config.section("train.scheduler")
    name = str(section.get("name")).lower()
    if name in {"none", "constant"}:
        return None
    if name != "cosine":
        raise ConfigError(
            f"Unknown train.scheduler.name '{name}'. Supported: cosine, constant, none."
        )

    warmup_epochs = int(section.get("warmup_epochs"))
    min_lr = float(section.get("min_lr"))
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    if warmup_epochs < 0 or warmup_epochs >= max(1, epochs):
        raise ConfigError(
            f"train.scheduler.warmup_epochs ({warmup_epochs}) must lie in [0, train.epochs)."
        )

    def lr_lambda(epoch: int, base_lr: float) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return (min_lr + (base_lr - min_lr) * cosine) / base_lr

    return LambdaLR(
        optimizer,
        lr_lambda=[lambda epoch, base=base: lr_lambda(epoch, base) for base in base_lrs],
    )


def build_loss(config: Config) -> nn.Module:
    """Build the training criterion from ``train.loss``."""
    section = config.section("train.loss")
    name = str(section.get("name")).lower()
    if name != "cross_entropy":
        raise ConfigError(
            f"Unknown train.loss.name '{name}'. Experiment 0A uses cross_entropy only."
        )
    return nn.CrossEntropyLoss(label_smoothing=float(section.get("label_smoothing")))


def build_amp_dtype(config: Config, key: str) -> torch.dtype:
    """Resolve an AMP dtype string (e.g. ``train.amp.dtype``)."""
    name = str(config.get(key)).lower()
    try:
        return _AMP_DTYPES[name]
    except KeyError as error:
        raise ConfigError(
            f"Unknown AMP dtype '{name}' at '{key}'. Supported: {', '.join(_AMP_DTYPES)}."
        ) from error
