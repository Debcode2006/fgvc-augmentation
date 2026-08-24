"""Baseline training loop.

One conventional model, trained once, on the internal training subset. Its only
purpose is to give the audit a stable frozen classifier; nothing here is a
research contribution and nothing here should grow to chase accuracy.
"""

from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import Config
from ..data.cub import CubMetadata
from ..data.datasets import CubImageDataset
from ..data.splits import InternalSplit
from ..logging_utils import get_logger
from ..models.checkpoint import Checkpoint, save_checkpoint
from ..models.factory import FeatureClassifier, build_model
from ..seeding import dataloader_generator, worker_init_fn
from ..transforms.pipeline import build_eval_transform, build_train_transform
from .optim import build_amp_dtype, build_loss, build_optimizer, build_scheduler

__all__ = ["BaselineTrainer", "EpochMetrics"]

logger = get_logger(__name__)


@dataclass
class EpochMetrics:
    """Metrics recorded for one training epoch."""

    epoch: int
    lr: float
    train_loss: float
    train_top1: float
    val_loss: float
    val_top1: float
    val_top5: float
    seconds: float


class BaselineTrainer:
    """Trains the frozen-baseline model for Experiment 0A."""

    def __init__(
        self,
        config: Config,
        metadata: CubMetadata,
        split: InternalSplit,
        device: torch.device,
        seed: int,
    ) -> None:
        self.config = config
        self.metadata = metadata
        self.split = split
        self.device = device
        self.seed = seed

        self.epochs = int(config.get("train.epochs"))
        self.model: FeatureClassifier = build_model(config).to(device)
        self.criterion: nn.Module = build_loss(config)
        self.optimizer = build_optimizer(self.model.parameters(), config)
        self.scheduler = build_scheduler(self.optimizer, config, epochs=self.epochs)

        self.amp_enabled = bool(config.get("train.amp.enabled")) and device.type == "cuda"
        self.amp_dtype = build_amp_dtype(config, "train.amp.dtype")
        self.scaler = torch.amp.GradScaler(
            device.type, enabled=self.amp_enabled and self.amp_dtype is torch.float16
        )
        grad_clip = config.get("train.grad_clip_norm", None)
        self.grad_clip_norm = float(grad_clip) if grad_clip is not None else None

        self.train_loader = self._build_loader(split.train_ids, training=True)
        self.val_loader = self._build_loader(split.val_ids, training=False)
        self.history: list[EpochMetrics] = []

        self.monitor = str(config.get("train.checkpoint.monitor"))
        self.monitor_mode = str(config.get("train.checkpoint.mode")).lower()
        self._best_value = -float("inf") if self.monitor_mode == "max" else float("inf")

    # -- data --------------------------------------------------------------

    def _build_loader(self, image_ids: tuple[int, ...], *, training: bool) -> DataLoader:
        config = self.config
        pipeline = build_train_transform(config) if training else build_eval_transform(config)
        dataset = CubImageDataset(self.metadata, image_ids, pipeline)

        num_workers = int(config.get("train.num_workers"))
        batch_size = int(
            config.get("train.batch_size" if training else "train.eval_batch_size")
        )
        loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": training,
            "num_workers": num_workers,
            "pin_memory": bool(config.get("train.pin_memory")) and self.device.type == "cuda",
            "drop_last": False,
        }
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = bool(config.get("train.persistent_workers"))
            loader_kwargs["prefetch_factor"] = int(config.get("train.prefetch_factor"))
            if bool(config.get("seed.seed_dataloader_workers")):
                loader_kwargs["worker_init_fn"] = worker_init_fn
        if training:
            loader_kwargs["generator"] = dataloader_generator(self.seed)

        logger.info(
            "%s loader: %d images, batch_size=%d, workers=%d.",
            "Train" if training else "Val", len(dataset), batch_size, num_workers,
        )
        return DataLoader(dataset, **loader_kwargs)

    # -- loop --------------------------------------------------------------

    def fit(self) -> Path:
        """Train for the configured number of epochs; return the best checkpoint path."""
        checkpoint_dir = self.config.path("train.checkpoint.dir")
        best_path = checkpoint_dir / str(self.config.get("train.checkpoint.best_name"))
        last_path = checkpoint_dir / str(self.config.get("train.checkpoint.last_name"))
        save_last = bool(self.config.get("train.checkpoint.save_last"))

        logger.info("Training baseline for %d epoch(s) on %s.", self.epochs, self.device)
        for epoch in range(self.epochs):
            started = time.perf_counter()
            lr = self.optimizer.param_groups[0]["lr"]
            train_loss, train_top1 = self._train_one_epoch(epoch)
            val_loss, val_top1, val_top5 = self.evaluate()
            if self.scheduler is not None:
                self.scheduler.step()

            metrics = EpochMetrics(
                epoch=epoch,
                lr=lr,
                train_loss=train_loss,
                train_top1=train_top1,
                val_loss=val_loss,
                val_top1=val_top1,
                val_top5=val_top5,
                seconds=time.perf_counter() - started,
            )
            self.history.append(metrics)
            logger.info(
                "epoch %2d/%d | lr %.2e | train loss %.4f top1 %.4f | val loss %.4f "
                "top1 %.4f top5 %.4f | %.1fs",
                epoch + 1, self.epochs, lr, train_loss, train_top1, val_loss, val_top1,
                val_top5, metrics.seconds,
            )

            if self._is_improvement(metrics):
                save_checkpoint(self._make_checkpoint(metrics, best=True), best_path)
            if save_last:
                save_checkpoint(self._make_checkpoint(metrics, best=False), last_path)

        self._write_history()
        logger.info("Best %s = %.4f; best checkpoint: %s", self.monitor, self._best_value, best_path)
        return best_path

    def _train_one_epoch(self, epoch: int) -> tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        for step, batch in enumerate(self.train_loader):
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(self.device.type, dtype=self.amp_dtype,
                                    enabled=self.amp_enabled):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            if self.grad_clip_norm is not None:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_size = labels.size(0)
            total_loss += float(loss.detach()) * batch_size
            total_correct += int((logits.detach().argmax(dim=1) == labels).sum())
            total_seen += batch_size

            if step % 50 == 0:
                logger.debug(
                    "  epoch %d step %d/%d loss %.4f",
                    epoch + 1, step, len(self.train_loader), float(loss.detach()),
                )

        return total_loss / max(1, total_seen), total_correct / max(1, total_seen)

    @torch.no_grad()
    def evaluate(self) -> tuple[float, float, float]:
        """Evaluate on the internal validation subset; returns (loss, top1, top5)."""
        self.model.eval()
        total_loss = 0.0
        total_top1 = 0
        total_top5 = 0
        total_seen = 0

        for batch in self.val_loader:
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)
            with torch.amp.autocast(self.device.type, dtype=self.amp_dtype,
                                    enabled=self.amp_enabled):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += float(loss) * batch_size
            top5 = logits.float().topk(min(5, logits.size(1)), dim=1).indices
            total_top1 += int((top5[:, 0] == labels).sum())
            total_top5 += int((top5 == labels.unsqueeze(1)).any(dim=1).sum())
            total_seen += batch_size

        denominator = max(1, total_seen)
        return total_loss / denominator, total_top1 / denominator, total_top5 / denominator

    # -- checkpointing -----------------------------------------------------

    def _is_improvement(self, metrics: EpochMetrics) -> bool:
        value = getattr(metrics, self.monitor, None)
        if value is None:
            raise ValueError(
                f"train.checkpoint.monitor='{self.monitor}' is not a recorded metric. "
                f"Available: {', '.join(sorted(asdict(metrics)))}."
            )
        improved = value > self._best_value if self.monitor_mode == "max" else value < self._best_value
        if improved:
            self._best_value = value
        return improved

    def _make_checkpoint(self, metrics: EpochMetrics, *, best: bool) -> Checkpoint:
        include_optimizer = bool(self.config.get("train.checkpoint.save_optimizer_state"))
        return Checkpoint(
            model_state={key: value.detach().cpu()
                         for key, value in self.model.state_dict().items()},
            optimizer_state=self.optimizer.state_dict() if include_optimizer else None,
            scheduler_state=(self.scheduler.state_dict()
                             if include_optimizer and self.scheduler is not None else None),
            config=self.config.as_dict(),
            epoch=metrics.epoch,
            seed=self.seed,
            metrics={
                "val_top1": metrics.val_top1,
                "val_top5": metrics.val_top5,
                "val_loss": metrics.val_loss,
                "train_top1": metrics.train_top1,
                "train_loss": metrics.train_loss,
            },
            metadata={
                "selection": "best" if best else "last",
                "monitor": self.monitor,
                "architecture": self.model.architecture,
                "feature_dim": self.model.feature_dim,
                "num_classes": self.model.num_classes,
                "split_fingerprint": self.split.fingerprint,
                "split_seed": self.split.seed,
                "num_train_images": self.split.num_train,
                "num_val_images": self.split.num_val,
                "device": str(self.device),
                "torch_version": torch.__version__,
                "amp_enabled": self.amp_enabled,
            },
        )

    def _write_history(self) -> None:
        path = self.config.path("train.history_path")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(self.history[0])))
            writer.writeheader()
            for metrics in self.history:
                writer.writerow(asdict(metrics))
        logger.info("Wrote training history to %s", path)
