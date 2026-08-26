"""The training loop.

Experiment 0A trains one conventional model, once, on the internal training
subset; its only purpose is to give the audit a stable frozen classifier.
Experiment 0B runs the *same* loop once per augmentation policy - same
architecture, optimizer, schedule, budget and checkpoint rule - varying nothing
but the training pipeline and the RNG seed. There is deliberately one trainer:
a second copy would be a second place for the two experiments to drift apart.

Callers that need policy-scoped behaviour supply :class:`TrainingArtifacts`
(where the checkpoints and history go), a training pipeline, a run label and any
extra checkpoint metadata. Omit them all and the trainer behaves exactly as
Experiment 0A's single-model run always has.

Nothing here is a research contribution and nothing here should grow to chase
accuracy.
"""

from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from PIL.Image import Image
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

__all__ = ["BaselineTrainer", "EpochMetrics", "TrainingArtifacts"]

logger = get_logger(__name__)


@dataclass
class EpochMetrics:
    """Metrics recorded for one training epoch."""

    epoch: int
    lr: float
    train_loss: float
    train_top1: float
    train_top5: float
    val_loss: float
    val_top1: float
    val_top5: float
    seconds: float


@dataclass(frozen=True)
class TrainingArtifacts:
    """Where one training run writes its checkpoints and per-epoch history.

    Experiment 0A reads these straight from ``train.checkpoint`` /
    ``train.history_path``; Experiment 0B scopes them per policy so that no arm
    can overwrite another's checkpoint.
    """

    checkpoint_dir: Path
    best_name: str
    last_name: str
    history_path: Path

    @classmethod
    def from_config(cls, config: Config) -> "TrainingArtifacts":
        """The single-model layout declared under ``train`` in the YAML file."""
        return cls(
            checkpoint_dir=config.path("train.checkpoint.dir"),
            best_name=str(config.get("train.checkpoint.best_name")),
            last_name=str(config.get("train.checkpoint.last_name")),
            history_path=config.path("train.history_path"),
        )

    @property
    def best_path(self) -> Path:
        return self.checkpoint_dir / self.best_name

    @property
    def last_path(self) -> Path:
        return self.checkpoint_dir / self.last_name


class BaselineTrainer:
    """Trains one model under one training pipeline.

    Used unchanged by Experiment 0A (a single baseline) and by Experiment 0B
    (once per augmentation policy).
    """

    def __init__(
        self,
        config: Config,
        metadata: CubMetadata,
        split: InternalSplit,
        device: torch.device,
        seed: int,
        *,
        run_name: str = "baseline",
        train_pipeline: Callable[[Image], torch.Tensor] | None = None,
        artifacts: TrainingArtifacts | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.metadata = metadata
        self.split = split
        self.device = device
        self.seed = seed
        self.run_name = run_name
        self.train_pipeline = train_pipeline
        self.artifacts = artifacts or TrainingArtifacts.from_config(config)
        self.extra_metadata = dict(extra_metadata or {})

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
        self.best_metrics: EpochMetrics | None = None

    @property
    def best_epoch(self) -> int:
        """Zero-based epoch that produced the selected checkpoint (-1 if none)."""
        return -1 if self.best_metrics is None else self.best_metrics.epoch

    # -- data --------------------------------------------------------------

    def _build_loader(self, image_ids: tuple[int, ...], *, training: bool) -> DataLoader:
        config = self.config
        if training:
            pipeline = (
                self.train_pipeline if self.train_pipeline is not None
                else build_train_transform(config)
            )
        else:
            pipeline = build_eval_transform(config)
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
        best_path = self.artifacts.best_path
        last_path = self.artifacts.last_path
        save_last = bool(self.config.get("train.checkpoint.save_last"))

        logger.info(
            "Training run '%s' for %d epoch(s) on %s (seed %d).",
            self.run_name, self.epochs, self.device, self.seed,
        )
        for epoch in range(self.epochs):
            started = time.perf_counter()
            lr = self.optimizer.param_groups[0]["lr"]
            train_loss, train_top1, train_top5 = self._train_one_epoch(epoch)
            val_loss, val_top1, val_top5 = self.evaluate()
            if self.scheduler is not None:
                self.scheduler.step()

            metrics = EpochMetrics(
                epoch=epoch,
                lr=lr,
                train_loss=train_loss,
                train_top1=train_top1,
                train_top5=train_top5,
                val_loss=val_loss,
                val_top1=val_top1,
                val_top5=val_top5,
                seconds=time.perf_counter() - started,
            )
            self._require_finite(metrics)
            self.history.append(metrics)
            logger.info(
                "[%s] epoch %2d/%d | lr %.2e | train loss %.4f top1 %.4f top5 %.4f | "
                "val loss %.4f top1 %.4f top5 %.4f | %.1fs",
                self.run_name, epoch + 1, self.epochs, lr, train_loss, train_top1, train_top5,
                val_loss, val_top1, val_top5, metrics.seconds,
            )

            if self._is_improvement(metrics):
                save_checkpoint(self._make_checkpoint(metrics, best=True), best_path)
            if save_last:
                save_checkpoint(self._make_checkpoint(metrics, best=False), last_path)

        self._write_history()
        logger.info(
            "[%s] best %s = %.4f at epoch %d; best checkpoint: %s",
            self.run_name, self.monitor, self._best_value, self.best_epoch + 1, best_path,
        )
        return best_path

    def _train_one_epoch(self, epoch: int) -> tuple[float, float, float]:
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_top5 = 0
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
            batch_loss = float(loss.detach())
            if batch_loss != batch_loss or batch_loss in (float("inf"), -float("inf")):
                raise RuntimeError(
                    f"[{self.run_name}] non-finite training loss ({batch_loss}) at epoch "
                    f"{epoch + 1}, step {step}. Training is diverging; inspect the "
                    "learning rate, AMP settings and the augmentation policy."
                )
            total_loss += batch_loss * batch_size
            scores = logits.detach().float().topk(min(5, logits.size(1)), dim=1).indices
            total_correct += int((scores[:, 0] == labels).sum())
            total_top5 += int((scores == labels.unsqueeze(1)).any(dim=1).sum())
            total_seen += batch_size

            if step % 50 == 0:
                logger.debug(
                    "  [%s] epoch %d step %d/%d loss %.4f",
                    self.run_name, epoch + 1, step, len(self.train_loader), batch_loss,
                )

        denominator = max(1, total_seen)
        return total_loss / denominator, total_correct / denominator, total_top5 / denominator

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

    def _require_finite(self, metrics: EpochMetrics) -> None:
        """Fail loudly on a NaN/Inf metric rather than checkpointing on it."""
        offenders = [
            key for key, value in asdict(metrics).items()
            if isinstance(value, float) and (value != value or value in (float("inf"), -float("inf")))
        ]
        if offenders:
            raise RuntimeError(
                f"[{self.run_name}] epoch {metrics.epoch + 1} produced non-finite "
                f"metric(s): {', '.join(offenders)}."
            )

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
            self.best_metrics = metrics
        return improved

    def _make_checkpoint(self, metrics: EpochMetrics, *, best: bool) -> Checkpoint:
        include_optimizer = bool(self.config.get("train.checkpoint.save_optimizer_state"))
        checkpoint = Checkpoint(
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
                "train_top5": metrics.train_top5,
                "train_loss": metrics.train_loss,
            },
            metadata={
                "run_name": self.run_name,
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
        checkpoint.metadata.update(self.extra_metadata)
        return checkpoint

    def _write_history(self) -> None:
        path = self.artifacts.history_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(self.history[0])))
            writer.writeheader()
            for metrics in self.history:
                writer.writerow(asdict(metrics))
        logger.info("Wrote training history to %s", path)
