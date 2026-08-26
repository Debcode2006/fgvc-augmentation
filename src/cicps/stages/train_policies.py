"""Stage: Experiment 0B's controlled augmentation-policy sweep.

Trains one independent model per policy declared under ``policies`` in the YAML
file. Every arm is identical except for the augmentation pipeline and the RNG
stream: same dataset, same internal split, same ImageNet-pretrained
initialisation, same optimizer, schedule, epoch budget, batch size, validation
preprocessing, loss and checkpoint rule. Nothing is tuned per policy, and no
Experiment 0A result influences what is trained - the 0A summary is joined
against these results afterwards, during analysis.

Each policy starts from a freshly built model, so nothing is carried over
between arms. The official CUB test portion is never loaded.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from random import Random
from typing import Any

import torch

from ..config import Config, ConfigError
from ..data.cub import CubMetadata
from ..data.splits import InternalSplit, resolve_split
from ..data.validation import validate_split
from ..data.cub import load_cub_metadata
from ..device import resolve_device
from ..logging_utils import additional_log_file, get_logger, setup_logging
from ..seeding import derive_run_seed, seed_everything, seed_run
from ..training.policies import AugmentationPolicy, build_policies, select_policies
from ..training.trainer import BaselineTrainer, TrainingArtifacts
from ..transforms.policy_pipeline import (
    build_policy_train_transform,
    describe_policy_pipeline,
)

__all__ = ["run_train_policies", "PolicyRunRecord", "RUN_COLUMNS"]

logger = get_logger(__name__)


@dataclass
class PolicyRunRecord:
    """One row of ``train_runs.csv``: the outcome of training one policy."""

    policy: str
    audit_transform: str
    is_baseline: bool
    description: str
    seed: int
    split_fingerprint: str
    num_train_images: int
    num_val_images: int
    epochs: int
    batch_size: int
    optimizer: str
    learning_rate: float
    weight_decay: float
    scheduler: str
    architecture: str
    pretrained_weights: str
    device: str
    application_probability: float
    best_epoch: int
    best_val_top1: float
    best_val_top5: float
    best_val_loss: float
    best_train_top1: float
    best_train_top5: float
    best_train_loss: float
    final_val_top1: float
    final_val_top5: float
    final_val_loss: float
    final_train_top1: float
    final_train_top5: float
    final_train_loss: float
    total_seconds: float
    checkpoint_path: str
    history_path: str
    completed_at: str = field(default="")


RUN_COLUMNS: tuple[str, ...] = tuple(PolicyRunRecord.__dataclass_fields__)
"""Canonical column order of ``train_runs.csv``."""


def run_train_policies(
    config: Config, *, policy_names: list[str] | None = None
) -> list[PolicyRunRecord]:
    """Train every declared policy (or just ``policy_names``); return their records."""
    setup_logging(config, stage="train")
    logger.info("Configuration: %s", config.source)
    logger.info("Experiment: %s", config.get("experiment.name"))

    master_seed = seed_everything(config)
    device = resolve_device(config)

    metadata = load_cub_metadata(config)
    split = resolve_split(metadata, config)
    report = validate_split(split, metadata, config)
    if not report.ok:
        report.log()
        raise RuntimeError("Refusing to train: the internal split failed validation.")

    train_ids, val_ids = _apply_subset(config, split, master_seed)
    logger.info(
        "Internal split %s: %d train / %d val images from the official train portion "
        "(official test set locked, %d images untouched).",
        split.fingerprint, len(train_ids), len(val_ids), len(metadata.official_test),
    )

    declared = build_policies(config)
    policies = select_policies(declared, policy_names)
    logger.info(
        "Declared policies (%d): %s",
        len(declared), ", ".join(policy.name for policy in declared),
    )
    if policy_names:
        logger.warning(
            "Execution scope restricted to %d of %d policies: %s. This is a re-run "
            "convenience only; the experiment definition is unchanged.",
            len(policies), len(declared), ", ".join(policy.name for policy in policies),
        )

    records: list[PolicyRunRecord] = []
    for index, policy in enumerate(policies, start=1):
        logger.info(
            "=== policy %d/%d: %s === %s",
            index, len(policies), policy.name, policy.summary_line(),
        )
        with additional_log_file(config, stage=f"train_{policy.name}"):
            records.append(
                _train_one_policy(
                    config, policy, metadata, split, train_ids, val_ids, device, master_seed
                )
            )

    _write_runs(config, records)
    _write_manifest(config, declared, policies, records, split, device, master_seed,
                    len(train_ids), len(val_ids))
    _log_records(records)
    return records


# ---------------------------------------------------------------------------
# One policy
# ---------------------------------------------------------------------------


def _train_one_policy(
    config: Config,
    policy: AugmentationPolicy,
    metadata: CubMetadata,
    split: InternalSplit,
    train_ids: tuple[int, ...],
    val_ids: tuple[int, ...],
    device: torch.device,
    master_seed: int,
) -> PolicyRunRecord:
    """Train exactly one policy from a fresh pretrained initialisation.

    Seeding happens in two deliberate steps.

    1. The *master* seed is installed while the model is built, so every arm
       starts from a bit-identical initialisation: the same ImageNet backbone and
       the same freshly drawn classification head. Initialisation is a control
       variable, not something that should differ between arms.
    2. The *policy* seed takes over for everything stochastic thereafter -
       shuffling order and the augmentation draws - so each arm has an
       independent RNG stream and adding or renaming a policy cannot perturb
       another one's randomness.
    """
    pipeline = build_policy_train_transform(config, policy)
    described = describe_policy_pipeline(pipeline)

    logger.info("policy       : %s", policy.name)
    logger.info("description  : %s", policy.description)
    logger.info("audit link   : %s", policy.audit_transform or "(reference arm)")
    logger.info("master seed  : %d -> resolved policy seed %d", master_seed, policy.seed)
    logger.info("device       : %s", device)
    logger.info("split        : fingerprint %s (seed %d)", split.fingerprint, split.seed)
    logger.info("train / val  : %d / %d images", len(train_ids), len(val_ids))
    logger.info("architecture : %s (weights %s, pretrained=%s)",
                config.get("model.architecture"), config.get("model.weights", None),
                config.get("model.pretrained"))
    logger.info("optimizer    : %s lr=%s weight_decay=%s",
                config.get("train.optimizer.name"), config.get("train.optimizer.lr"),
                config.get("train.optimizer.weight_decay"))
    logger.info("scheduler    : %s (warmup %s epochs, min_lr %s)",
                config.get("train.scheduler.name"), config.get("train.scheduler.warmup_epochs"),
                config.get("train.scheduler.min_lr"))
    logger.info("budget       : %s epochs, batch_size %s, monitor %s (%s)",
                config.get("train.epochs"), config.get("train.batch_size"),
                config.get("train.checkpoint.monitor"), config.get("train.checkpoint.mode"))
    logger.info("pipeline     : %s", json.dumps(described, default=str))

    artifacts = _artifacts_for(config, policy)
    scoped_split = InternalSplit(
        train_ids=train_ids,
        val_ids=val_ids,
        seed=split.seed,
        val_fraction=split.val_fraction,
        stratified=split.stratified,
        fingerprint=split.fingerprint,
        num_classes=split.num_classes,
        created_at=split.created_at,
    )

    logger.info(
        "Seeding model initialisation from the master seed %d so every policy starts "
        "identically.", master_seed,
    )
    seed_run(master_seed)
    trainer = BaselineTrainer(
        config, metadata, scoped_split, device, policy.seed,
        run_name=policy.name,
        train_pipeline=pipeline,
        artifacts=artifacts,
        extra_metadata={
            "experiment": str(config.get("experiment.name")),
            "policy": policy.describe(),
            "augmentation_pipeline": described,
            "master_seed": master_seed,
        },
    )
    logger.info(
        "Handing the RNG to this policy's own stream (seed %d) for shuffling and "
        "augmentation sampling.", policy.seed,
    )
    seed_run(policy.seed)
    checkpoint_path = trainer.fit()

    best = trainer.best_metrics
    if best is None:
        raise RuntimeError(
            f"Policy '{policy.name}' finished without selecting a checkpoint; "
            f"train.epochs is {config.get('train.epochs')}."
        )
    final = trainer.history[-1]
    return PolicyRunRecord(
        policy=policy.name,
        audit_transform=policy.audit_transform or "",
        is_baseline=policy.is_baseline,
        description=policy.description,
        seed=policy.seed,
        split_fingerprint=split.fingerprint,
        num_train_images=len(train_ids),
        num_val_images=len(val_ids),
        epochs=int(config.get("train.epochs")),
        batch_size=int(config.get("train.batch_size")),
        optimizer=str(config.get("train.optimizer.name")),
        learning_rate=float(config.get("train.optimizer.lr")),
        weight_decay=float(config.get("train.optimizer.weight_decay")),
        scheduler=str(config.get("train.scheduler.name")),
        architecture=str(config.get("model.architecture")),
        pretrained_weights=str(config.get("model.weights", None)),
        device=str(device),
        application_probability=policy.application_probability,
        best_epoch=best.epoch,
        best_val_top1=best.val_top1,
        best_val_top5=best.val_top5,
        best_val_loss=best.val_loss,
        best_train_top1=best.train_top1,
        best_train_top5=best.train_top5,
        best_train_loss=best.train_loss,
        final_val_top1=final.val_top1,
        final_val_top5=final.val_top5,
        final_val_loss=final.val_loss,
        final_train_top1=final.train_top1,
        final_train_top5=final.train_top5,
        final_train_loss=final.train_loss,
        total_seconds=sum(metrics.seconds for metrics in trainer.history),
        checkpoint_path=str(checkpoint_path),
        history_path=str(artifacts.history_path),
        completed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def _artifacts_for(config: Config, policy: AugmentationPolicy) -> TrainingArtifacts:
    """Per-policy checkpoint directory and history file; arms never collide.

    The checkpoint root and file names come from the existing
    ``train.checkpoint`` section - the same keys Experiment 0A uses - with one
    sub-directory per policy appended.
    """
    return TrainingArtifacts(
        checkpoint_dir=config.path("train.checkpoint.dir") / policy.name,
        best_name=str(config.get("train.checkpoint.best_name")),
        last_name=str(config.get("train.checkpoint.last_name")),
        history_path=config.path("policies.output.history_dir") / f"{policy.name}.csv",
    )


# ---------------------------------------------------------------------------
# Optional subsetting (smoke runs only)
# ---------------------------------------------------------------------------


def _apply_subset(
    config: Config, split: InternalSplit, master_seed: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Optionally cap the number of train/validation images.

    ``train.max_train_images`` / ``train.max_val_images`` exist so the smoke test
    can exercise the whole pipeline in minutes. They must be ``null`` for a
    scientific run, and the subset is drawn from the *master* seed - never from a
    policy seed - so every arm trains and validates on exactly the same images.
    """
    train_ids = _subsample(split.train_ids, config.get("train.max_train_images", None),
                           master_seed, "train")
    val_ids = _subsample(split.val_ids, config.get("train.max_val_images", None),
                         master_seed, "val")
    if len(train_ids) != split.num_train or len(val_ids) != split.num_val:
        logger.warning(
            "SUBSET ACTIVE: training on %d/%d and validating on %d/%d images. This is a "
            "plumbing configuration, not a scientific run.",
            len(train_ids), split.num_train, len(val_ids), split.num_val,
        )
    return train_ids, val_ids


def _subsample(ids: tuple[int, ...], limit: Any, master_seed: int, kind: str) -> tuple[int, ...]:
    if limit is None:
        return tuple(ids)
    limit = int(limit)
    if limit <= 0:
        raise ConfigError(f"train.max_{kind}_images must be a positive integer or null.")
    if limit >= len(ids):
        return tuple(ids)
    rng = Random(derive_run_seed(master_seed, "subset", kind))
    return tuple(sorted(rng.sample(sorted(ids), limit)))


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def _write_runs(config: Config, records: list[PolicyRunRecord]) -> Path:
    """Merge these runs into ``train_runs.csv``, preserving policies not re-run."""
    path = config.path("policies.output.runs_path")
    path.parent.mkdir(parents=True, exist_ok=True)

    merged: dict[str, dict[str, Any]] = {}
    if path.is_file():
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("policy"):
                    merged[row["policy"]] = {key: row.get(key, "") for key in RUN_COLUMNS}
    for record in records:
        merged[record.policy] = {key: value for key, value in asdict(record).items()}

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RUN_COLUMNS))
        writer.writeheader()
        for row in merged.values():
            writer.writerow(row)
    logger.info("Wrote %d policy run row(s) to %s", len(merged), path)
    return path


def _write_manifest(
    config: Config,
    declared: list[AugmentationPolicy],
    trained: list[AugmentationPolicy],
    records: list[PolicyRunRecord],
    split: InternalSplit,
    device: torch.device,
    master_seed: int,
    num_train: int,
    num_val: int,
) -> Path:
    """Persist the resolved configuration and the resolved policy definitions."""
    path = config.path("policies.output.manifest_path")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": config.get("experiment.name"),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "master_seed": master_seed,
        "device": str(device),
        "torch_version": torch.__version__,
        "split": {
            "path": str(config.path("split.path")),
            "fingerprint": split.fingerprint,
            "seed": split.seed,
            "val_fraction": split.val_fraction,
            "stratified": split.stratified,
            "num_train_declared": split.num_train,
            "num_val_declared": split.num_val,
            "num_train_used": num_train,
            "num_val_used": num_val,
        },
        "application_probability": float(config.get("policies.application_probability")),
        "baseline_policy": str(config.get("policies.baseline_name")),
        "policies_declared": [policy.describe() for policy in declared],
        "policies_trained_this_run": [policy.name for policy in trained],
        "runs": [asdict(record) for record in records],
        "config": config.as_dict(),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    logger.info("Wrote resolved configuration manifest to %s", path)
    return path


def _log_records(records: list[PolicyRunRecord]) -> None:
    logger.info("Policy sweep complete (%d run(s)):", len(records))
    for record in records:
        logger.info(
            "  %-28s seed %-12d best epoch %2d | val top1 %.4f top5 %.4f loss %.4f",
            record.policy, record.seed, record.best_epoch + 1, record.best_val_top1,
            record.best_val_top5, record.best_val_loss,
        )
