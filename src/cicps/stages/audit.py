"""Stage: the Experiment 0A transformation audit.

Loads a previously trained checkpoint, so the audit can be re-run any number of
times without retraining.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

from ..audit.engine import AuditEngine, AuditResult
from ..audit.records import table_path, write_table
from ..config import Config
from ..data.cub import load_cub_metadata
from ..data.splits import resolve_split
from ..data.validation import validate_split
from ..device import resolve_device
from ..logging_utils import get_logger, setup_logging
from ..models.checkpoint import load_checkpoint, warn_on_config_drift
from ..models.factory import load_model_for_audit
from ..seeding import seed_everything
from ..transforms.base import build_transform_suite

__all__ = ["run_audit", "AuditPaths"]

logger = get_logger(__name__)

_DRIFT_KEYS = (
    "model.architecture",
    "model.num_classes",
    "preprocess.image_size",
    "preprocess.resize_size",
    "preprocess.interpolation",
    "preprocess.normalize.mean",
    "preprocess.normalize.std",
    "split.path",
    "split.val_fraction",
)


@dataclass(frozen=True)
class AuditPaths:
    """Artefacts written by the audit stage."""

    records: Path
    originals: Path
    manifest: Path


def run_audit(config: Config, *, checkpoint_path: Path | None = None) -> AuditPaths:
    """Run the transformation audit and persist its dataframes."""
    setup_logging(config, stage="audit")
    logger.info("Configuration: %s", config.source)
    seed = seed_everything(config)
    device = resolve_device(config)

    metadata = load_cub_metadata(config)
    split = resolve_split(metadata, config)
    report = validate_split(split, metadata, config)
    if not report.ok:
        report.log()
        raise RuntimeError("Refusing to audit: the internal split failed validation.")

    path = checkpoint_path or config.path("audit.checkpoint")
    checkpoint = load_checkpoint(path, map_location="cpu")
    warn_on_config_drift(checkpoint, config, _DRIFT_KEYS)
    _warn_on_split_drift(checkpoint, split)

    model = load_model_for_audit(config, checkpoint.model_state, device)
    transforms = build_transform_suite(config)
    logger.info(
        "Auditing %d transformation(s): %s",
        len(transforms), ", ".join(transform.name for transform in transforms),
    )

    engine = AuditEngine(config, metadata, model, transforms, device)
    result = engine.run(split.val_ids)

    return _write_outputs(config, result, checkpoint_path=path, seed=seed,
                          checkpoint_metrics=checkpoint.metrics, device=device,
                          transforms=transforms, split_fingerprint=split.fingerprint)


def _write_outputs(
    config: Config,
    result: AuditResult,
    *,
    checkpoint_path: Path,
    seed: int,
    checkpoint_metrics: dict[str, float],
    device: torch.device,
    transforms: list,
    split_fingerprint: str,
) -> AuditPaths:
    """Persist the records, the frozen originals, and a run manifest."""
    output_dir = config.path("audit.output.dir")
    fmt = str(config.get("audit.output.format"))
    precision = int(config.get("audit.output.float_precision"))

    records_path = table_path(output_dir, str(config.get("audit.output.records_name")), fmt)
    originals_path = table_path(output_dir, str(config.get("audit.output.originals_name")), fmt)
    write_table(result.records, records_path, fmt=fmt, float_precision=precision)
    write_table(result.originals, originals_path, fmt=fmt, float_precision=precision)

    expected_rows = len(result.image_ids) * len(result.transform_names)
    if len(result.records) != expected_rows:
        raise RuntimeError(
            f"Audit produced {len(result.records)} rows but expected {expected_rows} "
            f"({len(result.image_ids)} images x {len(result.transform_names)} transformations)."
        )

    manifest_path = output_dir / "audit_manifest.json"
    manifest = {
        "experiment": config.get("experiment.name"),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "transform_param_seed": config.seeded("audit.transform_param_seed"),
        "device": str(device),
        "torch_version": torch.__version__,
        "checkpoint": str(checkpoint_path),
        "checkpoint_metrics": checkpoint_metrics,
        "split_fingerprint": split_fingerprint,
        "num_images": len(result.image_ids),
        "num_transformations": len(result.transform_names),
        "num_records": int(len(result.records)),
        "transformations": [transform.describe() for transform in transforms],
        "records_path": str(records_path),
        "originals_path": str(originals_path),
        "config": config.as_dict(),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, default=str)
    logger.info("Wrote audit manifest to %s", manifest_path)

    return AuditPaths(records=records_path, originals=originals_path, manifest=manifest_path)


def _warn_on_split_drift(checkpoint, split) -> None:
    """Warn when the checkpoint was trained against a different split."""
    trained_fingerprint = checkpoint.metadata.get("split_fingerprint")
    if trained_fingerprint and trained_fingerprint != split.fingerprint:
        logger.warning(
            "The checkpoint was trained on split fingerprint %s but the current split is %s. "
            "The audited 'validation' images may have been seen during training.",
            trained_fingerprint, split.fingerprint,
        )
