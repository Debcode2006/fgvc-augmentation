"""Stage: the Experiment 1A gradient-compatibility measurement.

Loads checkpoints written by Experiments 0A and 0B read-only, measures paired
clean/transformed gradients over the internal validation split, and persists the
per-sample records, the per-image clean reference and a run manifest.

Nothing here trains, and nothing here writes outside ``gradients.output.dir``.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

from ..audit.records import table_path, write_table
from ..config import Config, ConfigError
from ..data.cub import load_cub_metadata
from ..data.splits import resolve_split
from ..data.validation import validate_split
from ..device import resolve_device
from ..gradients.checkpoints import build_stages
from ..gradients.engine import GradientEngine
from ..logging_utils import get_logger, setup_logging
from ..seeding import seed_everything
from ..transforms.base import build_transform_suite

__all__ = ["run_gradients", "GradientPaths"]

logger = get_logger(__name__)


@dataclass(frozen=True)
class GradientPaths:
    """Artefacts written by the Experiment 1A measurement stage."""

    records: Path
    clean: Path
    manifest: Path


def run_gradients(config: Config, *, stage_names: list[str] | None = None) -> GradientPaths:
    """Run the Experiment 1A measurement and persist its dataframes."""
    setup_logging(config, stage="gradients")
    logger.info("Configuration: %s", config.source)
    logger.info("Experiment: %s", config.get("experiment.name"))

    seed = seed_everything(config)
    _apply_precision(config)
    device = resolve_device(config)

    metadata = load_cub_metadata(config)
    split = resolve_split(metadata, config)
    report = validate_split(split, metadata, config)
    if not report.ok:
        report.log()
        raise RuntimeError("Refusing to measure: the internal split failed validation.")
    logger.info(
        "Internal split %s: %d train / %d val images; official test portion locked "
        "(%d images untouched).",
        split.fingerprint, split.num_train, split.num_val, len(metadata.official_test),
    )

    transforms = build_transform_suite(config)
    stages = build_stages(config, {transform.name for transform in transforms})
    stages = _select_stages(stages, stage_names)
    logger.info(
        "Measuring %d checkpoint stage(s): %s",
        len(stages), ", ".join(stage.name for stage in stages),
    )
    logger.info(
        "Transformation suite (%d): %s",
        len(transforms), ", ".join(transform.name for transform in transforms),
    )

    subset = str(config.get("gradients.split_subset")).lower()
    if subset == "val":
        image_ids = split.val_ids
    elif subset == "train":
        logger.warning(
            "gradients.split_subset='train' measures the INTERNAL TRAINING images. A converged "
            "checkpoint has near-zero loss on those, so clean gradient norms collapse and the "
            "cosine becomes ill-conditioned. The default 'val' is also what Experiment 0A "
            "measured, which is what makes the two joinable."
        )
        image_ids = split.train_ids
    else:
        raise ConfigError(
            f"gradients.split_subset must be 'val' or 'train', got '{subset}'. The official "
            "CUB test set is locked and is never an option."
        )

    engine = GradientEngine(config, metadata, transforms, device)
    started = time.perf_counter()
    result = engine.run(image_ids, stages)
    elapsed = time.perf_counter() - started
    logger.info("Measurement complete in %.1f s (%d record rows).", elapsed, len(result.records))

    return _write_outputs(
        config, result, seed=seed, device=device, transforms=transforms,
        split_fingerprint=split.fingerprint, subset=subset, elapsed=elapsed,
        num_images=len(result.clean["image_id"].unique()),
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _apply_precision(config: Config) -> None:
    """Install the numerical-precision policy for the measurement.

    TF32 is disabled by default. It silently reduces matmul and convolution
    mantissas to 10 bits, which is fine for training and not fine for an
    instrument: it would put a floor on how exactly the ``identity`` control arm
    can reproduce a cosine of 1.0, and would make the reproduction of Experiment
    0A's margins hardware-dependent.
    """
    allow_tf32 = bool(config.get("gradients.precision.allow_tf32"))
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    dtype = str(config.get("gradients.precision.dtype")).lower()
    if dtype != "float32":
        raise ConfigError(
            f"gradients.precision.dtype must be 'float32', got '{dtype}'. Experiment 1A "
            "measures gradients; reduced precision would quantise the measurement."
        )
    logger.info("Precision: float32, autocast disabled, TF32 allowed = %s.", allow_tf32)


def _select_stages(stages: list, names: list[str] | None) -> list:
    """Restrict the run to ``names`` (execution scope only, as 0B's ``--policy`` is)."""
    if not names:
        return stages
    known = {stage.name: stage for stage in stages}
    unknown = [name for name in names if name not in known]
    if unknown:
        raise ConfigError(
            f"Unknown checkpoint stage name(s) {unknown}. Declared in the configuration: "
            f"{', '.join(known)}."
        )
    logger.warning(
        "Execution scope restricted to %d of %d stages: %s. This is a re-run convenience "
        "only; the experiment definition is unchanged.",
        len(names), len(stages), ", ".join(names),
    )
    return [known[name] for name in names]


def _git_commit() -> str | None:
    """Return the current git commit, or ``None`` outside a repository."""
    try:
        output = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - environment dependent
        return None
    return output.stdout.strip() or None


def _write_outputs(
    config: Config,
    result,
    *,
    seed: int,
    device: torch.device,
    transforms: list,
    split_fingerprint: str,
    subset: str,
    elapsed: float,
    num_images: int,
) -> GradientPaths:
    """Persist the records, the clean reference, and a full run manifest."""
    output_dir = config.path("gradients.output.dir")
    fmt = str(config.get("gradients.output.format"))
    precision = int(config.get("gradients.output.float_precision"))

    records_path = table_path(output_dir, str(config.get("gradients.output.records_name")), fmt)
    clean_path = table_path(output_dir, str(config.get("gradients.output.clean_name")), fmt)
    write_table(result.records, records_path, fmt=fmt, float_precision=precision)
    write_table(result.clean, clean_path, fmt=fmt, float_precision=precision)

    manifest_path = output_dir / str(config.get("gradients.output.manifest_name"))
    manifest = {
        "experiment": config.get("experiment.name"),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "seed": seed,
        "transform_param_seed": config.seeded("gradients.sampling.param_seed"),
        "realisations": int(config.get("gradients.sampling.realisations")),
        "sampling_protocol": (
            "Realisation 0 reuses Experiment 0A's frozen parameter key "
            "derive_seed(param_seed, image_id, transform_name) verbatim, so the transformed "
            "image is bit-identical to the one 0A measured. Realisations >= 1 extend the key "
            "and are independent draws."
        ),
        "split_subset": subset,
        "split_fingerprint": split_fingerprint,
        "num_images": num_images,
        "num_records": int(len(result.records)),
        "gradient_mode": "eval",
        "gradient_mode_rationale": (
            "model.eval(): frozen BatchNorm running statistics, so the clean and transformed "
            "passes differ only in pixels and a single-sample gradient is well defined."
        ),
        "precision": {
            "dtype": str(config.get("gradients.precision.dtype")),
            "allow_tf32": bool(config.get("gradients.precision.allow_tf32")),
            "autocast": False,
            "reduction_dtype": "float64",
        },
        "scopes": result.scopes,
        "checkpoints": [stage.describe() for stage in result.stages],
        "transformations": [transform.describe() for transform in transforms],
        "null_control": {
            "enabled": bool(config.get("gradients.null_control.enabled")),
            "definition": (
                "Cosine between this image's clean gradient and the previously processed "
                "image's clean gradient, under a seeded permutation of the traversal order. "
                "It calibrates what an unrelated learning signal scores at this scope."
            ),
        },
        "device": str(device),
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "cuda_device": (
            torch.cuda.get_device_name(0) if device.type == "cuda" else None
        ),
        "runtime_seconds": round(elapsed, 2),
        "warnings": list(result.warnings),
        "records_path": str(records_path),
        "clean_path": str(clean_path),
        "config": config.as_dict(),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, default=str)
    logger.info("Wrote gradient manifest to %s", manifest_path)

    return GradientPaths(records=records_path, clean=clean_path, manifest=manifest_path)
