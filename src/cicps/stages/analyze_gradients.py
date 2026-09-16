"""Stage: Experiment 1A analysis - summaries, relationships, joins and figures.

Reads only what the measurement stage wrote, plus two read-only references:
Experiment 0A's audit dataframe and Experiment 0B's run table. It retrains
nothing and re-measures nothing, so it can be re-run freely.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..analysis.gradient_plots import render_plots
from ..analysis.gradients import (
    describe_association,
    join_with_0a,
    join_with_0b,
    relate_to_delta_margin,
    summarize_gradients,
    summarize_null_control,
)
from ..audit.records import table_path, write_table
from ..config import Config, ConfigError
from ..gradients.checkpoints import build_stages
from ..gradients.records import validate_records
from ..logging_utils import get_logger, setup_logging
from ..transforms.base import build_transform_suite

__all__ = ["run_analyze_gradients", "GradientAnalysisPaths"]

logger = get_logger(__name__)


@dataclass(frozen=True)
class GradientAnalysisPaths:
    """Artefacts written by the Experiment 1A analysis stage."""

    summary: Path
    null_summary: Path
    relationships: Path
    joined_0b: Path | None
    association: Path | None
    plots: list[Path]


def run_analyze_gradients(config: Config) -> GradientAnalysisPaths:
    """Summarise the gradient records and render the Experiment 1A figures."""
    setup_logging(config, stage="analyze")
    logger.info("Configuration: %s", config.source)

    records = _read(_records_path(config, "records_name"), "gradient records")
    clean = _read(_records_path(config, "clean_name"), "clean reference")
    validate_records(records)
    logger.info("Loaded %d gradient row(s) and %d clean row(s).", len(records), len(clean))

    transforms = build_transform_suite(config)
    transform_order = [transform.name for transform in transforms]
    stages = build_stages(config, set(transform_order))
    checkpoint_order = [stage.name for stage in stages
                        if stage.name in set(records["checkpoint"].unique())]
    control = str(config.get("analysis.control_transform"))
    anchor = str(config.get("analysis.anchor_checkpoint"))
    if anchor not in checkpoint_order:
        raise ConfigError(
            f"analysis.anchor_checkpoint is '{anchor}' but that stage is not present in the "
            f"gradient records. Measured stages: {', '.join(checkpoint_order)}."
        )
    scopes = list(config.get("gradients.scopes").keys())
    seed = int(config.get("seed.value"))
    resamples = int(config.get("analysis.bootstrap.resamples"))
    confidence = float(config.get("analysis.bootstrap.confidence"))

    _check_control_arm(records, control, config)

    summary = summarize_gradients(
        records, quantiles=list(config.get("analysis.quantiles")), scopes=scopes,
        checkpoint_order=checkpoint_order, transform_order=transform_order,
        bootstrap_resamples=resamples, confidence=confidence, seed=seed,
    )
    null_summary = summarize_null_control(
        clean, checkpoint_order=checkpoint_order, bootstrap_resamples=resamples,
        confidence=confidence, seed=seed,
    )
    relationships = relate_to_delta_margin(
        records, checkpoint_order=checkpoint_order, transform_order=transform_order,
        control_transform=control,
    )

    output_dir = config.path("analysis.output.dir")
    fmt = str(config.get("analysis.output.format"))
    summary_path = table_path(output_dir, str(config.get("analysis.output.summary_name")), fmt)
    null_path = table_path(output_dir, str(config.get("analysis.output.null_name")), fmt)
    relationship_path = table_path(
        output_dir, str(config.get("analysis.output.relationships_name")), fmt)
    write_table(summary, summary_path, fmt=fmt)
    write_table(null_summary, null_path, fmt=fmt)
    write_table(relationships, relationship_path, fmt=fmt)

    joined_0a: pd.DataFrame | None = None
    if bool(config.get("analysis.join_0a.enabled")):
        joined_0a = join_with_0a(records, config, anchor)
        _verify_0a_reproduction(joined_0a, config)
        write_table(
            joined_0a,
            table_path(output_dir, str(config.get("analysis.join_0a.output_name")), fmt),
            fmt=fmt,
        )

    joined_0b: Path | None = None
    joined_0b_frame: pd.DataFrame | None = None
    association_path: Path | None = None
    if bool(config.get("analysis.join_0b.enabled")):
        joined_0b_frame = join_with_0b(summary, config, anchor)
        joined_0b = table_path(output_dir, str(config.get("analysis.join_0b.output_name")), fmt)
        write_table(joined_0b_frame, joined_0b, fmt=fmt)

        association = describe_association(
            joined_0b_frame,
            pairs=[
                ("grad_cosine_mean", "val_top1_delta_0b"),
                ("grad_norm_ratio_mean", "val_top1_delta_0b"),
                ("delta_margin_mean", "val_top1_delta_0b"),
                ("grad_cosine_mean", "delta_margin_mean"),
            ],
            methods=[str(method) for method in config.get("analysis.association.methods")],
            context=f"transformation-level, checkpoint '{anchor}'",
        )
        association_path = table_path(
            output_dir, str(config.get("analysis.association.output_name")), fmt)
        write_table(association, association_path, fmt=fmt)
        _log_association(association)

    plots = render_plots(
        records, summary, null_summary, joined_0a, joined_0b_frame, config,
        transform_order=transform_order, checkpoint_order=checkpoint_order,
        control_transform=control, anchor_checkpoint=anchor, scopes=scopes,
    )

    _log_summary(summary, null_summary, anchor)
    logger.info(
        "Analysis complete. Descriptive statistics only. Gradient alignment is evidence about "
        "learning-signal compatibility, not proof that an augmentation is good or bad, and any "
        "transformation-level coefficient here rests on five or six points."
    )
    return GradientAnalysisPaths(
        summary=summary_path,
        null_summary=null_path,
        relationships=relationship_path,
        joined_0b=joined_0b,
        association=association_path,
        plots=plots,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _records_path(config: Config, key: str) -> Path:
    return table_path(
        config.path("gradients.output.dir"),
        str(config.get(f"gradients.output.{key}")),
        str(config.get("gradients.output.format")),
    )


def _read(path: Path, what: str) -> pd.DataFrame:
    if not path.is_file():
        raise ConfigError(
            f"Experiment 1A {what} not found: {path}. Run the `gradients` stage first "
            "(`python -m cicps gradients --config config/experiment_1a.yaml`)."
        )
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def _check_control_arm(records: pd.DataFrame, control: str, config: Config) -> None:
    """The control arm is the correctness check on the whole gradient path.

    ``identity`` transforms nothing, so ``g_T`` must equal ``g_x`` and the cosine
    and norm ratio must both be 1. Any deviation beyond float64 round-off means
    the measurement path is non-deterministic and every other number is suspect -
    the same role ``identity``'s exactly-zero ``delta_margin`` plays in 0A.
    """
    arm = records[records["transform"] == control]
    if arm.empty:
        logger.warning(
            "No '%s' control rows found; the gradient path is unverified for this run.", control
        )
        return
    tolerance = float(config.get("analysis.control_tolerance"))
    cosine_error = (arm["grad_cosine"] - 1.0).abs().max()
    ratio_error = (arm["grad_norm_ratio"] - 1.0).abs().max()
    margin_error = arm["delta_margin"].abs().max()
    logger.info(
        "Control arm '%s' (%d rows): max |C_g - 1| = %.3e, max |R_g - 1| = %.3e, "
        "max |dM| = %.3e.", control, len(arm), cosine_error, ratio_error, margin_error,
    )
    if cosine_error > tolerance or ratio_error > tolerance:
        raise RuntimeError(
            f"The '{control}' control arm deviates from 1.0 by more than "
            f"{tolerance:g} (cosine {cosine_error:.3e}, norm ratio {ratio_error:.3e}). The "
            "gradient measurement path is not deterministic; do not report these numbers."
        )


def _verify_0a_reproduction(joined: pd.DataFrame, config: Config) -> None:
    """Check that the anchor stage reproduces Experiment 0A's per-image margins.

    Realisation 0 applies exactly 0A's frozen draw to exactly 0A's image at
    exactly the checkpoint 0A audited, so ``delta_margin`` must agree with 0A's
    to within floating-point reproduction error. A disagreement means the two
    experiments are not measuring the same perturbation, which would invalidate
    every comparison between them.
    """
    tolerance = float(config.get("analysis.join_0a.tolerance"))
    difference = (joined["delta_margin"] - joined["delta_margin_0a"]).abs()
    competitor_mismatch = int((joined["hard_competitor"] != joined["hard_competitor_0a"]).sum())
    logger.info(
        "Experiment 0A reproduction at the anchor stage: max |dM(1A) - dM(0A)| = %.3e, "
        "mean %.3e, hardest-competitor disagreements %d / %d.",
        difference.max(), difference.mean(), competitor_mismatch, len(joined),
    )
    if difference.max() > tolerance:
        logger.warning(
            "Experiment 1A's delta_margin differs from Experiment 0A's by up to %.3e, above the "
            "configured tolerance of %g. The two are then not measuring the same perturbation; "
            "inspect the checkpoint, the preprocessing and the transformation parameters before "
            "reading the joined analysis.",
            difference.max(), tolerance,
        )


def _log_summary(summary: pd.DataFrame, null_summary: pd.DataFrame, anchor: str) -> None:
    cell = summary[summary["checkpoint"] == anchor]
    logger.info("Experiment 1A summary at the anchor checkpoint '%s':", anchor)
    for _, row in cell.iterrows():
        logger.info(
            "  %-22s C_g %+.4f [%+.4f, %+.4f] | median %+.4f | frac C_g<0 %.3f | "
            "R_g %.3f | dM %+.4f",
            row["transform"], row["grad_cosine_mean"], row["grad_cosine_ci_low"],
            row["grad_cosine_ci_high"], row["grad_cosine_median"],
            row["frac_cosine_negative"], row["grad_norm_ratio_mean"], row["delta_margin_mean"],
        )
    for _, row in null_summary.iterrows():
        logger.info(
            "  cross-image null @ %-18s mean %+.4f [%+.4f, %+.4f] over %d pair(s)",
            row["checkpoint"], row["null_cosine_mean"], row["null_cosine_ci_low"],
            row["null_cosine_ci_high"], row["n_pairs"],
        )


def _log_association(association: pd.DataFrame) -> None:
    for _, row in association.iterrows():
        logger.info(
            "  association %-28s vs %-22s %-8s %+.3f (n = %d) - %s",
            row["x"], row["y"], row["method"], row["coefficient"], row["n_points"], row["note"],
        )
