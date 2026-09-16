"""Invariant assertions for a completed Experiment 1A run.

Configuration driven, so the same code validates a smoke run and a scientific
run. It reports *every* failure rather than stopping at the first, and exits
non-zero if any check fails.

    python scripts/check_experiment_1a.py --config config/experiment_1a.yaml
    python scripts/check_experiment_1a.py --config config/experiment_1a.yaml \\
        --override config/smoke_1a.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from random import Random

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cicps.config import Config, load_config, project_root  # noqa: E402
from cicps.data.cub import load_cub_metadata  # noqa: E402
from cicps.data.splits import resolve_split  # noqa: E402
from cicps.gradients.checkpoints import build_stages  # noqa: E402
from cicps.gradients.records import REQUIRED_RECORD_COLUMNS  # noqa: E402
from cicps.audit.records import encode_params  # noqa: E402
from cicps.gradients.datasets import realisation_seed  # noqa: E402
from cicps.transforms.base import build_transform_suite  # noqa: E402


class Checker:
    """Accumulates pass/fail results so one run reports every problem."""

    def __init__(self) -> None:
        self.passed = 0
        self.failures: list[str] = []

    def check(self, condition: bool, description: str) -> bool:
        if condition:
            self.passed += 1
            print(f"  PASS  {description}")
        else:
            self.failures.append(description)
            print(f"  FAIL  {description}")
        return bool(condition)

    def report(self) -> int:
        total = self.passed + len(self.failures)
        print(f"\n{self.passed}/{total} checks passed.")
        if self.failures:
            print("\nFailures:")
            for failure in self.failures:
                print(f"  - {failure}")
            return 1
        return 0


def _table(directory: Path, stem: str, fmt: str) -> Path:
    return directory / f"{stem}.{'csv' if fmt == 'csv' else 'parquet'}"


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/experiment_1a.yaml"))
    parser.add_argument("--override", type=Path, action="append", default=[])
    args = parser.parse_args(argv)

    path = args.config if args.config.is_absolute() else project_root() / args.config
    config: Config = load_config(path, overrides=args.override)
    checker = Checker()

    gradients_dir = config.path("gradients.output.dir")
    analysis_dir = config.path("analysis.output.dir")
    fmt = str(config.get("gradients.output.format"))
    analysis_fmt = str(config.get("analysis.output.format"))

    records_path = _table(gradients_dir, str(config.get("gradients.output.records_name")), fmt)
    clean_path = _table(gradients_dir, str(config.get("gradients.output.clean_name")), fmt)
    manifest_path = gradients_dir / str(config.get("gradients.output.manifest_name"))

    print("== artefacts ==")
    if not checker.check(records_path.is_file(), f"gradient records exist ({records_path.name})"):
        return checker.report()
    checker.check(clean_path.is_file(), f"clean reference exists ({clean_path.name})")
    checker.check(manifest_path.is_file(), f"run manifest exists ({manifest_path.name})")

    records = _read(records_path)
    clean = _read(clean_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    print("\n== schema ==")
    missing = [c for c in REQUIRED_RECORD_COLUMNS if c not in records.columns]
    checker.check(not missing, f"every mandated column present (missing: {missing})")
    scopes = list(config.get("gradients.scopes").keys())
    for scope in scopes:
        checker.check(f"cosine_{scope}" in records.columns, f"per-scope column cosine_{scope}")

    print("\n== the measurement is frozen and paired ==")
    checker.check(
        manifest.get("gradient_mode") == "eval",
        "gradients were measured in eval() mode (frozen BatchNorm statistics)",
    )
    checker.check(
        manifest.get("precision", {}).get("dtype") == "float32"
        and manifest.get("precision", {}).get("autocast") is False,
        "float32 with autocast disabled",
    )
    checker.check(
        manifest.get("precision", {}).get("reduction_dtype") == "float64",
        "inner products accumulated in float64",
    )

    stages = build_stages(config, {t.name for t in build_transform_suite(config)})
    declared = [stage.name for stage in stages]
    measured = list(records["checkpoint"].unique())
    checker.check(
        set(measured).issubset(set(declared)),
        f"every measured stage was declared in the configuration ({measured})",
    )
    checker.check(len(measured) >= 1, f"at least one checkpoint stage was measured ({len(measured)})")

    for stage in measured:
        stage_rows = records[records["checkpoint"] == stage]
        per_image = stage_rows.groupby("image_id").size().unique()
        checker.check(
            len(per_image) == 1,
            f"stage '{stage}': every image has the same number of rows ({per_image})",
        )
        competitors = stage_rows.groupby("image_id")["hard_competitor"].nunique().max()
        checker.check(
            competitors == 1,
            f"stage '{stage}': the hardest competitor is frozen per image (max distinct "
            f"= {competitors})",
        )
        clean_margins = stage_rows.groupby("image_id")["clean_margin"].nunique().max()
        checker.check(
            clean_margins == 1,
            f"stage '{stage}': one clean margin per image, shared across arms "
            f"(max distinct = {clean_margins})",
        )
        norms = stage_rows.groupby("image_id")["clean_grad_norm"].nunique().max()
        checker.check(
            norms == 1,
            f"stage '{stage}': the clean gradient is computed once per image "
            f"(max distinct norm = {norms})",
        )

    print("\n== the control arm ==")
    control = str(config.get("analysis.control_transform"))
    tolerance = float(config.get("analysis.control_tolerance"))
    arm = records[records["transform"] == control]
    checker.check(not arm.empty, f"the '{control}' control arm was measured")
    if not arm.empty:
        checker.check(
            float((arm["grad_cosine"] - 1.0).abs().max()) <= tolerance,
            f"control arm cosine == 1 within {tolerance:g} "
            f"(max deviation {float((arm['grad_cosine'] - 1.0).abs().max()):.3e})",
        )
        checker.check(
            float((arm["grad_norm_ratio"] - 1.0).abs().max()) <= tolerance,
            f"control arm norm ratio == 1 within {tolerance:g} "
            f"(max deviation {float((arm['grad_norm_ratio'] - 1.0).abs().max()):.3e})",
        )
        checker.check(
            float(arm["delta_margin"].abs().max()) == 0.0,
            "control arm delta_margin is exactly 0",
        )
        checker.check(
            bool(arm["prediction_consistent"].astype(bool).all()),
            "control arm prediction consistency is 100%",
        )

    print("\n== numerical sanity ==")
    cosine = records["grad_cosine"].to_numpy(dtype=np.float64)
    finite = cosine[np.isfinite(cosine)]
    checker.check(
        bool((finite >= -1.0 - 1e-9).all() and (finite <= 1.0 + 1e-9).all()),
        f"every cosine lies in [-1, 1] (range {finite.min():.4f} .. {finite.max():.4f})",
    )
    ratios = records["grad_norm_ratio"].to_numpy(dtype=np.float64)
    checker.check(
        bool((ratios[np.isfinite(ratios)] >= 0).all()), "every norm ratio is non-negative"
    )
    non_finite = int((~np.isfinite(cosine)).sum())
    degenerate = int(records["grad_degenerate"].astype(bool).sum())
    checker.check(
        non_finite == degenerate,
        f"every non-finite cosine is flagged degenerate ({non_finite} non-finite, "
        f"{degenerate} flagged)",
    )
    for column in ("clean_loss", "transformed_loss", "clean_grad_norm", "transformed_grad_norm"):
        checker.check(
            bool(np.isfinite(records[column].to_numpy(dtype=np.float64)).all()),
            f"'{column}' is finite on every row",
        )

    print("\n== the official test set is locked ==")
    metadata = load_cub_metadata(config)
    split = resolve_split(metadata, config)
    test_ids = {image.image_id for image in metadata.official_test}
    measured_ids = set(int(i) for i in records["image_id"].unique())
    checker.check(
        not (measured_ids & test_ids),
        f"no official test image was measured ({len(test_ids)} test images untouched)",
    )
    subset = str(config.get("gradients.split_subset"))
    expected = set(split.val_ids) if subset == "val" else set(split.train_ids)
    checker.check(
        measured_ids.issubset(expected),
        f"every measured image comes from the internal '{subset}' subset",
    )
    checker.check(
        manifest.get("split_fingerprint") == split.fingerprint,
        f"the run used the shared split fingerprint {split.fingerprint}",
    )

    print("\n== sampling protocol ==")
    expected_seed = config.seeded("gradients.sampling.param_seed")
    checker.check(
        int(manifest.get("transform_param_seed", -1)) == int(expected_seed),
        f"the run used transformation parameter seed {expected_seed}",
    )
    # Realisation 0 must re-derive Experiment 0A's exact frozen draw, otherwise
    # the per-image join compares two different perturbations.
    suite = {t.name: t for t in build_transform_suite(config)}
    sample = records[(records["realisation"] == 0) & (records["transform"] != "identity")]
    mismatches = 0
    for _, row in sample.head(200).iterrows():
        transform = suite[str(row["transform"])]
        expected = encode_params(
            transform.resolve(Random(realisation_seed(
                expected_seed, int(row["image_id"]), transform.name, 0)))
        )
        mismatches += int(expected != str(row["transform_params"]))
    checker.check(
        mismatches == 0,
        f"realisation-0 parameters re-derive exactly (checked {min(len(sample), 200)} rows, "
        f"{mismatches} mismatch)",
    )
    checker.check(
        0 in set(records["realisation"].unique()),
        "realisation 0 (Experiment 0A's frozen draw) was measured",
    )

    print("\n== analysis artefacts ==")
    summary_path = _table(
        analysis_dir, str(config.get("analysis.output.summary_name")), analysis_fmt)
    null_path = _table(analysis_dir, str(config.get("analysis.output.null_name")), analysis_fmt)
    if checker.check(summary_path.is_file(), "gradient summary exists"):
        summary = _read(summary_path)
        checker.check(
            set(summary["checkpoint"].unique()) == set(measured),
            "the summary covers every measured stage",
        )
        checker.check(
            bool((summary["grad_cosine_ci_low"] <= summary["grad_cosine_mean"]).all()
                 and (summary["grad_cosine_mean"] <= summary["grad_cosine_ci_high"]).all()),
            "every bootstrap interval brackets its mean",
        )
    if checker.check(null_path.is_file(), "null-control summary exists"):
        nulls = _read(null_path)
        checker.check(not nulls.empty, "the cross-image null control produced rows")

    for name in ("relationships_name",):
        path = _table(analysis_dir, str(config.get(f"analysis.output.{name}")), analysis_fmt)
        checker.check(path.is_file(), f"{path.name} exists")

    if bool(config.get("analysis.join_0a.enabled")):
        path = _table(analysis_dir, str(config.get("analysis.join_0a.output_name")), analysis_fmt)
        if checker.check(path.is_file(), "the Experiment 0A per-sample join exists"):
            joined = _read(path)
            difference = (joined["delta_margin"] - joined["delta_margin_0a"]).abs()
            limit = float(config.get("analysis.join_0a.tolerance"))
            checker.check(
                float(difference.max()) <= limit,
                f"1A reproduces 0A's per-image delta_margin within {limit:g} "
                f"(max {float(difference.max()):.3e})",
            )
            checker.check(
                int((joined["hard_competitor"] != joined["hard_competitor_0a"]).sum()) == 0,
                "the frozen hardest competitor agrees with Experiment 0A on every joined row",
            )

    if bool(config.get("analysis.join_0b.enabled")):
        path = _table(analysis_dir, str(config.get("analysis.join_0b.output_name")), analysis_fmt)
        if checker.check(path.is_file(), "the Experiment 0B downstream join exists"):
            joined = _read(path)
            checker.check(
                len(joined) >= 1 and "val_top1_delta_0b" in joined.columns,
                f"the 0B join carries the baseline-relative accuracy delta ({len(joined)} rows)",
            )

    print("\n== figures ==")
    plots_dir = config.path("analysis.plots.dir")
    suffix = str(config.get("analysis.plots.file_format"))
    for kind in config.get("analysis.plots.kinds"):
        figure = plots_dir / f"{kind}.{suffix}"
        checker.check(
            figure.is_file() and figure.stat().st_size > 0, f"{figure.name} exists and is non-empty"
        )

    print("\n== traceability ==")
    for key in ("created_at", "seed", "split_fingerprint", "scopes", "checkpoints",
                "transformations", "precision", "torch_version", "config"):
        checker.check(key in manifest, f"the manifest records '{key}'")

    return checker.report()


if __name__ == "__main__":
    raise SystemExit(main())
