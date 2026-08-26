"""Assert the invariants of an Experiment 0B run.

A pipeline that exits 0 has proved almost nothing: it may have trained one
policy twice, leaked an official test image, written a NaN metric, or silently
skipped a plot. This script checks the properties that actually matter, against
the artefacts a run left on disk, and exits non-zero with a list of failures.

It is configuration driven like every other entry point, so it validates a smoke
run and a scientific run with the same code:

    python scripts/check_smoke_0b.py --config config/experiment_0b.yaml \\
                                     --override config/smoke_0b.yaml

    python scripts/check_smoke_0b.py --config config/experiment_0b.yaml
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cicps.audit.records import table_path  # noqa: E402
from cicps.config import Config, load_config, project_root  # noqa: E402
from cicps.data.cub import load_cub_metadata  # noqa: E402
from cicps.data.splits import resolve_split  # noqa: E402
from cicps.stages.train_policies import _apply_subset  # noqa: E402
from cicps.training.policies import build_policies  # noqa: E402
from cicps.transforms.policy_pipeline import (  # noqa: E402
    build_policy_train_transform,
    describe_policy_pipeline,
)


class Checker:
    """Accumulates pass/fail results so every invariant is reported, not just the first."""

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.checks = 0

    def check(self, condition: bool, message: str) -> bool:
        self.checks += 1
        if condition:
            print(f"  ok    {message}")
        else:
            print(f"  FAIL  {message}")
            self.failures.append(message)
        return bool(condition)

    def section(self, title: str) -> None:
        print(f"\n{title}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("config/experiment_0b.yaml"))
    parser.add_argument("--override", type=Path, action="append", default=[])
    args = parser.parse_args(argv)

    config = load_config(_absolute(args.config), overrides=[_absolute(p) for p in args.override])
    print(f"Checking Experiment 0B artefacts for '{config.get('experiment.name')}'.")

    checker = Checker()
    policies = _check_policies(checker, config)
    _check_shared_initialisation(checker, config)
    runs = _check_runs(checker, config, policies)
    _check_no_test_leak(checker, config, runs)
    _check_checkpoints_and_histories(checker, config, policies)
    _check_analysis(checker, config, policies)

    print(f"\n{checker.checks - len(checker.failures)}/{checker.checks} checks passed.")
    if checker.failures:
        print("\nFAILED:")
        for failure in checker.failures:
            print(f"  - {failure}")
        return 1
    print("All Experiment 0B invariants hold.")
    return 0


# ---------------------------------------------------------------------------
# Policy construction
# ---------------------------------------------------------------------------


def _check_policies(checker: Checker, config: Config) -> list:
    checker.section("Policy construction")
    policies = build_policies(config)
    declared = [entry["name"] for entry in config.get("policies.list")]

    checker.check(len(policies) == len(declared),
                  f"all {len(declared)} declared policies were constructed")
    names = [policy.name for policy in policies]
    checker.check(len(set(names)) == len(names), "policy names are unique")
    checker.check(len(set(policy.seed for policy in policies)) == len(policies),
                  "every policy has a distinct derived seed")

    baseline_name = str(config.get("policies.baseline_name"))
    baseline = [policy for policy in policies if policy.name == baseline_name]
    checker.check(len(baseline) == 1 and baseline[0].is_baseline,
                  f"'{baseline_name}' is the unmodified reference arm")

    probability = float(config.get("policies.application_probability"))
    checker.check(
        all(policy.application_probability == probability for policy in policies),
        f"every policy shares one application probability ({probability})",
    )

    declared_by_name = {entry["name"]: entry for entry in config.get("policies.list")}
    for policy in policies:
        spec = declared_by_name[policy.name]
        described = describe_policy_pipeline(build_policy_train_transform(config, policy))
        if policy.is_baseline:
            checker.check(
                described["added_transform"] is None
                and described["crop"]["kind"] == "RandomResizedCrop",
                f"'{policy.name}': baseline recipe, no added transformation",
            )
            continue
        if spec.get("add"):
            added = described["added_transform"]
            checker.check(
                added is not None
                and added["type"] == spec["add"]["type"]
                and added["params"] == spec["add"].get("params", {})
                and added["application_probability"] == probability,
                f"'{policy.name}': adds {spec['add']['type']} with the declared parameters",
            )
        else:
            crop = described["crop"]
            expected = spec["override"]["random_resized_crop"]
            baseline_crop = config.get("train.augmentation.random_resized_crop")
            checker.check(
                crop["kind"] == "probabilistic_choice"
                and crop["probability"] == probability
                and crop["alternate"]["scale"] == [float(v) for v in expected["scale"]]
                and crop["alternate"]["ratio"] == [float(v) for v in expected["ratio"]]
                and crop["baseline"]["scale"] == [float(v) for v in baseline_crop["scale"]],
                f"'{policy.name}': substitutes the declared crop parameters "
                f"{expected['scale']} for the baseline's {baseline_crop['scale']} "
                f"with p={probability}",
            )
            checker.check(
                described["added_transform"] is None,
                f"'{policy.name}': overrides the baseline crop rather than stacking a second one",
            )
    return policies


def _check_shared_initialisation(checker: Checker, config: Config) -> None:
    """Every arm must start from the same weights, not just the same architecture.

    The training stage installs the master seed while the model is built, so the
    initialisation is a pure function of that seed. Building twice under it and
    comparing tensors verifies the mechanism without retraining anything.
    """
    import torch

    from cicps.models.factory import build_model
    from cicps.seeding import seed_run

    checker.section("Shared model initialisation")
    master_seed = int(config.get("seed.value"))
    states = []
    for _ in range(2):
        seed_run(master_seed)
        states.append({key: value.clone() for key, value in build_model(config).state_dict().items()})
    identical = states[0].keys() == states[1].keys() and all(
        torch.equal(states[0][key], states[1][key]) for key in states[0]
    )
    checker.check(
        identical,
        f"model initialisation is bit-identical under the master seed {master_seed}, so "
        "every policy starts from the same weights",
    )


# ---------------------------------------------------------------------------
# Training outcomes
# ---------------------------------------------------------------------------


def _check_runs(checker: Checker, config: Config, policies: list) -> pd.DataFrame:
    checker.section("Training runs (train_runs.csv)")
    path = config.path("policies.output.runs_path")
    if not checker.check(path.is_file(), f"{path} exists"):
        return pd.DataFrame()

    runs = pd.read_csv(path)
    names = [policy.name for policy in policies]
    checker.check(sorted(runs["policy"]) == sorted(names),
                  "every policy has exactly one run row")
    checker.check((runs["num_train_images"] > 0).all(), "training dataset is non-empty")
    checker.check((runs["num_val_images"] > 0).all(), "validation dataset is non-empty")
    checker.check(runs["split_fingerprint"].nunique() == 1,
                  "baseline and modified policies share one split fingerprint")

    for column in ("num_train_images", "num_val_images", "epochs", "batch_size",
                   "learning_rate", "weight_decay", "optimizer", "scheduler",
                   "architecture", "pretrained_weights", "application_probability"):
        checker.check(runs[column].astype(str).nunique() == 1,
                      f"every policy trained with the same {column}")

    for column in ("best_val_top1", "best_val_top5", "best_val_loss",
                   "final_train_loss", "final_val_loss"):
        checker.check(runs[column].map(_finite).all(), f"{column} is finite for every policy")
    checker.check(runs["best_epoch"].ge(0).all(), "every policy selected a checkpoint epoch")
    return runs


def _check_no_test_leak(checker: Checker, config: Config, runs: pd.DataFrame) -> None:
    checker.section("Official test-set lock")
    metadata = load_cub_metadata(config)
    split = resolve_split(metadata, config)
    train_ids, val_ids = _apply_subset(config, split, int(config.get("seed.value")))
    locked = {image.image_id for image in metadata.official_test}

    used = set(train_ids) | set(val_ids)
    checker.check(not (used & locked),
                  f"no official test image entered training or validation "
                  f"({len(locked)} test images untouched)")
    checker.check(not (set(train_ids) & set(val_ids)),
                  "training and validation subsets do not overlap")
    if not runs.empty:
        checker.check(
            int(runs["num_train_images"].iloc[0]) == len(train_ids)
            and int(runs["num_val_images"].iloc[0]) == len(val_ids),
            "recorded dataset sizes match the split the configuration resolves to",
        )


def _check_checkpoints_and_histories(checker: Checker, config: Config, policies: list) -> None:
    checker.section("Per-policy checkpoints and histories")
    checkpoint_root = config.path("train.checkpoint.dir")
    best_name = str(config.get("train.checkpoint.best_name"))
    history_dir = config.path("policies.output.history_dir")

    for policy in policies:
        checkpoint = checkpoint_root / policy.name / best_name
        checker.check(checkpoint.is_file(), f"'{policy.name}': checkpoint {checkpoint} exists")

        history_path = history_dir / f"{policy.name}.csv"
        if not checker.check(history_path.is_file(),
                             f"'{policy.name}': history {history_path} exists"):
            continue
        history = pd.read_csv(history_path)
        checker.check(len(history) >= 1,
                      f"'{policy.name}': at least one training epoch completed "
                      f"({len(history)} recorded)")
        finite = all(
            history[column].map(_finite).all()
            for column in ("train_loss", "train_top1", "train_top5",
                           "val_loss", "val_top1", "val_top5")
        )
        checker.check(finite, f"'{policy.name}': every recorded epoch metric is finite")


# ---------------------------------------------------------------------------
# Analysis artefacts
# ---------------------------------------------------------------------------


def _check_analysis(checker: Checker, config: Config, policies: list) -> None:
    checker.section("Analysis artefacts")
    fmt = str(config.get("analysis.output.format"))
    output_dir = config.path("analysis.output.dir")

    summary_path = table_path(output_dir, str(config.get("analysis.output.summary_name")), fmt)
    joined_path = table_path(output_dir, str(config.get("analysis.output.joined_name")), fmt)
    if checker.check(summary_path.is_file(), f"{summary_path} exists"):
        summary = pd.read_csv(summary_path)
        checker.check(sorted(summary["policy"]) == sorted(p.name for p in policies),
                      "every policy appears in the policy summary")
        checker.check("delta_best_val_top1_vs_baseline" in summary.columns,
                      "policy summary carries the baseline-relative delta")
    if checker.check(joined_path.is_file(), f"{joined_path} exists"):
        joined = pd.read_csv(joined_path)
        arms = [p for p in policies if not p.is_baseline]
        checker.check(len(joined) == len(arms),
                      f"joined table has one row per non-baseline arm ({len(arms)})")
        checker.check(joined["delta_margin_mean"].map(_finite).all(),
                      "every arm carries an Experiment 0A mean delta_margin")

    reference = config.path("analysis.reference_0a.summary_path")
    checker.check(reference.is_file(), f"Experiment 0A summary {reference} is present")

    plots_dir = config.path("analysis.plots.dir")
    extension = str(config.get("analysis.plots.file_format"))
    for kind in config.get("analysis.plots.kinds"):
        path = plots_dir / f"{kind}.{extension}"
        checker.check(path.is_file() and path.stat().st_size > 0, f"plot {path} exists")


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else (project_root() / path)


if __name__ == "__main__":
    raise SystemExit(main())
