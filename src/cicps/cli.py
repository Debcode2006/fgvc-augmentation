"""Command-line interface.

Every command takes ``--config`` (and optional ``--override`` files) and reads
all experiment settings from there. There are deliberately no per-parameter
command-line flags: the YAML file is the single source of truth, so a run is
described completely by the configuration it was given.

Experiment 0A - the frozen transformation audit::

    python -m cicps prepare  --config config/experiment_0a.yaml
    python -m cicps verify   --config config/experiment_0a.yaml
    python -m cicps train    --config config/experiment_0a.yaml
    python -m cicps audit    --config config/experiment_0a.yaml
    python -m cicps analyze  --config config/experiment_0a.yaml

Experiment 0B - the controlled downstream training comparison::

    python -m cicps prepare  --config config/experiment_0b.yaml
    python -m cicps train    --config config/experiment_0b.yaml
    python -m cicps analyze  --config config/experiment_0b.yaml

The commands are identical; the configuration decides what they do. A file that
declares a ``policies`` section is a 0B policy sweep, so ``train`` trains one
model per policy and ``analyze`` joins the sweep against Experiment 0A's summary.
There is no experiment-selecting command-line flag, because that would be an
experiment-defining value living outside the YAML file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .config import Config, ConfigError, load_config, project_root
from .stages import (
    run_analyze,
    run_analyze_policies,
    run_audit,
    run_prepare,
    run_train,
    run_train_policies,
    run_verify_transforms,
)

__all__ = ["main", "build_parser", "is_policy_experiment"]

DEFAULT_CONFIG = "config/experiment_0a.yaml"

POLICY_SECTION = "policies"
"""Presence of this configuration section marks a run as an Experiment 0B sweep."""


def is_policy_experiment(config: Config) -> bool:
    """True when the configuration declares an Experiment 0B policy sweep."""
    return POLICY_SECTION in config


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ``cicps`` command."""
    parser = argparse.ArgumentParser(
        prog="cicps",
        description=(
            "CICPS - Experiment 0A transformation audit and Experiment 0B augmentation "
            "policy comparison on CUB-200-2011."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        subparser.add_argument(
            "--config", type=Path, default=Path(DEFAULT_CONFIG),
            help=f"Path to the experiment YAML configuration (default: {DEFAULT_CONFIG}).",
        )
        subparser.add_argument(
            "--override", type=Path, action="append", default=[], metavar="YAML",
            help="Additional YAML file deep-merged over the base configuration. Repeatable.",
        )
        return subparser

    prepare = add_common(subparsers.add_parser(
        "prepare", help="Validate CUB and materialise the internal train/val split."))
    prepare.add_argument(
        "--overwrite", action="store_true",
        help="Rebuild the split manifest even if it already exists (invalidates checkpoints "
             "trained on the previous split).",
    )

    verify = add_common(subparsers.add_parser(
        "verify", help="Check that the audited transformations are deterministic per sample."))
    verify.add_argument(
        "--num-images", type=int, default=8,
        help="How many validation images to check (default: 8).",
    )

    train = add_common(subparsers.add_parser(
        "train",
        help="Train the baseline model (0A), or one model per augmentation policy (0B).",
    ))
    train.add_argument(
        "--policy", action="append", default=[], metavar="NAME",
        help="Experiment 0B only: train just this declared policy, leaving the other "
             "policies' checkpoints, histories and run rows in place. Repeatable. This is "
             "an execution-scope flag for re-runs; it cannot define a policy, only select "
             "one already declared in the configuration.",
    )

    audit = add_common(subparsers.add_parser(
        "audit", help="Run the transformation audit against a trained checkpoint."))
    audit.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Override audit.checkpoint for this run only.",
    )

    analyze = add_common(subparsers.add_parser(
        "analyze",
        help="Summarise the audit dataframe and plot delta-margin (0A), or summarise the "
             "policy sweep and join it against Experiment 0A (0B).",
    ))
    analyze.add_argument(
        "--records", type=Path, default=None,
        help="Experiment 0A only: override the audit dataframe path for this run.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = _load(args)
        if args.command == "prepare":
            run_prepare(config, overwrite=args.overwrite)
        elif args.command == "verify":
            run_verify_transforms(config, num_images=args.num_images)
        elif args.command == "train":
            if is_policy_experiment(config):
                run_train_policies(config, policy_names=list(args.policy))
            else:
                _reject_policy_flag(parser, args)
                run_train(config)
        elif args.command == "audit":
            run_audit(config, checkpoint_path=_resolve(config, args.checkpoint))
        elif args.command == "analyze":
            if is_policy_experiment(config):
                if args.records is not None:
                    parser.error(
                        "--records applies to the Experiment 0A audit dataframe; this "
                        "configuration declares a 'policies' section (Experiment 0B)."
                    )
                run_analyze_policies(config)
            else:
                run_analyze(config, records_path=_resolve(config, args.records))
        else:  # pragma: no cover - argparse enforces the choices
            parser.error(f"Unknown command '{args.command}'.")
    except (ConfigError, RuntimeError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


def _load(args: argparse.Namespace) -> Config:
    """Load the configuration, resolving a relative --config against the repo root."""
    path = args.config if args.config.is_absolute() else project_root() / args.config
    if not path.is_file() and args.config.is_file():
        path = args.config  # relative to the current working directory
    return load_config(path, overrides=args.override)


def _resolve(config: Config, path: Path | None) -> Path | None:
    return None if path is None else config.resolve(path)


def _reject_policy_flag(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """``--policy`` is meaningless without a policy sweep; say so rather than ignore it."""
    if getattr(args, "policy", None):
        parser.error(
            "--policy selects an Experiment 0B augmentation policy, but this configuration "
            f"declares no '{POLICY_SECTION}' section. Use a 0B configuration such as "
            "config/experiment_0b.yaml."
        )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
