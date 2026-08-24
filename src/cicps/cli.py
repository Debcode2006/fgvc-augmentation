"""Command-line interface.

Every command takes ``--config`` (and optional ``--override`` files) and reads
all experiment settings from there. There are deliberately no per-parameter
command-line flags: the YAML file is the single source of truth, so a run is
described completely by the configuration it was given.

    python -m cicps prepare  --config config/experiment_0a.yaml
    python -m cicps verify   --config config/experiment_0a.yaml
    python -m cicps train    --config config/experiment_0a.yaml
    python -m cicps audit    --config config/experiment_0a.yaml
    python -m cicps analyze  --config config/experiment_0a.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .config import Config, ConfigError, load_config, project_root
from .stages import run_analyze, run_audit, run_prepare, run_train, run_verify_transforms

__all__ = ["main", "build_parser"]

DEFAULT_CONFIG = "config/experiment_0a.yaml"


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ``cicps`` command."""
    parser = argparse.ArgumentParser(
        prog="cicps",
        description="CICPS - Experiment 0A transformation audit on CUB-200-2011.",
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

    add_common(subparsers.add_parser("train", help="Train the frozen baseline model."))

    audit = add_common(subparsers.add_parser(
        "audit", help="Run the transformation audit against a trained checkpoint."))
    audit.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Override audit.checkpoint for this run only.",
    )

    analyze = add_common(subparsers.add_parser(
        "analyze", help="Summarise the audit dataframe and render the delta-margin plots."))
    analyze.add_argument(
        "--records", type=Path, default=None,
        help="Override the audit dataframe path for this run only.",
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
            run_train(config)
        elif args.command == "audit":
            run_audit(config, checkpoint_path=_resolve(config, args.checkpoint))
        elif args.command == "analyze":
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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
