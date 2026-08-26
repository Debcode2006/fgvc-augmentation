"""Logging setup driven entirely by configuration."""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from typing import Iterator

from .config import Config

__all__ = ["setup_logging", "get_logger", "additional_log_file"]

_ROOT_LOGGER_NAME = "cicps"


def setup_logging(config: Config, stage: str) -> logging.Logger:
    """Configure and return the package logger for a pipeline ``stage``.

    Handlers are rebuilt on every call so that repeated in-process invocations
    (tests, notebooks) do not duplicate log lines.
    """
    level = getattr(logging, str(config.get("logging.level")).upper(), logging.INFO)
    formatter = logging.Formatter(
        fmt=config.get("logging.format"),
        datefmt=config.get("logging.date_format"),
    )

    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    if bool(config.get("logging.to_console")):
        console = logging.StreamHandler(stream=sys.stdout)
        console.setFormatter(formatter)
        console.setLevel(level)
        logger.addHandler(console)

    if bool(config.get("logging.to_file")):
        log_dir = config.path("logging.dir")
        log_dir.mkdir(parents=True, exist_ok=True)
        filename = str(config.get("logging.file_template")).format(stage=stage)
        file_handler = logging.FileHandler(log_dir / filename, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
        logger.info("Logging to %s", (log_dir / filename))

    return logger


@contextmanager
def additional_log_file(config: Config, stage: str) -> Iterator[logging.Logger]:
    """Temporarily tee the package logger into a second, stage-scoped file.

    Experiment 0B's policy sweep uses this so that one run produces both a
    combined ``train.log`` and a per-policy ``train_<policy>.log`` - which is
    what makes an OOM, a NaN loss or a wrong seed diagnosable per arm without
    scrolling through six interleaved runs.
    """
    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    if not bool(config.get("logging.to_file")):
        yield logger
        return

    log_dir = config.path("logging.dir")
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / str(config.get("logging.file_template")).format(stage=stage)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter(fmt=config.get("logging.format"),
                          datefmt=config.get("logging.date_format"))
    )
    handler.setLevel(logger.level)
    logger.addHandler(handler)
    logger.info("Also logging this run to %s", path)
    try:
        yield logger
    finally:
        logger.removeHandler(handler)
        handler.close()


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the package root."""
    suffix = name.split(".")[-1]
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{suffix}")
