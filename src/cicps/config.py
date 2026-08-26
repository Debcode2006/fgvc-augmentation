"""Configuration loading.

The YAML file under ``config/`` is the single source of truth for an
experiment. Python code *consumes* configuration; it never defines it.

``Config`` is a thin, read-only view over the parsed YAML that

* supports dotted lookup (``cfg.get("train.optimizer.lr")``),
* raises a clear, path-qualified error when a key is missing,
* resolves relative paths against the project root,

so that adding a new setting means editing YAML only - never this module.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import yaml

__all__ = ["Config", "ConfigError", "load_config", "project_root"]

_MISSING = object()


class ConfigError(ValueError):
    """Raised when configuration is missing, malformed, or contradictory."""


def project_root() -> Path:
    """Return the repository root (the parent of ``src/``)."""
    # .../<root>/src/cicps/config.py -> .../<root>
    return Path(__file__).resolve().parents[2]


class Config:
    """Read-only, dotted-path view over a parsed YAML configuration tree."""

    def __init__(self, data: Mapping[str, Any], *, root: Path | None = None,
                 source: Path | None = None, prefix: str = "") -> None:
        if not isinstance(data, Mapping):
            raise ConfigError(
                f"Configuration node '{prefix or '<root>'}' must be a mapping, "
                f"got {type(data).__name__}."
            )
        self._data: dict[str, Any] = dict(data)
        self._root = root if root is not None else project_root()
        self._source = source
        self._prefix = prefix

    # -- construction ------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path, *, overrides: Sequence[str | Path] = ()) -> "Config":
        """Load ``path``, then deep-merge each override file on top of it."""
        base_path = Path(path)
        if not base_path.is_file():
            raise ConfigError(f"Configuration file not found: {base_path}")
        data = _read_yaml(base_path)
        for override in overrides:
            override_path = Path(override)
            if not override_path.is_file():
                raise ConfigError(f"Override configuration file not found: {override_path}")
            data = _deep_merge(data, _read_yaml(override_path))
        return cls(data, source=base_path)

    # -- lookup ------------------------------------------------------------

    def get(self, dotted_key: str, default: Any = _MISSING) -> Any:
        """Return the value at ``dotted_key``.

        Raises ``ConfigError`` naming the full path when the key is absent and
        no ``default`` was supplied.
        """
        node: Any = self._data
        walked: list[str] = []
        for part in dotted_key.split("."):
            walked.append(part)
            if not isinstance(node, Mapping) or part not in node:
                if default is not _MISSING:
                    return default
                raise ConfigError(
                    f"Missing configuration key '{self._qualify(dotted_key)}'"
                    f" (resolved as far as '{self._qualify('.'.join(walked[:-1]))}')"
                    + (f" in {self._source}" if self._source else "")
                    + ". Add it to the YAML configuration; it must not be hard-coded."
                )
            node = node[part]
        return copy.deepcopy(node) if isinstance(node, (dict, list)) else node

    def section(self, dotted_key: str) -> "Config":
        """Return a nested mapping as a ``Config``."""
        value = self.get(dotted_key)
        return Config(value, root=self._root, source=self._source,
                      prefix=self._qualify(dotted_key))

    def sections(self, dotted_key: str) -> list["Config"]:
        """Return a list-of-mappings node as a list of ``Config`` objects."""
        value = self.get(dotted_key)
        if not isinstance(value, list):
            raise ConfigError(
                f"Configuration key '{self._qualify(dotted_key)}' must be a list, "
                f"got {type(value).__name__}."
            )
        return [
            Config(item, root=self._root, source=self._source,
                   prefix=f"{self._qualify(dotted_key)}[{i}]")
            for i, item in enumerate(value)
        ]

    def path(self, dotted_key: str, default: Any = _MISSING) -> Path:
        """Return the value at ``dotted_key`` as a path resolved against root."""
        value = self.get(dotted_key, default)
        if value is None:
            raise ConfigError(
                f"Configuration key '{self._qualify(dotted_key)}' must name a path, got null."
            )
        return self.resolve(value)

    def resolve(self, value: str | Path) -> Path:
        """Resolve ``value`` against the project root when it is relative."""
        candidate = Path(value)
        return candidate if candidate.is_absolute() else (self._root / candidate)

    def seeded(self, dotted_key: str, fallback_key: str = "seed.value") -> int:
        """Return an optional seed override, falling back to the master seed."""
        value = self.get(dotted_key, None)
        return int(self.get(fallback_key) if value is None else value)

    # -- dict-ish behaviour ------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        """Return a deep copy of the underlying mapping."""
        return copy.deepcopy(self._data)

    def __contains__(self, dotted_key: object) -> bool:
        # A distinct sentinel: passing _MISSING itself would read as "no default
        # supplied" inside get(), which would raise on exactly the absent keys
        # this is meant to report as False.
        absent = object()
        if not isinstance(dotted_key, str):
            return False
        return self.get(dotted_key, absent) is not absent

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        origin = f" source={self._source}" if self._source else ""
        return f"<Config keys={sorted(self._data)}{origin}>"

    # -- properties --------------------------------------------------------

    @property
    def root(self) -> Path:
        """Project root that relative paths resolve against."""
        return self._root

    @property
    def source(self) -> Path | None:
        """Path of the YAML file this configuration was loaded from."""
        return self._source

    # -- internals ---------------------------------------------------------

    def _qualify(self, dotted_key: str) -> str:
        if not dotted_key:
            return self._prefix or "<root>"
        return f"{self._prefix}.{dotted_key}" if self._prefix else dotted_key


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        raise ConfigError(f"Configuration file is empty: {path}")
    if not isinstance(data, Mapping):
        raise ConfigError(
            f"Configuration file must contain a top-level mapping: {path}"
        )
    return dict(data)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (override wins)."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path, *, overrides: Sequence[str | Path] = ()) -> Config:
    """Convenience wrapper around :meth:`Config.from_yaml`."""
    return Config.from_yaml(path, overrides=overrides)
