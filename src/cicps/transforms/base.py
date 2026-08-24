"""Deterministic transformation interface and registry.

Determinism contract
--------------------
A transformation in Experiment 0A must be a *fixed function of the sample*. If
``T`` were re-sampled on every evaluation, ``delta_margin`` would mix the effect
of the transformation with the effect of the random draw, and the audit would no
longer measure what it claims to measure.

The contract is enforced by splitting each transformation into two phases:

``resolve(rng)``
    Draws this sample's parameters from ``rng``, a :class:`random.Random` seeded
    by ``(audit.transform_param_seed, image_id, transform name)``. Same seed and
    same image id therefore always yield the same parameters - across runs,
    processes, machines and DataLoader worker counts.

``apply_pil`` / ``apply_tensor``
    Pure functions of ``(input, resolved parameters)``. They must never touch a
    global RNG.

Parameters vary *between* samples (a single fixed rotation angle for the whole
dataset would be a much weaker probe) but are frozen *for* a sample.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from random import Random
from typing import Any, Callable, ClassVar, Mapping, Sequence

import torch
from PIL.Image import Image

from ..config import Config, ConfigError

__all__ = [
    "DeterministicTransform",
    "register_transform",
    "build_transform",
    "build_transform_suite",
    "available_transform_types",
]

_REGISTRY: dict[str, type["DeterministicTransform"]] = {}


class DeterministicTransform(ABC):
    """Base class for an audited transformation.

    Subclasses declare a ``type_name`` (matched against ``type`` in the YAML
    ``transformations`` list) and implement :meth:`resolve` plus at least one of
    :meth:`apply_pil` / :meth:`apply_tensor`.
    """

    type_name: ClassVar[str] = ""

    def __init__(self, name: str, params: Mapping[str, Any] | None = None) -> None:
        self.name = name
        self.params: dict[str, Any] = dict(params or {})
        self._validate()

    # -- to be implemented by subclasses -----------------------------------

    def _validate(self) -> None:
        """Validate ``self.params``; raise :class:`ConfigError` when invalid."""

    @abstractmethod
    def resolve(self, rng: Random) -> dict[str, Any]:
        """Return this sample's frozen parameters, drawn from ``rng``."""

    def apply_pil(self, image: Image, resolved: Mapping[str, Any]) -> Image:
        """Image-space stage, applied to the canonical 224x224 view."""
        return image

    def apply_tensor(
        self, tensor: torch.Tensor, resolved: Mapping[str, Any]
    ) -> torch.Tensor:
        """Tensor-space stage, applied after ToTensor + Normalize."""
        return tensor

    # -- helpers -----------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        """Configuration-level description, for manifests and logs."""
        return {"name": self.name, "type": self.type_name, "params": dict(self.params)}

    def _require(self, key: str) -> Any:
        if key not in self.params:
            raise ConfigError(
                f"Transformation '{self.name}' (type '{self.type_name}') requires "
                f"params.{key}."
            )
        return self.params[key]

    def _range(self, key: str, *, non_negative: bool = False) -> tuple[float, float]:
        """Read a ``[low, high]`` parameter pair with a clear error message."""
        value = self._require(key)
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
            raise ConfigError(
                f"Transformation '{self.name}': params.{key} must be a two-element "
                f"[min, max] list, got {value!r}."
            )
        low, high = float(value[0]), float(value[1])
        if low > high:
            raise ConfigError(
                f"Transformation '{self.name}': params.{key} has min > max ({low} > {high})."
            )
        if non_negative and low < 0:
            raise ConfigError(
                f"Transformation '{self.name}': params.{key} must be non-negative, got {value!r}."
            )
        return low, high

    def _non_negative(self, key: str) -> float:
        value = float(self._require(key))
        if value < 0:
            raise ConfigError(
                f"Transformation '{self.name}': params.{key} must be non-negative, got {value}."
            )
        return value

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} name={self.name!r} params={self.params!r}>"


def register_transform(
    type_name: str,
) -> Callable[[type[DeterministicTransform]], type[DeterministicTransform]]:
    """Class decorator registering a transformation implementation."""

    def decorator(cls: type[DeterministicTransform]) -> type[DeterministicTransform]:
        if type_name in _REGISTRY:
            raise ValueError(f"Transformation type '{type_name}' is already registered.")
        cls.type_name = type_name
        _REGISTRY[type_name] = cls
        return cls

    return decorator


def available_transform_types() -> list[str]:
    """Sorted list of registered transformation type names."""
    return sorted(_REGISTRY)


def build_transform(name: str, type_name: str, params: Mapping[str, Any]) -> DeterministicTransform:
    """Instantiate one registered transformation."""
    try:
        cls = _REGISTRY[type_name]
    except KeyError as error:
        raise ConfigError(
            f"Unknown transformation type '{type_name}' for transformation '{name}'. "
            f"Registered types: {', '.join(available_transform_types())}."
        ) from error
    return cls(name=name, params=params)


def build_transform_suite(config: Config) -> list[DeterministicTransform]:
    """Build the audited transformation suite from ``transformations`` in YAML."""
    specs = config.sections("transformations")
    if not specs:
        raise ConfigError("Configuration key 'transformations' must list at least one entry.")

    suite: list[DeterministicTransform] = []
    seen: set[str] = set()
    for spec in specs:
        name = str(spec.get("name"))
        if name in seen:
            raise ConfigError(f"Duplicate transformation name '{name}' in 'transformations'.")
        seen.add(name)
        suite.append(build_transform(name, str(spec.get("type")), spec.get("params", {})))
    return suite
