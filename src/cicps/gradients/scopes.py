"""Parameter scopes for gradient measurement.

A *scope* is a named subset of the model's trainable parameters over which a
gradient inner product is accumulated. Experiment 1A reports several at once
because the marginal cost is nil - the backward pass has already run - while the
interpretation differs sharply:

``full``
    Every trainable parameter. This is what the optimizer actually moves, so it
    is Experiment 1A's primary scope.
``head`` / ``backbone``
    The classification layer versus the representation trunk. A whole-model
    cosine is dominated by whichever block carries the larger gradient norm, so
    "the head disagrees" and "the features disagree" must be separable.
``stem`` / ``layer1`` ... ``layer4``
    Depth-resolved detail; supporting evidence only.

Scopes are declared in YAML as ``name -> list of regular expressions`` matched
against parameter names, so a different architecture needs a configuration edit
rather than a code change. Every scope must be non-empty; a pattern that matches
nothing is a silent measurement of nothing and therefore raises.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch.nn as nn

from ..config import Config, ConfigError
from ..logging_utils import get_logger

__all__ = ["ParameterScope", "ScopeSet", "build_scopes"]

logger = get_logger(__name__)


@dataclass(frozen=True)
class ParameterScope:
    """One named subset of the trainable parameters."""

    name: str
    patterns: tuple[str, ...]
    """Regular expressions matched (via ``re.search``) against parameter names."""

    indices: tuple[int, ...]
    """Positions into the trainable-parameter list this scope covers."""

    num_parameters: int
    """Total number of scalar parameters in the scope."""

    def describe(self) -> dict:
        return {
            "name": self.name,
            "patterns": list(self.patterns),
            "num_tensors": len(self.indices),
            "num_parameters": self.num_parameters,
        }


@dataclass(frozen=True)
class ScopeSet:
    """The ordered collection of scopes measured in one run."""

    scopes: tuple[ParameterScope, ...]
    parameter_names: tuple[str, ...]
    """Names of the trainable parameters, in the order gradients are read."""

    primary: str
    """Name of the scope treated as Experiment 1A's primary measurement."""

    def __len__(self) -> int:
        return len(self.scopes)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(scope.name for scope in self.scopes)

    def describe(self) -> dict:
        return {
            "primary": self.primary,
            "num_trainable_tensors": len(self.parameter_names),
            "scopes": [scope.describe() for scope in self.scopes],
        }


def build_scopes(model: nn.Module, config: Config) -> ScopeSet:
    """Build the scope set declared under ``gradients.scopes`` in the YAML file."""
    declared: Mapping[str, Sequence[str]] = config.get("gradients.scopes")
    if not isinstance(declared, Mapping) or not declared:
        raise ConfigError(
            "Configuration key 'gradients.scopes' must be a non-empty mapping of "
            "scope name -> list of parameter-name regular expressions."
        )
    primary = str(config.get("gradients.primary_scope"))
    if primary not in declared:
        raise ConfigError(
            f"gradients.primary_scope is '{primary}' but that scope is not declared. "
            f"Declared scopes: {', '.join(declared)}."
        )

    named = [(name, parameter) for name, parameter in model.named_parameters()
             if parameter.requires_grad]
    if not named:
        raise ConfigError("The model exposes no trainable parameters to measure.")
    parameter_names = tuple(name for name, _ in named)

    scopes: list[ParameterScope] = []
    for name, patterns in declared.items():
        if isinstance(patterns, str) or not isinstance(patterns, Sequence) or not patterns:
            raise ConfigError(
                f"gradients.scopes.{name} must be a non-empty list of regular expressions, "
                f"got {patterns!r}."
            )
        compiled = [re.compile(str(pattern)) for pattern in patterns]
        indices = tuple(
            index for index, parameter_name in enumerate(parameter_names)
            if any(expression.search(parameter_name) for expression in compiled)
        )
        if not indices:
            raise ConfigError(
                f"gradients.scopes.{name} matches no trainable parameter. Patterns: "
                f"{list(patterns)}. Available parameter names start with: "
                f"{', '.join(sorted({n.split('.')[0] for n in parameter_names}))}."
            )
        count = sum(int(named[index][1].numel()) for index in indices)
        scopes.append(
            ParameterScope(name=str(name), patterns=tuple(str(p) for p in patterns),
                           indices=indices, num_parameters=count)
        )

    scope_set = ScopeSet(scopes=tuple(scopes), parameter_names=parameter_names, primary=primary)
    for scope in scope_set.scopes:
        logger.info(
            "Gradient scope %-10s : %3d tensor(s), %9d parameter(s)%s",
            scope.name, len(scope.indices), scope.num_parameters,
            "  [primary]" if scope.name == primary else "",
        )
    return scope_set
