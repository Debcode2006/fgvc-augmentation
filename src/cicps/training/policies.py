"""Training augmentation policies for Experiment 0B.

Experiment 0A audited transformations against a *frozen* model, with parameters
frozen per ``(image, transform)`` pair. Experiment 0B asks a different question -
what happens when one of those transformations is used as a *training*
augmentation - so here the same transformation implementations are sampled
stochastically, once per image per epoch, exactly as an ordinary augmentation
would be.

A policy is deliberately expressed as

    the standard baseline recipe  +  at most one controlled modification

so that every arm differs from ``baseline`` by exactly one thing. Two kinds of
modification are supported, both declared in YAML:

``add``
    Append one audited transformation, applied with probability
    ``policies.application_probability``. Used by the colour-jitter, rotation,
    blur and erasing arms.

``override``
    Replace the parameters of a stage the baseline *already contains*, again
    taken with probability ``policies.application_probability``. Used by the
    crop arm: the baseline already crops, so stacking a second independent crop
    would confound "0A crop strength" with "two crops". Horizontal flip needs no
    arm at all - it is already part of the baseline.

Nothing in this module inspects Experiment 0A's results. Policies are declared
in the configuration and fixed before any 0B model is trained.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..config import Config, ConfigError
from ..logging_utils import get_logger
from ..seeding import derive_run_seed
from ..transforms.base import available_transform_types

__all__ = ["AugmentationPolicy", "build_policies", "select_policies"]

logger = get_logger(__name__)

_OVERRIDABLE_STAGES = ("random_resized_crop", "horizontal_flip")
"""Baseline stages a policy may re-parameterise, named as in ``train.augmentation``."""


@dataclass(frozen=True)
class AugmentationPolicy:
    """One training augmentation policy: the baseline plus one modification."""

    name: str
    """Stable machine-readable identifier; appears in every 0B output."""

    description: str
    """Human-readable summary, copied into logs and manifests."""

    audit_transform: str | None
    """Name of the Experiment 0A transformation this arm corresponds to.

    ``None`` for the reference ``baseline`` policy. Used *only* to join 0B
    results against the 0A summary during analysis - never to build the policy.
    """

    seed: int
    """Policy-specific RNG seed, derived from the master seed and the name."""

    application_probability: float
    """Probability with which the modification is applied to a training image."""

    add: dict[str, Any] | None = None
    """``{type, params}`` of an audited transformation appended to the baseline."""

    override: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Replacement parameters for baseline stages, keyed by stage name."""

    @property
    def is_baseline(self) -> bool:
        """True when the policy is the unmodified standard recipe."""
        return self.add is None and not self.override

    def describe(self) -> dict[str, Any]:
        """Configuration-level description, for manifests and logs."""
        return {
            "name": self.name,
            "description": self.description,
            "audit_transform": self.audit_transform,
            "seed": self.seed,
            "application_probability": self.application_probability,
            "add": None if self.add is None else dict(self.add),
            "override": {key: dict(value) for key, value in self.override.items()},
            "is_baseline": self.is_baseline,
        }

    def summary_line(self) -> str:
        """One-line rendering of what this policy actually does."""
        if self.is_baseline:
            return "baseline recipe only (reference arm)"
        if self.add is not None:
            return (
                f"baseline + {self.add['type']}(p={self.application_probability}, "
                f"params={self.add['params']})"
            )
        stages = ", ".join(
            f"{stage}({params})" for stage, params in sorted(self.override.items())
        )
        return f"baseline with {stages} substituted with p={self.application_probability}"


def build_policies(config: Config) -> list[AugmentationPolicy]:
    """Build the policy sweep declared under ``policies`` in the YAML file.

    Raises :class:`ConfigError` on duplicate names, an unknown transformation
    type, a policy that changes more than one thing, a missing baseline arm, or
    an out-of-range application probability.
    """
    probability = float(config.get("policies.application_probability"))
    if not 0.0 <= probability <= 1.0:
        raise ConfigError(
            f"policies.application_probability must lie in [0, 1], got {probability}."
        )
    baseline_name = str(config.get("policies.baseline_name"))
    master_seed = int(config.get("seed.value"))
    registered = set(available_transform_types())

    policies: list[AugmentationPolicy] = []
    seen: set[str] = set()
    for spec in config.sections("policies.list"):
        name = str(spec.get("name")).strip()
        if not name:
            raise ConfigError("Every entry of 'policies.list' needs a non-empty 'name'.")
        if name in seen:
            raise ConfigError(f"Duplicate policy name '{name}' in 'policies.list'.")
        seen.add(name)

        add = _read_add(spec, name, registered)
        override = _read_override(spec, name)
        if add is not None and override:
            raise ConfigError(
                f"Policy '{name}' declares both 'add' and 'override'. A 0B arm must differ "
                "from the baseline by exactly one modification."
            )

        policies.append(
            AugmentationPolicy(
                name=name,
                description=str(spec.get("description", "")),
                audit_transform=_read_audit_transform(spec, name, add, override),
                seed=derive_run_seed(master_seed, "policy", name),
                application_probability=probability,
                add=add,
                override=override,
            )
        )

    if not policies:
        raise ConfigError("Configuration key 'policies.list' must declare at least one policy.")
    if baseline_name not in seen:
        raise ConfigError(
            f"policies.baseline_name is '{baseline_name}' but no policy of that name is "
            f"declared. Declared: {', '.join(sorted(seen))}."
        )
    reference = next(policy for policy in policies if policy.name == baseline_name)
    if not reference.is_baseline:
        raise ConfigError(
            f"The reference policy '{baseline_name}' must declare neither 'add' nor "
            "'override': it is the unmodified standard recipe every other arm is compared to."
        )
    return policies


def select_policies(
    policies: list[AugmentationPolicy], names: list[str] | None
) -> list[AugmentationPolicy]:
    """Restrict a policy sweep to ``names`` (execution scope only).

    Used by ``--policy`` to re-run a single arm without touching the others. It
    cannot invent a policy: every name must already be declared in the YAML, so
    the experiment definition still lives entirely in the configuration.
    """
    if not names:
        return policies
    known = {policy.name: policy for policy in policies}
    unknown = [name for name in names if name not in known]
    if unknown:
        raise ConfigError(
            f"Unknown policy name(s) {unknown}. Declared in the configuration: "
            f"{', '.join(known)}."
        )
    return [known[name] for name in names]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _read_add(spec: Config, name: str, registered: set[str]) -> dict[str, Any] | None:
    """Parse the optional ``add`` block of a policy declaration."""
    raw = spec.get("add", None)
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ConfigError(f"Policy '{name}': 'add' must be a mapping, got {type(raw).__name__}.")
    if "type" not in raw:
        raise ConfigError(f"Policy '{name}': 'add' requires a 'type'.")
    type_name = str(raw["type"])
    if type_name not in registered:
        raise ConfigError(
            f"Policy '{name}': unknown transformation type '{type_name}'. "
            f"Registered types: {', '.join(sorted(registered))}."
        )
    params = raw.get("params", {}) or {}
    if not isinstance(params, Mapping):
        raise ConfigError(f"Policy '{name}': 'add.params' must be a mapping.")
    return {"type": type_name, "params": dict(params)}


def _read_override(spec: Config, name: str) -> dict[str, dict[str, Any]]:
    """Parse the optional ``override`` block of a policy declaration."""
    raw = spec.get("override", None)
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ConfigError(
            f"Policy '{name}': 'override' must be a mapping of baseline stage name to "
            f"parameters, got {type(raw).__name__}."
        )
    unknown = [stage for stage in raw if stage not in _OVERRIDABLE_STAGES]
    if unknown:
        raise ConfigError(
            f"Policy '{name}': cannot override baseline stage(s) {unknown}. "
            f"Overridable stages: {', '.join(_OVERRIDABLE_STAGES)}."
        )
    return {str(stage): dict(params or {}) for stage, params in raw.items()}


def _read_audit_transform(
    spec: Config, name: str, add: dict[str, Any] | None, override: Mapping[str, Any]
) -> str | None:
    """Read the 0A transformation name this arm is joined against during analysis."""
    value = spec.get("audit_transform", None)
    if value is None:
        if add is not None or override:
            raise ConfigError(
                f"Policy '{name}' modifies the baseline but declares no 'audit_transform'. "
                "Name the Experiment 0A transformation it corresponds to so the 0A/0B join "
                "is explicit rather than inferred from the policy name."
            )
        return None
    return str(value)
