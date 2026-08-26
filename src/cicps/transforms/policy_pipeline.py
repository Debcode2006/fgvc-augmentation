"""Stochastic training pipelines for Experiment 0B's augmentation policies.

Experiment 0A applies an audited transformation as a *deterministic probe*: its
parameters are frozen per ``(image, transform)`` pair so that ``delta_margin``
measures one fixed perturbation. Experiment 0B uses the same transformations for
their ordinary purpose - data augmentation - and therefore samples them afresh
every time an image is drawn, exactly as a conventional recipe would.

The transformation *implementations* are shared: this module wraps the very same
:class:`~cicps.transforms.base.DeterministicTransform` objects that the audit
uses, calling ``resolve`` against the process-global :mod:`random` stream rather
than against a per-sample frozen generator. Strengths therefore cannot drift
apart between the two experiments, while the sampling semantics stay appropriate
to each.

Reproducibility comes from the existing seeding system rather than from freezing
parameters: :func:`~cicps.seeding.seed_run` seeds the global stream in the main
process and :func:`~cicps.seeding.worker_init_fn` seeds it in every DataLoader
worker, so a policy run replays identically given the same seed, split and
worker count.
"""

from __future__ import annotations

import random as _random
from random import Random
from typing import Any, Callable

import torch
from PIL.Image import Image

from ..config import Config, ConfigError
from ..logging_utils import get_logger
from ..training.policies import AugmentationPolicy
from .base import DeterministicTransform, build_transform
from .pipeline import (
    CanonicalView,
    Normalizer,
    build_canonical_view,
    build_normalizer,
    resolve_interpolation,
)

__all__ = [
    "GlobalRandomStream",
    "StochasticAugmentation",
    "ProbabilisticChoice",
    "PolicyTrainPipeline",
    "build_policy_train_transform",
    "describe_policy_pipeline",
]

logger = get_logger(__name__)


class GlobalRandomStream(Random):
    """A :class:`random.Random` view onto the process-global :mod:`random` stream.

    The audited transformations take a ``Random`` in ``resolve``. During training
    we want their draws to come from the stream the repository already seeds -
    ``random.seed(...)`` in the main process, ``worker_init_fn`` in every
    DataLoader worker - so that augmentation varies across epochs and workers
    while staying reproducible from the master seed. Delegating ``random()`` is
    the documented way to re-target a ``Random`` subclass.
    """

    def random(self) -> float:  # noqa: D102 - see class docstring
        return _random.random()

    def seed(self, *args: Any, **kwargs: Any) -> None:
        """No-op: the underlying stream is seeded by :mod:`cicps.seeding`."""

    def getstate(self) -> tuple:  # pragma: no cover - keeps the object picklable
        return ()

    def setstate(self, state: Any) -> None:  # pragma: no cover
        return None


class StochasticAugmentation:
    """One audited transformation applied with probability ``p`` during training.

    ``draw`` performs the Bernoulli trial *and* the parameter draw in one step so
    that the PIL-space and tensor-space halves of a transformation (colour jitter
    is PIL-space, random erasing is tensor-space) always see the same resolved
    parameters for a given sample.
    """

    def __init__(self, transform: DeterministicTransform, probability: float) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ConfigError(
                f"Application probability for '{transform.name}' must lie in [0, 1], "
                f"got {probability}."
            )
        self.transform = transform
        self.probability = float(probability)
        self._rng = GlobalRandomStream()

    def draw(self) -> dict[str, Any] | None:
        """Return freshly sampled parameters, or ``None`` when not applied."""
        if self.probability < 1.0 and self._rng.random() >= self.probability:
            return None
        return self.transform.resolve(self._rng)

    def apply_pil(self, image: Image, resolved: dict[str, Any]) -> Image:
        return self.transform.apply_pil(image, resolved)

    def apply_tensor(self, tensor: torch.Tensor, resolved: dict[str, Any]) -> torch.Tensor:
        return self.transform.apply_tensor(tensor, resolved)

    def describe(self) -> dict[str, Any]:
        described = self.transform.describe()
        described["application_probability"] = self.probability
        return described


class ProbabilisticChoice:
    """Pick between a baseline stage and a re-parameterised one with probability ``p``.

    Used by the crop arm. The baseline already contains a ``RandomResizedCrop``,
    so the controlled way to introduce Experiment 0A's crop strength is to
    *substitute* its parameters on a ``p`` fraction of images rather than stack a
    second independent crop on top. That keeps the fraction of images receiving
    the audited perturbation identical (``p``) across every non-baseline arm.
    """

    def __init__(self, base: Callable, alternate: Callable, probability: float) -> None:
        self.base = base
        self.alternate = alternate
        self.probability = float(probability)
        self._rng = GlobalRandomStream()

    def __call__(self, image: Image) -> Image:
        chosen = (
            self.alternate
            if self.probability >= 1.0 or self._rng.random() < self.probability
            else self.base
        )
        return chosen(image)


class PolicyTrainPipeline:
    """Baseline recipe plus at most one policy modification.

    Stage order mirrors Experiment 0A's audit pipeline - geometry/photometry in
    PIL space, then normalisation, then any tensor-space stage - so a
    transformation behaves the same way in both experiments.

    A class rather than a closure so that datasets holding it stay picklable:
    Windows and macOS spawn DataLoader workers, which pickle the dataset.
    """

    def __init__(
        self,
        crop: Callable[[Image], Image],
        flip: Callable[[Image], Image] | None,
        normalize: Normalizer,
        augmentation: StochasticAugmentation | None = None,
    ) -> None:
        self.crop = crop
        self.flip = flip
        self.normalize = normalize
        self.augmentation = augmentation

    def __call__(self, image: Image) -> torch.Tensor:
        image = self.crop(image)
        if self.flip is not None:
            image = self.flip(image)

        resolved = self.augmentation.draw() if self.augmentation is not None else None
        if resolved is not None:
            image = self.augmentation.apply_pil(image, resolved)

        tensor = self.normalize(image)
        if resolved is not None:
            tensor = self.augmentation.apply_tensor(tensor, resolved)
        return tensor


def build_policy_train_transform(
    config: Config, policy: AugmentationPolicy
) -> PolicyTrainPipeline:
    """Build the training pipeline for one Experiment 0B policy.

    Every arm starts from the *same* ``train.augmentation`` baseline; the policy
    supplies either an appended transformation (``add``) or replacement
    parameters for a baseline stage (``override``), never both.
    """
    from torchvision.transforms import RandomHorizontalFlip

    crop_cfg = dict(config.get("train.augmentation.random_resized_crop"))
    flip_cfg = dict(config.get("train.augmentation.horizontal_flip"))
    probability = policy.application_probability

    crop: Callable[[Image], Image] = (
        _build_crop(config, crop_cfg) if bool(crop_cfg.get("enabled"))
        else build_canonical_view(config)
    )
    crop_override = policy.override.get("random_resized_crop")
    if crop_override is not None:
        if not bool(crop_cfg.get("enabled")):
            raise ConfigError(
                f"Policy '{policy.name}' overrides random_resized_crop, but "
                "train.augmentation.random_resized_crop.enabled is false, so there is no "
                "baseline crop to substitute."
            )
        crop = ProbabilisticChoice(
            base=crop,
            alternate=_build_crop(config, {**crop_cfg, **crop_override}),
            probability=probability,
        )

    flip_params = {**flip_cfg, **policy.override.get("horizontal_flip", {})}
    flip = (
        RandomHorizontalFlip(p=float(flip_params["p"]))
        if bool(flip_params.get("enabled"))
        else None
    )

    augmentation = None
    if policy.add is not None:
        augmentation = StochasticAugmentation(
            build_transform(
                name=policy.audit_transform or policy.name,
                type_name=str(policy.add["type"]),
                params=policy.add.get("params", {}),
            ),
            probability=probability,
        )

    return PolicyTrainPipeline(
        crop=crop, flip=flip, normalize=build_normalizer(config), augmentation=augmentation
    )


def describe_policy_pipeline(pipeline: PolicyTrainPipeline) -> dict[str, Any]:
    """Serialisable description of a built pipeline, for logs and manifests.

    Describing the *built* object rather than the configuration is deliberate: it
    is what proves that the pipeline a policy actually trained under is the one
    the YAML declared.
    """
    return {
        "crop": _describe_stage(pipeline.crop),
        "flip": _describe_stage(pipeline.flip),
        "normalize": {
            "mean": list(pipeline.normalize.mean),
            "std": list(pipeline.normalize.std),
        },
        "added_transform": (
            None if pipeline.augmentation is None else pipeline.augmentation.describe()
        ),
    }


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _build_crop(config: Config, params: dict[str, Any]):
    """Build a torchvision ``RandomResizedCrop`` from baseline/override parameters."""
    from torchvision.transforms import RandomResizedCrop

    missing = [key for key in ("scale", "ratio") if key not in params]
    if missing:
        raise ConfigError(
            f"random_resized_crop parameters are missing {missing}; got {sorted(params)}."
        )
    return RandomResizedCrop(
        size=int(config.get("preprocess.image_size")),
        scale=tuple(float(value) for value in params["scale"]),
        ratio=tuple(float(value) for value in params["ratio"]),
        interpolation=resolve_interpolation(
            params.get("interpolation", config.get("preprocess.interpolation"))
        ),
        antialias=True,
    )


def _describe_stage(stage: Any) -> Any:
    """Render one built pipeline stage as plain data."""
    if stage is None:
        return None
    if isinstance(stage, ProbabilisticChoice):
        return {
            "kind": "probabilistic_choice",
            "probability": stage.probability,
            "baseline": _describe_stage(stage.base),
            "alternate": _describe_stage(stage.alternate),
        }
    if isinstance(stage, CanonicalView):
        return {
            "kind": "canonical_view",
            "image_size": stage.image_size,
            "resize_size": stage.resize_size,
        }
    described: dict[str, Any] = {"kind": type(stage).__name__, "repr": repr(stage)}
    # Pull the parameters out structurally rather than leaving them inside the
    # repr string, so a manifest can be diffed and a test can assert on them.
    for attribute in ("size", "scale", "ratio", "p"):
        if hasattr(stage, attribute):
            value = getattr(stage, attribute)
            described[attribute] = list(value) if isinstance(value, (list, tuple)) else value
    return described
