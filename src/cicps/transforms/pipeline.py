"""Image preprocessing pipelines.

The *canonical view* is the deterministic ``resize -> centre crop`` that turns a
raw CUB JPEG into the fixed square the network sees. Experiment 0A applies its
audited transformations on top of that view, which is what makes ``identity`` a
true no-op and makes every other transformation differ from ``identity`` by
exactly one operation.
"""

from __future__ import annotations

from typing import Callable

import torch
from PIL.Image import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from ..config import Config, ConfigError

__all__ = [
    "CanonicalView",
    "Normalizer",
    "EvalPipeline",
    "TrainPipeline",
    "build_canonical_view",
    "build_normalizer",
    "build_eval_transform",
    "build_train_transform",
    "resolve_interpolation",
]

_INTERPOLATION_MODES = {
    "nearest": InterpolationMode.NEAREST,
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "lanczos": InterpolationMode.LANCZOS,
}


def resolve_interpolation(name: str) -> InterpolationMode:
    """Map a configuration string to a torchvision interpolation mode."""
    try:
        return _INTERPOLATION_MODES[str(name).lower()]
    except KeyError as error:
        raise ConfigError(
            f"Unknown interpolation '{name}'. Choose one of: "
            f"{', '.join(sorted(_INTERPOLATION_MODES))}."
        ) from error


class CanonicalView:
    """Deterministic ``resize -> centre crop`` producing the square model view."""

    def __init__(self, image_size: int, resize_size: int, interpolation: InterpolationMode) -> None:
        if image_size <= 0 or resize_size <= 0:
            raise ConfigError("preprocess.image_size and preprocess.resize_size must be positive.")
        if resize_size < image_size:
            raise ConfigError(
                f"preprocess.resize_size ({resize_size}) must be >= preprocess.image_size "
                f"({image_size}); otherwise the centre crop would upsample."
            )
        self.image_size = image_size
        self.resize_size = resize_size
        self.interpolation = interpolation

    def __call__(self, image: Image) -> Image:
        resized = TF.resize(image, self.resize_size, interpolation=self.interpolation,
                            antialias=True)
        return TF.center_crop(resized, [self.image_size, self.image_size])


class Normalizer:
    """``PIL -> float tensor`` conversion with ImageNet-style normalisation."""

    def __init__(self, mean: list[float], std: list[float]) -> None:
        if len(mean) != len(std):
            raise ConfigError("preprocess.normalize.mean and .std must have equal length.")
        self.mean = [float(value) for value in mean]
        self.std = [float(value) for value in std]

    def __call__(self, image: Image) -> torch.Tensor:
        return TF.normalize(TF.to_tensor(image), self.mean, self.std)


def build_canonical_view(config: Config) -> CanonicalView:
    """Build the canonical view from the ``preprocess`` configuration section."""
    return CanonicalView(
        image_size=int(config.get("preprocess.image_size")),
        resize_size=int(config.get("preprocess.resize_size")),
        interpolation=resolve_interpolation(config.get("preprocess.interpolation")),
    )


def build_normalizer(config: Config) -> Normalizer:
    """Build the tensor normaliser from the ``preprocess`` configuration section."""
    return Normalizer(
        mean=list(config.get("preprocess.normalize.mean")),
        std=list(config.get("preprocess.normalize.std")),
    )


class EvalPipeline:
    """Deterministic evaluation pipeline: canonical view then normalisation.

    A class rather than a closure so that datasets holding it stay picklable -
    Windows and macOS spawn DataLoader workers, which pickle the dataset.
    """

    def __init__(self, view: CanonicalView, normalize: Normalizer) -> None:
        self.view = view
        self.normalize = normalize

    def __call__(self, image: Image) -> torch.Tensor:
        return self.normalize(self.view(image))


class TrainPipeline:
    """Stochastic training pipeline for the *baseline* model.

    Picklable for the same reason as :class:`EvalPipeline`.
    """

    def __init__(self, crop: Callable[[Image], Image], flip: Callable[[Image], Image] | None,
                 normalize: Normalizer) -> None:
        self.crop = crop
        self.flip = flip
        self.normalize = normalize

    def __call__(self, image: Image) -> torch.Tensor:
        image = self.crop(image)
        if self.flip is not None:
            image = self.flip(image)
        return self.normalize(image)


def build_eval_transform(config: Config) -> EvalPipeline:
    """Build the deterministic evaluation pipeline."""
    return EvalPipeline(build_canonical_view(config), build_normalizer(config))


def build_train_transform(config: Config) -> TrainPipeline:
    """Build the baseline training pipeline from ``train.augmentation``.

    This is the ordinary training recipe. It is deliberately unrelated to the
    audited transformation suite: the audit measures a frozen model, and this
    only decides how that model is trained.
    """
    from torchvision.transforms import RandomHorizontalFlip, RandomResizedCrop

    crop_cfg = config.section("train.augmentation.random_resized_crop")
    flip_cfg = config.section("train.augmentation.horizontal_flip")

    crop = (
        RandomResizedCrop(
            size=int(config.get("preprocess.image_size")),
            scale=tuple(float(value) for value in crop_cfg.get("scale")),
            ratio=tuple(float(value) for value in crop_cfg.get("ratio")),
            interpolation=resolve_interpolation(config.get("preprocess.interpolation")),
            antialias=True,
        )
        if bool(crop_cfg.get("enabled"))
        else build_canonical_view(config)
    )
    flip = RandomHorizontalFlip(p=float(flip_cfg.get("p"))) if bool(flip_cfg.get("enabled")) else None
    return TrainPipeline(crop, flip, build_normalizer(config))
