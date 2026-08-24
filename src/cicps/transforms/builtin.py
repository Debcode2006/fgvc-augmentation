"""The seven transformations audited by Experiment 0A.

    1. identity              5. random_resized_crop
    2. horizontal_flip       6. gaussian_blur
    3. color_jitter          7. random_erasing
    4. rotation

Every strength/range comes from the YAML configuration. Nothing here decides
whether a transformation is "safe" or "destructive" - that is precisely the
question the audit is meant to answer empirically.

Geometry parameters are resolved as *fractions of the view* rather than pixel
counts, so a transformation's frozen parameters stay meaningful if the canonical
view size is reconfigured, and ``resolve`` stays a pure function of the RNG.
"""

from __future__ import annotations

import math
from random import Random
from typing import Any, Mapping

import torch
from PIL.Image import Image
from torchvision.transforms import functional as TF

from ..config import ConfigError
from .base import DeterministicTransform, register_transform
from .pipeline import resolve_interpolation

__all__ = [
    "Identity",
    "HorizontalFlip",
    "ColorJitter",
    "Rotation",
    "RandomResizedCrop",
    "GaussianBlur",
    "RandomErasing",
]

_MAX_GEOMETRY_ATTEMPTS = 10
"""Rejection-sampling budget, matching torchvision's crop/erase parameter search."""


@register_transform("identity")
class Identity(DeterministicTransform):
    """The control arm: the canonical view, untouched.

    ``delta_margin`` for this transformation is the numerical noise floor of the
    entire measurement and should be ~0 by construction.
    """

    def resolve(self, rng: Random) -> dict[str, Any]:
        return {}


@register_transform("horizontal_flip")
class HorizontalFlip(DeterministicTransform):
    """Always-applied left-right mirror (no flip probability).

    A probabilistic flip would leave roughly half the audited samples identical
    to ``identity`` and dilute the measurement, so the audit applies it outright.
    """

    def resolve(self, rng: Random) -> dict[str, Any]:
        return {"flipped": True}

    def apply_pil(self, image: Image, resolved: Mapping[str, Any]) -> Image:
        return TF.hflip(image)


@register_transform("color_jitter")
class ColorJitter(DeterministicTransform):
    """Photometric jitter with per-sample frozen factors.

    Unlike ``torchvision.transforms.ColorJitter``, the four adjustments are
    applied in a fixed order - brightness, contrast, saturation, hue - because a
    randomly permuted order would be a second, unmeasured source of variation.
    """

    _FACTOR_KEYS = ("brightness", "contrast", "saturation")

    def _validate(self) -> None:
        for key in self._FACTOR_KEYS:
            value = self._non_negative(key)
            if value > 1.0:
                raise ConfigError(
                    f"Transformation '{self.name}': params.{key} must lie in [0, 1], got {value}."
                )
        hue = self._non_negative("hue")
        if hue > 0.5:
            raise ConfigError(
                f"Transformation '{self.name}': params.hue must lie in [0, 0.5], got {hue}."
            )

    def resolve(self, rng: Random) -> dict[str, Any]:
        resolved = {
            key: rng.uniform(max(0.0, 1.0 - float(self.params[key])),
                             1.0 + float(self.params[key]))
            for key in self._FACTOR_KEYS
        }
        hue = float(self.params["hue"])
        resolved["hue"] = rng.uniform(-hue, hue)
        return resolved

    def apply_pil(self, image: Image, resolved: Mapping[str, Any]) -> Image:
        image = TF.adjust_brightness(image, resolved["brightness"])
        image = TF.adjust_contrast(image, resolved["contrast"])
        image = TF.adjust_saturation(image, resolved["saturation"])
        return TF.adjust_hue(image, resolved["hue"])


@register_transform("rotation")
class Rotation(DeterministicTransform):
    """In-plane rotation by a frozen angle drawn from ``[-degrees, +degrees]``."""

    def _validate(self) -> None:
        self._non_negative("degrees")

    def resolve(self, rng: Random) -> dict[str, Any]:
        degrees = float(self.params["degrees"])
        return {"angle": rng.uniform(-degrees, degrees)}

    def apply_pil(self, image: Image, resolved: Mapping[str, Any]) -> Image:
        return TF.rotate(
            image,
            angle=resolved["angle"],
            interpolation=resolve_interpolation(self.params.get("interpolation", "bilinear")),
            expand=bool(self.params.get("expand", False)),
            fill=_as_fill(self.params.get("fill", 0)),
        )


@register_transform("random_resized_crop")
class RandomResizedCrop(DeterministicTransform):
    """Crop a frozen sub-region of the view and resize it back to the view size.

    Parameters are stored as fractions of the view (``top``, ``left``,
    ``height``, ``width`` in ``[0, 1]``), so they are resolution independent.
    """

    def _validate(self) -> None:
        scale_low, scale_high = self._range("scale", non_negative=True)
        if scale_high > 1.0 or scale_low <= 0.0:
            raise ConfigError(
                f"Transformation '{self.name}': params.scale must lie in (0, 1], "
                f"got [{scale_low}, {scale_high}]."
            )
        ratio_low, _ = self._range("ratio", non_negative=True)
        if ratio_low <= 0.0:
            raise ConfigError(
                f"Transformation '{self.name}': params.ratio entries must be positive."
            )

    def resolve(self, rng: Random) -> dict[str, Any]:
        scale_low, scale_high = self._range("scale")
        ratio_low, ratio_high = self._range("ratio")
        log_ratio = (math.log(ratio_low), math.log(ratio_high))

        for _ in range(_MAX_GEOMETRY_ATTEMPTS):
            area_fraction = rng.uniform(scale_low, scale_high)
            aspect = math.exp(rng.uniform(*log_ratio))
            width = math.sqrt(area_fraction * aspect)
            height = math.sqrt(area_fraction / aspect)
            if 0.0 < width <= 1.0 and 0.0 < height <= 1.0:
                return {
                    "top": rng.uniform(0.0, 1.0 - height),
                    "left": rng.uniform(0.0, 1.0 - width),
                    "height": height,
                    "width": width,
                    "area_fraction": area_fraction,
                    "aspect": aspect,
                    "fallback": False,
                }

        # Centre-crop fallback, mirroring torchvision when rejection sampling fails.
        # The canonical view is square, so its aspect ratio is 1.
        if ratio_low > 1.0:
            height, width = 1.0 / ratio_low, 1.0
        elif ratio_high < 1.0:
            height, width = 1.0, ratio_high
        else:
            height = width = 1.0
        return {
            "top": (1.0 - height) / 2.0,
            "left": (1.0 - width) / 2.0,
            "height": height,
            "width": width,
            "area_fraction": height * width,
            "aspect": width / height,
            "fallback": True,
        }

    def apply_pil(self, image: Image, resolved: Mapping[str, Any]) -> Image:
        view_width, view_height = image.size
        top, left, height, width = _to_pixels(resolved, view_height, view_width)
        return TF.resized_crop(
            image,
            top=top,
            left=left,
            height=height,
            width=width,
            size=[view_height, view_width],
            interpolation=resolve_interpolation(self.params.get("interpolation", "bilinear")),
            antialias=True,
        )


@register_transform("gaussian_blur")
class GaussianBlur(DeterministicTransform):
    """Gaussian blur with a frozen sigma drawn from the configured range."""

    def _validate(self) -> None:
        kernel_size = int(self._require("kernel_size"))
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ConfigError(
                f"Transformation '{self.name}': params.kernel_size must be a positive odd "
                f"integer, got {kernel_size}."
            )
        low, _ = self._range("sigma", non_negative=True)
        if low <= 0.0:
            raise ConfigError(
                f"Transformation '{self.name}': params.sigma entries must be positive."
            )

    def resolve(self, rng: Random) -> dict[str, Any]:
        return {"sigma": rng.uniform(*self._range("sigma"))}

    def apply_pil(self, image: Image, resolved: Mapping[str, Any]) -> Image:
        kernel_size = int(self.params["kernel_size"])
        return TF.gaussian_blur(image, kernel_size=[kernel_size, kernel_size],
                                sigma=[resolved["sigma"], resolved["sigma"]])


@register_transform("random_erasing")
class RandomErasing(DeterministicTransform):
    """Erase a frozen rectangular region of the normalised tensor.

    Applied in tensor space (after normalisation), matching
    ``torchvision.transforms.RandomErasing``. ``params.value`` is therefore
    expressed in normalised units: ``0.0`` is the per-channel ImageNet mean.
    """

    def _validate(self) -> None:
        scale_low, scale_high = self._range("scale", non_negative=True)
        if scale_high > 1.0 or scale_low <= 0.0:
            raise ConfigError(
                f"Transformation '{self.name}': params.scale must lie in (0, 1], "
                f"got [{scale_low}, {scale_high}]."
            )
        ratio_low, _ = self._range("ratio", non_negative=True)
        if ratio_low <= 0.0:
            raise ConfigError(
                f"Transformation '{self.name}': params.ratio entries must be positive."
            )

    def resolve(self, rng: Random) -> dict[str, Any]:
        scale_low, scale_high = self._range("scale")
        ratio_low, ratio_high = self._range("ratio")
        log_ratio = (math.log(ratio_low), math.log(ratio_high))

        for _ in range(_MAX_GEOMETRY_ATTEMPTS):
            area_fraction = rng.uniform(scale_low, scale_high)
            aspect = math.exp(rng.uniform(*log_ratio))
            height = math.sqrt(area_fraction * aspect)
            width = math.sqrt(area_fraction / aspect)
            if 0.0 < height < 1.0 and 0.0 < width < 1.0:
                return {
                    "top": rng.uniform(0.0, 1.0 - height),
                    "left": rng.uniform(0.0, 1.0 - width),
                    "height": height,
                    "width": width,
                    "area_fraction": area_fraction,
                    "erased": True,
                }

        # No valid region was found; leave the sample untouched rather than
        # silently erasing something outside the configured envelope.
        return {"top": 0.0, "left": 0.0, "height": 0.0, "width": 0.0,
                "area_fraction": 0.0, "erased": False}

    def apply_tensor(self, tensor: torch.Tensor, resolved: Mapping[str, Any]) -> torch.Tensor:
        if not resolved.get("erased", False):
            return tensor
        view_height, view_width = tensor.shape[-2], tensor.shape[-1]
        top, left, height, width = _to_pixels(resolved, view_height, view_width)
        if height <= 0 or width <= 0:
            return tensor
        erased = tensor.clone()
        erased[..., top:top + height, left:left + width] = float(self.params.get("value", 0.0))
        return erased


def _to_pixels(
    resolved: Mapping[str, Any], view_height: int, view_width: int
) -> tuple[int, int, int, int]:
    """Convert fractional ``(top, left, height, width)`` to in-bounds pixels."""
    height = max(1, int(round(resolved["height"] * view_height)))
    width = max(1, int(round(resolved["width"] * view_width)))
    height = min(height, view_height)
    width = min(width, view_width)
    top = min(int(round(resolved["top"] * view_height)), view_height - height)
    left = min(int(round(resolved["left"] * view_width)), view_width - width)
    return max(0, top), max(0, left), height, width


def _as_fill(value: Any) -> Any:
    """Normalise a YAML fill value into what ``TF.rotate`` accepts."""
    if isinstance(value, (list, tuple)):
        return [float(component) for component in value]
    return float(value)
