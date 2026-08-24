"""Baseline model construction.

Experiment 0A deliberately uses a conventional ImageNet-pretrained ResNet-18
with a cross-entropy head. The model is an *instrument*, not a contribution: it
only has to provide a stable discriminative representation against which
augmentation-induced change can be measured.

:class:`FeatureClassifier` exposes logits and the penultimate (post-global-pool)
representation from a single forward pass, so the audit never runs the backbone
twice to obtain both.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torchvision.models as tv_models

from ..config import Config, ConfigError
from ..logging_utils import get_logger

__all__ = ["FeatureClassifier", "build_model", "load_model_for_audit"]

logger = get_logger(__name__)

_SUPPORTED_ARCHITECTURES = {
    # architecture -> (torchvision builder, weights enum name)
    "resnet18": ("resnet18", "ResNet18_Weights"),
    "resnet34": ("resnet34", "ResNet34_Weights"),
    "resnet50": ("resnet50", "ResNet50_Weights"),
}


class FeatureClassifier(nn.Module):
    """A backbone whose classifier head is separated from its feature trunk."""

    def __init__(self, backbone: nn.Module, classifier: nn.Linear, architecture: str) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.architecture = architecture

    @property
    def feature_dim(self) -> int:
        """Width of the penultimate representation."""
        return int(self.classifier.in_features)

    @property
    def num_classes(self) -> int:
        return int(self.classifier.out_features)

    def features(self, images: torch.Tensor) -> torch.Tensor:
        """Return the flattened penultimate representation ``z = f(x)``."""
        return torch.flatten(self.backbone(images), 1)

    def forward(
        self, images: torch.Tensor, *, return_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Return logits, and optionally the penultimate features alongside."""
        representation = self.features(images)
        logits = self.classifier(representation)
        return (logits, representation) if return_features else logits


def build_model(config: Config) -> FeatureClassifier:
    """Build the baseline model described by the ``model`` configuration section."""
    architecture = str(config.get("model.architecture")).lower()
    if architecture not in _SUPPORTED_ARCHITECTURES:
        raise ConfigError(
            f"Unsupported model.architecture '{architecture}'. Supported: "
            f"{', '.join(sorted(_SUPPORTED_ARCHITECTURES))}."
        )
    builder_name, weights_enum_name = _SUPPORTED_ARCHITECTURES[architecture]
    pretrained = bool(config.get("model.pretrained"))
    num_classes = int(config.get("model.num_classes"))

    weights = _resolve_weights(config, weights_enum_name) if pretrained else None
    network: nn.Module = getattr(tv_models, builder_name)(weights=weights)

    in_features = int(network.fc.in_features)
    expected_dim = config.get("model.feature_dim", None)
    if expected_dim is not None and int(expected_dim) != in_features:
        raise ConfigError(
            f"model.feature_dim is {expected_dim} but {architecture} produces {in_features}-d "
            "features. Set it to null to infer automatically."
        )

    classifier = nn.Linear(in_features, num_classes)
    # Drop the ImageNet head; the trunk ends at global average pooling.
    network.fc = nn.Identity()

    model = FeatureClassifier(backbone=network, classifier=classifier, architecture=architecture)
    logger.info(
        "Built %s (pretrained=%s, weights=%s, feature_dim=%d, num_classes=%d, params=%.2fM).",
        architecture, pretrained, getattr(weights, "name", None), in_features, num_classes,
        sum(parameter.numel() for parameter in model.parameters()) / 1e6,
    )
    return model


def _resolve_weights(config: Config, weights_enum_name: str) -> Any:
    """Resolve ``model.weights`` into a torchvision weights enum member."""
    enum = getattr(tv_models, weights_enum_name)
    requested = config.get("model.weights", None)
    if requested is None:
        return enum.DEFAULT
    try:
        return getattr(enum, str(requested))
    except AttributeError as error:
        available = [name for name in dir(enum) if name.isupper()]
        raise ConfigError(
            f"Unknown model.weights '{requested}' for {weights_enum_name}. "
            f"Available: {', '.join(sorted(available))}."
        ) from error


def load_model_for_audit(
    config: Config, state_dict: dict[str, torch.Tensor], device: torch.device
) -> FeatureClassifier:
    """Rebuild the architecture, load trained weights, freeze and move to device.

    ``strict=True`` on purpose: an architecture/checkpoint mismatch must fail
    rather than silently auditing a partly random model.
    """
    model = build_model(_without_pretrained(config))
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _without_pretrained(config: Config) -> Config:
    """Return a copy of ``config`` with pretrained weights disabled.

    Downloading ImageNet weights only to overwrite them with the checkpoint is
    pure waste, and would fail on an offline machine.
    """
    data = config.as_dict()
    data.setdefault("model", {})["pretrained"] = False
    return Config(data, root=config.root, source=config.source)
