"""Paired clean/transformed views for the Experiment 1A gradient measurement.

Experiment 1A compares ``g_x`` and ``g_T`` for the *same* image, so the clean
view and every transformed view must come out of one dataset item. That also
decodes each JPEG exactly once instead of once per transformation, which is what
keeps the measurement input-bound rather than decode-bound.

Transformation sampling protocol
--------------------------------
Realisation ``r = 0`` re-uses Experiment 0A's frozen parameter key verbatim,

    ``derive_seed(param_seed, image_id, transform_name)``

so the transformed image Experiment 1A differentiates is *bit-identical* to the
one Experiment 0A measured ``delta_margin`` on. That is what makes the two
experiments joinable per image rather than only per transformation.

Additional realisations ``r >= 1`` extend the key with the realisation index, so
they are independent draws that leave ``r = 0`` untouched. They exist to estimate
how much of the spread in a transformation's gradient behaviour comes from the
parameter draw rather than from the image; the default is a single realisation,
because more of them multiply the cost without changing what is being asked.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from ..audit.records import encode_params
from ..config import ConfigError
from ..data.cub import CubMetadata
from ..data.datasets import load_rgb_image
from ..seeding import derive_seed
from ..transforms.base import DeterministicTransform
from ..transforms.pipeline import CanonicalView, Normalizer

__all__ = ["PairedViewDataset", "PairedBatch", "collate_paired", "realisation_seed"]


def realisation_seed(param_seed: int, image_id: int, transform_name: str, realisation: int) -> int:
    """Per-sample transformation seed, backwards compatible with Experiment 0A.

    ``realisation == 0`` reproduces Experiment 0A's key exactly; any other value
    appends the index so the extra draws are independent of it.
    """
    if int(realisation) == 0:
        return derive_seed(int(param_seed), int(image_id), str(transform_name))
    return derive_seed(
        int(param_seed), int(image_id), str(transform_name), "realisation", int(realisation)
    )


@dataclass
class PairedBatch:
    """One collated batch of paired views."""

    image_ids: list[int]
    labels: torch.Tensor
    """(B,) contiguous true labels."""

    clean: torch.Tensor
    """(B, 3, H, W) canonical views."""

    views: torch.Tensor
    """(B, K, 3, H, W) transformed views, K = len(transform_names) * realisations."""

    view_transform: list[str]
    """(K,) transformation name for each view slot."""

    view_realisation: list[int]
    """(K,) realisation index for each view slot."""

    view_params: list[list[str]]
    """(B, K) JSON-encoded resolved parameters."""

    def __len__(self) -> int:
        return len(self.image_ids)


class PairedViewDataset(Dataset):
    """CUB subset yielding the canonical view plus every transformed view.

    The canonical view is produced by the same ``CanonicalView`` + ``Normalizer``
    pair Experiment 0A uses, and each transformed view by the same
    ``apply_pil`` -> normalise -> ``apply_tensor`` order, so a view here is the
    same tensor Experiment 0A's audit built.
    """

    def __init__(
        self,
        metadata: CubMetadata,
        image_ids: Sequence[int],
        canonical_view: CanonicalView,
        normalizer: Normalizer,
        transforms: Sequence[DeterministicTransform],
        param_seed: int,
        realisations: int = 1,
    ) -> None:
        if not image_ids:
            raise ConfigError("Cannot build a paired-view dataset over an empty image id list.")
        if not transforms:
            raise ConfigError("Cannot build a paired-view dataset with no transformations.")
        if int(realisations) < 1:
            raise ConfigError(
                f"gradients.sampling.realisations must be >= 1, got {realisations}."
            )
        self.metadata = metadata
        self.image_ids = tuple(int(image_id) for image_id in image_ids)
        self.records = tuple(metadata.subset(self.image_ids))
        self.canonical_view = canonical_view
        self.normalizer = normalizer
        self.transforms = list(transforms)
        self.param_seed = int(param_seed)
        self.realisations = int(realisations)

        self.view_transform: list[str] = []
        self.view_realisation: list[int] = []
        for transform in self.transforms:
            for realisation in range(self.realisations):
                self.view_transform.append(transform.name)
                self.view_realisation.append(realisation)

    def __len__(self) -> int:
        return len(self.records)

    @property
    def num_views(self) -> int:
        return len(self.view_transform)

    def resolve_params(
        self, image_id: int, transform: DeterministicTransform, realisation: int
    ) -> dict[str, Any]:
        """Frozen parameters for one ``(image, transform, realisation)`` triple."""
        seed = realisation_seed(self.param_seed, image_id, transform.name, realisation)
        return transform.resolve(Random(seed))

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        source = load_rgb_image(self.metadata.absolute_path(record))
        canonical = self.canonical_view(source)

        views: list[torch.Tensor] = []
        params: list[str] = []
        for transform in self.transforms:
            for realisation in range(self.realisations):
                resolved = self.resolve_params(record.image_id, transform, realisation)
                view = transform.apply_pil(canonical, resolved)
                tensor = self.normalizer(view)
                views.append(transform.apply_tensor(tensor, resolved))
                params.append(encode_params(resolved))

        return {
            "image_id": int(record.image_id),
            "label": int(record.label),
            "clean": self.normalizer(canonical),
            "views": torch.stack(views, dim=0),
            "params": params,
        }


def collate_paired(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate paired items, keeping the JSON parameter strings as plain lists."""
    return {
        "image_ids": [item["image_id"] for item in items],
        "labels": torch.tensor([item["label"] for item in items], dtype=torch.long),
        "clean": torch.stack([item["clean"] for item in items], dim=0),
        "views": torch.stack([item["views"] for item in items], dim=0),
        "params": [item["params"] for item in items],
    }
