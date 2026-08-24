"""Stage: verify that the audited transformations are deterministic.

Experiment 0A only measures what it claims to measure if ``T(x)`` is a fixed
function of the sample. This stage checks that property directly, on real
images, without needing a trained model:

* resolving a sample's parameters twice yields identical parameters;
* materialising a sample twice yields bit-identical tensors, including through
  fresh dataset objects (i.e. no reliance on cached state);
* ``identity`` reproduces the canonical evaluation view exactly;
* every non-identity transformation actually changes the tensor, so a
  mis-specified strength cannot silently turn an arm into a second control.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..config import Config
from ..data.cub import load_cub_metadata
from ..data.datasets import CubImageDataset, DeterministicTransformDataset
from ..data.splits import resolve_split
from ..logging_utils import get_logger, setup_logging
from ..transforms.base import build_transform_suite
from ..transforms.pipeline import build_canonical_view, build_eval_transform, build_normalizer

__all__ = ["run_verify_transforms", "TransformCheck"]

logger = get_logger(__name__)

_IDENTITY_TYPE = "identity"


@dataclass
class TransformCheck:
    """Outcome of the determinism checks for one transformation."""

    name: str
    params_reproducible: bool
    tensor_reproducible: bool
    differs_from_identity: bool
    max_abs_difference: float

    @property
    def ok(self) -> bool:
        return self.params_reproducible and self.tensor_reproducible


def run_verify_transforms(config: Config, *, num_images: int = 8) -> list[TransformCheck]:
    """Run the determinism checks over the first ``num_images`` validation images."""
    setup_logging(config, stage="verify")
    logger.info("Configuration: %s", config.source)

    metadata = load_cub_metadata(config)
    split = resolve_split(metadata, config)
    image_ids = list(split.val_ids)[:num_images]
    logger.info("Verifying determinism on %d validation image(s).", len(image_ids))

    canonical_view = build_canonical_view(config)
    normalizer = build_normalizer(config)
    param_seed = config.seeded("audit.transform_param_seed")

    reference = _stack(CubImageDataset(metadata, image_ids, build_eval_transform(config)))

    def make_dataset(transform) -> DeterministicTransformDataset:
        return DeterministicTransformDataset(
            metadata=metadata,
            image_ids=image_ids,
            canonical_view=canonical_view,
            normalizer=normalizer,
            transform=transform,
            param_seed=param_seed,
        )

    checks: list[TransformCheck] = []
    for transform in build_transform_suite(config):
        # Two independently constructed datasets: reproducibility must not come
        # from cached state inside one object.
        first, second = make_dataset(transform), make_dataset(transform)
        params_reproducible = all(
            first.resolve_params(image_id) == second.resolve_params(image_id)
            for image_id in image_ids
        )
        left, right = _stack(first), _stack(second)
        tensor_reproducible = bool(torch.equal(left, right))
        difference = float((left - reference).abs().max())
        is_identity = transform.type_name == _IDENTITY_TYPE

        check = TransformCheck(
            name=transform.name,
            params_reproducible=params_reproducible,
            tensor_reproducible=tensor_reproducible,
            differs_from_identity=difference > 0.0,
            max_abs_difference=difference,
        )
        checks.append(check)

        logger.info(
            "%-22s params_reproducible=%s tensor_reproducible=%s max|T(x)-x|=%.6f",
            check.name, check.params_reproducible, check.tensor_reproducible,
            check.max_abs_difference,
        )
        if is_identity and check.differs_from_identity:
            logger.error(
                "'%s' is declared as the identity control but changes the tensor "
                "(max difference %.6g).", check.name, check.max_abs_difference,
            )
            check.tensor_reproducible = False  # force a non-zero exit
        if not is_identity and not check.differs_from_identity:
            logger.warning(
                "'%s' leaves every sample unchanged; its configured strength makes it a "
                "second control arm rather than a transformation.", check.name,
            )

    failed = [check.name for check in checks if not check.ok]
    if failed:
        raise RuntimeError(
            "Determinism verification failed for: " + ", ".join(failed)
        )
    logger.info("All %d transformation(s) are deterministic per sample.", len(checks))
    return checks


def _stack(dataset) -> torch.Tensor:
    """Materialise a whole small dataset into one tensor."""
    return torch.stack([dataset[index]["image"] for index in range(len(dataset))])
