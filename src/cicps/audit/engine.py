"""The Experiment 0A audit engine.

Executes exactly the defined procedure and nothing else:

1. Evaluate every internal-validation image ONCE in its original (canonical)
   form, recording probabilities and penultimate features.
2. From the ORIGINAL probabilities only, select and freeze the hardest competing
   class ``c(x) = argmax_{k != y} p(k|x)`` and compute ``M_yc(x)``.
3. For each configured transformation, evaluate ``T(x)`` in batches, score it
   against the SAME frozen pair ``(y, c(x))``, and compute
   ``delta_margin = M_yc(T(x)) - M_yc(x)`` plus the supporting metrics.
4. Emit one record per (image, transformation).

The original representation is computed once and reused for every
transformation; the frozen competitor is never recomputed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from ..config import Config, ConfigError
from ..data.cub import CubMetadata
from ..data.datasets import CubImageDataset, DeterministicTransformDataset
from ..logging_utils import get_logger
from ..models.factory import FeatureClassifier
from ..transforms.base import DeterministicTransform
from ..transforms.pipeline import (
    build_canonical_view,
    build_eval_transform,
    build_normalizer,
)
from .competitor import OriginalState, compute_original_state, pairwise_margin
from .inference import InferenceResult, build_audit_loader, run_inference
from .metrics import MetricContext, compute_supporting_metrics
from .records import encode_params, order_columns, validate_schema

__all__ = ["AuditEngine", "AuditResult"]

logger = get_logger(__name__)


@dataclass
class AuditResult:
    """Output of one audit run."""

    records: pd.DataFrame
    """One row per (validation image, transformation)."""

    originals: pd.DataFrame
    """One row per validation image: frozen competitor, original margin, ..."""

    image_ids: tuple[int, ...]
    """Audited validation image ids, in evaluation order."""

    transform_names: tuple[str, ...]
    """Audited transformation names, in configuration order."""


class AuditEngine:
    """Runs the transformation audit against a frozen baseline model."""

    def __init__(
        self,
        config: Config,
        metadata: CubMetadata,
        model: FeatureClassifier,
        transforms: Sequence[DeterministicTransform],
        device: torch.device,
    ) -> None:
        if not transforms:
            raise ConfigError("The audit needs at least one transformation.")
        self.config = config
        self.metadata = metadata
        self.model = model
        self.transforms = list(transforms)
        self.device = device

        self.param_seed = config.seeded("audit.transform_param_seed")
        self.progress = bool(config.get("audit.progress"))
        self.amp_enabled = bool(config.get("audit.amp.enabled"))
        self.amp_dtype = getattr(torch, str(config.get("audit.amp.dtype")))

        self.canonical_view = build_canonical_view(config)
        self.normalizer = build_normalizer(config)

    # -- public API --------------------------------------------------------

    def run(self, image_ids: Sequence[int]) -> AuditResult:
        """Audit ``image_ids`` (the internal validation subset) end to end."""
        image_ids = self._limit(image_ids)
        logger.info(
            "Auditing %d validation image(s) x %d transformation(s) = %d record(s).",
            len(image_ids), len(self.transforms), len(image_ids) * len(self.transforms),
        )

        original = self._evaluate_original(image_ids)
        logger.info(
            "Original pass: top-1 accuracy %.4f, mean margin %.4f (median %.4f).",
            original.accuracy, float(original.margin.mean()), float(np.median(original.margin)),
        )

        frames = [self._audit_transform(transform, original) for transform in self.transforms]
        records = pd.concat(frames, ignore_index=True)
        records = order_columns(records)
        validate_schema(records)

        return AuditResult(
            records=records,
            originals=self._originals_frame(original),
            image_ids=tuple(int(image_id) for image_id in original.image_ids),
            transform_names=tuple(transform.name for transform in self.transforms),
        )

    # -- stages ------------------------------------------------------------

    def _limit(self, image_ids: Sequence[int]) -> list[int]:
        """Apply the optional ``audit.max_images`` cap (smoke tests only)."""
        ordered = list(image_ids)
        cap = self.config.get("audit.max_images", None)
        if cap is None:
            return ordered
        cap = int(cap)
        if cap <= 0:
            raise ConfigError(f"audit.max_images must be positive or null, got {cap}.")
        if cap < len(ordered):
            logger.warning(
                "audit.max_images=%d truncates the audit to the first %d of %d validation "
                "images. This is a smoke-test setting; use null for a full run.",
                cap, cap, len(ordered),
            )
            return ordered[:cap]
        return ordered

    def _evaluate_original(self, image_ids: Sequence[int]) -> OriginalState:
        """Single pass over the untransformed canonical views."""
        dataset = CubImageDataset(self.metadata, image_ids, build_eval_transform(self.config))
        loader = build_audit_loader(dataset, self.config, self.device)
        result = self._infer(loader, "original")
        return compute_original_state(
            image_ids=result.image_ids,
            true_labels=result.labels,
            probs=result.probs,
            features=result.features,
        )

    def _audit_transform(
        self, transform: DeterministicTransform, original: OriginalState
    ) -> pd.DataFrame:
        """Evaluate one transformation and build its records."""
        dataset = DeterministicTransformDataset(
            metadata=self.metadata,
            image_ids=[int(image_id) for image_id in original.image_ids],
            canonical_view=self.canonical_view,
            normalizer=self.normalizer,
            transform=transform,
            param_seed=self.param_seed,
        )
        loader = build_audit_loader(dataset, self.config, self.device)
        result = self._infer(loader, transform.name).aligned_to(original.image_ids)

        if not np.array_equal(result.labels, original.true_labels):
            raise RuntimeError(
                f"Label mismatch between the original and '{transform.name}' passes; "
                "the audit would be comparing different images."
            )

        rows = np.arange(len(original))
        # The competitor comes from `original`, never from `result` - this is the
        # line that keeps the audit faithful to the experiment's definition.
        augmented_true_prob = result.probs[rows, original.true_labels].astype(np.float64)
        augmented_competitor_prob = (
            result.probs[rows, original.competitor_labels].astype(np.float64)
        )
        augmented_margin = pairwise_margin(augmented_true_prob, augmented_competitor_prob)
        delta_margin = augmented_margin - original.margin

        context = MetricContext(
            transform_name=transform.name,
            original=original,
            augmented_probs=result.probs,
            augmented_features=result.features,
            augmented_predictions=result.predictions,
            augmented_true_prob=augmented_true_prob,
            augmented_competitor_prob=augmented_competitor_prob,
            augmented_margin=augmented_margin,
            delta_margin=delta_margin,
        )
        supporting = compute_supporting_metrics(context)

        frame = pd.DataFrame(
            {
                "image_id": original.image_ids,
                "image_path": [self.metadata.by_id(int(i)).relative_path
                               for i in original.image_ids],
                "true_class": original.true_labels + 1,
                "true_label": original.true_labels,
                "true_class_name": [self.metadata.class_name(label=int(label))
                                    for label in original.true_labels],
                "hard_competitor": original.competitor_labels + 1,
                "hard_competitor_label": original.competitor_labels,
                "hard_competitor_name": [self.metadata.class_name(label=int(label))
                                         for label in original.competitor_labels],
                "original_true_prob": original.true_prob,
                "original_competitor_prob": original.competitor_prob,
                "original_margin": original.margin,
                "original_prediction": original.predictions + 1,
                "transform": transform.name,
                "transform_type": transform.type_name,
                "transform_params": [
                    encode_params(dataset.resolve_params(int(image_id)))
                    for image_id in original.image_ids
                ],
                "augmented_true_prob": augmented_true_prob,
                "augmented_competitor_prob": augmented_competitor_prob,
                "augmented_margin": augmented_margin,
                "augmented_prediction": result.predictions + 1,
                "delta_margin": delta_margin,
                **supporting,
            }
        )
        logger.info(
            "%-22s mean dM %+.5f | median dM %+.5f | frac dM<0 %.3f | mean cos %.4f | "
            "pred-consistency %.3f | acc %.4f -> %.4f",
            transform.name,
            float(delta_margin.mean()),
            float(np.median(delta_margin)),
            float((delta_margin < 0).mean()),
            float(np.asarray(supporting["feature_cosine"]).mean()),
            float(np.asarray(supporting["prediction_consistent"]).mean()),
            original.accuracy,
            float(np.asarray(supporting["correct_augmented"]).mean()),
        )
        return frame

    def _originals_frame(self, original: OriginalState) -> pd.DataFrame:
        """Per-image frozen originals, persisted alongside the audit records."""
        return pd.DataFrame(
            {
                "image_id": original.image_ids,
                "image_path": [self.metadata.by_id(int(i)).relative_path
                               for i in original.image_ids],
                "true_class": original.true_labels + 1,
                "true_label": original.true_labels,
                "true_class_name": [self.metadata.class_name(label=int(label))
                                    for label in original.true_labels],
                "hard_competitor": original.competitor_labels + 1,
                "hard_competitor_label": original.competitor_labels,
                "hard_competitor_name": [self.metadata.class_name(label=int(label))
                                         for label in original.competitor_labels],
                "original_true_prob": original.true_prob,
                "original_competitor_prob": original.competitor_prob,
                "original_margin": original.margin,
                "original_prediction": original.predictions + 1,
                "correct_original": original.correct,
            }
        )

    def _infer(self, loader, description: str) -> InferenceResult:
        return run_inference(
            self.model,
            loader,
            self.device,
            amp_enabled=self.amp_enabled,
            amp_dtype=self.amp_dtype,
            progress=self.progress,
            description=description,
        )
