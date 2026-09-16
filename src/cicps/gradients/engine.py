"""The Experiment 1A gradient-compatibility engine.

For one frozen checkpoint :math:`\\theta`, one image :math:`(x, y)` and one
transformation :math:`T`, the engine computes

.. math::

    g_x = \\nabla_\\theta L(x, y), \\qquad g_T = \\nabla_\\theta L(T(x), y),

    C_g = \\frac{\\langle g_x, g_T \\rangle}{\\lVert g_x \\rVert \\lVert g_T \\rVert},
    \\qquad R_g = \\frac{\\lVert g_T \\rVert}{\\lVert g_x \\rVert},

alongside the zeroth-order quantities Experiment 0A measured on the same image -
the pairwise margin against the frozen hardest competitor, and its change.

Four properties of this procedure are load-bearing.

**The checkpoint is frozen in value, not in differentiability.** Gradients are
taken *with respect to* the parameters; no optimizer is constructed, ``step()``
is never called, and the model's full state (parameters *and* buffers) is
fingerprinted before and after every stage to prove nothing moved.

**``eval()`` mode, deliberately.** ResNet-18 contains BatchNorm and no dropout.
In ``train()`` mode a single-sample gradient would have BatchNorm normalise the
image by its own statistics, which is degenerate, would make the result depend on
what else happened to share the batch, and would mutate the running statistics.
In ``eval()`` mode the frozen running statistics are used, so the clean and
transformed passes differ by exactly one thing: the pixels. The price is that the
measured gradient is the eval-mode gradient rather than literally the gradient a
training step would have taken; that is recorded as a limitation, not hidden.

**Per-sample, and genuinely paired.** Every comparison is between two views of
the *same* image at the *same* checkpoint. The clean pass is computed once per
image and reused across every transformation, exactly as Experiment 0A reuses its
original pass, so a transformation arm can never be compared against a
differently-computed clean reference.

**float32 throughout, no autocast.** The gradient is the measurement instrument
here, as the probability was in Experiment 0A. Reductions accumulate in float64.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import Config, ConfigError
from ..data.cub import CubMetadata
from ..logging_utils import get_logger
from ..models.factory import FeatureClassifier
from ..seeding import derive_run_seed, worker_init_fn
from ..training.optim import build_loss
from ..transforms.base import DeterministicTransform
from ..transforms.pipeline import build_canonical_view, build_normalizer
from .accumulate import GradientVector, capture_gradients, compare, scope_norms
from .checkpoints import CheckpointStage, load_stage_model, state_fingerprint
from .datasets import PairedViewDataset, collate_paired
from .records import order_records, scope_column, validate_records
from .scopes import ScopeSet, build_scopes

__all__ = ["GradientEngine", "GradientResult"]

logger = get_logger(__name__)


@dataclass
class GradientResult:
    """Output of one Experiment 1A run."""

    records: pd.DataFrame
    """One row per (stage, transformation, realisation, image)."""

    clean: pd.DataFrame
    """One row per (stage, image): the shared clean pass and the null control."""

    stages: tuple[CheckpointStage, ...]
    """Stages actually measured, enriched with epoch and checkpoint metrics."""

    scopes: dict
    """Serialisable description of the measured parameter scopes."""

    warnings: tuple[str, ...]
    """Non-fatal anomalies worth carrying into the manifest."""


class GradientEngine:
    """Measures gradient compatibility across checkpoints and transformations."""

    def __init__(
        self,
        config: Config,
        metadata: CubMetadata,
        transforms: Sequence[DeterministicTransform],
        device: torch.device,
    ) -> None:
        if not transforms:
            raise ConfigError("Experiment 1A needs at least one transformation.")
        self.config = config
        self.metadata = metadata
        self.transforms = {transform.name: transform for transform in transforms}
        self.transform_order = [transform.name for transform in transforms]
        self.device = device

        self.param_seed = config.seeded("gradients.sampling.param_seed")
        self.realisations = int(config.get("gradients.sampling.realisations"))
        self.batch_size = int(config.get("gradients.batch_size"))
        self.num_workers = int(config.get("gradients.num_workers"))
        self.progress = bool(config.get("gradients.progress"))
        self.null_enabled = bool(config.get("gradients.null_control.enabled"))

        self.criterion: nn.Module = build_loss(config)
        self.canonical_view = build_canonical_view(config)
        self.normalizer = build_normalizer(config)
        self._warnings: list[str] = []

    # -- public API --------------------------------------------------------

    def run(self, image_ids: Sequence[int], stages: Sequence[CheckpointStage]) -> GradientResult:
        """Measure every stage over ``image_ids``; return the assembled tables."""
        ordered_ids = self._order_images(image_ids)
        logger.info(
            "Experiment 1A: %d image(s) x %d stage(s), %d realisation(s) per transformation.",
            len(ordered_ids), len(stages), self.realisations,
        )

        record_frames: list[pd.DataFrame] = []
        clean_frames: list[pd.DataFrame] = []
        measured: list[CheckpointStage] = []
        scopes_described: dict = {}

        for index, stage in enumerate(stages, start=1):
            logger.info("=== stage %d/%d: %s === %s", index, len(stages), stage.name,
                        stage.description)
            model, enriched = load_stage_model(stage, self.config, self.device)
            scopes = build_scopes(model, self.config)
            if not scopes_described:
                scopes_described = scopes.describe()

            before = state_fingerprint(model)
            records, clean = self._measure_stage(model, enriched, scopes, ordered_ids)
            after = state_fingerprint(model)
            if before != after:
                raise RuntimeError(
                    f"Stage '{stage.name}': the model state changed during measurement "
                    f"({before} -> {after}). Experiment 1A must never modify a checkpoint."
                )
            logger.info(
                "Stage '%s': model state verified unchanged (fingerprint %s).",
                stage.name, before,
            )

            record_frames.append(records)
            clean_frames.append(clean)
            measured.append(enriched)
            del model
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        records = order_records(pd.concat(record_frames, ignore_index=True))
        validate_records(records)
        return GradientResult(
            records=records,
            clean=pd.concat(clean_frames, ignore_index=True),
            stages=tuple(measured),
            scopes=scopes_described,
            warnings=tuple(self._warnings),
        )

    # -- one stage ---------------------------------------------------------

    def _measure_stage(
        self,
        model: FeatureClassifier,
        stage: CheckpointStage,
        scopes: ScopeSet,
        image_ids: Sequence[int],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run the full paired measurement for one checkpoint stage."""
        names = list(stage.transforms) if stage.transforms else list(self.transform_order)
        transforms = [self.transforms[name] for name in names]
        logger.info("Stage '%s': measuring %d transformation(s): %s",
                    stage.name, len(transforms), ", ".join(names))

        dataset = PairedViewDataset(
            metadata=self.metadata,
            image_ids=image_ids,
            canonical_view=self.canonical_view,
            normalizer=self.normalizer,
            transforms=transforms,
            param_seed=self.param_seed,
            realisations=self.realisations,
        )
        loader = self._build_loader(dataset)
        parameters = [p for p in model.parameters() if p.requires_grad]

        rows: list[dict] = []
        clean_rows: list[dict] = []
        previous: tuple[int, int, GradientVector] | None = None
        degenerate_count = 0

        for batch in self._maybe_progress(loader, stage.name):
            clean_images = batch["clean"].to(self.device, non_blocking=True)
            views = batch["views"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            for position, image_id in enumerate(batch["image_ids"]):
                label = int(labels[position])
                primary = self._primary_index(scopes)
                clean_state, clean_gradient = self._clean_pass(
                    model, parameters, clean_images[position], label
                )

                null_cosine = float("nan")
                null_partner = -1
                null_same_class = False
                if self.null_enabled and previous is not None:
                    partner_id, partner_label, partner_gradient = previous
                    null_cosine = compare(
                        partner_gradient, clean_gradient, scopes
                    ).cosine[primary]
                    null_partner = partner_id
                    null_same_class = partner_label == label

                clean_rows.append({
                    "checkpoint": stage.name,
                    "checkpoint_kind": stage.kind,
                    "checkpoint_epoch": stage.epoch,
                    "image_id": int(image_id),
                    "true_class": label + 1,
                    "true_label": label,
                    "hard_competitor": clean_state["competitor"] + 1,
                    "hard_competitor_label": clean_state["competitor"],
                    "clean_loss": clean_state["loss"],
                    "clean_true_prob": clean_state["true_prob"],
                    "clean_competitor_prob": clean_state["competitor_prob"],
                    "clean_margin": clean_state["margin"],
                    "clean_prediction": clean_state["prediction"] + 1,
                    "correct_clean": clean_state["correct"],
                    "clean_grad_norm": scope_norms(clean_gradient, scopes)[primary],
                    "null_cosine": null_cosine,
                    "null_partner_image_id": null_partner,
                    "null_partner_same_class": null_same_class,
                })

                for slot in range(dataset.num_views):
                    statistics, transformed = self._transformed_pass(
                        model, parameters, views[position, slot], label,
                        clean_gradient, clean_state, scopes,
                    )
                    if statistics.degenerate:
                        degenerate_count += 1
                    rows.append(self._row(
                        stage, scopes, dataset, batch, position, slot,
                        image_id, label, clean_state, transformed, statistics,
                    ))

                if self.null_enabled:
                    previous = (int(image_id), label, clean_gradient)

        if degenerate_count:
            message = (
                f"Stage '{stage.name}': {degenerate_count} pair(s) had a numerically zero "
                "gradient in the primary scope; their cosine and norm ratio are NaN."
            )
            logger.warning("%s", message)
            self._warnings.append(message)

        records = pd.DataFrame(rows)
        self._log_stage_summary(stage, records, scopes)
        return records, pd.DataFrame(clean_rows)

    # -- passes ------------------------------------------------------------

    def _clean_pass(
        self, model: FeatureClassifier, parameters: list, image: torch.Tensor, label: int
    ) -> tuple[dict, GradientVector]:
        """Forward/backward the canonical view; freeze the hardest competitor.

        The competitor is selected from the clean image at *this* checkpoint,
        following Experiment 0A's rule, and is then held fixed for every
        transformation of this image - so ``delta_margin`` compares like with
        like within a stage.
        """
        target = torch.tensor([label], device=self.device)
        model.zero_grad(set_to_none=True)
        logits, features = model(image.unsqueeze(0), return_features=True)
        loss = self.criterion(logits, target)
        loss.backward()
        gradient = capture_gradients(parameters, loss)

        probabilities = torch.softmax(logits.detach().float(), dim=1)[0]
        masked = probabilities.clone()
        masked[label] = -float("inf")
        competitor = int(masked.argmax())
        prediction = int(probabilities.argmax())
        true_prob = float(probabilities[label])
        competitor_prob = float(probabilities[competitor])

        state = {
            "loss": float(loss.detach()),
            "competitor": competitor,
            "true_prob": true_prob,
            "competitor_prob": competitor_prob,
            "margin": true_prob - competitor_prob,
            "prediction": prediction,
            "correct": prediction == label,
            "features": features.detach().float()[0],
        }
        return state, gradient

    def _transformed_pass(
        self,
        model: FeatureClassifier,
        parameters: list,
        view: torch.Tensor,
        label: int,
        clean_gradient: GradientVector,
        clean_state: dict,
        scopes: ScopeSet,
    ):
        """Forward/backward one transformed view and compare it to the clean one."""
        target = torch.tensor([label], device=self.device)
        model.zero_grad(set_to_none=True)
        logits, features = model(view.unsqueeze(0), return_features=True)
        loss = self.criterion(logits, target)
        loss.backward()
        gradient = capture_gradients(parameters, loss)

        probabilities = torch.softmax(logits.detach().float(), dim=1)[0]
        competitor = clean_state["competitor"]
        true_prob = float(probabilities[label])
        competitor_prob = float(probabilities[competitor])
        prediction = int(probabilities.argmax())

        transformed = {
            "loss": float(loss.detach()),
            "true_prob": true_prob,
            "competitor_prob": competitor_prob,
            "margin": true_prob - competitor_prob,
            "prediction": prediction,
            "correct": prediction == label,
            "feature_cosine": float(
                torch.nn.functional.cosine_similarity(
                    clean_state["features"].unsqueeze(0),
                    features.detach().float(),
                    dim=1,
                )[0]
            ),
        }
        statistics = compare(clean_gradient, gradient, scopes)
        return statistics, transformed

    # -- row assembly ------------------------------------------------------

    def _row(
        self, stage, scopes, dataset, batch, position, slot,
        image_id, label, clean_state, transformed, statistics,
    ) -> dict:
        primary = self._primary_index(scopes)
        transform_name = dataset.view_transform[slot]
        row = {
            "checkpoint": stage.name,
            "checkpoint_kind": stage.kind,
            "checkpoint_epoch": stage.epoch,
            "checkpoint_policy": stage.policy or "",
            "image_id": int(image_id),
            "image_path": self.metadata.by_id(int(image_id)).relative_path,
            "true_class": label + 1,
            "true_label": label,
            "true_class_name": self.metadata.class_name(label=label),
            "hard_competitor": clean_state["competitor"] + 1,
            "hard_competitor_label": clean_state["competitor"],
            "transform": transform_name,
            "transform_type": self.transforms[transform_name].type_name,
            "realisation": dataset.view_realisation[slot],
            "transform_params": batch["params"][position][slot],
            "clean_loss": clean_state["loss"],
            "transformed_loss": transformed["loss"],
            "delta_loss": transformed["loss"] - clean_state["loss"],
            "clean_true_prob": clean_state["true_prob"],
            "transformed_true_prob": transformed["true_prob"],
            "clean_competitor_prob": clean_state["competitor_prob"],
            "transformed_competitor_prob": transformed["competitor_prob"],
            "clean_margin": clean_state["margin"],
            "transformed_margin": transformed["margin"],
            "delta_margin": transformed["margin"] - clean_state["margin"],
            "grad_cosine": statistics.cosine[primary],
            "grad_norm_ratio": statistics.norm_ratio[primary],
            "clean_grad_norm": statistics.norm_left[primary],
            "transformed_grad_norm": statistics.norm_right[primary],
            "grad_degenerate": statistics.degenerate,
            "feature_cosine": transformed["feature_cosine"],
            "prediction_consistent": transformed["prediction"] == clean_state["prediction"],
            "correct_clean": clean_state["correct"],
            "correct_transformed": transformed["correct"],
        }
        for index, name in enumerate(scopes.names):
            row[scope_column("cosine", name)] = statistics.cosine[index]
            row[scope_column("norm_ratio", name)] = statistics.norm_ratio[index]
            row[scope_column("clean_norm", name)] = statistics.norm_left[index]
            row[scope_column("transformed_norm", name)] = statistics.norm_right[index]
        return row

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _primary_index(scopes: ScopeSet) -> int:
        return scopes.names.index(scopes.primary)

    def _order_images(self, image_ids: Sequence[int]) -> list[int]:
        """Apply the optional cap, then permute for the cross-image null control.

        The null control pairs each image's clean gradient with the previously
        processed image's. Traversing the validation ids in their natural order
        would make that partner almost always a *same-class* neighbour, because
        CUB ids are grouped by species - which would measure something quite
        different from "an unrelated image". A seeded permutation removes that
        structure without adding a single backward pass; the partner actually
        used is recorded per row either way.
        """
        ordered = [int(image_id) for image_id in image_ids]
        cap = self.config.get("gradients.max_images", None)
        if cap is not None:
            cap = int(cap)
            if cap <= 0:
                raise ConfigError(f"gradients.max_images must be positive or null, got {cap}.")
            if cap < len(ordered):
                logger.warning(
                    "gradients.max_images=%d truncates the measurement to %d of %d validation "
                    "images. This is a smoke-test setting; use null for a scientific run.",
                    cap, cap, len(ordered),
                )
                ordered = ordered[:cap]

        seed = derive_run_seed(int(self.config.get("seed.value")), "gradient_image_order")
        rng = Random(seed)
        permuted = list(ordered)
        rng.shuffle(permuted)
        logger.info(
            "Image traversal order permuted with seed %d (for the cross-image null control); "
            "the paired clean/transformed measurement is unaffected.", seed,
        )
        return permuted

    def _build_loader(self, dataset: PairedViewDataset) -> DataLoader:
        kwargs = {
            "batch_size": self.batch_size,
            "shuffle": False,
            "num_workers": self.num_workers,
            "pin_memory": bool(self.config.get("gradients.pin_memory"))
            and self.device.type == "cuda",
            "drop_last": False,
            "collate_fn": collate_paired,
        }
        if self.num_workers > 0 and bool(self.config.get("seed.seed_dataloader_workers")):
            kwargs["worker_init_fn"] = worker_init_fn
        return DataLoader(dataset, **kwargs)

    def _maybe_progress(self, loader: DataLoader, description: str):
        if not self.progress:
            return loader
        try:
            from tqdm.auto import tqdm
        except ImportError:  # pragma: no cover - tqdm is optional
            return loader
        return tqdm(loader, desc=description, leave=False)

    def _log_stage_summary(self, stage: CheckpointStage, records: pd.DataFrame,
                           scopes: ScopeSet) -> None:
        if records.empty:
            return
        for name in records["transform"].drop_duplicates():
            group = records[records["transform"] == name]
            logger.info(
                "%-14s %-22s mean Cg %+.4f | median Cg %+.4f | mean Rg %.4f | "
                "mean dM %+.5f | mean clean loss %.4f",
                stage.name, name,
                float(np.nanmean(group["grad_cosine"])),
                float(np.nanmedian(group["grad_cosine"])),
                float(np.nanmean(group["grad_norm_ratio"])),
                float(group["delta_margin"].mean()),
                float(group["clean_loss"].mean()),
            )
