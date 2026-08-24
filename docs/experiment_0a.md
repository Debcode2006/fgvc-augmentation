# Experiment 0A — The Transformation Audit

**A falsification / mechanism test on CUB-200-2011.**

---

## 1. Experiment title

**Experiment 0A — Transformation Audit: measuring augmentation-induced change in
fine-grained discriminative separation.**

---

## 2. Research motivation

Standard image augmentation is applied to fine-grained recognition datasets as if
every transformation were equally harmless. But fine-grained classification does
not hinge on global appearance — it hinges on small, localised, *discriminative*
evidence: the exact shade of a throat patch, a wing-bar, the shape of a bill.

That kind of evidence is plausibly fragile in a way global appearance is not. A
crop that removes a wing-bar, a blur that erases fine barring, or a colour jitter
that shifts a diagnostic hue may leave an image that still obviously depicts *a
bird*, and still obviously depicts *roughly that kind of bird*, while destroying
exactly the evidence that separates the true species from its nearest look-alike.

If that is true, augmentation policies chosen on global heuristics are silently
destroying label-relevant signal, and there is room for an augmentation method
that is aware of what it is destroying. Before proposing such a method, the
underlying phenomenon has to be shown to exist and to be measurable.

## 3. Research question

The programme-level question is:

> Can the preservation of fine-grained discriminative evidence under image
> augmentation be quantified?

The narrower question this experiment answers is:

> **Do different image transformations cause measurably different changes in an
> image's discriminative separation from its closest competing class?**

## 4. What this experiment IS testing

* Whether a *pairwise* discriminative quantity — the probability margin between
  an image's true class and its single hardest competing class — moves
  measurably when a transformation is applied.
* Whether that movement **differs across transformations**.
* Whether the effect is measurable at all with a plain, honestly-trained
  baseline model and no special machinery.

This is a **mechanism test**, and it is designed to be **falsifiable**: it can
come back negative (see §33–34).

## 5. What this experiment is NOT testing

* It is **not** DPGA (Discriminative-Preservation-Guided Augmentation). No
  augmentation is selected, ranked, filtered or weighted by anything measured
  here.
* It is **not** Experiment 0B. No model is trained under a different
  augmentation policy, and no downstream accuracy comparison is made.
* It is **not** a claim that any transformation is "good" or "bad". The
  transformation list is deliberately unordered and the analysis makes no
  ranking claim.
* It is **not** a benchmark. The baseline exists to be an instrument, not a
  result. Nothing here should be tuned to raise accuracy.
* It does **not** establish causality between margin damage and downstream
  accuracy. That link is future work.

---

## 6. Dataset

**CUB-200-2011** (Caltech-UCSD Birds-200-2011).

| Property | Value |
| --- | --- |
| Images | 11,788 |
| Classes (species) | 200 |
| Official train | 5,994 |
| Official test | 5,794 |
| Task | Fine-grained classification |

The dataset is used **exactly as distributed**. Nothing is copied, moved, or
re-foldered into `train/` and `test/` directories. The official metadata files
are the only authority on labels and splits.

## 7. Dataset directory structure

```
data/CUB_200_2011/
├── images/
│   ├── 001.Black_footed_Albatross/
│   │   ├── Black_Footed_Albatross_0046_18.jpg
│   │   └── ...
│   └── ... (200 species directories)
├── images.txt                 # <image_id> <relative_path>
├── image_class_labels.txt     # <image_id> <class_id>          (class_id ∈ 1..200)
├── train_test_split.txt       # <image_id> <is_official_train>  (1 = train, 0 = test)
├── classes.txt                # <class_id> <class_name>
├── bounding_boxes.txt         # unused by Experiment 0A
├── parts/                     # unused by Experiment 0A (relevant to future work)
├── attributes/                # unused by Experiment 0A
└── README
```

Only `images.txt`, `image_class_labels.txt`, `train_test_split.txt`,
`classes.txt` and the image files themselves are read.

## 8. Official train/test split

`train_test_split.txt` partitions the 11,788 images into 5,994 official-train
and 5,794 official-test images. Experiment 0A reads this file and treats it as
final.

## 9. Internal train/validation split

The internal split is carved **out of the official train portion only**:

* ≈ **90 %** internal train → 5,394 images
* ≈ **10 %** internal validation → 600 images
* **Stratified by species**: all 200 classes appear in both subsets.
* **Deterministic**: each class draws from its own RNG keyed by
  `(seed, "internal_split", class_id)`, so the assignment of one class does not
  depend on any other class.
* **Seeded**: master seed `42` by default (`seed.value`, overridable by
  `split.seed`).
* **Persisted**: written to
  `outputs/experiment_0a/splits/internal_split.json` and reused verbatim by
  every later stage and every later experiment.

Each class contributes `min(max(1, round(n_class × val_fraction)), n_class − 1)`
validation images, which guarantees at least one train and one validation image
per class. With CUB's 29–30 images per class this yields exactly 3 validation
images per class → 600 total.

The manifest stores a **fingerprint** of the population and the split settings.
Loading a manifest whose fingerprint disagrees with the current dataset or
configuration is a hard error, not a warning — otherwise a checkpoint could
silently be audited on a validation set it was trained on.

## 10. Why the test set is locked

The official test set is reserved for the *final* evaluation of the eventual
method. It must therefore not influence any intermediate decision. In
Experiment 0A it is never used for:

training · validation · threshold selection · augmentation analysis ·
transformation ranking · hyper-parameter selection · debugging by performance

The dataset validation stage explicitly checks that no official-test image id
entered the internal split, and reports the locked count in the log.

---

## 11. Baseline architecture

| Component | Choice |
| --- | --- |
| Backbone | ResNet-18, ImageNet-pretrained (`IMAGENET1K_V1`) |
| Head | Single linear layer, 512 → 200 |
| Input | 224 × 224 RGB |
| Loss | Cross-entropy |
| Features | Penultimate = post-global-average-pool, 512-d |

Deliberately excluded: Swin, ViT, custom FGVC architectures, contrastive or
metric learning, retrieval, prototypes, attention modules, Mixup, CutMix, DPGA.

**The model is an instrument, not a contribution.** Its only job is to provide a
stable discriminative probability and feature representation against which
augmentation-induced change can be measured. Making it stronger would not make
Experiment 0A more informative.

`FeatureClassifier` separates the trunk from the head so that logits and the
penultimate representation come out of **one** forward pass — the audit never
runs the backbone twice.

## 12. Training configuration

Defaults (all live in `config/experiment_0a.yaml`, none in Python):

| Setting | Default |
| --- | --- |
| Optimizer | AdamW, lr 3e-4, weight decay 0.05 |
| Schedule | Cosine decay, 1 warmup epoch, min lr 1e-6 |
| Epochs | 20 |
| Batch size | 32 (eval 64) |
| Mixed precision | fp16 autocast, enabled on CUDA |
| Training augmentation | RandomResizedCrop(scale 0.7–1.0) + HorizontalFlip(p 0.5) |
| Seed | 42 |
| Checkpoint selection | Best `val_top1` |

The training-time augmentation is the ordinary recipe used to fit the baseline.
It is **unrelated** to the audited transformation suite: the audit measures a
*frozen* model, and this only decides how that model got there.

### Reference run

For calibration when reproducing: on an RTX 3050 Laptop (4 GB) the defaults
above train in ≈ 7 minutes (≈ 20 s/epoch, `num_workers: 4`) and reach
**76.2 % internal-validation top-1 / 93.5 % top-5** at epoch 17. The full audit
over 600 × 7 takes ≈ 7 minutes. A materially lower baseline accuracy suggests
something is wrong with the split or the preprocessing; the audit itself does
not require any particular accuracy, but a near-random model would make the
margins uninformative.

---

## 13. Exact transformation list

The audit measures exactly seven transformations:

| # | Name | Type | Default parameters |
| --- | --- | --- | --- |
| 1 | `identity` | `identity` | — (control arm) |
| 2 | `horizontal_flip` | `horizontal_flip` | always applied |
| 3 | `color_jitter` | `color_jitter` | brightness/contrast/saturation 0.4, hue 0.1 |
| 4 | `rotation` | `rotation` | ±15° |
| 5 | `random_resized_crop` | `random_resized_crop` | scale 0.5–0.9, ratio 0.75–1.333 |
| 6 | `gaussian_blur` | `gaussian_blur` | kernel 9, sigma 1.0–2.0 |
| 7 | `random_erasing` | `random_erasing` | scale 0.05–0.2, ratio 0.3–3.3 |

**No ordering of "safe" versus "destructive" is assumed.** Discovering whether
such an ordering exists is the point of the experiment.

### Where transformations are applied

```
raw JPEG
  → Resize(256) → CenterCrop(224)      ← the "canonical view"
  → T.apply_pil(...)                    ← flip / jitter / rotate / crop / blur
  → ToTensor → Normalize(ImageNet)
  → T.apply_tensor(...)                 ← random erasing
  → model
```

Applying transformations *on top of* the canonical view is what makes `identity`
a true no-op and makes every other arm differ from `identity` by exactly one
operation. `random_erasing` acts in normalised tensor space (matching
torchvision), so its `value: 0.0` means "the per-channel ImageNet mean colour".

## 14. Determinism requirements

A transformation in Experiment 0A must be a **fixed function of the sample**. If
`T` were re-sampled on every evaluation, `ΔM` would confound the effect of the
transformation with the effect of the random draw.

The contract is enforced structurally by splitting each transformation in two:

* **`resolve(rng)`** draws this sample's parameters from a `random.Random`
  seeded by `derive_seed(audit.transform_param_seed, image_id, transform_name)`
  — a BLAKE2b-based derivation, *not* Python's salted `hash()`. Same seed +
  same image id ⇒ same parameters, across runs, processes, machines, batch
  sizes and DataLoader worker counts.
* **`apply_pil` / `apply_tensor`** are pure functions of
  `(input, resolved parameters)` and never touch a global RNG.

Parameters vary *between* samples — a single fixed rotation angle for the whole
dataset would be a much weaker probe — but are frozen *for* each sample. The
resolved parameters of every row are stored in the `transform_params` column, so
any sample's exact transformation can be inspected after the fact.

Two implementation details support this:

* `ColorJitter` applies its four adjustments in a **fixed order**
  (brightness → contrast → saturation → hue), unlike torchvision, which permutes
  the order randomly.
* `HorizontalFlip` is **always** applied rather than applied with probability
  0.5, which would otherwise leave half the samples identical to `identity`.

The `verify` command checks all of this on real images (§25).

---

## 15. Original hard-competitor definition

For a validation image `x` with true label `y`, the hardest competing class is
taken from the **original image only**:

$$c(x) = \arg\max_{k \neq y} \; p(k \mid x)$$

`c(x)` is then **frozen** for every transformation of that image:

```
Original:  true class = A,  hard competitor = B
identity            → compare A vs B
horizontal_flip     → compare A vs B
color_jitter        → compare A vs B
rotation            → compare A vs B
random_resized_crop → compare A vs B
gaussian_blur       → compare A vs B
random_erasing      → compare A vs B
```

Even if the strongest incorrect class of `T(x)` becomes some class C, that does
**not** change `c(x)`. Re-selecting the competitor per transformation would
redefine the measured quantity and make `ΔM` incomparable across arms.

The true class is masked out before the arg-max, so the competitor is genuinely
the strongest *other* class whether or not the model's top-1 is correct.

## 16. Pairwise margin definition

$$M_{yc}(x) = p_y(x) - p_c(x)$$

where `p_y` is the softmax probability of the true class and `p_c` that of the
frozen hardest competitor. Softmax is always computed in **float32** — the
margins are the measurement instrument, and half precision would quantise small
margins away. For the same reason `audit.amp.enabled` defaults to `false`.

For a transformed image:

$$M_{yc}(T(x)) = p_y(T(x)) - p_c(T(x))$$

## 17. ΔM definition

$$\boxed{\;\Delta M = M_{yc}(T(x)) - M_{yc}(x)\;}$$

**This is the primary metric of Experiment 0A.** It is computed explicitly in
the audit engine and is deliberately *not* pluggable — feature cosine,
prediction consistency, entropy, top-1 confidence, generic feature distance and
KL divergence are all supporting metrics at best, and none of them may replace
it.

## 18. Interpretation of ΔM

| Observation | Reading |
| --- | --- |
| `ΔM ≈ 0` | Discriminative separation preserved |
| `ΔM < 0` | The transformation weakens separation |
| Strongly negative `ΔM` | Potentially destructive transformation |
| `ΔM > 0` | The transformation strengthens *apparent* separation |

Caveats that must accompany any reading:

* `ΔM > 0` means the model's separation grew, which is not the same as evidence
  being created. It may reflect the removal of a confusing region, or ordinary
  model noise.
* `identity` gives the **noise floor** of the whole measurement. With the
  pipeline as implemented it is exactly `0.0`, because identity is a bitwise
  no-op on the canonical view. Any non-zero spread on other arms is therefore
  attributable to the transformation, not to evaluation jitter.
* `ΔM` is bounded in `[-2, 2]` and is not scale-free: an image whose original
  margin is 0.01 cannot lose much, while one at 0.99 can. Per-class and
  margin-stratified analyses are future work.

## 19. Supporting metrics

Supporting evidence only — none of these is the novelty, and none replaces `ΔM`.

**A. Feature consistency.** With `z = f(x)` and `zT = f(T(x))` the penultimate
(post-pool, 512-d) representations:

$$C_f = \cos(z, z_T) = \frac{z \cdot z_T}{\lVert z \rVert \, \lVert z_T \rVert}$$

This says whether the representation moved — not whether the discriminative
evidence for `y` against `c(x)` survived.

**B. Prediction consistency.** $C_p = \mathbb{1}[\arg\max p(x) = \arg\max p(T(x))]$, stored as a boolean.

**C. Original correctness.** $\mathbb{1}[\arg\max p(x) = y]$.

**D. Transformed correctness.** $\mathbb{1}[\arg\max p(T(x)) = y]$.

Adding a new supporting metric: write a function mapping a `MetricContext` to
named columns, decorate it with `@register_metric("name")` in
`src/cicps/audit/metrics.py`, and the engine emits those columns automatically.

---

## 20. Exact dataframe schema

**One logical row per `validation image × transformation`.** With the default
configuration: 600 × 7 = **4,200 rows**.

Class identifiers appear in three forms so nothing downstream has to guess an
offset:

* `*_class` — official CUB class id, **1..200** (as in `classes.txt`)
* `*_label` — contiguous model index, **0..199** (`label = class_id − 1`)
* `*_class_name` — species name, e.g. `001.Black_footed_Albatross`

| Column | Type | Meaning |
| --- | --- | --- |
| `image_id` | int | Official CUB image id (from `images.txt`) |
| `image_path` | str | Path relative to `images/` |
| **`true_class`** | int | True species, CUB id 1..200 |
| `true_label` | int | True species, contiguous 0..199 |
| `true_class_name` | str | True species name |
| **`hard_competitor`** | int | Frozen `c(x)`, CUB id 1..200 |
| `hard_competitor_label` | int | Frozen `c(x)`, contiguous 0..199 |
| `hard_competitor_name` | str | Frozen competitor's species name |
| **`original_true_prob`** | float | `p_y(x)` |
| **`original_competitor_prob`** | float | `p_c(x)` |
| **`original_margin`** | float | `M_yc(x)` |
| `original_prediction` | int | `argmax_k p(k|x)`, CUB id |
| **`transform`** | str | Transformation name |
| `transform_type` | str | Registered implementation type |
| `transform_params` | str | JSON of this sample's frozen parameters |
| **`augmented_true_prob`** | float | `p_y(T(x))` |
| **`augmented_competitor_prob`** | float | `p_c(T(x))`, frozen `c(x)` |
| **`augmented_margin`** | float | `M_yc(T(x))` |
| `augmented_prediction` | int | `argmax_k p(k|T(x))`, CUB id |
| **`delta_margin`** | float | **`ΔM` — the primary metric** |
| **`feature_cosine`** | float | `C_f` |
| **`prediction_consistent`** | bool | `C_p` |
| **`correct_original`** | bool | `argmax p(x) == y` |
| **`correct_augmented`** | bool | `argmax p(T(x)) == y` |

Bold columns are the fields Experiment 0A mandates; `validate_schema()` enforces
their presence on every write and read. Extra columns may be added; none of the
mandated ones may be removed.

A second dataframe, `original_predictions.csv`, holds **one row per validation
image** with the frozen originals (competitor, probabilities, margin,
correctness) for convenience.

## 21. Pipeline / logic flow

```
ONE NORMAL BASELINE MODEL
        ↓
Official CUB training split                       (5,994 images)
        ↓
Internal ~90/10 stratified train/validation split (5,394 / 600)
        ↓
Train ImageNet-pretrained ResNet-18
        ↓
FREEZE the trained baseline                       (eval(), no_grad, requires_grad=False)
        ↓
For every validation image x:
        ↓
    Evaluate the ORIGINAL image        ──── one batched pass, computed ONCE
        ↓
    Identify true class y
        ↓
    Identify hardest competitor  c(x) = argmax_{k≠y} p(k|x)
        ↓
    FREEZE c(x)
        ↓
    For each of the 7 deterministic transformations T:
        ↓
        Evaluate T(x)                  ──── one batched pass per transformation
        ↓
        M(x)      = p_y(x)    − p_c(x)          (reused, not recomputed)
        M(T(x))   = p_y(T(x)) − p_c(T(x))       (frozen c(x))
        ΔM        = M(T(x)) − M(x)
        ↓
        Supporting metrics: C_f, C_p, correctness
        ↓
    Emit one dataframe row per image × transformation
        ↓
Summary statistics by transformation
        ↓
ΔM distribution plots
```

The original representation is computed **once** and reused across all seven
arms; only the transformed passes are repeated. Inference runs under
`torch.no_grad()` in `eval()` mode with configurable batch sizes, on GPU when
available and on CPU otherwise.

---

## 22. Repository code structure

```
cicps-augmentation/
├── config/
│   ├── experiment_0a.yaml       # THE source of truth for the experiment
│   └── smoke_0a.yaml            # override for a fast end-to-end plumbing test
├── data/CUB_200_2011/           # dataset, exactly as distributed
├── docs/experiment_0a.md        # this handbook
├── outputs/                     # generated artefacts (git-ignorable)
├── src/cicps/
│   ├── cli.py                   # argparse CLI; every command is config-driven
│   ├── __main__.py              # `python -m cicps`
│   ├── config.py                # YAML loading, dotted lookup, path resolution
│   ├── seeding.py               # global seeding + per-sample seed derivation
│   ├── logging_utils.py         # configuration-driven logging
│   ├── device.py                # device resolution with explicit CPU fallback
│   ├── data/
│   │   ├── cub.py               # official CUB metadata parsing
│   │   ├── splits.py            # deterministic stratified internal split
│   │   ├── datasets.py          # torch datasets (plain pipeline / deterministic T)
│   │   └── validation.py        # dataset + split integrity checks
│   ├── models/
│   │   ├── factory.py           # FeatureClassifier, build_model
│   │   └── checkpoint.py        # save/load, config-drift warnings
│   ├── transforms/
│   │   ├── base.py              # DeterministicTransform ABC + registry
│   │   ├── builtin.py           # the seven transformations
│   │   └── pipeline.py          # canonical view, normaliser, train pipeline
│   ├── training/
│   │   ├── optim.py             # optimizer / scheduler / loss factories
│   │   └── trainer.py           # baseline training loop
│   ├── audit/
│   │   ├── inference.py         # batched no_grad inference → probs + features
│   │   ├── competitor.py        # frozen competitor + pairwise margin
│   │   ├── metrics.py           # supporting-metric registry
│   │   ├── engine.py            # the audit engine
│   │   └── records.py           # dataframe schema + serialisation
│   ├── analysis/
│   │   ├── summary.py           # per-transformation descriptive statistics
│   │   └── plots.py             # ΔM distribution plots
│   └── stages/                  # prepare / verify / train / audit / analyze
├── pyproject.toml
└── requirements.txt
```

Separation of responsibility is deliberate: Experiment 0B and a future DPGA can
reuse the config, seeding, CUB parsing, split, model, checkpoint, inference,
transform-registry and metric-registry layers without touching Experiment 0A.

## 23. Configuration structure

`config/experiment_0a.yaml` is the **only** source of experiment configuration.
No path, size, hyper-parameter, transformation strength, seed, or output name is
hard-coded in Python. Missing keys raise a path-qualified `ConfigError` rather
than silently defaulting.

| Section | Controls |
| --- | --- |
| `experiment` | Name and description |
| `seed` | Master seed, cudnn flags, deterministic algorithms, worker seeding |
| `dataset` | CUB root, metadata filenames, expected counts, validation strictness |
| `split` | `val_fraction`, stratification, split seed, manifest path, overwrite |
| `model` | Architecture, pretrained flag, weights enum, `num_classes` |
| `preprocess` | `image_size`, `resize_size`, interpolation, normalisation |
| `train` | Epochs, batch sizes, workers, AMP, optimizer, scheduler, loss, training augmentation, checkpointing, history path |
| `audit` | Checkpoint, batch size, workers, AMP, `max_images`, param seed, output paths/format |
| `transformations` | **The seven audited transformations and their strengths** |
| `analysis` | Quantiles, filtering, output paths, plot kinds/order/size/format |
| `runtime` | Device selection, `require_device` |
| `logging` | Level, directory, file template, format |

Overrides are deep-merged YAML files:
`--config base.yaml --override tweak.yaml` (repeatable). This keeps a run fully
described by the files it was given — there are intentionally **no**
per-parameter command-line flags.

---

## 24. How to train the baseline

```bash
# 0. Environment (once)
python -m venv .venv
.venv\Scripts\activate                     # Windows;  source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .

# 1. Validate the dataset and materialise the internal split
python -m cicps prepare --config config/experiment_0a.yaml

# 2. Train the baseline
python -m cicps train --config config/experiment_0a.yaml
```

`train` writes `outputs/experiment_0a/checkpoints/baseline_best.pt` (selected by
`val_top1`) and `baseline_last.pt`, plus a per-epoch history CSV.

## 25. How to run the audit

```bash
# Optional but recommended: prove the seven transformations are deterministic
python -m cicps verify --config config/experiment_0a.yaml --num-images 8

# The transformation audit itself
python -m cicps audit --config config/experiment_0a.yaml
```

Training and auditing are **separable stages**. The audit loads the checkpoint
independently, so it can be re-run any number of times — with different
transformation strengths, for instance — without retraining. Use
`--checkpoint PATH` to audit a specific checkpoint for one run only.

## 26. How to regenerate results

```bash
python -m cicps analyze --config config/experiment_0a.yaml
```

The full sequence from a clean checkout:

```bash
python -m cicps prepare --config config/experiment_0a.yaml
python -m cicps verify  --config config/experiment_0a.yaml
python -m cicps train   --config config/experiment_0a.yaml
python -m cicps audit   --config config/experiment_0a.yaml
python -m cicps analyze --config config/experiment_0a.yaml
```

A fast end-to-end plumbing check (1 epoch, 24 audited images, ~3 minutes) that
writes to `outputs/smoke_0a/` and never touches real outputs:

```bash
python -m cicps train   --config config/experiment_0a.yaml --override config/smoke_0a.yaml
python -m cicps audit   --config config/experiment_0a.yaml --override config/smoke_0a.yaml
python -m cicps analyze --config config/experiment_0a.yaml --override config/smoke_0a.yaml
```

**The smoke run is not a scientific run.** One epoch on a truncated audit set
produces a near-random baseline; its numbers are meaningless and must never be
reported.

## 27. Output locations

```
outputs/experiment_0a/
├── splits/internal_split.json               # deterministic split manifest + fingerprint
├── checkpoints/
│   ├── baseline_best.pt                     # selected by val_top1
│   └── baseline_last.pt
├── logs/
│   ├── prepare.log · verify.log · train.log · audit.log · analyze.log
│   └── train_history.csv                    # per-epoch metrics
├── audit/
│   ├── transformation_audit.csv             # THE audit dataframe (600 × 7 rows)
│   ├── original_predictions.csv             # one row per validation image
│   └── audit_manifest.json                  # config snapshot, seeds, checkpoint, transforms
└── analysis/
    ├── transformation_summary.csv           # per-transformation statistics
    └── plots/
        ├── delta_margin_violin.png
        ├── delta_margin_box.png
        └── delta_margin_hist.png
```

Every one of these paths is configurable; none is hard-coded.

## 28. How to generate plots

Plots are produced by `analyze`. Configure them under `analysis.plots`:

* `kinds` — any subset of `violin`, `box`, `hist`, `ecdf`
* `order` — `config` (declaration order; makes **no** claim about safety) or
  `median` (sorted by median `ΔM`; descriptive only)
* `dpi`, `figsize`, `file_format`, `hist_bins`
* `delta_margin_limit` — symmetric axis clamp, or `null` for data-driven limits

The primary visualisation is the **distribution of `ΔM` for each of the seven
transformations**, with a dashed zero reference line.

## 29. Expected artifacts

1. Baseline training logs (`logs/train.log`, `logs/train_history.csv`)
2. Baseline checkpoints (`baseline_best.pt`, `baseline_last.pt`)
3. Deterministic split information (`splits/internal_split.json`)
4. The transformation audit dataframe (`audit/transformation_audit.csv`)
5. Per-transformation summary statistics (`analysis/transformation_summary.csv`)
6. `ΔM` distribution plots (`analysis/plots/`)
7. A run manifest (`audit/audit_manifest.json`)

### Summary table columns

Per transformation: `n_samples`, `delta_margin_mean`, `delta_margin_median`,
`delta_margin_std`, `delta_margin_min`/`max`, `delta_margin_mean_abs`,
configurable quantiles (default 1/5/25/50/75/95/99 %), `frac_delta_negative`,
`frac_delta_positive`, `original_margin_mean`, `augmented_margin_mean`,
`feature_cosine_mean`/`median`, `prediction_consistency_rate`,
`accuracy_original`, `accuracy_augmented`, `accuracy_delta`.

> **These are descriptive statistics only.** Experiment 0A implements **no**
> hypothesis test. Nothing in this table licenses a claim of statistical
> significance, and a higher mean `ΔM` does not make a transformation "better".

## 30. Reproducibility instructions

Seeds are set for Python, NumPy, torch and CUDA from `seed.value` (default 42),
and `PYTHONHASHSEED` is set. `cudnn.deterministic` is on and `cudnn.benchmark`
is off by default. `seed.deterministic_algorithms` additionally enables
`torch.use_deterministic_algorithms(True, warn_only=True)`.

Guaranteed reproducible given the same configuration and seed:

* the internal train/validation split (also fingerprint-checked on load);
* the frozen transformation parameters for every `(image, transform)` pair;
* the audit dataframe — verified bit-identical across separate process runs;
* DataLoader worker seeding and shuffling order.

**Documented residual non-determinism:**

* GPU floating-point reductions (cuDNN/cuBLAS kernel selection, atomics) can
  differ across GPU models, driver versions and library versions. Training
  losses may therefore differ in the last decimals on different hardware, and a
  *retrained* checkpoint will not be bitwise identical to one trained elsewhere.
  The audit *given a fixed checkpoint* is stable.
* `seed.deterministic_algorithms: true` uses `warn_only=True`, so an op with no
  deterministic kernel warns and proceeds rather than aborting.
* Mixed precision is enabled for training (`train.amp.enabled`). It is
  **disabled** for the audit, so the reported margins are fp32.
* Pillow/libjpeg version differences can change decoded pixels by ±1 level.
* CSV serialisation rounds to `audit.output.float_precision` significant digits;
  use Parquet if exact float round-tripping matters.

## 31. Common failure modes

| Symptom | Cause | Fix |
| --- | --- | --- |
| `CUB root not found` | `dataset.root` wrong | Point it at the directory containing `images.txt` |
| `Missing required CUB metadata file(s)` | Partial download | Re-extract the dataset |
| `Split manifest ... was created for a different dataset population` | Dataset or split settings changed under an existing manifest | `prepare --overwrite` — **note it invalidates checkpoints trained on the old split** |
| `Checkpoint not found` | Audit run before training | Run `train`, or point `audit.checkpoint` elsewhere |
| `train.scheduler.warmup_epochs must lie in [0, train.epochs)` | Warmup ≥ epochs (e.g. 1-epoch smoke run) | Set `warmup_epochs: 0` |
| CUDA out of memory | 4 GB-class GPU | Lower `train.batch_size` to 16, `audit.batch_size` to 32 |
| `Missing configuration key '...'` | Key absent from YAML | Add it — values must not be hard-coded |
| `OFFICIAL TEST LEAK` in validation | A manifest was hand-edited | Regenerate with `prepare --overwrite` |
| Audit is very slow | Dataloading, not compute | Raise `audit.num_workers`; on Windows try `0` if worker startup dominates |
| Warning: *"leaves every sample unchanged"* | A transformation's strength is degenerate | Check its `params` in the YAML |

## 32. Troubleshooting

* **Check what actually ran.** `outputs/experiment_0a/audit/audit_manifest.json`
  contains the full configuration snapshot, seeds, checkpoint path, checkpoint
  metrics, split fingerprint and every transformation's declared parameters.
* **Check the split is the one you think.** The manifest's `fingerprint` and the
  checkpoint's `metadata.split_fingerprint` must match; a mismatch is logged as
  a warning at audit time.
* **Check determinism directly.** `python -m cicps verify` reports, per
  transformation, whether parameters and tensors reproduce and how far `T(x)`
  moves from `x`.
* **Check the control arm.** `identity` must show `ΔM` exactly `0.0`,
  `feature_cosine` `1.0` and prediction consistency `1.0`. If it does not, the
  evaluation path is non-deterministic and every other number is suspect.
* **Check the competitor really is frozen.** Group the audit dataframe by
  `image_id`: `hard_competitor` and `original_margin` must each take exactly one
  distinct value per image, and every image must have exactly 7 rows.
* **Configuration drift** between the checkpoint and the current YAML
  (architecture, preprocessing, split) is logged as an explicit warning.

---

## 33. What results would SUPPORT the hypothesis

The hypothesis under test is that discriminative preservation is *measurable and
transformation-dependent*. Supporting evidence would look like:

* **Separation between arms.** The `ΔM` distributions of different
  transformations are visibly different in location and/or spread — well beyond
  the `identity` noise floor of `0.0`.
* **A meaningful negative tail.** Some transformations produce a substantial
  `frac_delta_negative` and a heavy left tail, i.e. they routinely destroy
  discriminative separation for a non-trivial share of images.
* **Non-redundancy with existing signals.** `ΔM` is not simply a monotone
  function of `feature_cosine` or of prediction consistency — that is, images
  can keep their representation broadly intact while their *pairwise* separation
  collapses. This is what would make a *discriminative-preservation* criterion
  worth having, as opposed to a generic feature-distance criterion.
* **Heterogeneity within a transformation.** Large per-image variance inside one
  transformation would mean the damage is *sample-dependent*, which is the
  precondition for any per-sample augmentation policy.

## 34. What results would WEAKEN or FALSIFY the hypothesis

* **No separation.** All seven `ΔM` distributions are effectively
  indistinguishable — augmentation choice does not measurably affect pairwise
  discriminative separation on this data with this model.
* **Pure noise.** `ΔM` distributions are centred near zero with spread
  comparable to what plausible evaluation noise would produce, and no arm shows
  a systematic shift.
* **Full redundancy.** `ΔM` is almost perfectly predicted by `feature_cosine` or
  by prediction consistency. Then nothing new is being measured, and a generic
  consistency criterion would do the same job with less machinery.
* **Homogeneity.** `ΔM` depends only on *which* transformation was applied and
  essentially not on the image. Then a fixed global augmentation policy is
  already optimal and per-sample guidance has no headroom.

A negative result here is a legitimate and useful outcome: it would falsify the
premise of the proposed method before any effort is spent building it. **Do not
tune the experiment until it produces a positive result.**

## 35. DPGA and Experiment 0B are NOT implemented

Explicitly and by design, this repository currently contains **no**
implementation of:

* **DPGA** (Discriminative-Preservation-Guided Augmentation) — no candidate
  generation, no preservation assessment, no ranking, filtering or weighting of
  transformations, and no augmentation-guided training.
* **Experiment 0B** — no training of separate models under different
  augmentation policies, and no downstream usefulness comparison.
* Any analysis of CUB part annotations, segmentation overlap, attention or
  discriminative-region localisation, class-level margin damage, the
  margin-damage ↔ accuracy-degradation relationship, multiple seeds, or
  additional datasets.

The architecture is intended to make these additions possible without rewriting
Experiment 0A — a new transformation is a registry entry plus a YAML block, and
a new metric is a registry entry — but none of them exists yet.

## 36. Clear next step after Experiment 0A

1. **Run the full audit and read the `ΔM` distributions** against §33–34. Decide
   honestly whether the phenomenon is present, absent, or redundant with the
   supporting metrics.
2. If present, the immediate next step is **Experiment 0B**: train separate
   baselines under different augmentation policies on the same internal split
   and test whether the margin damage measured in 0A predicts downstream
   usefulness. That is the step which converts a measurement into a claim about
   *consequences* — and it is the necessary precondition for DPGA.
3. If absent, report the negative result and revisit the premise before building
   anything on top of it.

The official CUB test set stays locked throughout.
