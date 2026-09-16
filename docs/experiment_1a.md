# Experiment 1A — Gradient Compatibility Analysis

**A diagnostic measurement on frozen CUB-200-2011 checkpoints.**

---

## 1. Research context

The programme has run three steps.

**Experiment 0A — the transformation audit.** One ResNet-18 was trained, frozen,
and probed with seven deterministic transformations. For each validation image
`x` with true class `y`, the hardest competing class was fixed from the original
image, `c(x) = argmax_{k≠y} p(k|x)`, and the change in the pairwise margin
measured:

```
ΔM = [p_y(T(x)) − p_c(T(x))] − [p_y(x) − p_c(x)]
```

It established that transformations damage fine-grained discriminative
separation by measurably *different* amounts — mean `ΔM` from +0.035
(`horizontal_flip`) to −0.211 (`gaussian_blur`) — and that this is not simply
feature drift in disguise (`corr(ΔM, feature_cosine)` ≈ 0.34, R² ≈ 0.12). Full
details: [`docs/experiment_0a.md`](experiment_0a.md).

**Experiment 0B — the policy comparison.** Six models were trained identically
apart from one augmentation each. The result was a **null**: the frozen-model
damage measured by 0A did not predict downstream usefulness. Gaussian blur, by
far the worst-audited transformation, produced only the second-mildest accuracy
drop; rotation, the mildest-audited of the five, trained worse than three of the
four other modified arms. Pearson r between 0A `ΔM` and the 0B accuracy delta was
**0.200** over five points. Full details:
[`docs/experiment_0b.md`](experiment_0b.md).

**Experiment 1A — this experiment.** 0B left a specific gap: a transformation can
damage a frozen model's output without that damage carrying over to training. 1A
asks what a *different view of the same frozen checkpoints* shows — not what the
transformation does to the model's answer, but what it does to the model's
**learning signal**.

```
0A: what a transformation does to a frozen model's OUTPUT      (zeroth order)
0B: what happens when a model is TRAINED with it               (the endpoint)
1A: what it does to a frozen model's GRADIENT                  (first order)
```

## 2. Motivation

### Frozen perturbation vs. learning signal, in plain terms

Experiment 0A asked a question about *answers*. It showed the model a blurred
bird and checked how much less certain the model became that this was a
Blue-winged Warbler rather than its nearest look-alike. That is a question about
what the model currently believes.

Training does not consume beliefs. It consumes **corrections**. When a training
image is presented, what actually changes the model is the gradient — the
direction in which every weight should move to make this image's label more
likely. Two images can produce the same amount of "being wrong" while
recommending completely different corrections, and two images can produce very
different amounts of "being wrong" while recommending the *same* correction, only
more or less strongly.

So the natural question left by 0B is:

> When the model gets a transformed image, does it get told to move in a
> direction compatible with what the clean image would have told it, or somewhere
> else entirely?

That is a first-order property of the same frozen checkpoint, and 0A never
measured it.

### What this cannot do

This must be stated up front, because it constrains every conclusion below.

`C_g` is an **instantaneous** quantity at a fixed `θ`, exactly as `ΔM` is. 0B's
outcome is the endpoint of a 20-epoch trajectory under AdamW with weight decay,
a cosine schedule and BatchNorm statistics that update as training proceeds. No
instantaneous measurement can be *shown* to determine that endpoint, and
certainly not from five transformation points. 1A is therefore framed as a
**diagnostic**: does the learning-signal view carry information the margin view
does not? It is not framed as the explanation of the 0A→0B gap.

Second, **`C_g` has no a priori sign-mapping to usefulness.** A high cosine means
the transformed sample pushes the update where the clean sample already pushed
it. That reads as "harmless" and as "redundant" equally well — and redundancy is
precisely what makes an augmentation *useless*. A low cosine could be new
information or genuine conflict. The experiment measures; it does not rank.

## 3. Research question

> At different stages of an already-trained model, do clean and transformed
> versions of the same image produce different degrees of gradient alignment and
> gradient magnitude, and does this behaviour differ systematically across
> transformations?

Subsidiary questions, all falsifiable:

1. Is `C_g` distinguishable from the cross-image null — i.e. does pairing carry
   any signal at all?
2. Do the transformations separate on `C_g`, and does that ordering differ from
   their ordering on `ΔM`? (**non-redundancy**)
3. Does gradient compatibility change between an untrained and a converged model?
   (**Pattern D**)
4. Does a model trained *with* a transformation show higher compatibility with
   it? (**Candidate B, the adaptation probe**)

## 4. Hypotheses

**Candidate A — learning signal.** The gap 0B left is explained, at least in
part, by a property of the gradient rather than of the output margin.

*Supported by:* transformations separating on `C_g` well beyond the null, with an
ordering that is **not** a repackaging of `ΔM`.

*Weakened by:* `C_g` being a near-monotone function of `ΔM` (then it is `ΔM` in
different clothes), or all arms sitting at the null (then the paired structure
carries nothing), or all arms being indistinguishable.

**Candidate B — representation evolution.** What matters is not the initial
gradient but what training under a transformation eventually produces.

1A can only *probe* this, via the adaptation stages, and cannot test it properly
— that is Experiment 1B.

The four patterns from the research proposal (A: negative `ΔM` + high `C_g`; B:
negative `ΔM` + low `C_g`; C: near-zero `ΔM` + low `C_g`; D: alignment changes
over training) are **interpretive templates, not predictions**. No code path
encodes an expectation about which holds.

## 5. Mathematical definitions

Let `θ` be the parameters of a frozen checkpoint, `L` the cross-entropy loss, and
`(x, y)` an internal-validation image and its true label.

**Clean and transformed gradients.**

```
g_x = ∇_θ L(x, y)              g_T = ∇_θ L(T(x), y)
```

**Gradient cosine (primary).**

```
C_g = ⟨g_x, g_T⟩ / (‖g_x‖ · ‖g_T‖)          C_g ∈ [−1, 1]
```

`C_g = 1`: the transformed sample recommends exactly the clean sample's update
direction. `C_g = 0`: the two recommendations are orthogonal. `C_g < 0`: they
conflict.

**Gradient norms and norm ratio (primary).**

```
R_g = ‖g_T‖ / ‖g_x‖
```

`R_g > 1`: the transformed sample is a *stronger* corrective signal, whatever its
direction.

**Scoped versions.** For a parameter scope `S ⊆ θ`, the same formulas restricted
to `S`. Reported for `full` (primary), `backbone`, `head`, `stem`,
`layer1…layer4`.

**Margin and ΔM (as in 0A), recomputed at each checkpoint.**

```
c(x) = argmax_{k≠y} p(k|x)        (from the CLEAN image, at this checkpoint)
M(x) = p_y(x) − p_c(x)
ΔM   = M(T(x)) − M(x)             (competitor frozen across arms)
```

Note `c(x)` is checkpoint-specific: it is selected from the clean image *at that
stage*. Only at the anchor stage does it coincide with 0A's competitor, and there
it is verified to agree on every row.

**Cross-image null (the scale reference).**

```
C_null = cos(g_{x_i}, g_{x_j}),   i ≠ j, both CLEAN, seeded random partner
```

**Supporting quantities.** Clean/transformed loss and their difference, clean and
transformed true/competitor probability, feature cosine
`C_f = cos(f(x), f(T(x)))`, prediction consistency
`1[argmax p(x) = argmax p(T(x))]`, and clean/transformed correctness.

## 6. Experimental setup

| Item | Value |
| --- | --- |
| Dataset | CUB-200-2011, exactly as distributed |
| Population | The **internal validation** subset, 600 images — 0A's population |
| Split | 0A's manifest verbatim, fingerprint `2c3a0390542d2a57` |
| Official test set | **Locked.** Never loaded; asserted by the invariant script |
| Model | ResNet-18, 512-d penultimate, 200-way linear head |
| Loss | Cross-entropy, label smoothing 0.0 (0B's training loss) |
| Mode | `model.eval()` — see §6.2 |
| Precision | float32, no autocast, TF32 off, float64 reductions |
| Transformations | 0A's exact seven, byte-identical parameters |
| Sampling | Realisation 0 = 0A's frozen per-sample draw |
| Gradient scope | 8 scopes measured; `full` is primary |
| Batch size | 8 images per loader item; the gradient is per-sample |
| Seed | 42 throughout |

### 6.1 Checkpoint stages

The shared trainer persists only `best` and `last` per run, so **no
intra-training checkpoint exists anywhere in the repository**. The stages below
are what is honestly available:

| Stage | Source | Epoch | val top-1 |
| --- | --- | --- | --- |
| `init` | reconstructed under master seed 42 | 0 | — |
| `experiment_0a_best` | `outputs/experiment_0a/checkpoints/baseline_best.pt` | 17/20 | 76.17 % |
| `0b_baseline_best` | `outputs/experiment_0b/checkpoints/baseline/best.pt` | 14/20 | 78.17 % |
| `0b_baseline_last` | `outputs/experiment_0b/checkpoints/baseline/last.pt` | 20/20 | 77.17 % |

Plus five **adaptation probe** stages — each 0B policy's own checkpoint,
measured on that policy's own transformation plus the identity control.

`init` is not loaded from disk; it is rebuilt by installing the master seed and
constructing the model, which is exactly what 0B does before training each arm.
This was verified to produce bit-identical weights across repeated builds, and
the resulting state is fingerprinted into the manifest.

This spans *untrained → converged → post-peak*. It **cannot** resolve a training
trajectory. Producing one requires per-epoch checkpointing during training, which
is Experiment 1B's job.

### 6.2 Why `eval()` mode

ResNet-18 contains BatchNorm and no dropout.

In `train()` mode a single-sample gradient would have BatchNorm normalise the
image by *its own* statistics — degenerate, dependent on whatever else shared the
batch, and it would mutate the running statistics of a checkpoint we are
committed to leaving untouched.

In `eval()` mode the frozen running statistics are used, so the clean and
transformed passes differ by exactly one thing: the pixels. The cost is that the
measured gradient is the eval-mode gradient, not literally the gradient a
training step would have taken. That is a real limitation (§15), stated rather
than hidden.

### 6.3 Transformation sampling protocol

Required by the proposal to be documented explicitly.

* **The same transformed realisation is reused, not resampled.** Realisation 0
  re-uses 0A's parameter key `derive_seed(param_seed, image_id, transform_name)`
  verbatim, so the transformed image 1A differentiates is bit-identical to the
  one 0A measured `ΔM` on.
* **Why:** it makes the two experiments joinable *per image*, not merely per
  transformation. A scatter of `C_g` against `ΔM` is then a within-sample
  comparison of two views of one perturbation. Resampling would confound the
  transformation with the draw, exactly as 0A argued (§14 there).
* **Multiple realisations are supported and off by default.**
  `gradients.sampling.realisations: R` adds independent draws keyed by an
  extended seed, leaving realisation 0 untouched. Default 1: more realisations
  multiply cost by `R` without changing the question being asked. Raise it only
  to ask how much spread comes from the draw rather than from the image.

### 6.4 The cross-image null control

A cosine in ~11.2 M dimensions has no intuitive scale. Without a reference,
`C_g = 0.2` cannot be called high or low.

Two references bracket it:

* **Upper:** the `identity` arm. It transforms nothing, so `g_T = g_x` and
  `C_g = 1` exactly. This doubles as a bit-exact correctness check on the whole
  gradient path — the role `identity`'s exactly-zero `ΔM` plays in 0A.
* **Lower:** the cross-image null. Each image's clean gradient is compared with
  the *previously processed* image's clean gradient. Traversal order is permuted
  with a derived seed, because CUB image ids are grouped by species and the
  natural order would make the partner a same-class neighbour. It costs **zero**
  extra backward passes.

## 7. Implementation

New package `src/cicps/gradients/`, deliberately generic so Experiment 1B can
reuse it unchanged:

```
gradients/
├── scopes.py       named parameter subsets from YAML regexes; validates non-empty
├── accumulate.py   online float64 inner products; caches a gradient's own norms
├── checkpoints.py  stage resolution, the reconstructed init, state fingerprinting
├── datasets.py     paired clean + all-transformed views, one JPEG decode per image
├── records.py      dataframe schemas
└── engine.py       the measurement
```

plus `analysis/gradients.py` (summaries, bootstrap, relationships, joins),
`analysis/gradient_plots.py`, `stages/gradients.py` and
`stages/analyze_gradients.py`.

**Reused unchanged:** config, seeding, logging, device resolution, CUB parsing,
the split, the model factory, the checkpoint loader, the transformation registry,
`audit/records.py`'s table writers. Nothing is duplicated.

### Efficiency

The naive shape of this experiment is `images × transforms × 2` backward passes.
Two structural choices cut that:

* **The clean pass is computed once per image** and compared against all seven
  transformed gradients — `1 + K` backward passes per image, not `2K`. The clean
  gradient's own per-parameter squared norms are cached on the vector, so the
  most expensive reduction is not repeated seven times.
* **One JPEG decode per image**, producing the clean view and every transformed
  view in the same dataset item.

Per scope, only three float64 scalars are ever formed — `⟨g_x, g_T⟩`, `‖g_x‖²`,
`‖g_T‖²`. **No raw gradient vector is ever written to disk.** Storing them would
be hundreds of gigabytes and would add nothing.

### Safety

* No optimizer is constructed and `step()` is never called.
* Every stage's full `state_dict` — parameters **and buffers** — is BLAKE2b
  fingerprinted before and after measurement, and a mismatch raises. Buffers are
  included because an accidental `train()` mode would move BatchNorm running
  statistics without touching a single weight.
* 1A never writes into `outputs/experiment_0a/` or `outputs/experiment_0b/`.

## 8. Configuration

`config/experiment_1a.yaml` is the single source of truth; a missing key raises.

| Section | Controls |
| --- | --- |
| `experiment` | Name and description |
| `seed` | Master seed, cudnn flags, worker seeding |
| `dataset` | CUB root, metadata filenames, expected counts, validation strictness |
| `split` | Path to the **shared** 0A split manifest |
| `model` | Architecture, weights, `num_classes` — must match the checkpoints |
| `preprocess` | Canonical view and normalisation, identical to 0A/0B |
| `train.loss` | The loss being differentiated (nothing else in `train` is read) |
| `gradients.split_subset` | `val` (default) or `train`; the test set is not an option |
| `gradients.max_images` | Cap for smoke runs; `null` for a scientific run |
| `gradients.batch_size` / `num_workers` / `pin_memory` / `progress` | Loading |
| `gradients.precision` | `dtype` (float32 only) and `allow_tf32` |
| `gradients.sampling` | `param_seed`, `realisations` (§6.3) |
| `gradients.null_control.enabled` | The cross-image null (§6.4) |
| `gradients.scopes` | Scope name → parameter-name regexes |
| `gradients.primary_scope` | Which scope is the primary metric |
| `gradients.checkpoints` | **The checkpoint stages**, and per-stage transform subsets |
| `gradients.output` | Directory, format, table names, float precision |
| `transformations` | 0A's exact seven — must stay byte-identical |
| `analysis.anchor_checkpoint` | The stage the joins and single-stage figures use |
| `analysis.control_transform` / `control_tolerance` | The `identity` assertion |
| `analysis.bootstrap` | Resamples and confidence level |
| `analysis.join_0a` / `join_0b` | The read-only reference paths and tolerances |
| `analysis.plots` | Figure kinds, size, dpi, format, directory |
| `runtime` / `logging` | Device selection; log level, directory, format |

`config/smoke_1a.yaml` is a deep-merged override running the whole pipeline on 12
images and three stages.

## 9. Logging and reproducibility

Every run writes `outputs/experiment_1a/gradients/gradient_manifest.json`,
recording: timestamp, **git commit**, master seed, transformation parameter seed,
the sampling protocol in words, split subset and fingerprint, image and record
counts, gradient mode (`eval`) *and the reason for it*, the full precision policy
(dtype, TF32, autocast, reduction dtype), every scope with its tensor and
parameter counts, every checkpoint stage with its path, epoch and validation
metrics, every transformation with its resolved parameters, the null-control
definition, device, CUDA device name, torch/Python/platform versions, runtime
seconds, any warnings, and the **complete resolved configuration**.

Guaranteed reproducible given the same configuration, seed and checkpoints: the
split, the per-sample transformation parameters, the traversal permutation, the
null-control pairings, and the bootstrap intervals (seeded via BLAKE2b, never
Python's per-process-salted `hash()`).

**Documented residual non-determinism.** 1A's `ΔM` at the anchor stage does not
reproduce 0A's bit-for-bit. 0A ran batched fp32 inference under the cuDNN TF32
default of its day; 1A runs per-sample with TF32 off. The discrepancy was
measured directly, and it moves when TF32 or the batch size is changed, while the
frozen competitor agrees on every row — so it is the residual GPU
non-determinism already documented in
[`docs/experiment_0a.md`](experiment_0a.md) §30, not a different perturbation.
Measured values are in §12.

## 10. Outputs

```
outputs/experiment_1a/
├── gradients/
│   ├── gradient_records.csv      one row per (stage, transform, realisation, image)
│   ├── clean_reference.csv       one row per (stage, image): clean pass + null control
│   └── gradient_manifest.json    full provenance (§9)
├── analysis/
│   ├── gradient_summary.csv          per (stage, transform): the primary metrics + CIs
│   ├── null_control_summary.csv      per stage: the cross-image null scale reference
│   ├── relationship_summary.csv      within-transformation associations across images
│   ├── joined_1a_0a_per_sample.csv   per-image join to 0A's ΔM at the anchor stage
│   ├── joined_1a_0b_summary.csv      per-transformation join to 0B's downstream result
│   ├── association_exploratory.csv   transformation-level coefficients, n attached
│   └── plots/*.png                   the eight figures
└── logs/
    ├── gradients.log
    └── analyze.log
```

## 11. Metrics

| Metric | Status | Reading |
| --- | --- | --- |
| `grad_cosine` (`C_g`, full scope) | **primary** | Compatibility of the transformed learning signal with the clean one. Not a quality score. |
| `grad_norm_ratio` (`R_g`) | **primary** | Relative strength of the transformed signal. Heavy-tailed — read the median. |
| `cosine_head` / `cosine_backbone` | reported alongside | Whether disagreement sits in the decision layer or the representation |
| `cosine_stem`…`cosine_layer4` | supporting | Depth-resolved detail |
| `null_cosine` | **scale reference** | What two *unrelated* learning signals score |
| `delta_margin` | supporting / bridge | 0A's primary metric, recomputed at each stage |
| `clean_loss`, `transformed_loss`, `delta_loss` | supporting | How hard each view is |
| `feature_cosine` | supporting | Generic representation drift |
| `prediction_consistent`, `correct_*` | supporting | Behavioural change |

`identity` must give `C_g = 1`, `R_g = 1`, `ΔM = 0`. The analysis stage **fails
the run** if it does not; every other number would be suspect.

Reading rules, inherited from 0A and 0B:

* A high `C_g` is not "good". It is compatible with "redundant".
* Transformation-level coefficients rest on five or six points. Report `n`; never
  call them significant.
* Smoke-run numbers are never reported.

## 12. Results

**Run date:** 2026-09-15. **Hardware:** RTX 3050 Laptop (4 GB). **Seed:** 42.
**Scale:** 600 validation images × 9 checkpoint stages = **22,800 gradient rows**
and 5,400 clean rows, measured in **1,679 s** (28 min).
`scripts/check_experiment_1a.py` reports **99/99 checks passed**.

### 12.1 Validity checks

| Check | Result |
| --- | --- |
| `identity` control: `C_g` | **exactly 1.0** (max deviation 0.000e+00) |
| `identity` control: `R_g` | **exactly 1.0** (max deviation 0.000e+00) |
| `identity` control: `ΔM` | exactly 0.0; prediction consistency 100 % |
| Model state unchanged per stage | verified by BLAKE2b fingerprint, all 9 stages |
| Realisation-0 parameters re-derive 0A's draw | 0 mismatches |
| 0A `ΔM` reproduced at the anchor stage | max 2.5e-3, **mean 2.3e-4**, over 4,200 rows |
| Frozen hardest competitor agrees with 0A | **4,200 / 4,200** |
| Official test images touched | **0** (5,794 locked) |

The anchor stage reproduces 0A's published per-transformation means to five
decimals (e.g. `gaussian_blur` −0.21115 here vs −0.211149 in 0A; `color_jitter`
−0.14533 vs −0.145339). The residual per-image discrepancy is the documented GPU
non-determinism of §9.

### 12.2 The scale of a cosine: the two controls

| Reference | `C_g` |
| --- | --- |
| `identity` (the same image) | 1.0000 |
| **Cross-image null** (unrelated images, anchor stage) | **+0.0036** [−0.0032, +0.0103] |

Across all nine stages the null sits between **−0.0003 and +0.0095**, with
intervals spanning zero at eight of them (n = 599 pairs each). **Two unrelated
images produce essentially orthogonal learning signals.** Every paired value
below — 0.50 to 0.85 — is therefore enormous relative to chance: a transformed
image is nowhere near an unrelated one. Without this control none of those
numbers could be called large.

### 12.3 Primary result — anchor stage (`experiment_0a_best`, 600 images/arm)

| Transform | mean `C_g` | 95 % CI | median `C_g` | q05 | median `R_g` | q95 `R_g` | mean `ΔM` |
| --- | ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| identity | 1.0000 | — | 1.0000 | 1.000 | 1.000 | 1.00 | 0.0000 |
| rotation | **0.7665** | [0.759, 0.774] | 0.7758 | 0.609 | 1.094 | 7.4 | −0.0072 |
| random_erasing | 0.7531 | [0.738, 0.767] | 0.8090 | 0.352 | 1.245 | 45.5 | −0.1216 |
| random_resized_crop | 0.6853 | [0.677, 0.695] | 0.7056 | 0.461 | 1.290 | 20.1 | −0.0593 |
| color_jitter | 0.6813 | [0.667, 0.696] | 0.7246 | 0.329 | 1.711 | 70.3 | −0.1453 |
| horizontal_flip | 0.6495 | [0.644, 0.656] | 0.6566 | 0.517 | 1.009 | 4.3 | +0.0349 |
| gaussian_blur | **0.5031** | [0.491, 0.515] | 0.5001 | 0.248 | 2.500 | 160.3 | −0.2111 |

* **The arms separate clearly**, with non-overlapping bootstrap intervals for most
  pairs, and all far above the null.
* **`C_g < 0` is vanishingly rare** — 0.2 % of rows for `color_jitter`, 0 % for
  every other arm. Transformed images essentially never recommend an *opposing*
  update; they recommend a partially different one.
* **`R_g` is heavy-tailed and must be read at the median.** Mean `R_g` reaches
  43.8 for `color_jitter` — driven by images whose clean loss is near zero, so the
  ratio explodes. The median tells the real story: blur 2.50, jitter 1.71, flip
  1.01.

### 12.4 Is the gradient view redundant? (within-transformation, n = 600 each)

| Relationship | pooled Pearson | pooled Spearman | per-arm range (Pearson) |
| --- | ---: | ---: | --- |
| `C_g` vs `ΔM` | **+0.185** | +0.148 | −0.093 … +0.348 |
| `C_g` vs `feature_cosine` | **+0.770** | +0.760 | +0.416 … +0.882 |
| `R_g` vs `ΔM` | −0.127 | **−0.787** | ρ: −0.727 … −0.828 |

Three findings, one of them awkward:

1. **`C_g` is *not* a repackaging of `ΔM`.** Per image, R² ≈ 0.034 pooled. The two
   views of the same perturbation on the same image are close to unrelated.
2. **But `C_g` is largely predictable from `feature_cosine`** — R² ≈ 0.59 pooled,
   up to 0.78 for `random_erasing`. For comparison, 0A found `ΔM` vs
   `feature_cosine` at only R² ≈ 0.12. **The expensive gradient measurement sits
   substantially closer to generic representation drift than 0A's cheap margin
   metric did.** This is a negative result for the novelty of `C_g`.
3. **`R_g` is strongly rank-redundant with `ΔM`** (ρ ≈ −0.79, consistently within
   every arm): the more the margin is damaged, the louder the transformed
   gradient. Mechanically unsurprising — a higher loss produces a larger gradient
   — so `R_g` adds little beyond "this transformed image is harder".

### 12.5 Does compatibility change as the model learns? (Pattern D)

Mean `C_g`, baseline lineage:

| Transform | `init` | `0a_best` | `0b_baseline_best` | `0b_baseline_last` |
| --- | ---: | ---: | ---: | ---: |
| color_jitter | **0.8091** | 0.6813 | 0.6861 | 0.6753 |
| random_erasing | 0.7600 | 0.7531 | 0.7450 | 0.7426 |
| rotation | 0.7386 | **0.7665** | 0.7783 | 0.7757 |
| random_resized_crop | 0.7149 | 0.6853 | 0.6971 | 0.6950 |
| horizontal_flip | 0.6716 | 0.6495 | 0.6500 | 0.6475 |
| gaussian_blur | 0.6488 | **0.5030** | 0.5367 | 0.5368 |

Median `R_g` over the same stages: blur **1.09 → 2.50 → 2.72 → 2.76**; jitter
1.03 → 1.71 → 1.59 → 1.64; flip 1.00 → 1.01 → 0.99 → 1.00.

* **Alignment does change with training, and not uniformly.** Five arms lose
  alignment (blur −0.146, jitter −0.128); `rotation` *gains* it (+0.028). The
  ordering itself is reshuffled: `color_jitter` is the **best**-aligned arm at
  init and only mid-table once trained.
* **Training makes damaging transforms louder**, not merely more divergent:
  blur's median `R_g` more than doubles.
* **Best vs last barely differ** — the structure is stable near convergence, so
  the interesting variation is early, exactly where this experiment has no
  checkpoints.

**Pattern C is clearly present at `init`.** There, `ΔM` is 0.001–0.003 for *every*
arm — an untrained model has no discriminative structure to damage, so the margin
view is blind — while `C_g` spans 0.649 to 0.809 and orders the transformations.
The gradient view sees structure exactly where the margin view sees nothing.

### 12.6 Where in the network does the disagreement live?

Mean `C_g` by scope, anchor stage:

| Transform | full | head | backbone | stem | layer1 | layer4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rotation | 0.767 | 0.937 | 0.760 | 0.709 | 0.734 | 0.802 |
| random_erasing | 0.753 | 0.893 | 0.747 | 0.733 | 0.735 | 0.765 |
| color_jitter | 0.681 | 0.867 | 0.674 | 0.572 | 0.616 | 0.733 |
| gaussian_blur | 0.503 | 0.804 | 0.493 | **0.374** | 0.418 | 0.599 |

**The disagreement is a representation phenomenon, not a decision-layer one.** The
classifier head stays highly aligned for every transformation (0.80–0.94) while
the backbone tracks the full-model value almost exactly — unsurprising, since the
backbone holds 99 % of the parameters. Disagreement is **deepest in the early
layers** and recovers with depth (blur: 0.374 at the stem, 0.599 at layer4). That
is the expected signature of a pixel-level perturbation: it disrupts early filters
most, and the network partially re-converges downstream.

### 12.7 Adaptation probe (Candidate B)

Each transformation measured on the baseline model vs on the 0B model *trained
with that transformation*:

| Transform | `C_g` baseline | `C_g` own-trained | Δ`C_g` | median `R_g` base → own | mean `ΔM` base → own | acc. Δ base → own |
| --- | ---: | ---: | ---: | :---: | :---: | :---: |
| gaussian_blur | 0.5367 | **0.7900** | **+0.253** | 2.72 → 1.14 | −0.208 → **+0.003** | −0.233 → −0.030 |
| color_jitter | 0.6861 | **0.8517** | **+0.166** | 1.59 → 1.08 | −0.150 → −0.012 | −0.193 → −0.027 |
| random_erasing | 0.7450 | 0.8201 | +0.075 | 1.25 → 1.06 | −0.123 → −0.045 | −0.155 → −0.065 |
| rotation | 0.7783 | 0.8090 | +0.031 | 1.13 → 1.03 | −0.024 → +0.017 | −0.043 → −0.018 |
| random_resized_crop | 0.6971 | 0.7042 | +0.007 | 1.29 → 1.26 | −0.084 → −0.054 | −0.090 → −0.073 |

**Every transformation becomes more compatible with a model trained on it**, on
every axis simultaneously — alignment up, norm ratio toward 1, margin damage
toward 0, accuracy loss shrunk. And **the size of the adaptation tracks the size
of the original damage**: blur and jitter, the two most disrupted on the baseline
model, adapt by far the most. Gaussian blur goes from the worst-aligned, most
margin-damaging transformation in the entire experiment to nearly harmless
(`ΔM` = +0.003) on the model that trained with it.

This is the single clearest signal in Experiment 1A — and it is **confounded**
(§15.7): a policy's checkpoint differs from the baseline's in its whole training
trajectory, not only in the transformation.

### 12.8 Does any of it predict Experiment 0B? (n = 5, descriptive only)

| Predictor (anchor stage) | vs 0B Δ val top-1: Pearson | Spearman |
| --- | ---: | ---: |
| mean `C_g` | **−0.236** | −0.100 |
| mean `R_g` | −0.535 | −0.500 |
| mean `ΔM` (0A's metric) | +0.200 | +0.200 |

**No gradient quantity predicts downstream usefulness better than 0A's `ΔM` did,
and `C_g` is if anything weakly *negatively* associated with it.** The arm with
the lowest alignment by a wide margin (blur, `C_g` = 0.50) produced the
second-*mildest* downstream drop, while the only arm that beat the baseline
(crop, +0.5 pp) sits mid-table at 0.70. Five points, one seed each — these
coefficients are unstable by construction and no significance test is run.

## 13. Visual analysis

Eight figures in `outputs/experiment_1a/analysis/plots/`, each answering one
question:

| Figure | Question | What it shows |
| --- | --- | --- |
| `grad_cosine_distribution.png` | How does the learning signal differ by transformation? | Violins per arm with both reference levels drawn — 1.0 (identity) and the cross-image null |
| `grad_norm_ratio_distribution.png` | Stronger or weaker signal? | The heavy right tails that make the mean `R_g` unreadable |
| `grad_cosine_vs_delta_margin.png` | Redundant with 0A? | 3,600 paired points, near-flat — the §12.4 result |
| `grad_cosine_vs_feature_cosine.png` | Redundant with feature drift? | A clear positive band — the awkward §12.4 result |
| `grad_cosine_by_checkpoint.png` | Does it change with training? | Five arms fall, rotation rises, null flat at ~0 |
| `grad_norm_ratio_by_checkpoint.png` | Does signal strength change? | Blur's median ratio more than doubling |
| `grad_cosine_vs_downstream.png` | Does it predict 0B? | Five labelled points, no trend |
| `scope_heatmap.png` | Where does disagreement live? | Head aligned, stem least aligned |

## 14. Interpretation

**What was measured.** For 600 held-out images at 9 frozen checkpoints, the cosine
and norm ratio between the parameter gradient of a clean image and of its
transformed counterpart — the same images, the same transformations and the same
frozen per-sample draws Experiment 0A used.

**What the results mean.**

1. **Gradient compatibility is real, large, and transformation-specific.**
   Unrelated images produce orthogonal learning signals (`C_g` ≈ 0.004), while
   clean/transformed pairs score 0.50–0.85 and separate cleanly by transformation.
   The measurement works, and the `identity` arm proves the path is exact.

2. **Transformed images almost never *conflict* with clean ones.** Negative
   cosines are essentially absent. The intuition behind Candidate A — that a
   damaging transformation might push the parameters the *wrong way* — is not what
   happens on this model. What varies is how much of the clean update's direction
   is retained, and how loudly it is stated.

3. **`C_g` is genuinely different from `ΔM` per image, but largely overlaps with
   feature cosine.** This is the most consequential negative finding. 0A earned
   its metric by showing `ΔM` was *not* a restatement of feature drift
   (R² ≈ 0.12). `C_g` fails that same test (R² ≈ 0.59). A cheap forward-pass
   quantity recovers most of what an expensive per-sample backward pass tells us.

4. **The margin view and the gradient view genuinely diverge at `init`.** With no
   discriminative structure to damage, `ΔM` ≈ 0 for every arm while `C_g` orders
   them across a 0.16 range. Pattern C is real — but it appears where the model is
   untrained, which limits how much can be built on it.

5. **Compatibility is dynamic, and re-orders.** `color_jitter` is the best-aligned
   transformation on an untrained network and mid-table on a trained one; blur's
   alignment collapses while its signal strength doubles. An augmentation's
   relationship to the learning signal is a property of the *model state*, not of
   the transformation alone.

6. **The disagreement is an early-layer representation effect**, not a
   decision-layer one — the head stays aligned at 0.80–0.94 throughout.

7. **None of it closes the 0A→0B gap.** Against 0B's downstream deltas, `C_g`
   scores r = −0.24 where `ΔM` scored +0.20. Substituting a first-order
   measurement for a zeroth-order one did not buy predictive power.

**The most likely reading of 7, stated as a hypothesis and not a result:** the
premise that a *good* augmentation is a *compatible* one may simply be wrong. High
`C_g` means the transformed sample recommends the update the clean sample already
recommended — which is redundancy, and a redundant augmentation teaches nothing
new. The weakly negative association is at least consistent with that, and so is
the observation that blur — lowest alignment, biggest adaptation headroom —
trained better than three arms that disturbed the model far less. 1A cannot test
this; it has five points.

**The strongest positive signal is Candidate B, not Candidate A.** Every
transformation is markedly more compatible with a model trained on it, and the
transformations that were most disrupted adapt most (blur: `C_g` +0.25, `ΔM`
−0.208 → +0.003). Whatever determines augmentation usefulness looks like a
property of *what training does with the perturbation*, not of the perturbation's
instantaneous effect on a fixed model. That is a statement about the trajectory,
and 1A deliberately measured only its endpoints.

## 15. Limitations

1. **Instantaneous, not dynamic.** Both `C_g` and `ΔM` are properties of a fixed
   `θ`. 0B's outcome is a trajectory endpoint. No correspondence found here is
   evidence of a mechanism.
2. **No training trajectory.** Only `best`/`last` checkpoints exist, so the
   checkpoint axis is untrained → converged → post-peak, not a curve. Pattern D
   can be answered only coarsely.
3. **`eval()`-mode gradients** (§6.2) are not literally training-step gradients:
   BatchNorm uses frozen statistics and the real update is a batch average, not a
   per-sample gradient.
4. **Held-out images.** Gradients are measured on validation images — the update
   a held-out image *would* suggest. Measuring on training images is supported
   but ill-conditioned at a converged checkpoint (near-zero clean loss).
5. **One realisation by default.** The spread reported mixes image-to-image
   variation with the single frozen parameter draw per image.
6. **Five transformation points, one seed each**, for anything joined to 0B —
   the same limitation that made 0B inconclusive. It is not repaired here.
7. **The adaptation probe is confounded.** A policy's checkpoint differs from the
   baseline's in its entire training trajectory, not only in the transformation.
8. **One dataset, one architecture, one budget.**
9. **Cosine compresses.** Two gradients can share direction while differing
   enormously in the subspace that matters for one class; `C_g` will not see it.
10. **Scope arbitrariness.** `full` weights every parameter equally, which is not
    how AdamW moves them — it rescales per-parameter by second-moment estimates.
    A preconditioned cosine would be a different, also defensible, choice.

## 16. Decision

**Candidate A is not supported as an explanation of the 0A→0B gap.**

Precisely what is and is not established:

* **Established:** gradient compatibility is measurable to bit-exact control
  precision, is far above the unrelated-image null, differs systematically and
  substantially across transformations, is not a per-image repackaging of `ΔM`,
  changes with training and re-orders the transformations as it does, and lives
  in the early representation rather than the classifier head.
* **Not established, and the evidence points the other way:** that gradient
  compatibility explains or predicts downstream augmentation usefulness. It
  correlates with 0B's outcome at r = −0.24 (n = 5) — no better than the metric
  it was meant to improve on, and in the opposite direction to the experiment's
  motivating intuition.
* **A partial strike against the metric itself:** `C_g` is ~59 % explained by
  feature cosine, a quantity obtainable from a forward pass. It is a costlier
  route to much of the same information.
* **Inconclusive but the most promising thread:** Candidate B. The adaptation
  probe is consistent and large on every axis, and confounded by design.

This is a **negative result for the first-order hypothesis and a positive result
for the measurement infrastructure**, in the same spirit as 0B's null. It was
possible for this experiment to come back the other way: non-overlapping
intervals and an n = 600 per-arm sample would have detected a real
`C_g` ↔ downstream relationship had one existed at this scale.

## 17. Implications for Experiment 1B

Directed by what 1A actually found — including what it found *not* to be worth
measuring.

**1. The checkpoint axis is the binding constraint — fix it first.** Every
interesting quantity here changed between `init` and convergence and was then
flat (best vs last differ by < 0.01). All the structure is in training, and the
repository has no checkpoint there. 1B's first requirement is **per-epoch
checkpointing during training** — a small, additive change to the shared trainer
(a configurable save-every-N), and the single highest-value thing 1B can do.

**2. Make the adaptation probe a controlled experiment.** 1A's probe is
suggestive and confounded. 1B should track, epoch by epoch within *one* training
run, how `C_g(T)` evolves for the transformation that run is training with versus
transformations it is not. That removes the cross-model confound entirely and
turns the strongest 1A signal into a real measurement.

**3. Drop or demote what proved redundant.** Per-sample `R_g` is ρ ≈ −0.79 with
`ΔM` and adds little; per-sample `C_g` is R² ≈ 0.59 with feature cosine. 1B
should not pay full per-sample backward-pass cost for both. Measure `C_g` on a
**subsample** for trajectory purposes and spend the saved budget on the
representation-structure metrics 1A did not have: intra-class compactness,
inter-class separation, and class-level structure under transformation.

**4. Measure early layers preferentially.** Disagreement concentrated in the stem
and layer1 and washed out by layer4. A layer-resolved representation analysis
should weight where the effect actually is.

**5. Address the seed problem.** Every downstream claim in 0B and 1A rests on five
transformations with one seed each. Multiple seeds per policy remain the
outstanding methodological debt; no amount of extra instrumentation substitutes
for it.

**6. Test the redundancy hypothesis directly.** The reading in §14 — that
compatible augmentation is *redundant* augmentation — is currently speculation
fitted to five points. It makes a checkable prediction: augmentation strength
titrated so that `C_g` decreases should improve downstream accuracy up to a point
and then hurt. That is a dosage experiment, cheap relative to what has been
built, and it would test the mechanism rather than correlate endpoints.

**What 1B should not do:** build a DPGA-style method on `C_g`. On this evidence a
method that selected augmentations for gradient compatibility would have
preferred rotation (`C_g` = 0.77, which trained −2.2 pp) over blur (`C_g` = 0.50,
which trained −1.2 pp) — the same failure mode 0B already exposed for `ΔM`.

---

## Appendix A — Commands

```bash
# Prerequisites: Experiment 0A's audit and Experiment 0B's training must have run.
python -m cicps gradients --config config/experiment_1a.yaml   # the measurement
python -m cicps analyze   --config config/experiment_1a.yaml   # summaries, joins, figures
python scripts/check_experiment_1a.py --config config/experiment_1a.yaml

# Re-run one stage without disturbing the others (execution scope only):
python -m cicps gradients --config config/experiment_1a.yaml --stage 0b_baseline_best

# Fast end-to-end plumbing check (12 images, 3 stages; NOT a scientific run):
python -m cicps gradients --config config/experiment_1a.yaml --override config/smoke_1a.yaml
python -m cicps analyze   --config config/experiment_1a.yaml --override config/smoke_1a.yaml
python scripts/check_experiment_1a.py --config config/experiment_1a.yaml --override config/smoke_1a.yaml
```

There is no experiment-selecting flag. A configuration declaring a `gradients`
section is an Experiment 1A run, exactly as a `policies` section makes one a 0B
sweep.

## Appendix B — Code added and changed

**New:**

```text
config/experiment_1a.yaml               the source of truth for 1A
config/smoke_1a.yaml                    tiny end-to-end override
docs/experiment_1a.md                   this handbook
scripts/check_experiment_1a.py          invariant assertions for any 1A run
src/cicps/gradients/                    scopes · accumulate · checkpoints · datasets · records · engine
src/cicps/analysis/gradients.py         summaries, bootstrap CIs, relationships, joins
src/cicps/analysis/gradient_plots.py    the eight 1A figures
src/cicps/stages/gradients.py           the measurement stage
src/cicps/stages/analyze_gradients.py   the analysis stage
```

**Extended (backwards compatible; 0A and 0B behaviour unchanged):**

```text
src/cicps/cli.py             adds the `gradients` command and --stage; analyze dispatches on
                             the `gradients` section
src/cicps/stages/__init__.py exports the two new stages
```

**Not duplicated:** the model factory, the checkpoint format, the transformation
registry, the split logic, the CUB parser, the device resolver, the table
writers, the logging and seeding systems.

Experiments 0A and 0B were re-run after these changes: 0A's analysis reproduces
its published summary, and `scripts/check_smoke_0b.py` still reports 75/75.
