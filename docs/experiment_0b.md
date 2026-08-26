# Experiment 0B — The Augmentation Policy Comparison

**A controlled downstream training comparison on CUB-200-2011.**

---

## 1. Experiment title

**Experiment 0B — Augmentation Policy Comparison: does the transformation-level
discriminative damage measured by Experiment 0A correspond to anything
downstream?**

---

## 2. Motivation

Experiment 0A established a phenomenon:

> Different image transformations produce heterogeneous changes in fine-grained
> discriminative separation when applied to a frozen classifier.

Its per-transformation mean `ΔM` spans a wide range (see §21), and prediction
consistency and feature cosine move with it. That is a real, measurable,
reproducible effect on a frozen model.

But a frozen-model audit is not a training experiment. Nothing in 0A shows that
a transformation which damages the margin of an *already-trained* network is
therefore a worse *training* augmentation. Those are different claims, and the
second one does not follow from the first. In fact there is a well-known reason
to expect the opposite: a perturbation that a network currently handles badly is
exactly the kind of perturbation that training on it might teach the network to
handle.

The whole research programme hinges on which of those is true. If margin damage
under a frozen audit says nothing about downstream usefulness, then a method
that selects augmentations to preserve margin is built on sand. So the link has
to be tested before any such method is proposed, and it has to be tested in a
way that can come back negative.

## 3. Relationship to Experiment 0A

| | Experiment 0A | Experiment 0B |
| --- | --- | --- |
| Model | One, frozen | Six, trained independently |
| Transformations | Applied as a **deterministic probe** at audit time | Applied as **stochastic training augmentation** |
| Parameters | Frozen per `(image, transform)` pair | Re-sampled each time an image is drawn |
| Primary metric | `ΔM` per image per transformation | Internal-validation top-1 per policy |
| Question | Does the phenomenon exist? | Does it matter for training? |
| Output | `transformation_summary.csv` | `policy_summary.csv`, `joined_0a_0b_summary.csv` |

0B **consumes** 0A in exactly one place: the analysis stage reads
`outputs/experiment_0a/analysis/transformation_summary.csv` to put each 0A
transformation's `ΔM` statistics next to the downstream result of the policy
that used it. The file is opened read-only. **Experiment 0B never writes into
`outputs/experiment_0a/`.**

Shared machinery — configuration, seeding, CUB parsing, the split, the model
factory, checkpointing, the trainer, the transformation registry — is reused, not
duplicated. The two experiments are one codebase with one trainer and one
transformation registry, differing in what they ask.

> **0B does not use `ΔM` to choose augmentation policies.** The 0A measurement
> is used *afterward, for analysis only*. It is never a training signal, never a
> selection criterion, never a filter, and never a weight.

## 4. Research question

> **Does the transformation-specific discriminative damage measured by
> Experiment 0A have a measurable relationship with downstream usefulness when
> those transformations are used during model training?**

Operationally:

> If a transformation causes stronger negative `ΔM` under the frozen-model
> audit, does incorporating that transformation into the training augmentation
> policy produce a different downstream recognition outcome?

## 5. Hypotheses

The experiment is stated so that all three outcomes are publishable and none is
privileged by the implementation.

**H1 (relevance).** Transformations differing in 0A `ΔM` produce
correspondingly different downstream validation outcomes when added to a fixed
training recipe.

**H0 (null).** Downstream outcomes do not correspond to 0A `ΔM`. Margin damage
under a frozen audit is real but carries no downstream signal under this
training setup.

**H2 (inversion).** Transformations with *worse* 0A `ΔM` produce *better*
downstream models, because a perturbation the frozen model handles badly is a
useful training perturbation.

No code path encodes an expectation about which holds. The analysis stage
computes the join and the descriptive statistics and stops there; §23 gives the
reading rules.

## 6. What 0B tests

* Whether adding exactly one audited transformation to a fixed, conventional
  training recipe changes internal-validation top-1 / top-5.
* Whether the pattern of those changes across transformations corresponds to
  the pattern of 0A `ΔM` across the same transformations.
* Whether any such correspondence is visible at all at this scale — six runs,
  five transformation points, one seed per policy.

## 7. What 0B does NOT test

* It is **not** DPGA. No augmentation is selected, ranked, filtered, weighted or
  scheduled by `ΔM` or by anything else measured at run time.
* It is **not** an adaptive, learned or per-image augmentation policy. Every
  policy is a fixed recipe declared in YAML before any training happens.
* It does **not** establish causality. A correspondence between 0A `ΔM` and 0B
  accuracy is an association across five transformations, nothing more (§28).
* It is **not** a benchmark or a leaderboard. Accuracy here is an instrument for
  comparing policies under one fixed budget, not a result to be maximised.
* It does **not** test the official CUB test set. That set stays locked (§10).
* It does **not** test augmentation *dosage*. The application probability is
  held constant across arms on purpose (§13).
* It does **not** separate policy effect from seed noise. One seed per policy is
  a real limitation (§24).

---

## 8. Dataset

**CUB-200-2011**, used exactly as distributed — the same dataset, the same
directory layout and the same metadata files as Experiment 0A. See
[`docs/experiment_0a.md`](experiment_0a.md) §6–§8 for the full description; 0B
adds nothing and changes nothing.

| Property | Value |
| --- | --- |
| Images | 11,788 |
| Classes (species) | 200 |
| Official train | 5,994 |
| Official test | 5,794 (locked) |

## 9. Split policy

Experiment 0B reuses **exactly** Experiment 0A's internal split, by pointing at
the same manifest:

```yaml
split:
  path: outputs/experiment_0a/splits/internal_split.json
```

This is deliberate and load-bearing:

* every policy trains and validates on byte-identical image sets, so the split
  cannot be a confound between arms;
* 0A's audit and 0B's comparison speak about the same 600 validation images, so
  the join in §21 relates two measurements over the same population;
* the manifest is fingerprint-checked on load (`splits.py::_verify_split`), so a
  silently changed dataset or split setting raises instead of being tolerated.

The resulting split is 5,394 internal-train / 600 internal-validation images,
stratified by species, seed 42, fingerprint `2c3a0390542d2a57`. The fingerprint
is recorded in every checkpoint, in `train_runs.csv`, and in
`manifests/resolved_config.json`, and the analysis stage warns loudly if the
arms do not all share it.

`prepare` may be run with the 0B configuration; because the manifest already
exists and `split.overwrite` is `false`, it loads and validates rather than
rebuilding.

## 10. Official test-set lock

The official CUB test portion is **never loaded by any stage of Experiment 0B**.

* The internal split is carved from the official *train* portion only, and
  `validate_split` fails the run if any id falls outside it.
* Checkpoint selection uses internal-validation top-1, never test performance.
* No stage reads `train_test_split.txt` rows flagged as test, other than to
  count and report them as untouched.
* `scripts/check_smoke_0b.py` asserts the property directly against the image
  ids a run actually resolved: *"no official test image entered training or
  validation (5794 test images untouched)"*.

## 11. Baseline model

Identical for every policy, and identical to Experiment 0A's:

| Setting | Value |
| --- | --- |
| Architecture | ResNet-18 |
| Initialisation | ImageNet, `IMAGENET1K_V1` |
| Head | fresh `nn.Linear(512, 200)` |
| Feature dim | 512 |

Every policy builds its own fresh model. Nothing is carried over between arms —
no fine-tuning of one arm from another, no shared optimizer state, no warm start.

**Initialisation is a control variable.** The training stage installs the
*master* seed while the model is built, so all six arms start from bit-identical
weights (the same pretrained backbone *and* the same randomly drawn head), and
only then hands the RNG over to the policy-specific stream (§16). The invariant
script verifies this by rebuilding twice under the master seed and comparing
tensors.

## 12. Training setup

Copied unchanged from `config/experiment_0a.yaml`. Held constant across every
policy — no per-policy tuning of any kind.

| Setting | Value |
| --- | --- |
| Optimizer | AdamW, lr 3e-4, weight decay 0.05, betas (0.9, 0.999) |
| Schedule | Cosine decay, 1 warmup epoch, min lr 1e-6 |
| Epochs | 20 |
| Batch size | 32 (eval 64) |
| Loss | Cross-entropy, label smoothing 0.0 |
| Mixed precision | fp16 autocast, enabled on CUDA |
| Gradient clipping | disabled |
| Validation preprocessing | deterministic canonical view (resize 256 → centre-crop 224 → normalise) |
| Checkpoint selection | best `val_top1` |

Validation preprocessing is never touched by a policy: a policy changes only
what happens during *training*. Every arm is therefore evaluated through exactly
the same deterministic pipeline.

## 13. Exact augmentation policies

Six arms. Every arm contains the **same standard baseline recipe**, and differs
from `baseline` by exactly one modification.

### The standard baseline recipe

```text
RandomResizedCrop(size=224, scale=[0.7, 1.0], ratio=[0.75, 1.3333])
+ RandomHorizontalFlip(p=0.5)
+ Normalize(ImageNet mean/std)
```

This is Experiment 0A's training recipe verbatim (`train.augmentation`).

### The six policies

| Policy | Modification | 0A transformation joined for analysis |
| --- | --- | --- |
| `baseline` | none — reference arm | — |
| `baseline_color_jitter` | `add` colour jitter, p = 0.5 | `color_jitter` |
| `baseline_rotation` | `add` rotation, p = 0.5 | `rotation` |
| `baseline_crop` | `override` the baseline crop parameters, p = 0.5 | `random_resized_crop` |
| `baseline_gaussian_blur` | `add` Gaussian blur, p = 0.5 | `gaussian_blur` |
| `baseline_random_erasing` | `add` random erasing, p = 0.5 | `random_erasing` |

**There is deliberately no horizontal-flip arm.** Horizontal flip is already
part of the standard baseline recipe, so a "baseline + horizontal flip" policy
would either duplicate an existing stage or be identical to the baseline.

### Stage order

```text
crop → flip → [policy transformation, PIL space] → normalise → [policy transformation, tensor space]
```

This mirrors Experiment 0A's audit pipeline exactly (canonical view → `apply_pil`
→ normalise → `apply_tensor`), so a transformation behaves identically in both
experiments. Colour jitter, rotation and blur act in PIL space; random erasing
acts in normalised tensor space, matching torchvision.

### Application probability

```yaml
policies:
  application_probability: 0.5
```

Every non-baseline arm applies its single modification with **p = 0.5**, the
same value for all of them. The value lives only in the configuration; it is
never hard-coded in Python.

Why 0.5:

* Experiment 0A applied its transformations deterministically because it was
  probing a frozen model. As a *training* augmentation a transformation needs an
  explicit application probability, and the recipe this repository already uses
  states its own as `0.5` (`train.augmentation.horizontal_flip.p`). 0.5 is
  therefore the existing convention rather than a new tuning knob.
* At p = 0.5 each arm is an even mixture of "baseline" and "baseline plus the
  audited transformation", so an arm can neither collapse onto the baseline
  (p = 0) nor stop being a controlled increment on it (p = 1).
* It is **identical across all arms on purpose.** Tuning it per policy would
  confound transformation identity with augmentation dosage — exactly the
  confound this experiment exists to avoid.

## 14. Policy comparison rationale

### Why "baseline + one transformation" and not "one transformation per arm"

Replacing the whole recipe per arm would confound the effect of the added
transformation with the effect of *removing* `RandomResizedCrop` and
`HorizontalFlip`. An arm that scored badly would be uninterpretable: was it the
blur, or the missing crop? Adding one transformation on top of a fixed recipe
isolates the increment, which is the quantity the research question is about.

### Why the crop arm overrides rather than adds

The baseline **already contains** a `RandomResizedCrop`. Stacking a second,
independent crop on top of it would measure "two sequential crops", not
"Experiment 0A's crop strength", and would shrink the effective field of view far
beyond either configuration.

The controlled implementation is to **substitute** the 0A crop parameters for
the baseline crop parameters, on the same `p` fraction of images:

```text
with probability p:      RandomResizedCrop(scale=[0.5, 0.9],  ratio=[0.75, 1.3333])   # 0A strength
otherwise:               RandomResizedCrop(scale=[0.7, 1.0],  ratio=[0.75, 1.3333])   # baseline
```

This keeps the arm structurally parallel to the others: in every non-baseline
policy, a `p` fraction of training images receive the audited perturbation and
the remaining `1 - p` receive the plain baseline treatment. The fraction of
images touched by the intervention is identical across all five arms.

The `override` mechanism is a first-class part of the policy abstraction
(`src/cicps/training/policies.py`), not a special case: a policy declares either
`add` (append a transformation the baseline lacks) or `override`
(re-parameterise a stage the baseline already has), never both, and the
configuration loader rejects a policy that declares both.

## 15. Transformation parameter values

Every strength is Experiment 0A's audited value, unchanged, so that the `ΔM` a
transformation earned in 0A refers to the same perturbation 0B trains with.

### Colour jitter
```yaml
brightness: 0.4
contrast: 0.4
saturation: 0.4
hue: 0.1
```

### Rotation
```yaml
degrees: 15.0
interpolation: bilinear
expand: false
fill: 0
```

### Random resized crop (the `baseline_crop` override)
```yaml
scale: [0.5, 0.9]
ratio: [0.75, 1.3333]
interpolation: bilinear
```

### Gaussian blur
```yaml
kernel_size: 9
sigma: [1.0, 2.0]
```

### Random erasing
```yaml
scale: [0.05, 0.2]
ratio: [0.3, 3.3]
value: 0.0     # normalised tensor space; 0.0 == the ImageNet mean colour
```

### Baseline stages
```yaml
random_resized_crop: {scale: [0.7, 1.0], ratio: [0.75, 1.3333]}
horizontal_flip:     {p: 0.5}
```

**Strengths were fixed before any 0B model was trained and must not be retuned
after seeing a downstream number** (§17).

## 16. Seed and reproducibility policy

### Deterministic vs. stochastic — the key distinction

Experiment 0A makes each audited transformation **deterministic per
`(image, transform)`** so that `ΔM` measures one fixed perturbation rather than
a random draw. That mechanism is right for an audit and **wrong for training**:
an augmentation that applies the same fixed rotation to a given image in every
epoch is not really augmenting.

Experiment 0B therefore **does not** force the audit's per-image freezing onto
the training pipeline. It reuses the same transformation *implementations* —
literally the same `DeterministicTransform` subclasses from the shared registry —
but calls `resolve()` against the process-global `random` stream instead of a
per-sample frozen generator (`src/cicps/transforms/policy_pipeline.py`). Strengths
therefore cannot drift apart between the two experiments, while the sampling
semantics stay appropriate to each.

Reproducibility comes from the **existing global seeding system**, not from
freezing parameters:

* `seed_run()` seeds Python / NumPy / torch / CUDA in the main process;
* `worker_init_fn` seeds every DataLoader worker from the parent generator;
* `dataloader_generator(policy_seed)` fixes the shuffling order.

Given the same configuration, code, split, policy and seed, a policy run replays
identically. This was verified in practice: re-running a single policy after the
full sweep reproduced its `best_val_top1` exactly.

### Policy-derived seeds

```text
policy_seed = blake2b(master_seed, "policy", <policy name>) mod 2^31
```

implemented as `seeding.derive_run_seed`, built on the existing
`seeding.derive_seed` (BLAKE2b, **not** Python's per-process-salted `hash()`).
Consequences:

* the same master seed and policy name always give the same policy seed, across
  runs, processes and machines;
* every policy has an **independent** RNG stream, so adding or renaming a policy
  cannot silently perturb another policy's randomness;
* the resolved seed is logged, written into `train_runs.csv`, and stored in
  every checkpoint.

Resolved seeds under `seed.value: 42`:

| Policy | Seed |
| --- | --- |
| `baseline` | 1880975077 |
| `baseline_color_jitter` | 503065231 |
| `baseline_rotation` | 69590260 |
| `baseline_crop` | 1179721965 |
| `baseline_gaussian_blur` | 1659421884 |
| `baseline_random_erasing` | 1505275385 |

### The two-step seeding sequence

Each policy is seeded twice, on purpose:

1. **master seed installed → model built.** Initialisation is a control
   variable, so every arm starts from bit-identical weights (§11).
2. **policy seed installed → training runs.** Shuffling order and augmentation
   draws get an independent per-policy stream.

Both steps are logged explicitly at the start of every policy run.

### Residual non-determinism

The same caveats as Experiment 0A apply: GPU floating-point reductions can differ
across GPU models, driver and library versions, so a *retrained* checkpoint will
not be bitwise identical to one trained on different hardware. `cudnn.deterministic`
is on and `cudnn.benchmark` off by default. Mixed precision is enabled for
training. See [`docs/experiment_0a.md`](experiment_0a.md) §30.

## 17. Analysis safeguard — the transformation set is predetermined

The transformation set is fixed:

```text
color_jitter · rotation · random_resized_crop · gaussian_blur · random_erasing
```

* **The 0A results must not be used to choose which 0B policies are included.**
* Gaussian blur is **not** dropped for looking destructive.
* No transformation is added for looking interesting.
* No strength is retuned after seeing a downstream result.
* No policy gets extra epochs, a different learning rate, a different
  architecture or a different checkpoint rule.

0B is a predetermined follow-up experiment. If a future variation is wanted, it
is a new experiment with a new configuration file, not an edit to this one.

## 18. Training procedure

```text
for each policy declared in policies.list:
    resolve policy seed from the master seed and the policy name
    build the policy training pipeline (baseline + one modification)
    install the MASTER seed
    build a fresh ImageNet-pretrained ResNet-18            # identical across arms
    install the POLICY seed
    train for train.epochs, evaluating after every epoch   # identical budget
    save best (and last) checkpoint into checkpoints/<policy>/
    write per-epoch history to logs/history/<policy>.csv
    append/replace this policy's row in logs/train_runs.csv
write manifests/resolved_config.json
```

Six independent runs. One model is never mutated into another.

There is exactly **one trainer** (`training/trainer.py`), shared with Experiment
0A. It gained optional, defaulted parameters — a training pipeline, a
`TrainingArtifacts` output location, a run label and extra checkpoint metadata —
so that 0B can scope it per policy. Omit them all and it behaves exactly as 0A's
single-model run always has. There is no `train_0b.py`, no `train_policy_1.py`,
no duplicated model and no duplicated transformation registry.

## 19. Checkpoint selection

**Best internal-validation top-1**, for every policy, via the existing
centralised mechanism (`train.checkpoint.monitor: val_top1`, `mode: max`).

* Never the official test set.
* Never a 0A quantity.
* Never a per-policy rule.

Each policy writes into its own directory, so arms cannot overwrite one another:

```text
outputs/experiment_0b/checkpoints/<policy>/best.pt
outputs/experiment_0b/checkpoints/<policy>/last.pt
```

Each checkpoint carries the model state, optimizer and scheduler state, the full
resolved configuration, the epoch, the policy seed, the metrics, the split
fingerprint, the device, and — new for 0B — the resolved policy definition and
the resolved augmentation pipeline, so a checkpoint can be traced to exactly the
augmentation it was trained under.

## 20. Metrics

### Per epoch (`logs/history/<policy>.csv`)

`epoch`, `lr`, `train_loss`, `train_top1`, `train_top5`, `val_loss`, `val_top1`,
`val_top5`, `seconds`.

> `train_top5` is a small additive change to the shared trainer. Experiment 0A's
> history CSV gains the same column; nothing in 0A consumes that file
> programmatically, and no 0A result changes.

### Per policy (`logs/train_runs.csv`)

Identity and configuration: `policy`, `audit_transform`, `is_baseline`,
`description`, `seed`, `split_fingerprint`, `num_train_images`,
`num_val_images`, `epochs`, `batch_size`, `optimizer`, `learning_rate`,
`weight_decay`, `scheduler`, `architecture`, `pretrained_weights`, `device`,
`application_probability`.

Outcome: `best_epoch`, `best_val_top1`, `best_val_top5`, `best_val_loss`,
`best_train_top1`, `best_train_top5`, `best_train_loss`, `final_val_top1`,
`final_val_top5`, `final_val_loss`, `final_train_top1`, `final_train_top5`,
`final_train_loss`, `total_seconds`, `checkpoint_path`, `history_path`,
`completed_at`.

### Guards

Non-finite losses and non-finite epoch metrics raise immediately, naming the
policy and epoch, rather than being checkpointed and silently carried into the
analysis.

**No metric is computed on the official CUB test set.**

## 21. Relationship between 0A and 0B — the joined analysis

The headline analysis is **not** "which policy scored highest". It is the join.

For each transformation the analysis retrieves, from Experiment 0A's existing
summary: mean `ΔM`, median `ΔM`, `frac_delta_negative`, mean |`ΔM`|, mean feature
cosine, prediction-consistency rate (and the remaining 0A columns). It puts them
next to that transformation's 0B policy result: best validation top-1 and top-5,
validation loss, and the change relative to the `baseline` arm.

The link comes from `policies.list[].audit_transform` in the configuration —
declared alongside the policy — so the correspondence is explicit rather than
inferred from a policy name.

`outputs/experiment_0b/analysis/joined_0a_0b_summary.csv`, conceptually:

| Transformation | 0A mean ΔM | 0A median ΔM | 0A negative fraction | 0B val top-1 | Δ val top-1 vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| color_jitter | … | … | … | … | … |
| rotation | … | … | … | … | … |
| random_resized_crop | … | … | … | … | … |
| gaussian_blur | … | … | … | … | … |
| random_erasing | … | … | … | … | … |

For reference, the 0A column values this join draws on (from the completed 0A
run, 600 validation images × 7 arms):

| Transformation | mean ΔM | median ΔM | frac ΔM < 0 | feature cosine | pred. consistency |
| --- | ---: | ---: | ---: | ---: | ---: |
| horizontal_flip | +0.0349 | +0.0013 | 0.442 | 0.962 | 0.883 |
| rotation | −0.0072 | −0.0007 | 0.530 | 0.960 | 0.865 |
| random_resized_crop | −0.0593 | −0.0046 | 0.568 | 0.938 | 0.790 |
| random_erasing | −0.1216 | −0.0149 | 0.650 | 0.927 | 0.745 |
| color_jitter | −0.1453 | −0.0441 | 0.683 | 0.919 | 0.695 |
| gaussian_blur | −0.2111 | −0.1458 | 0.713 | 0.865 | 0.610 |

**These are inputs to the join, not predictions.** The analysis must reveal
whether a relationship exists; it must not assume one.

### Exploratory association

Optionally (`analysis.association.enabled`), Pearson and Spearman coefficients
between `delta_margin_mean` and the 0B outcomes are written to
`association_exploratory.csv` and logged. Every row carries `n_points` and the
literal string *"exploratory / descriptive only; no significance test"*.

There are **five** transformation points. No p-value is computed and no
hypothesis test is run — deliberately (§29 of the specification, and §24 below).
A coefficient over five points is unstable by construction and must never be
reported as significance.

## 22. Required plots

Written to `outputs/experiment_0b/analysis/plots/`, all configuration driven via
`analysis.plots.kinds`.

| File | Content |
| --- | --- |
| `validation_top1_by_policy.png` | Best validation top-1 per policy; baseline drawn in grey with an explicit dashed reference line |
| `validation_top5_by_policy.png` | Same, for top-5 |
| `validation_curves.png` | Validation top-1 per epoch, one line per policy; baseline dashed |
| `training_curves.png` | Training top-1 and validation loss per epoch, one line per policy — shows whether differences arise early or emerge late |
| `delta_margin_vs_accuracy_delta.png` | **The key plot.** x = 0A mean `ΔM`, y = change in 0B validation top-1 vs. baseline. One labelled point per transformation, zero reference lines on both axes |
| `delta_margin_vs_accuracy.png` | Same, against absolute validation top-1, with the baseline as a horizontal reference |

The two scatter plots carry `n = <count> transformations` in the title and the
caption *"Descriptive only. Association is not causation, and any trend line over
N points is a visual aid, not evidence."* on the figure itself. The optional
least-squares trend line (`analysis.plots.trend_line`) is exactly that — a visual
aid.

> **Axis note.** The two bar charts are **not** zero-anchored. Differences
> between arms are small next to the absolute accuracy, and a zero-anchored bar
> chart would hide them entirely. Read the reference line, not the bar heights.

## 23. Possible outcomes and their interpretation

The implementation encodes no expectation. All three readings below are
legitimate results and none may be suppressed.

### Outcome A — Strong support

Transformations with more negative 0A `ΔM` consistently produce poorer
downstream outcomes.

> 0A's discriminative-preservation measurement appears relevant to downstream
> augmentation usefulness.

This would motivate a later experiment asking whether discriminative-preservation
information can *guide* augmentation selection. It would not by itself establish
that mechanism.

### Outcome B — Weak or no relationship

0A `ΔM` differs substantially across transformations, but downstream policy
performance does not correspond to it.

> Transformation-induced margin damage exists, but the 0A measurement is not
> sufficient to predict augmentation usefulness under this training setup.

**This is a scientifically valuable result and must not be hidden.** It would
say that the frozen-model audit, on its own, is not a basis for an augmentation
method — which is exactly what the programme needs to know before building one.

### Outcome C — Opposite relationship

Transformations with worse 0A `ΔM` produce *better* downstream models.

> The transformation may be damaging to the frozen model's current
> representation while acting as a useful training perturbation that improves
> learned robustness.

This would be the most important outcome of the three, because it would
demonstrate directly why a naive "preserve margin at all costs" method could be
wrong.

### Reading rules

* Report the direction and the magnitude, and report `n`.
* Do not call a five-point pattern significant.
* Do not report a smoke-run number as a result, ever.
* Do not describe any arm as "the best augmentation".
* Distinguish "was associated with" from "caused" (§28 of the specification).

## 24. Limitations

1. **One seed per policy.** Part of the difference between arms is training
   noise, not policy effect. With six runs there is no variance estimate. The
   honest fix — several seeds per policy and a within-policy spread — is out of
   scope for this first implementation and is the obvious next refinement.
2. **Five transformation points.** Any correlation is descriptive. This is why
   no significance test is implemented.
3. **One dataset, one architecture, one budget.** Nothing here generalises
   beyond CUB-200-2011 with a 20-epoch ImageNet-pretrained ResNet-18.
4. **One application probability.** Held constant on purpose, but that means 0B
   says nothing about how these transformations behave at other dosages.
5. **Internal validation only.** The official test set stays locked, so
   validation top-1 is an internal comparison metric, not a generalisation
   estimate. Checkpoint selection also uses it, which makes the reported best
   value mildly optimistic in the usual way — equally so for every arm.
6. **The join is across transformations, not within.** Five paired
   `(ΔM, accuracy)` observations cannot separate transformation identity from
   any other property those transformations happen to share.
7. **0A `ΔM` was measured on a model trained with the baseline recipe.** The 0A
   frozen model is the baseline policy's kind of model, so the audit's reference
   point is not neutral with respect to the arms.

## 25. Smoke-test procedure

`config/smoke_0b.yaml` runs the **entire** 0B pipeline on a tiny workload:
configuration loading → split loading and validation → policy construction →
dataset construction → training → validation → checkpoint saving → history
generation → 0A-result loading → joined analysis → all required plots → summary
CSVs.

It trains **all six policies** (2 epochs, 240 train / 120 validation images),
because "does policy construction work for every arm" is one of the things it
checks. Everything is written under `outputs/smoke_0b/`, never mixed with
scientific outputs.

```bash
python -m cicps prepare --config config/experiment_0b.yaml --override config/smoke_0b.yaml
python -m cicps train   --config config/experiment_0b.yaml --override config/smoke_0b.yaml
python -m cicps analyze --config config/experiment_0b.yaml --override config/smoke_0b.yaml
python scripts/check_smoke_0b.py --config config/experiment_0b.yaml --override config/smoke_0b.yaml
```

> **The smoke run is not a scientific run.** Two epochs on a few hundred images
> produce a near-random model; the numbers are meaningless, the differences
> between policies are noise, and neither may ever be reported.

### The smoke test asserts, it does not merely exit 0

`scripts/check_smoke_0b.py` is configuration driven, so it validates a smoke run
and a scientific run with the same code. It checks, and reports every failure
rather than stopping at the first:

* all six policies were constructed, with unique names and distinct seeds;
* each policy has the expected augmentation configuration — the declared
  transformation type and parameters for `add` arms, and for the crop arm that
  the declared parameters were *substituted* for the baseline's rather than a
  second crop stacked on;
* the baseline arm is unmodified, and every arm shares one application
  probability;
* model initialisation is bit-identical under the master seed;
* training and validation datasets are non-empty;
* one or more epochs completed for every policy;
* every recorded loss and metric is finite;
* every policy produced a checkpoint and a history record;
* every policy shares the same split fingerprint, dataset sizes, epochs, batch
  size, optimizer, learning rate, weight decay, scheduler, architecture and
  pretrained weights;
* **no official test image entered the training or validation datasets**, and
  train/val do not overlap;
* `policy_summary.csv` and `joined_0a_0b_summary.csv` exist, cover every policy,
  and carry the baseline-relative delta and the 0A `ΔM`;
* every required plot exists and is non-empty.

Last run: **75/75 checks passed.**

## 26. Scientific-run commands

```bash
# 0. Environment (once)
python -m venv .venv
.venv\Scripts\activate                     # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .

# 1. Validate the dataset and the shared internal split
python -m cicps prepare --config config/experiment_0b.yaml

# 2. Train all six policies (one command, six independent runs)
python -m cicps train --config config/experiment_0b.yaml

# 3. Summarise the sweep and join it against Experiment 0A
python -m cicps analyze --config config/experiment_0b.yaml

# 4. Verify the invariants of the completed run
python scripts/check_smoke_0b.py --config config/experiment_0b.yaml
```

Experiment 0B **requires that Experiment 0A's analysis has already been run**,
because step 3 reads `outputs/experiment_0a/analysis/transformation_summary.csv`.
It opens that file read-only.

### There is no experiment-selecting flag

`train` and `analyze` are the same commands for both experiments. The
configuration decides: a file that declares a `policies` section is a 0B policy
sweep. Putting an experiment selector on the command line would be an
experiment-defining value living outside the YAML file.

### Re-running only the analysis

```bash
python -m cicps analyze --config config/experiment_0b.yaml
```

Reads `logs/train_runs.csv` and `logs/history/*.csv`; retrains nothing.

### Re-running one policy without destroying previous outputs

```bash
python -m cicps train --config config/experiment_0b.yaml --policy baseline_gaussian_blur
```

`--policy` is repeatable. It is an **execution-scope** flag for re-runs, not an
experiment-defining one: it can only select a policy already declared in the
configuration, it cannot invent or modify one, and using it logs a warning. The
run rewrites only that policy's checkpoint directory and history file, and
*merges* its row into `train_runs.csv`, leaving every other policy's row intact.
Passing it with a 0A configuration is rejected with an explanatory error.

### GPU requirements

The same as Experiment 0A: ResNet-18 at 224 px with batch size 32 and fp16
autocast fits in a 4 GB card. The reference 0A run used an RTX 3050 Laptop
(4 GB). CPU-only will work and will be very slow. 0B is six sequential single-GPU
runs; it needs no more memory than 0A, only more time.

### Expected runtime

**The scientific 0B run has not been executed in this repository, so no measured
runtime exists for it.** What *is* measured:

* Experiment 0A's completed 20-epoch run: **20.8 s/epoch, 415 s total** on an
  RTX 3050 Laptop (4 GB) with `num_workers: 4`.
* The 0B smoke sweep (2 epochs × 240 images × 6 policies, `num_workers: 0`):
  **42 s of training in total**.

0B trains six models under 0A's exact budget, so on comparable hardware expect
roughly **six times the 0A single-model time — on the order of 40–45 minutes** —
plus a few seconds for the analysis. That figure is an **extrapolation from the
0A reference run, not a measurement**, and the per-epoch cost of the heavier arms
(blur in particular) will be somewhat higher than the baseline's.

## 27. Output directory structure

```text
outputs/experiment_0b/
├── checkpoints/
│   ├── baseline/{best.pt, last.pt}
│   ├── baseline_color_jitter/{best.pt, last.pt}
│   ├── baseline_rotation/{best.pt, last.pt}
│   ├── baseline_crop/{best.pt, last.pt}
│   ├── baseline_gaussian_blur/{best.pt, last.pt}
│   └── baseline_random_erasing/{best.pt, last.pt}
│
├── logs/
│   ├── prepare.log
│   ├── train.log                       # the whole sweep
│   ├── train_baseline.log              # one per policy, for per-arm diagnosis
│   ├── train_baseline_color_jitter.log
│   ├── train_baseline_rotation.log
│   ├── train_baseline_crop.log
│   ├── train_baseline_gaussian_blur.log
│   ├── train_baseline_random_erasing.log
│   ├── analyze.log
│   ├── train_runs.csv                  # one row per policy
│   └── history/<policy>.csv            # per-epoch metrics, one file per policy
│
├── analysis/
│   ├── policy_summary.csv
│   ├── joined_0a_0b_summary.csv
│   ├── association_exploratory.csv
│   └── plots/
│       ├── validation_top1_by_policy.png
│       ├── validation_top5_by_policy.png
│       ├── validation_curves.png
│       ├── training_curves.png
│       ├── delta_margin_vs_accuracy_delta.png
│       └── delta_margin_vs_accuracy.png
│
└── manifests/
    └── resolved_config.json            # resolved config, policies, seeds, split, runs
```

Smoke outputs mirror this under `outputs/smoke_0b/` and are git-ignored. Every
path is configurable; none is hard-coded.

### What each policy log makes diagnosable

Each per-policy log records the experiment name, policy name, description, the
0A transformation it links to, the master seed and the resolved policy seed, the
device, the split fingerprint and split seed, train/validation image counts, the
architecture and pretrained weights, the optimizer, learning rate, weight decay,
scheduler and warmup, the epoch budget, batch size and checkpoint monitor, the
fully resolved augmentation pipeline as JSON, per-epoch progress, and the best
epoch with its validation top-1/top-5. An OOM, a NaN loss, a missing checkpoint,
a wrong split, a wrong policy, a wrong seed, a wrong device or accidental
test-set use is diagnosable from the log without rerunning anything.

## 28. Reproducibility checklist

- [x] Single master seed in the configuration (`seed.value: 42`)
- [x] Policy seeds derived deterministically via BLAKE2b, never `hash()`
- [x] Every policy's resolved seed logged and persisted
- [x] Independent RNG stream per policy
- [x] Model initialisation identical across policies, seeded from the master seed
- [x] DataLoader workers seeded
- [x] Shuffling order fixed by a per-policy generator
- [x] Split loaded from a fingerprinted manifest, shared with Experiment 0A
- [x] Split fingerprint recorded in every checkpoint, run row and manifest
- [x] Full resolved configuration saved to `manifests/resolved_config.json`
- [x] Resolved augmentation pipeline recorded per policy and per checkpoint
- [x] Configuration is the single source of truth; no experiment value in Python
- [x] Official CUB test set never loaded; asserted by the invariant script
- [x] `cudnn.deterministic` on, `cudnn.benchmark` off
- [x] Residual non-determinism documented (§16)

## 29. What experiment follows 0B

Deliberately stated as a branch, not a plan, because the next step depends on
which outcome 0B produces.

**If Outcome A (support).** The next experiment asks whether
discriminative-preservation information can *guide* augmentation selection —
i.e. the first genuine DPGA prototype — and must first establish the effect
survives multiple seeds and a second architecture.

**If Outcome B (null).** The measurement is not yet a usable signal. The next
step is diagnostic: is `ΔM` measured on the wrong model (a frozen baseline
rather than the model being trained), at the wrong granularity (per
transformation rather than per image), or is discriminative preservation simply
not what determines augmentation usefulness?

**If Outcome C (inversion).** The most informative branch. The next experiment
would characterise *when* margin damage is useful — presumably a
difficulty/robustness trade-off — and would be strong evidence against any naive
margin-preserving method, which is worth knowing before one is built.

In all three cases the immediate methodological follow-up is the same: **repeat
0B with several seeds per policy**, so that the difference between arms can be
separated from training noise (§24.1).

---

## 30. Scientific run results (executed 2026-08-26)

The scientific run described in §26 was executed end-to-end on the reference
hardware (RTX 3050 Laptop, 4 GB) against `config/experiment_0b.yaml` with no
overrides. `scripts/check_smoke_0b.py` reported **75/75 checks passed** — all
six arms share the split fingerprint `2c3a0390542d2a57`, dataset sizes,
epoch/batch/optimizer/scheduler/architecture settings, every recorded metric is
finite, no official test image entered training or validation, and every
required checkpoint, history file and plot exists.

### Per-policy result (best internal-validation top-1)

| Policy | 0A transform | Best val top-1 | Δ vs baseline | Best val top-5 | Best epoch |
| --- | --- | ---: | ---: | ---: | ---: |
| `baseline` | — | 0.7817 | — | 0.9300 | 14 |
| `baseline_crop` | random_resized_crop | 0.7867 | **+0.0050** | 0.9417 | 19 |
| `baseline_gaussian_blur` | gaussian_blur | 0.7700 | −0.0117 | 0.9367 | 19 |
| `baseline_rotation` | rotation | 0.7600 | −0.0217 | 0.9383 | 15 |
| `baseline_random_erasing` | random_erasing | 0.7583 | −0.0233 | 0.9267 | 19 |
| `baseline_color_jitter` | color_jitter | 0.7383 | −0.0433 | 0.9183 | 16 |

Only `baseline_crop` matched or exceeded the baseline; every other arm lost
between roughly 1 and 4 accuracy points. With one seed per policy (§24.1) this
ordering carries training noise as well as policy effect and must not be read
as a ranking of "good" vs "bad" augmentations.

### The 0A/0B join

| Transformation | 0A mean ΔM | 0A frac ΔM<0 | 0B Δ val top-1 vs baseline |
| --- | ---: | ---: | ---: |
| gaussian_blur | −0.2111 (worst) | 0.713 | −0.0117 |
| color_jitter | −0.1453 | 0.683 | −0.0433 (worst) |
| random_erasing | −0.1216 | 0.650 | −0.0233 |
| random_resized_crop | −0.0593 | 0.568 | +0.0050 (best) |
| rotation | −0.0072 (least negative) | 0.530 | −0.0217 |

Exploratory association (`association_exploratory.csv`, n = 5 transformation
points, descriptive only, no significance test): Pearson r = **0.200**,
Spearman ρ = **0.200**, both between 0A `delta_margin_mean` and the 0B
baseline-relative accuracy delta.

### Reading against §23's three outcomes

The ranking by 0A damage (worst → least: blur, color_jitter, random_erasing,
crop, rotation) does **not** track the ranking by downstream accuracy loss
(worst → least: color_jitter, random_erasing, rotation, blur, crop). The
transformation with by far the most negative 0A `ΔM` — Gaussian blur, more than
40% more negative than the next-worst arm — produced only the **second-mildest**
downstream accuracy drop, beaten only by the crop arm. Conversely, rotation,
which had the mildest 0A damage of the five (`ΔM` an order of magnitude closer
to zero than blur's), produced one of the larger downstream losses. The
correlation across all five points is weak and positive (r ≈ 0.2) — far too
close to zero, and computed on far too few points (§24.2), to read as
confirmation of anything.

**This is Outcome B (§23): a weak-or-no relationship, with a visible element of
Outcome C for the blur arm specifically.** Concretely:

* **H0 (null) is the best-supported reading of the data as a whole.**
  Transformation-induced margin damage under the frozen-model audit does not
  correspond, in any consistent way, to how a training arm built around that
  same transformation performs downstream at this scale (five points, one seed
  each, 20-epoch budget).
* **H1 (relevance) is not supported.** If anything, the transformation that
  looked most damaging to the frozen model (blur) was one of the least harmful
  to actually train with.
* **H2 (inversion) is not established either**, and must not be claimed from
  this run: only one arm (crop) beat the baseline at all, and blur still lost
  accuracy relative to baseline — it simply lost less than three of the four
  other arms, despite having the worst audit score by a wide margin. A true
  inversion result would need worse-audited transformations to reliably produce
  *better* downstream models, which did not happen here.

Per §23's reading rules: this reports direction and magnitude with `n = 5`, does
not call the pattern significant, does not name any arm "the best augmentation"
(the crop arm's +0.005 is within plausible single-seed noise), and treats the
0A/0B relationship as association, not causation.

### What this means for the research programme

Consistent with the motivation in §2: **the frozen-model audit, on its own, is
not yet a basis for selecting training augmentations.** A method that picked
augmentations by preserving 0A `ΔM` would, on this evidence, have down-ranked
Gaussian blur — the transformation that turned out to be one of the more
usable training perturbations — while up-ranking rotation, which trained worse
than three of the four other modified arms. This is exactly the scientifically
valuable negative result flagged in §23 Outcome B, and per §29 the honest next
step is diagnostic (is `ΔM` measured on the wrong model, at the wrong
granularity, or simply not the right signal for augmentation usefulness) and
methodological (repeat with several seeds per policy before drawing any firm
conclusion from five single-seed points).

Full artefacts: `outputs/experiment_0b/analysis/policy_summary.csv`,
`joined_0a_0b_summary.csv`, `association_exploratory.csv`, and the six plots
under `outputs/experiment_0b/analysis/plots/`.

---

## Appendix A — Configuration structure

`config/experiment_0b.yaml` is the only source of experiment configuration.

| Section | Controls |
| --- | --- |
| `experiment` | Name and description |
| `seed` | Master seed, cudnn flags, deterministic algorithms, worker seeding |
| `dataset` | CUB root, metadata filenames, expected counts, validation strictness |
| `split` | Path to the **shared** 0A split manifest, fraction, stratification, seed |
| `model` | Architecture, pretrained flag, weights enum, `num_classes` |
| `preprocess` | `image_size`, `resize_size`, interpolation, normalisation |
| `train` | Epochs, batch sizes, workers, AMP, optimizer, scheduler, loss, **the standard baseline augmentation recipe**, optional subset caps, checkpointing |
| `policies` | **The six arms**, the shared application probability, the baseline arm's name, output paths |
| `analysis` | 0A reference summary path, exploratory association settings, output paths, plot kinds/size/format |
| `runtime` | Device selection, `require_device` |
| `logging` | Level, directory, file template, format |

`train.max_train_images` / `train.max_val_images` cap the internal split for
smoke runs. They must be `null` for a scientific run; when set, the subset is
drawn from the **master** seed so every policy sees identical images, and the
training stage logs a prominent `SUBSET ACTIVE` warning.

## Appendix B — Code added and changed

**New:**

```text
config/experiment_0b.yaml               the source of truth for 0B
config/smoke_0b.yaml                    tiny end-to-end override
docs/experiment_0b.md                   this handbook
scripts/check_smoke_0b.py               invariant assertions for any 0B run
src/cicps/training/policies.py          AugmentationPolicy, build_policies, select_policies
src/cicps/transforms/policy_pipeline.py stochastic training wrappers over the shared registry
src/cicps/stages/train_policies.py      the policy sweep
src/cicps/stages/analyze_policies.py    policy summary + the 0A/0B join
src/cicps/analysis/policies.py          summary, join, exploratory association
src/cicps/analysis/policy_plots.py      the six 0B figures
```

**Extended (backwards compatible; Experiment 0A behaviour unchanged):**

```text
src/cicps/cli.py             dispatches on the `policies` section; adds --policy (execution scope)
src/cicps/seeding.py         derive_run_seed, seed_run
src/cicps/logging_utils.py   additional_log_file (per-policy log tee)
src/cicps/training/trainer.py  train_top5; TrainingArtifacts; optional pipeline/artifacts/run
                               name/extra metadata; best-epoch tracking; non-finite guards
src/cicps/config.py          __contains__ bug fix (see below)
```

**Not duplicated:** the trainer, the model factory, the checkpoint format, the
transformation registry, the split logic, the CUB parser, the device resolver.

### One pre-existing bug fixed

`Config.__contains__` passed the module-private `_MISSING` sentinel as `get()`'s
default, which `get()` reads as *"no default supplied"* — so `key in config`
raised `ConfigError` on exactly the absent keys it was supposed to report as
`False`. Nothing previously exercised it. It is now fixed with a distinct
sentinel, which is what lets `train`/`analyze` dispatch on the presence of the
`policies` section.
