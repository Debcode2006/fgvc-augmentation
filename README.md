# CICPS — Augmentation and Fine-Grained Discriminative Evidence

Research code for measuring how image augmentation affects **fine-grained
discriminative evidence** on CUB-200-2011.

Programme question:

> Can the preservation of fine-grained discriminative evidence under image
> augmentation be quantified?

## Status

| Stage | State |
| --- | --- |
| **Experiment 0A — transformation audit** | ✅ implemented |
| **Experiment 0B — augmentation policies vs. downstream usefulness** | ✅ implemented |
| DPGA — Discriminative-Preservation-Guided Augmentation | ❌ not implemented |

**Experiment 0A** is a falsification / mechanism test. It freezes one ordinary
ResNet-18 baseline and asks whether seven deterministic transformations cause
measurably *different* changes in an image's probability margin against its
frozen hardest competing class:

$$\Delta M = \big(p_y(T(x)) - p_c(T(x))\big) - \big(p_y(x) - p_c(x)\big)$$

Full details: **[`docs/experiment_0a.md`](docs/experiment_0a.md)**.

**Experiment 0B** is a controlled downstream training comparison. It asks
whether the transformation-level damage 0A measured on a *frozen* model
corresponds to anything when the same transformations are used to *train*. Six
models are trained independently and identically — same split, same
initialisation, same optimizer, same budget, same checkpoint rule — differing
only in the training augmentation policy:

```text
baseline                  RandomResizedCrop(0.7–1.0) + HorizontalFlip(p 0.5)
baseline_color_jitter     baseline + colour jitter        (p 0.5)
baseline_rotation         baseline + rotation             (p 0.5)
baseline_crop             baseline with 0A crop strength  (p 0.5)
baseline_gaussian_blur    baseline + Gaussian blur        (p 0.5)
baseline_random_erasing   baseline + random erasing       (p 0.5)
```

The 0A summary is then joined against the downstream results — **afterwards, for
analysis only**. `ΔM` is never a training signal and never selects an
augmentation. 0B is *not* DPGA. Full details:
**[`docs/experiment_0b.md`](docs/experiment_0b.md)**.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate                     # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

Drop `--extra-index-url` for a CPU-only install. CUB-200-2011 is expected at
`data/CUB_200_2011/` in its original layout (not re-foldered into train/test).

## Running Experiment 0A

```bash
python -m cicps prepare --config config/experiment_0a.yaml   # validate dataset + build split
python -m cicps verify  --config config/experiment_0a.yaml   # prove transformations are deterministic
python -m cicps train   --config config/experiment_0a.yaml   # train the frozen baseline
python -m cicps audit   --config config/experiment_0a.yaml   # the transformation audit
python -m cicps analyze --config config/experiment_0a.yaml   # summary table + ΔM plots
```

Fast end-to-end plumbing check (~3 min, writes to `outputs/smoke_0a/`, not a
scientific run):

```bash
python -m cicps train   --config config/experiment_0a.yaml --override config/smoke_0a.yaml
python -m cicps audit   --config config/experiment_0a.yaml --override config/smoke_0a.yaml
python -m cicps analyze --config config/experiment_0a.yaml --override config/smoke_0a.yaml
```

## Running Experiment 0B

Requires Experiment 0A's `analyze` to have been run: the 0B analysis reads
`outputs/experiment_0a/analysis/transformation_summary.csv` (read-only).

```bash
python -m cicps prepare --config config/experiment_0b.yaml   # validate dataset + shared split
python -m cicps train   --config config/experiment_0b.yaml   # all six policies, one command
python -m cicps analyze --config config/experiment_0b.yaml   # policy summary + 0A/0B join + plots
python scripts/check_smoke_0b.py --config config/experiment_0b.yaml   # assert the invariants
```

The commands are the same as 0A's; the **configuration** decides what they do. A
config declaring a `policies` section is a 0B policy sweep. Re-run one arm
without disturbing the others:

```bash
python -m cicps train --config config/experiment_0b.yaml --policy baseline_gaussian_blur
```

Fast end-to-end plumbing check (writes to `outputs/smoke_0b/`, not a scientific
run):

```bash
python -m cicps prepare --config config/experiment_0b.yaml --override config/smoke_0b.yaml
python -m cicps train   --config config/experiment_0b.yaml --override config/smoke_0b.yaml
python -m cicps analyze --config config/experiment_0b.yaml --override config/smoke_0b.yaml
python scripts/check_smoke_0b.py --config config/experiment_0b.yaml --override config/smoke_0b.yaml
```

## Ground rules

* **`config/experiment_0a.yaml` is the single source of truth.** No experiment
  value is hard-coded in Python; a missing key raises rather than defaulting.
* **The official CUB test set is locked.** Train and validation are carved from
  the official *train* portion only, stratified by species, seeded, and
  fingerprinted. The validation stage fails loudly on any leak.
* **The audit is deterministic.** Transformation parameters are frozen per
  `(image, transform)` pair; the audit dataframe is reproducible bit-for-bit
  across runs.
* **The hardest competitor is chosen from the original image and frozen** for
  every transformation of that image.
* **`ΔM` is the primary metric.** Feature cosine and prediction consistency are
  supporting evidence, never substitutes.
* **No ranking claims.** The analysis is descriptive; no significance test is
  implemented, and a higher mean `ΔM` does not make a transformation "better".
* **0B is a controlled comparison, not a search.** Every policy is identical
  except for one augmentation change; nothing is tuned per policy, and the
  transformation set is predetermined — it is not chosen after seeing 0A's
  results.
* **`ΔM` is never a training signal.** 0B joins the 0A summary to its own
  results during analysis only. Nothing selects, ranks, weights or filters an
  augmentation by anything measured.

## Layout

```
config/      experiment + smoke-override YAML (0A and 0B)
data/        CUB-200-2011, as distributed
docs/        experiment handbooks (0A, 0B)
outputs/     generated: splits, checkpoints, logs, dataframes, manifests, plots
scripts/     invariant checks for a completed run
src/cicps/   config · seeding · data · models · transforms · training · audit · analysis · stages
```
