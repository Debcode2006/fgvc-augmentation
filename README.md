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
| Experiment 0B — augmentation policies vs. downstream usefulness | ❌ not implemented |
| DPGA — Discriminative-Preservation-Guided Augmentation | ❌ not implemented |

**Experiment 0A** is a falsification / mechanism test. It freezes one ordinary
ResNet-18 baseline and asks whether seven deterministic transformations cause
measurably *different* changes in an image's probability margin against its
frozen hardest competing class:

$$\Delta M = \big(p_y(T(x)) - p_c(T(x))\big) - \big(p_y(x) - p_c(x)\big)$$

Full details: **[`docs/experiment_0a.md`](docs/experiment_0a.md)**.

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

## Layout

```
config/      experiment + smoke-override YAML
data/        CUB-200-2011, as distributed
docs/        experiment handbook
outputs/     generated: splits, checkpoints, logs, audit dataframes, plots
src/cicps/   config · seeding · data · models · transforms · training · audit · analysis · stages
```
