# csc-2555-project
## When Demographic Parity “Improves” Under Shift: A Simulation Study of Fairness Metric Distortion

This repository contains the source code for the CSC 2555 course project (Summer 2026).

**Author:** Meixuan Chen

---

## Project Motivation

Many popular group fairness metrics are defined under a fixed data distribution. However, in real
applications the distribution often drifts: group proportions change, feature distributions shift, and
label noise may become asymmetric. In such settings, it is unclear how to interpret changes in
fairness metrics: an apparent improvement in demographic parity might be an artifact of shift
rather than a genuinely fairer model, while error-based metrics such as equalized odds may degrade
sharply.

In this project, I systematically explore how different types of controlled distribution shift affect
the behavior of standard group fairness metrics, even when the underlying classifier is kept fixed.

---

## Repository Structure

- `src/`
  - `data_generator.py`: synthetic dataset construction under different group/feature/label shifts.
  - `shifts.py`: definitions of shift types and utilities for applying them.
  - `metrics.py`: implementations of group fairness metrics (e.g., demographic parity, equalized odds).
  - `utils.py`: shared helper functions.
- `configs/`
  - `experiment_config.yaml`: main configuration for sweep ranges, metrics, and other hyperparameters.
- `experiments/`
  - `run_sweep.py`: main experiment driver; runs metric evaluation across a grid of shift parameters.
  - `plot_phase_diagrams.py`: generates phase diagrams and other summary plots from sweep results.
- `notebooks/`
  - `figures.ipynb`: notebook for interactive figure tweaking and additional exploratory plots.
- `data/`
  - `cached/`: cached synthetic datasets (no real-world data are used).
- `outputs/`
  - `logs/` (recommended): raw sweep outputs (e.g., CSV/JSON with metric values).
  - `figures/` (recommended): generated plots and phase diagrams used in the report.
- `paper/`
  - `project proposal.pdf`: original project proposal.
  - `draft.md`: working draft of the project report.

---

### How to run this project

1. install dependencies

```bash
pip install -r requirements.txt
```

2. run experiments

```bash
python experiments/run_sweep.py --config configs/experiment_config.yaml
```

3. generate results and plot figures( outputs/results/, outputs/figures/）

```bash
python experiments/plot_phase_diagrams.py

```