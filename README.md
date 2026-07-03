# csc-2555-project
## When Fairness Metrics Disagree Under Shift: A Simulation Study of Metric Distortion and Mitigation Trade-offs

This repository contains the source code for the CSC 2555 course project (Summer 2026).

**Author:** Meixuan Chen

---

## Repository Structure

- `src/`
  - `data_generator.py`: synthetic dataset construction under different group/feature/label shifts.
  - `shifts.py`: definitions of shift types and utilities for applying them.
  - `reweighting.py`: KDE density-ratio importance weighting utilities.
  - `adjust_threshold.py`: group-specific threshold post-processing utilities.
  - `metrics.py`: implementations of group fairness metrics (e.g., demographic parity, equalized odds).
  - `utils.py`: shared helper functions.
- `configs/`
  - `experiment_config.yaml`: main configuration for sweep ranges, metrics, and other hyperparameters.
- `experiments/`
  - `run_sweep.py`: evaluates baseline, KDE reweighting, TPR threshold adjustment, and target retraining.
  - `plot_phase_diagrams.py`: generates phase diagrams, mitigation comparisons, and supplementary plots.
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
