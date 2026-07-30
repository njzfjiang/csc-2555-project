# csc-2555-project
## When Fairness Metrics Disagree Under Shift: A Stylized Case Study of a Medical Follow-up Risk Score under Hospital Deployment Shift

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
  - `run_sweep.py`: evaluates baseline and target-data access regimes over three coupled seeds, plus a five-seed joint prior/label-noise danger-zone grid with clean and recorded labels.
  - `plot_phase_diagrams.py`: plots three-seed means and standard-deviation bands, the primary 2D danger-zone diagram, the matched covariate-shift access-regime comparison, and supplementary plots.
- `notebooks/`
  - `figures.ipynb`: notebook for interactive figure tweaking and additional exploratory plots.
- `data/`
  - `cached/`: cached synthetic datasets (no real-world data are used).
- `outputs/`
  - `logs/` (recommended): schema-v5 JSON outputs with means, standard deviations, and raw per-seed metric values.
  - `figures/` (recommended): generated plots and phase diagrams used in the report.
- `paper/`
  - `project proposal.pdf`: original project proposal.
  - `draft.md`: working draft of the project report.
  - `report.tex`: current report source.

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

3. generate figures from the newest log

```bash
python experiments/plot_phase_diagrams.py
```
