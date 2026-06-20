# csc-2555-project
## When Demographic Parity “Improves” Under Shift: A Simulation Study of Fairness Metric Distortion

This is the source code repo for CSC 2555 project for Summer 2026.
Group member: Meixuan Chen

### Project Motivation
Many popular group fairness metrics are defined under a fixed data distribution. However, in real
applications the distribution often drifts: group proportions change, feature distributions shift, and
label noise may become asymmetric. In such settings, it is unclear how to interpret changes in
fairness metrics: an apparent improvement in demographic parity might be an artifact of shift
rather than a genuinely fairer model, while error-based metrics such as equalized odds may degrade
sharply.
In this project, I systematically explore how different types of controlled distribution shift affect
the behavior of standard group fairness metrics, even when the underlying classifier is kept fixed.

### How to run this project

```bash
# 1. install dependencies
pip install -r requirements.txt

# 2. run experiments
python experiments/run_sweep.py --config configs/experiment_config.yaml

# 3. generate results and plot figures( outputs/results/, outputs/figures/）
python experiments/plot_phase_diagrams.py

```