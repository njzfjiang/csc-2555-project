# When Demographic Parity “Improves” Under Shift: A Simulation Study of Fairness Metric Distortion

## 1. Introduction

Group fairness metrics such as demographic parity and equalized odds are usually defined under a
fixed data-generating process. However, real-world deployments rarely enjoy such stability:
population composition drifts over time, feature distributions shift across domains, and label noise
may become asymmetric across groups.

In this project, we ask:

- How do standard group fairness metrics behave under controlled distribution shifts, even when the
  underlying classifier remains fixed?
- Under what conditions can demographic parity appear to “improve” purely as an artifact of shift,
  while error-based metrics degrade?

We address these questions through a synthetic simulation framework that allows us to systematically
sweep over different types and magnitudes of distribution shift.

## 2. Method

### 2.1 Synthetic data construction

We construct a family of synthetic binary classification problems with a binary sensitive attribute
\(A \in \{0, 1\}\), features \(X\), and label \(Y \in \{0, 1\}\). The data-generating process is
implemented in `src/data_generator.py` and parameterized by:

- group proportions \(P(A=1)\),
- class-conditional feature distributions \(P(X \mid A, Y)\),
- label noise rates that may differ across groups.

This setup allows us to introduce controlled shifts in:

- **Group mix shift:** changing \(P(A=1)\) while keeping within-group conditionals fixed.
- **Feature shift:** shifting the means or variances of \(P(X \mid A, Y)\) for one or both groups.
- **Label noise shift:** varying group-specific label noise rates.

### 2.2 Fairness and calibration metrics

We focus on two standard group fairness metrics implemented via `fairlearn.metrics`
(`src/metrics.py`):

- **Demographic parity difference (DP diff).** We measure the absolute difference in positive
  prediction rates between the two groups:
  \[
    \lvert \Pr(\hat{Y}=1 \mid A=0) - \Pr(\hat{Y}=1 \mid A=1) \rvert.
  \]
  A value of 0 corresponds to perfect demographic parity. We compute this quantity using
  `fairlearn.metrics.demographic_parity_difference`.

- **Equalized odds difference (EO diff).** We measure the worst-case discrepancy across groups in
  both true positive and false positive rates:
  \[
    \max\big(
      \lvert \mathrm{TPR}_0 - \mathrm{TPR}_1 \rvert,
      \lvert \mathrm{FPR}_0 - \mathrm{FPR}_1 \rvert
    \big).
  \]
  A value of 0 corresponds to perfect equalized odds. We compute this quantity using
  `fairlearn.metrics.equalized_odds_difference`.

In addition to group fairness, we track **probability calibration** at both the global and group
levels. We compute the **expected calibration error (ECE)** following a standard binned estimator.
Given predicted probabilities \(p_i\) and labels \(y_i\), we partition the interval \([0,1]\)
into \(B\) bins, and for each bin \(b\) compute the average predicted confidence and empirical
accuracy. The ECE is then

\[
  \mathrm{ECE}
  = \sum_{b=1}^B \frac{n_b}{N}
    \big\lvert \mathrm{acc}(b) - \mathrm{conf}(b) \big\rvert,
\]

where \(n_b\) is the number of points in bin \(b\), \(\mathrm{acc}(b)\) is the average label,
and \(\mathrm{conf}(b)\) is the average predicted probability in that bin.

To capture **group-wise calibration**, we compute ECE separately for each group and report the
**ECE gap**, defined as the difference between the worst- and best-calibrated group:

\[
  \mathrm{ECE\ Gap}
  = \max_{a} \mathrm{ECE}_a - \min_{a} \mathrm{ECE}_a.
\]

This is implemented in `src/metrics.py` as `calculate_group_ece_metrics`, which returns both the
per-group ECEs and their gap.

## 3. Experimental Setup

### 3.1 Shift parameter grid

We define a grid over shift parameters in `configs/experiment_config.yaml`. The main dimensions
include:

- group proportion \(P(A=1)\) ranging from \(...\) to \(...\),
- feature shift magnitude (e.g., mean shift \(\delta\) in one feature),
- label noise rates for each group.

The script `experiments/run_sweep.py` iterates over this grid and, for each configuration:

1. Generates a synthetic test set under the specified shift.
2. Applies the fixed classifier \(h\).
3. Computes all metrics and logs them to `outputs/`.

### 3.2 Implementation details and hyperparameters

Key implementation choices include:

- number of samples per configuration (e.g., \(N = ...\)),
- number of random seeds or repetitions, if any,
- classifier type and its training procedure (if trained),
- any regularization or thresholding choices.

(Here you can fill in the actual values once you finalize them.)

## 4. Results

### 4.1 Quantitative summaries

We aggregate the sweep outputs into tables and bar plots showing how each metric changes as a
function of shift parameters. For each shift type, we report:

- overall accuracy or error rate,
- demographic parity gap,
- equalized odds gaps (TPR and FPR differences),
- *(other metrics you compute).*

These results are produced by `experiments/plot_phase_diagrams.py` and visualized further in
`notebooks/figures.ipynb`.

*(Here you will insert concrete tables and plots later.)*

### 4.2 Phase diagrams of metric distortion

To highlight non-monotonic and counterintuitive behavior, we construct “phase diagrams” where each
point corresponds to a particular shift configuration and is colored by the value of a fairness
metric.

We emphasize regions where:

- demographic parity appears to improve (gap shrinks),
- while error-based metrics (e.g., EO gaps or group-specific error rates) worsen.

These diagrams illustrate that improvements in one fairness metric under shift need not correspond
to genuinely fairer behavior.

## 5. Discussion

Our simulations reveal several patterns:

- There exist regimes where demographic parity “improves” purely because group proportions or label
  noise shift, rather than because the classifier treats groups more similarly.
- Error-based metrics such as equalized odds can degrade sharply under certain shifts, even when the
  classifier is unchanged.
- Different metrics can therefore disagree not only in level but in trend as the environment drifts.

We discuss the implications of these findings for:

- interpreting fairness metrics in non-stationary environments,
- monitoring deployed systems under distribution shift,
- designing robustness checks for fairness evaluations.

(You can add bullet points for each concrete phenomenon you observe once experiments are done.)

## 6. Conclusion and Future Work

We presented a synthetic simulation framework for studying how standard group fairness metrics
behave under controlled distribution shifts, holding the classifier fixed. Our experiments show
that:

- apparent improvements in demographic parity under shift can be misleading,
- error-based metrics provide complementary information but can themselves be unstable.

Future directions include:

- extending the framework to more realistic feature spaces and multi-class settings,
- incorporating adaptive classifiers that are re-trained or updated under shift,
- exploring causal and counterfactual perspectives on fairness metrics under non-stationarity.

