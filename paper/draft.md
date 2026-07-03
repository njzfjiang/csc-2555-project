# When Fairness Metrics Disagree Under Shift: A Simulation Study of Metric Distortion and Mitigation Trade-offs

## 1. Introduction

Group fairness metrics such as demographic parity and equalized odds are usually defined under a
fixed data-generating process. However, real-world deployments rarely enjoy such stability:
population composition drifts over time, feature distributions shift across domains, and label noise
may become asymmetric across groups.

In this project, we ask:

- How do standard group fairness metrics behave under controlled distribution shifts, even when the
  underlying classifier remains fixed?
- Under what conditions can demographic parity remain stable or appear comparatively favorable
  while error-based metrics degrade?
- How do importance reweighting, post-hoc threshold adjustment, and target-distribution retraining
  affect the observed fairness distortion, and what trade-offs do they incur?

We address these questions through a synthetic simulation framework that allows us to systematically
sweep over different types and magnitudes of distribution shift.

## 2. Method

### 2.1 Distribution shift mechanisms

We consider three families of controlled distribution shift, each implemented as a wrapper around
the base data generator `generate_data` (`src/shifts.py`). For the diagnostic baseline, the source
classifier \(h\) is held fixed and only the test-time distribution changes. Mitigation and
supplementary experiments additionally evaluate reweighted training, post-hoc thresholding, and
target-distribution retraining.

#### 2.1.1 Group-conditioned base-rate shift

The function `group_shift(severity)` varies the group-conditioned base rates of the positive class
while keeping the feature generator otherwise fixed. Concretely, we define

- \( \Pr(Y=1 \mid S=0) = 0.3 + 0.2 \cdot \alpha \),
- \( \Pr(Y=1 \mid S=1) = 0.3 - 0.2 \cdot \alpha \),

with \(\alpha \in [0,1]\). Increasing the severity therefore simultaneously increases the
positive rate for group A and decreases it for group B. The function `group_shift` passes the
corresponding base rates to `generate_data` and returns the resulting features \(X\), labels \(Y\),
and group labels \(S\).

This family of shifts induces **asymmetric changes in label prevalence across groups** while keeping
the conditional feature generator \(P(X\mid Y,S)\) fixed. Because \(X\) depends on \(Y\), its
marginal distribution can still change as the base rates change.

#### 2.1.2 Group-specific covariate shift

The function `covariate_shift(severity, group)` implements a **covariate shift** that affects the
scale of a particular feature dimension for one group only. We treat `severity` as a multiplicative
scale factor applied to the standard deviation of that feature for the chosen group, with
\(\mathrm{severity} \in [1, 4]\). Formally,

- if `group='A'`, we set `cov_scale_a = severity`, `cov_scale_b = 1.0`;
- if `group='B'`, we set `cov_scale_a = 1.0`, `cov_scale_b = severity`.

These parameters are passed to `generate_data`, which scales the feature standard deviation
accordingly. The labels and group labels themselves are left unchanged.

This family of shifts isolates the effect of **within-group feature distribution changes** on
fairness metrics, without directly altering label noise or group proportions.

#### 2.1.3 Group-specific label noise shift

Finally, the function `label_shift(severity)` modifies the **label noise** asymmetrically across
groups. Here, `severity` (chosen in \([0, 0.3]\)) controls the probability with which positive labels
for group A and, at half that rate, negative labels for group B are flipped. Concretely, we call

- `generate_data(flip_noise_a=severity)`,

which flips positive labels in group A with probability \(\beta\) and negative labels in group B
with probability \(\beta/2\). Features are generated before these flips, so the marginal \(X\)
distribution is unchanged even though the relationship between observed labels and features is
altered asymmetrically.

Together, these three mechanisms allow us to separately study:

- changes in group-specific base rates,
- changes in group-specific feature distributions,
- changes in group-specific label noise,

and to observe how standard fairness and calibration metrics respond when each dimension is varied
in isolation.

### 2.2 Fairness and calibration metrics

We focus on three standard group fairness quantities implemented via `fairlearn.metrics`
(`src/metrics.py`):

- **Demographic parity difference (DP diff).** We measure the absolute difference in positive
  prediction rates between the two groups:
  \[
    \lvert \Pr(\hat{Y}=1 \mid S=0) - \Pr(\hat{Y}=1 \mid S=1) \rvert.
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

- **True-positive-rate gap (TPR gap).** We separately report
  \(\lvert \mathrm{TPR}_0-\mathrm{TPR}_1\rvert\), the equal-opportunity violation. This is necessary
  for interpreting the threshold-adjustment experiment: a smaller TPR gap can coincide with a
  larger EO difference if the false-positive-rate gap grows.

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

We also report balanced accuracy to expose utility changes that may accompany an apparent fairness
improvement.


### 2.3 Mitigation strategies

To evaluate whether fairness distortion can be detected or corrected, we implement three post-hoc
and training-time interventions:

- **KDE Importance Reweighting.** We estimate the density ratio
  \(w(x)=\hat{p}_{\text{target}}(x)/\hat{p}_{\text{train}}(x)\) using an independent unlabeled target
  adaptation sample and kernel density estimation (`src/reweighting.py`). The features are jointly
  standardized, weights are clipped to \([0.1,10]\), and the logistic-regression loss is reweighted.
  This approach is most directly justified under covariate shift.

- **Post-hoc Threshold Adjustment (Equal Opportunity).** Using an independent labeled calibration
  set, we take the larger group TPR at threshold 0.5 as the target and choose a deterministic
  threshold for each group on a grid with spacing 0.01 (`src/adjust_threshold.py`). This attempts to
  raise the lower-TPR group without retraining the classifier. It does not enforce full equalized
  odds and can increase the FPR gap.

- **Target-Distribution Retraining.** As a supplementary reference, we retrain the same model
  family on an independent sample from the shifted distribution and evaluate it on the shifted test
  set. This measures what target-distribution ERM can attain relative to a frozen source model; it
  is not a universal fairness optimum.

## 3. Experimental Setup

### 3.1 Shift parameter grid


We sweep over the following grids:

- Group-conditioned base-rate shift: \(\alpha \in [0, 1]\) with 11 steps.
- Covariate shift: \(\gamma \in [1, 4]\) with 10 steps.
- Asymmetric label noise: \(\beta \in [0, 0.3]\) with 10 steps.

### 3.2 Implementation details

- Sample size: source training, shifted evaluation, and target-retraining datasets each use
  \(N=5000\). KDE adaptation and threshold calibration each use independent samples of size 1000.
- Classifier: Logistic regression with `max_iter=1000` and default L2 regularization (`C=1.0`).
- Fairness metrics: computed via `fairlearn.metrics`; ECE uses 10 equal-width bins.
- Repetitions: all results are reported for a fixed random seed (\(42\)); diagnostics (ESS and
  thresholds) are logged for transparency.


## 4. Results

### 4.1 Key empirical findings

Our main results are summarized below. Baseline phase diagrams, mitigation comparisons, and the
target-retraining supplementary figure are saved in `outputs/figures/`.

**Diagnosis (RQ1 & RQ2):**
- Under **group-conditioned base-rate shift**, DP increases from 0.01 to 0.26 as \(\alpha\) goes from
  0 to 1, while the TPR gap remains between approximately 0.03 and 0.06. The two metric families
  therefore react very differently to the same prevalence change.
- Under **covariate shift**, DP grows from 0.01 to 0.18, EO reaches 0.30, and the TPR gap reaches
  0.12 at \(\gamma=4\). In this regime both DP and error-based fairness degrade, but at different
  rates.
- Under **asymmetric label noise**, DP stays artificially flat at 0.01 even as the TPR gap worsens
  from 0.03 to 0.18. This is the most deceptive regime: a static DP value masks a severe degradation
  in equalized odds.

**Mitigation (RQ3):**
- **KDE Reweighting:** Nearly ineffective. The ESS fraction ranges from 0.762 to 0.950 across the
  experiments, so clipping does not collapse the effective sample, but the resulting DP and EO
  curves remain close to the frozen baseline.
- **Threshold Adjustment:** Under covariate shift it reduces the maximum TPR gap from 0.124 to 0.018,
  yet the maximum EO difference rises from 0.298 to 0.408 because the FPR gap grows. Under label
  shift it lowers the maximum TPR gap from 0.176 to 0.030 but raises DP from approximately 0.012 to
  as much as 0.178; at \(\beta=0.3\), the calibrated threshold for group B is 0.16.
- **Target Retraining:** Under severe covariate shift (\(\gamma=4\)), balanced accuracy falls from
  69.8% for the frozen source model to 53.7% after target retraining. This indicates a limitation of
  the present logistic model and ERM objective under the shifted distribution, rather than a
  universal separability limit. Under label noise, retraining keeps DP low but only modestly changes
  EO.

### 4.2 Phase diagrams of metric distortion

To highlight non-monotonic and counterintuitive behavior, we construct “phase diagrams” where each
point corresponds to a particular shift configuration and is colored by the value of a fairness
metric.

We emphasize regions where demographic parity remains stable or changes slowly while error-based
metrics such as EO and the TPR gap worsen. The mitigation plots also show the converse: explicitly
reducing the TPR gap can increase DP or the FPR-driven EO difference. Thus, a favorable movement in
one fairness quantity need not correspond to uniformly fairer behavior.

## 5. Discussion

Our simulations reveal several patterns:

- A stable or comparatively favorable demographic-parity value can mask large changes in
  error-based fairness; the asymmetric-label-noise experiment is the clearest example.
- Error-based metrics such as equalized odds can degrade sharply under certain shifts, even when the
  classifier is unchanged.
- Different metrics can therefore disagree not only in level but in trend as the environment drifts.

We discuss the implications of these findings for:

- interpreting fairness metrics in non-stationary environments,
- monitoring deployed systems under distribution shift,
- designing robustness checks for fairness evaluations.


## 6. Conclusion and Future Work

We presented a synthetic simulation framework for studying how standard group fairness metrics
behave under controlled distribution shifts. The diagnostic baseline holds the classifier fixed,
while mitigation and supplementary experiments alter training or decision thresholds. Our
experiments show that:

- a stable or comparatively favorable demographic-parity value can be misleading,
- error-based metrics provide complementary information but can themselves be unstable.

Future directions include:

- extending the framework to more realistic feature spaces and multi-class settings,
- repeating the sweeps over multiple random seeds and reporting uncertainty bands,
- comparing richer model families and randomized equalized-odds post-processing,
- exploring causal and counterfactual perspectives on fairness metrics under non-stationarity.
