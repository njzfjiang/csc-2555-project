import numpy as np

from src.data_generator import generate_data


def group_shift(
    severity=0.5,
    seed=42,
    num_samples=1000,
    num_features=2,
    prior_a=0.5,
    base_rate=0.3,
    delta=0.2,
):
    """
    Progressively change group-conditional base rates.

    The conditional feature generator P(X | Y, S) and group proportion P(S)
    remain fixed. Only P(Y=1 | S) changes with severity alpha.
    
    Parameters:
    - severity: A float in [0, 1] that controls the magnitude of the shift (alpha).
    - seed: Random seed for reproducibility.
    
    Returns:
    - X_shifted: Shifted feature matrix
    - y: Labels
    - s: Group membership
    """
    if not 0 <= severity <= 1:
        raise ValueError("severity must be in [0, 1]")
    new_base_rate_a = base_rate + severity * delta
    new_base_rate_b = base_rate - severity * delta
    if not (0 <= new_base_rate_a <= 1 and 0 <= new_base_rate_b <= 1):
        raise ValueError("base_rate +/- severity * delta must remain in [0, 1]")

    X_shifted, y, s = generate_data(
        num_samples=num_samples,
        num_features=num_features,
        prior_a=prior_a,
        base_rate_a=new_base_rate_a,
        base_rate_b=new_base_rate_b,
        seed=seed,
    )
    
    return X_shifted, y, s

def covariate_shift(
    severity=1.0,
    group='A',
    seed=42,
    num_samples=1000,
    num_features=2,
    prior_a=0.5,
    base_rate_a=0.3,
    base_rate_b=0.3,
):
    """Apply a pure covariate shift to feature 1 of one group.

    The target feature distribution is generated first by scaling deviations
    from the source marginal mean. Labels are then sampled from the fixed
    source conditional P(Y=1 | X, S), so the severity changes P(X, S) without
    changing the label mechanism. At severity 1, this recovers the source
    distribution.
    
    Parameters:
    - severity: A float in [1, 4] that controls the magnitude of the shift (gamma).
    - group: 'A' or 'B' indicating which group to apply the shift to.
    - seed: Random seed for reproducibility.
    
    Returns:
    - X_shifted: Shifted feature matrix
    - y: Labels
    - s: Group membership
    """
    if severity < 1:
        raise ValueError("severity must be at least 1")
    if group not in ('A', 'B'):
        raise ValueError("group must be 'A' or 'B'")
    if num_features < 1:
        raise ValueError("num_features must be at least 1")
    if not (0 < base_rate_a < 1 and 0 < base_rate_b < 1):
        raise ValueError("base rates must lie strictly between 0 and 1")

    rng = np.random.RandomState(seed)
    s = rng.choice([0, 1], size=num_samples, p=[prior_a, 1 - prior_a])
    base_rates = np.where(s == 0, base_rate_a, base_rate_b)

    # Draw from the source marginal P(X | S), then transform X for the target
    # group. The temporary labels are used only to sample the source mixture.
    mixture_labels = rng.binomial(1, base_rates)
    source_means = 2 * base_rates - 1
    X_shifted = np.zeros((num_samples, num_features))
    X_shifted[:, 0] = rng.normal(
        loc=np.where(mixture_labels == 1, 1.0, -1.0),
        scale=1.0,
    )
    target_value = 0 if group == 'A' else 1
    target_mask = s == target_value
    X_shifted[target_mask, 0] = (
        source_means[target_mask]
        + severity
        * (X_shifted[target_mask, 0] - source_means[target_mask])
    )
    for feature in range(1, num_features):
        X_shifted[:, feature] = rng.normal(0, 1, size=num_samples)

    # For the source Gaussian mixture, the log likelihood ratio contributed by
    # the informative feature is 2*x. Keeping this conditional fixed makes the
    # experiment a genuine covariate shift.
    log_prior_odds = np.log(base_rates / (1 - base_rates))
    positive_probability = 1 / (
        1 + np.exp(-(log_prior_odds + 2 * X_shifted[:, 0]))
    )
    y = rng.binomial(1, positive_probability)
    return X_shifted, y, s

def label_shift(
    severity=0.1,
    seed=42,
    num_samples=1000,
    num_features=2,
    prior_a=0.5,
    base_rate_a=0.3,
    base_rate_b=0.3,
):
    """
    Apply a label shift to the labels y based on group membership s.
    
    Parameters:
    - severity: A float in [0, 0.3] that controls the magnitude of the shift (beta).
    - seed: Random seed for reproducibility.
    
    Returns:
    - X: Feature matrix (unchanged)
    - y_shifted: Shifted labels
    - s: Group membership
    """
    # Group A's positive examples are flipped with higher probability
    X, y_shifted, s = generate_data(
        num_samples=num_samples,
        num_features=num_features,
        prior_a=prior_a,
        base_rate_a=base_rate_a,
        base_rate_b=base_rate_b,
        flip_noise_a=severity,
        seed=seed,
    )
    return X, y_shifted, s


def joint_prior_label_shift(
    alpha=0.0,
    beta=0.0,
    seed=42,
    num_samples=1000,
    num_features=2,
    prior_a=0.5,
    base_rate=0.3,
    delta=0.2,
):
    """Jointly apply group-conditioned prior shift and asymmetric label noise.

    Clean labels Y* are sampled using the alpha-dependent group base rates and
    generate the features. Recorded labels Y_tilde are then obtained by
    applying beta-dependent asymmetric flips. Both label versions are returned
    so evaluation can distinguish metric disagreement from label corruption.
    """
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    if not 0 <= beta <= 1:
        raise ValueError("beta must be in [0, 1]")
    base_rate_a = base_rate + alpha * delta
    base_rate_b = base_rate - alpha * delta
    if not (0 <= base_rate_a <= 1 and 0 <= base_rate_b <= 1):
        raise ValueError("base_rate +/- alpha * delta must remain in [0, 1]")

    X, y_observed, s, y_clean = generate_data(
        num_samples=num_samples,
        num_features=num_features,
        prior_a=prior_a,
        base_rate_a=base_rate_a,
        base_rate_b=base_rate_b,
        flip_noise_a=beta,
        seed=seed,
        return_clean_labels=True,
    )
    return X, y_observed, y_clean, s
