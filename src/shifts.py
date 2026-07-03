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
    """
    Apply a covariate shift to feature 1 of a specified group.
    
    Parameters:
    - severity: A float in [1, 4] that controls the magnitude of the shift (gamma).
    - group: 'A' or 'B' indicating which group to apply the shift to.
    - seed: Random seed for reproducibility.
    
    Returns:
    - X_shifted: Shifted feature matrix
    - y: Labels
    - s: Group membership
    """
    if group == 'A':
        cov_a, cov_b = severity, 1.0
    elif group == 'B':  # group == 'B'
        cov_a, cov_b = 1.0, severity
    else:
        raise ValueError("group must be 'A' or 'B'")
    
    X_shifted, y, s = generate_data(
        num_samples=num_samples,
        num_features=num_features,
        prior_a=prior_a,
        base_rate_a=base_rate_a,
        base_rate_b=base_rate_b,
        cov_scale_a=cov_a,
        cov_scale_b=cov_b,
        seed=seed,
    )
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
