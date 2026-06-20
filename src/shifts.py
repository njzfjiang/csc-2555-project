from src.data_generator import generate_data

def group_shift(severity=0.5, seed=42):
    """
    Apply a group-specific shift to the features X based on group membership s.
    
    Parameters:
    - severity: A float in [0, 1] that controls the magnitude of the shift (alpha).
    - seed: Random seed for reproducibility.
    
    Returns:
    - X_shifted: Shifted feature matrix
    - y: Labels
    - s: Group membership
    """
    new_base_rate_a = 0.3 + severity * 0.2  # P(Y=1|A) increases with severity
    new_base_rate_b = 0.3 - severity * 0.2  # P(Y=1|B) decreases with severity

    X_shifted, y, s = generate_data(base_rate_a=new_base_rate_a, base_rate_b=new_base_rate_b, seed=seed)
    
    return X_shifted, y, s

def covariate_shift(severity=1.0, group='A', seed=42):
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
    
    X_shifted, y, s = generate_data(cov_scale_a=cov_a, cov_scale_b=cov_b, seed=seed)
    return X_shifted, y, s

def label_shift(severity=0.1, seed=42):
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
    X, y_shifted, s = generate_data(flip_noise_a=severity, seed=seed)
    return X, y_shifted, s