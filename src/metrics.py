import numpy as np
import fairlearn.metrics as fm

def demographic_parity_difference(y_true, y_pred, s):
    """
    Calculate the demographic parity difference between groups.
    
    DP Difference = |P(Y_pred=1|S=0) - P(Y_pred=1|S=1)|
    
    A value of 0 indicates perfect demographic parity.
    """
    return fm.demographic_parity_difference(y_true, y_pred, sensitive_features=s)

def equalized_odds_difference(y_true, y_pred, s):
    """
    Calculate the equalized odds difference between groups.
    
    EO Difference = max(|FPR_0 - FPR_1|, |TPR_0 - TPR_1|)
    
    A value of 0 indicates perfect equalized odds.
    """
    return fm.equalized_odds_difference(y_true, y_pred, sensitive_features=s)

def compute_ece(probs, labels, bins=10):
    """Computes Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, bins + 1)
    bin_indices = np.digitize(probs, bin_boundaries, right=True) - 1
    bin_indices = np.clip(bin_indices, 0, bins - 1)
    
    ece = 0.0
    n_total = len(probs)
    
    for b in range(bins):
        mask = bin_indices == b
        if np.any(mask):
            bin_size = np.sum(mask)
            bin_acc = np.mean(labels[mask])
            bin_conf = np.mean(probs[mask])
            ece += (bin_size / n_total) * np.abs(bin_acc - bin_conf)
            
    return ece

def calculate_group_ece_metrics(probs, labels, groups, bins=10):
    """
    Calculates ECE per group and the ECE Gap between the max and min calibrated groups.
    """
    unique_groups = np.unique(groups)
    group_eces = {}
    
    # Calculate ECE for each group
    for g in unique_groups:
        g_mask = groups == g
        if np.sum(g_mask) > 0:
            group_eces[g] = compute_ece(probs[g_mask], labels[g_mask], bins=bins)
            
    # Calculate ECE Gap
    ece_values = list(group_eces.values())
    ece_gap = np.max(ece_values) - np.min(ece_values)
    
    return group_eces, ece_gap