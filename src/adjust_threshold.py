import numpy as np
from sklearn.metrics import confusion_matrix


def _validate_inputs(scores, group, y_true=None, grid_resolution=0.01):
    scores = np.asarray(scores, dtype=float)
    group = np.asarray(group)
    if scores.ndim != 1 or group.ndim != 1 or len(scores) != len(group):
        raise ValueError("scores and group must be one-dimensional and aligned")
    if len(scores) == 0 or not np.isfinite(scores).all():
        raise ValueError("scores must be non-empty and finite")
    if not 0 < grid_resolution <= 1:
        raise ValueError("grid_resolution must be in (0, 1]")
    if y_true is not None:
        y_true = np.asarray(y_true)
        if y_true.ndim != 1 or len(y_true) != len(scores):
            raise ValueError("y_true must be one-dimensional and aligned")
        if not set(np.unique(y_true)).issubset({0, 1}):
            raise ValueError("y_true must be binary")
    return scores, group, y_true


def _threshold_grid(grid_resolution):
    return np.linspace(0.0, 1.0, int(np.ceil(1.0 / grid_resolution)) + 1)


def _rates(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if tp + fn else np.nan
    fpr = fp / (fp + tn) if fp + tn else np.nan
    return tpr, fpr


def apply_group_thresholds(y_pred_proba, group, thresholds):
    """Apply pre-fitted group-specific thresholds to probability scores."""
    scores, group, _ = _validate_inputs(y_pred_proba, group)
    predictions = np.empty(len(scores), dtype=int)
    for g in np.unique(group):
        if g not in thresholds:
            raise ValueError(f"Missing threshold for group {g!r}")
        mask = group == g
        predictions[mask] = (scores[mask] >= thresholds[g]).astype(int)
    return predictions


def adjust_threshold_equal_opportunity(
    y_true,
    y_pred_proba,
    group,
    target_tpr=None,
    grid_resolution=0.01,
):
    """Fit deterministic group thresholds that approximately equalize TPR.

    If ``target_tpr`` is omitted, the best group TPR at threshold 0.5 is used,
    so the procedure raises the lower-TPR group instead of lowering the better
    group. Ties in TPR distance are broken by calibration error.
    """
    scores, group, y_true = _validate_inputs(
        y_pred_proba, group, y_true=y_true, grid_resolution=grid_resolution
    )
    unique_groups = np.unique(group)
    baseline_tprs = []
    for g in unique_groups:
        mask = group == g
        tpr, _ = _rates(y_true[mask], (scores[mask] >= 0.5).astype(int))
        if np.isnan(tpr):
            raise ValueError(f"Group {g!r} has no positive calibration labels")
        baseline_tprs.append(tpr)
    if target_tpr is None:
        target_tpr = max(baseline_tprs)
    if not 0 <= target_tpr <= 1:
        raise ValueError("target_tpr must be in [0, 1]")

    thresholds = {}
    for g in unique_groups:
        mask = group == g
        y_g, scores_g = y_true[mask], scores[mask]
        candidates = []
        for threshold in _threshold_grid(grid_resolution):
            predictions = (scores_g >= threshold).astype(int)
            tpr, _ = _rates(y_g, predictions)
            error = np.mean(predictions != y_g)
            candidates.append((abs(tpr - target_tpr), error, abs(threshold - 0.5), threshold))
        thresholds[g] = min(candidates)[-1]
    return thresholds


def adjust_threshold_equalized_odds(y_true, y_pred_proba, group, grid_resolution=0.01):
    """
    Fit deterministic thresholds toward a common TPR/FPR operating point.

    This is only an approximation: exact equalized odds may require randomized
    post-processing when the groups' ROC curves do not share an operating point.
    """
    y_pred_proba, group, y_true = _validate_inputs(
        y_pred_proba, group, y_true=y_true, grid_resolution=grid_resolution
    )
    unique_groups = np.unique(group)
    
    target_tpr, target_fpr = 0, 0
    total_samples = 0
    for g in unique_groups:
        mask = (group == g)
        y_true_g = y_true[mask]
        y_pred_g = (y_pred_proba[mask] >= 0.5).astype(int)
        tpr, fpr = _rates(y_true_g, y_pred_g)
        n = len(y_true_g)

        target_tpr += (0 if np.isnan(tpr) else tpr) * n
        target_fpr += (0 if np.isnan(fpr) else fpr) * n
        total_samples += n
    target_tpr /= total_samples
    target_fpr /= total_samples

    best_thresholds = {}
    for g in unique_groups:
        mask = (group == g)
        y_true_g = y_true[mask]
        scores_g = y_pred_proba[mask]
        best_score = float('inf')
        best_t = 0.5
        
        for t in _threshold_grid(grid_resolution):
            y_pred_g = (scores_g >= t).astype(int)
            tpr, fpr = _rates(y_true_g, y_pred_g)
            tpr = 0 if np.isnan(tpr) else tpr
            fpr = 0 if np.isnan(fpr) else fpr
           
            diff = (tpr - target_tpr)**2 + (fpr - target_fpr)**2
            if diff < best_score:
                best_score = diff
                best_t = t
        best_thresholds[g] = best_t
    return best_thresholds

def adjust_threshold_demographic_parity(y_pred_proba, group, target_rate=None, grid_resolution=0.01):
    """
    Adjusts the decision threshold for each group to achieve demographic parity.
    """
    y_pred_proba, group, _ = _validate_inputs(
        y_pred_proba, group, grid_resolution=grid_resolution
    )
    unique_groups = np.unique(group)
    if target_rate is None:
        # Target selection rate: can be the global average selection rate
        global_pred = (y_pred_proba >= 0.5).astype(int)
        target_rate = np.mean(global_pred)

    best_thresholds = {}
    for g in unique_groups:
        mask = (group == g)
        scores_g = y_pred_proba[mask]
        best_score = float('inf')
        best_t = 0.5
        for t in _threshold_grid(grid_resolution):
            selection_rate = np.mean((scores_g >= t).astype(int))
            diff = abs(selection_rate - target_rate)
            if diff < best_score:
                best_score = diff
                best_t = t
        best_thresholds[g] = best_t
    return best_thresholds
