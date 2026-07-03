import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from src.utils import load_config
from src.data_generator import generate_data
from src.shifts import group_shift, covariate_shift, label_shift
from src.reweighting import compute_importance_weights_kde, effective_sample_size
from src.adjust_threshold import (
    adjust_threshold_equal_opportunity,
    apply_group_thresholds,
)
from src.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    true_positive_rate_difference,
    calculate_group_ece_metrics
)


METRIC_KEYS = ("dp", "eo", "tpr_gap", "ece_gap", "balanced_accuracy")
METHOD_KEYS = (
    "baseline",
    "kde_reweighting",
    "threshold_tpr",
    "target_retrained",
)


def train_classifier(X_train, y_train, X_test, seed=42, sample_weight=None,
                     model_config=None):
    """
    Train a logistic regression classifier on unshifted training data 
    and return predictions and probabilities on test data.
    
    Parameters:
    -----------
    X_train : array-like
        Unshifted training features
    y_train : array-like
        Unshifted training labels
    X_test : array-like
        Test features (may be shifted)
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    y_pred : array-like
        Binary predictions on test data
    y_proba : array-like
        Predicted probabilities for class 1 on test data
    """
    model_config = model_config or {}
    clf = LogisticRegression(
        random_state=model_config.get('random_state', seed),
        max_iter=model_config.get('max_iter', 1000),
        solver=model_config.get('solver', 'lbfgs'),
    )
    clf.fit(X_train, y_train, sample_weight=sample_weight)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    return y_pred, y_proba


def evaluate_shift(X, y, s, y_pred, y_proba, ece_bins=10):
    """
    Evaluate DP, EO, and ECE gap for a given dataset and predictions.
    """
    dp_diff = demographic_parity_difference(y, y_pred, s)
    eo_diff = equalized_odds_difference(y, y_pred, s)
    tpr_gap = true_positive_rate_difference(y, y_pred, s)
    _, ece_gap = calculate_group_ece_metrics(y_proba, y, s, bins=ece_bins)
    
    return {
        'dp': np.abs(dp_diff),  # Use absolute value for visualization
        'eo': np.abs(eo_diff),
        'tpr_gap': np.abs(tpr_gap),
        'ece_gap': ece_gap,
        'balanced_accuracy': balanced_accuracy_score(y, y_pred),
    }


def _empty_shift_results():
    return {
        method: {metric: [] for metric in METRIC_KEYS}
        for method in METHOD_KEYS
    } | {
        'diagnostics': {
            'ess': [],
            'ess_fraction': [],
            'weight_min': [],
            'weight_max': [],
            'thresholds': [],
        }
    }


def _data_kwargs(data_config, num_samples):
    return {
        'num_samples': num_samples,
        'num_features': data_config.get('num_features', 2),
        'prior_a': data_config.get('prior_a', 0.5),
        'base_rate_a': data_config.get('base_rate_a', 0.3),
        'base_rate_b': data_config.get('base_rate_b', 0.3),
    }


def _generate_shift(shift_type, severity, data_config, shift_config,
                    seed, num_samples):
    kwargs = _data_kwargs(data_config, num_samples)
    if shift_type == 'group_shift':
        if not np.isclose(kwargs['base_rate_a'], kwargs['base_rate_b']):
            raise ValueError(
                'group_shift requires equal baseline base_rate_a/base_rate_b'
            )
        base_rate = kwargs.pop('base_rate_a')
        kwargs.pop('base_rate_b')
        return group_shift(
            severity=severity,
            seed=seed,
            base_rate=base_rate,
            delta=shift_config.get('delta', 0.2),
            **kwargs,
        )
    if shift_type == 'covariate_shift':
        return covariate_shift(
            severity=severity,
            group=shift_config.get('target_group', 'A'),
            seed=seed,
            **kwargs,
        )
    if shift_type == 'label_shift':
        return label_shift(severity=severity, seed=seed, **kwargs)
    raise ValueError(f'Unknown shift type: {shift_type}')


def _evaluate_severity(
    results,
    shift_type,
    severity,
    shift_config,
    data_config,
    model_config,
    reweighting_config,
    threshold_config,
    supplementary_config,
    X_train,
    y_train,
    seed,
    ece_bins,
):
    X_test, y_test, s_test = _generate_shift(
        shift_type,
        severity,
        data_config,
        shift_config,
        seed,
        data_config['num_samples'],
    )
    y_pred, y_proba = train_classifier(
        X_train, y_train, X_test, seed=seed, model_config=model_config
    )
    baseline = evaluate_shift(
        X_test, y_test, s_test, y_pred, y_proba, ece_bins=ece_bins
    )

    adaptation_seed = seed + reweighting_config.get('adaptation_seed_offset', 10000)
    X_adapt, _, _ = _generate_shift(
        shift_type,
        severity,
        data_config,
        shift_config,
        adaptation_seed,
        reweighting_config.get('adaptation_num_samples', 1000),
    )
    weights = compute_importance_weights_kde(
        X_train,
        X_adapt,
        bandwidth=reweighting_config.get('bandwidth', 0.3),
        clip_min=reweighting_config.get('clip_min', 0.1),
        clip_max=reweighting_config.get('clip_max', 10.0),
        max_fit_samples=reweighting_config.get('max_fit_samples', 1000),
        random_state=seed,
    )
    weighted_pred, weighted_proba = train_classifier(
        X_train,
        y_train,
        X_test,
        seed=seed,
        sample_weight=weights,
        model_config=model_config,
    )
    reweighted = evaluate_shift(
        X_test,
        y_test,
        s_test,
        weighted_pred,
        weighted_proba,
        ece_bins=ece_bins,
    )

    # Fit post-processing thresholds on an independent labeled calibration set;
    # the evaluation labels above are never used for threshold selection.
    calibration_seed = seed + threshold_config.get('calibration_seed_offset', 30000)
    X_cal, y_cal, s_cal = _generate_shift(
        shift_type,
        severity,
        data_config,
        shift_config,
        calibration_seed,
        threshold_config.get('calibration_num_samples', 1000),
    )
    _, calibration_proba = train_classifier(
        X_train, y_train, X_cal, seed=seed, model_config=model_config
    )
    thresholds = adjust_threshold_equal_opportunity(
        y_cal,
        calibration_proba,
        s_cal,
        grid_resolution=threshold_config.get('grid_resolution', 0.01),
    )
    threshold_pred = apply_group_thresholds(y_proba, s_test, thresholds)
    threshold_metrics = evaluate_shift(
        X_test,
        y_test,
        s_test,
        threshold_pred,
        y_proba,
        ece_bins=ece_bins,
    )

    # Supplementary target-retraining oracle: train and evaluate on independent
    # samples from the same shifted distribution.
    target_train_seed = seed + supplementary_config.get('train_seed_offset', 20000)
    X_target_train, y_target_train, _ = _generate_shift(
        shift_type,
        severity,
        data_config,
        shift_config,
        target_train_seed,
        supplementary_config.get('train_num_samples', data_config['num_samples']),
    )
    target_pred, target_proba = train_classifier(
        X_target_train,
        y_target_train,
        X_test,
        seed=seed,
        model_config=model_config,
    )
    target_retrained = evaluate_shift(
        X_test,
        y_test,
        s_test,
        target_pred,
        target_proba,
        ece_bins=ece_bins,
    )

    for metric in METRIC_KEYS:
        results[shift_type]['baseline'][metric].append(baseline[metric])
        results[shift_type]['kde_reweighting'][metric].append(reweighted[metric])
        results[shift_type]['threshold_tpr'][metric].append(
            threshold_metrics[metric]
        )
        results[shift_type]['target_retrained'][metric].append(
            target_retrained[metric]
        )
    ess = effective_sample_size(weights)
    diagnostics = results[shift_type]['diagnostics']
    diagnostics['ess'].append(ess)
    diagnostics['ess_fraction'].append(ess / len(weights))
    diagnostics['weight_min'].append(float(weights.min()))
    diagnostics['weight_max'].append(float(weights.max()))
    diagnostics['thresholds'].append(
        {str(group): float(value) for group, value in thresholds.items()}
    )


def generate_phase_diagram_data(config, seed=42):
    """
    Generate data for phase diagrams across three shift types.
    
    Parameters:
    -----------
    config : dict
        Experiment configuration dictionary
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    dict, array, array, array
        Phase diagram matrices for each metric/shift type, and severity arrays
    """

    group_cfg = config['shifts']['group_shift']
    cov_cfg = config['shifts']['covariate_shift']
    label_cfg = config['shifts']['label_shift']
    
    alphas = np.linspace(group_cfg['severity_min'], group_cfg['severity_max'], 
                         int(group_cfg['num_steps']))
    gammas = np.linspace(cov_cfg['severity_min'], cov_cfg['severity_max'], 
                         int(cov_cfg['num_steps']))
    betas = np.linspace(label_cfg['severity_min'], label_cfg['severity_max'], 
                        int(label_cfg['num_steps']))
    
    data_cfg = config['data']
    
    # Base data for training (unshifted)
    X_train, y_train, s_train = generate_data(
        num_samples=data_cfg['num_samples'],
        prior_a=data_cfg['prior_a'],
        base_rate_a=data_cfg['base_rate_a'],
        base_rate_b=data_cfg['base_rate_b'],
        seed=seed
    )
    
    target_group = cov_cfg.get('target_group', 'A')
    print(f"Generating phase diagram data with target group for covariate shift: {target_group}")
    
    # Initialize results matrices
    results = {
        'group_shift': _empty_shift_results(),
        'covariate_shift': _empty_shift_results(),
        'label_shift': _empty_shift_results(),
    }
    model_cfg = config.get('model', {})
    reweighting_cfg = config.get('reweighting', {})
    threshold_cfg = config.get('threshold_adjustment', {})
    supplementary_cfg = config.get('supplementary', {})
    ece_bins = config.get('metrics', {}).get('ece_bins', 10)
    
    print("Generating phase diagram data...")
    
    # Group Shift
    print("  Group shift...", end='', flush=True)
    for severity in alphas:
        _evaluate_severity(
            results, 'group_shift', severity, group_cfg, data_cfg, model_cfg,
            reweighting_cfg, threshold_cfg, supplementary_cfg,
            X_train, y_train, seed, ece_bins
        )
    print(" done")
    
    # Covariate Shift
    print("  Covariate shift...", end='', flush=True)
    for severity in gammas:
        _evaluate_severity(
            results, 'covariate_shift', severity, cov_cfg, data_cfg, model_cfg,
            reweighting_cfg, threshold_cfg, supplementary_cfg,
            X_train, y_train, seed, ece_bins
        )
    print(" done")
    
    # Label Shift
    print("  Label shift...", end='', flush=True)
    for severity in betas:
        _evaluate_severity(
            results, 'label_shift', severity, label_cfg, data_cfg, model_cfg,
            reweighting_cfg, threshold_cfg, supplementary_cfg,
            X_train, y_train, seed, ece_bins
        )
    print(" done")
    
    return results, alphas, gammas, betas


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def save_results(results, alphas, gammas, betas, log_dir='outputs/logs'):
    """
    Save experimental results to log directory.
    
    Parameters:
    -----------
    results : dict
        Fairness metrics results
    alphas, gammas, betas : array-like
        Severity levels for each shift type
    log_dir : str, optional
        Directory to save logs. Default: 'outputs/logs'
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamp for the log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'sweep_results_{timestamp}.json')
    
    # Prepare data for JSON serialization
    output_data = {
        'schema_version': 3,
        'timestamp': timestamp,
        'alphas': alphas.tolist() if isinstance(alphas, np.ndarray) else list(alphas),
        'gammas': gammas.tolist() if isinstance(gammas, np.ndarray) else list(gammas),
        'betas': betas.tolist() if isinstance(betas, np.ndarray) else list(betas),
        'results': _json_ready(results),
    }
    
    # Save to JSON
    with open(log_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Results saved to {log_file}")
    return log_file


def run_sweep(config, seed=42):
    """
    Run experimental sweep across shift types and compute fairness metrics.
    
    This function evaluates how different shift scenarios affect fairness metrics
    (Demographic Parity, Equalized Odds, and ECE Gap).
    """
    print("\n" + "="*60)
    print("FAIRNESS METRICS EXPERIMENTAL SWEEP")
    print("="*60 + "\n")
    
    # Generate phase diagram data
    results, alphas, gammas, betas = generate_phase_diagram_data(config, seed=seed)
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60 + "\n")
    
    for shift_type in ('group_shift', 'covariate_shift', 'label_shift'):
        print(f"{shift_type.upper()}:")
        for method in METHOD_KEYS:
            dp = results[shift_type][method]['dp']
            eo = results[shift_type][method]['eo']
            tpr_gap = results[shift_type][method]['tpr_gap']
            print(
                f"  {method}: DP [{min(dp):.4f}, {max(dp):.4f}], "
                f"EO [{min(eo):.4f}, {max(eo):.4f}], "
                f"TPR gap [{min(tpr_gap):.4f}, {max(tpr_gap):.4f}]"
            )
        ess = results[shift_type]['diagnostics']['ess_fraction']
        print(f"  ESS fraction: [{min(ess):.3f}, {max(ess):.3f}]\n")
    
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60 + "\n")
    
    # Save results to logs directory
    log_dir = config.get('output', {}).get('log_dir', 'outputs/logs')
    save_results(results, alphas, gammas, betas, log_dir=log_dir)
    
    print("\n" + "="*60 + "\n")
    
    return results, alphas, gammas, betas


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/experiment_config.yaml',
                        help='Path to experiment config file')
    args = parser.parse_args()
    
    config = load_config(args.config)
    seed = config.get('experiment', {}).get('seed', 42)
    run_sweep(config, seed=seed)
