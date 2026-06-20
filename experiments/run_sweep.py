import numpy as np
import os
import json
import argparse
from datetime import datetime
from sklearn.linear_model import LogisticRegression

from src.utils import load_config
from src.data_generator import generate_data
from src.shifts import group_shift, covariate_shift, label_shift
from src.metrics import (
    demographic_parity_difference, 
    equalized_odds_difference, 
    calculate_group_ece_metrics
)


def train_classifier(X_train, y_train, X_test, seed=42):
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
    clf = LogisticRegression(random_state=seed, max_iter=1000, solver='lbfgs')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    return y_pred, y_proba


def evaluate_shift(X, y, s, y_pred, y_proba):
    """
    Evaluate DP, EO, and ECE gap for a given dataset and predictions.
    """
    dp_diff = demographic_parity_difference(y, y_pred, s)
    eo_diff = equalized_odds_difference(y, y_pred, s)
    _, ece_gap = calculate_group_ece_metrics(y_proba, y, s, bins=10)
    
    return {
        'dp': np.abs(dp_diff),  # Use absolute value for visualization
        'eo': np.abs(eo_diff),
        'ece_gap': ece_gap
    }


def generate_phase_diagram_data(config,seed=42):
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
        'group_shift': {'dp': [], 'eo': [], 'ece_gap': []},
        'covariate_shift': {'dp': [], 'eo': [], 'ece_gap': []},
        'label_shift': {'dp': [], 'eo': [], 'ece_gap': []}
    }
    
    print("Generating phase diagram data...")
    
    # Group Shift
    print("  Group shift...", end='', flush=True)
    for severity in alphas:
        X_test, y_test, s_test = group_shift(severity=severity, seed=seed)
        y_pred, y_proba = train_classifier(X_train, y_train, X_test, seed=seed)
        metrics = evaluate_shift(X_test, y_test, s_test, y_pred, y_proba)
        for metric in ['dp', 'eo', 'ece_gap']:
            results['group_shift'][metric].append(metrics[metric])
    print(" ✓")
    
    # Covariate Shift
    print("  Covariate shift...", end='', flush=True)
    for severity in gammas:
        X_test, y_test, s_test = covariate_shift(severity=severity, group=target_group, seed=seed)
        y_pred, y_proba = train_classifier(X_train, y_train, X_test, seed=seed)
        metrics = evaluate_shift(X_test, y_test, s_test, y_pred, y_proba)
        for metric in ['dp', 'eo', 'ece_gap']:
            results['covariate_shift'][metric].append(metrics[metric])
    print(" ✓")
    
    # Label Shift
    print("  Label shift...", end='', flush=True)
    for severity in betas:
        X_test, y_test, s_test = label_shift(severity=severity, seed=seed)
        y_pred, y_proba = train_classifier(X_train, y_train, X_test, seed=seed)
        metrics = evaluate_shift(X_test, y_test, s_test, y_pred, y_proba)
        for metric in ['dp', 'eo', 'ece_gap']:
            results['label_shift'][metric].append(metrics[metric])
    print(" ✓")
    
    return results, alphas, gammas, betas


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
        'timestamp': timestamp,
        'alphas': alphas.tolist() if isinstance(alphas, np.ndarray) else list(alphas),
        'gammas': gammas.tolist() if isinstance(gammas, np.ndarray) else list(gammas),
        'betas': betas.tolist() if isinstance(betas, np.ndarray) else list(betas),
        'results': {
            shift_type: {
                metric: [float(v) for v in values]
                for metric, values in metrics_dict.items()
            }
            for shift_type, metrics_dict in results.items()
        }
    }
    
    # Save to JSON
    with open(log_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✓ Results saved to {log_file}")
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
    
    # Print summary statistics
    print("GROUP SHIFT:")
    print(f"  DP range: [{min(results['group_shift']['dp']):.4f}, {max(results['group_shift']['dp']):.4f}]")
    print(f"  EO range: [{min(results['group_shift']['eo']):.4f}, {max(results['group_shift']['eo']):.4f}]")
    print(f"  ECE Gap range: [{min(results['group_shift']['ece_gap']):.4f}, {max(results['group_shift']['ece_gap']):.4f}]")
    
    print("\nCOVARIATE SHIFT:")
    print(f"  DP range: [{min(results['covariate_shift']['dp']):.4f}, {max(results['covariate_shift']['dp']):.4f}]")
    print(f"  EO range: [{min(results['covariate_shift']['eo']):.4f}, {max(results['covariate_shift']['eo']):.4f}]")
    print(f"  ECE Gap range: [{min(results['covariate_shift']['ece_gap']):.4f}, {max(results['covariate_shift']['ece_gap']):.4f}]")
    
    print("\nLABEL SHIFT:")
    print(f"  DP range: [{min(results['label_shift']['dp']):.4f}, {max(results['label_shift']['dp']):.4f}]")
    print(f"  EO range: [{min(results['label_shift']['eo']):.4f}, {max(results['label_shift']['eo']):.4f}]")
    print(f"  ECE Gap range: [{min(results['label_shift']['ece_gap']):.4f}, {max(results['label_shift']['ece_gap']):.4f}]")
    
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60 + "\n")
    
    # Save results to logs directory
    save_results(results, alphas, gammas, betas, log_dir='outputs/logs')
    
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