import numpy as np
import os
import json
from datetime import datetime
from sklearn.linear_model import LogisticRegression

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


def generate_phase_diagram_data(alphas=None, gammas=None, betas=None, seed=42):
    """
    Generate data for phase diagrams across three shift types.
    
    Parameters:
    -----------
    alphas : array-like, optional
        Group shift severity levels. Default: np.linspace(0.0, 0.5, 8)
    gammas : array-like, optional
        Covariate shift severity levels. Default: np.linspace(1.0, 4.0, 8)
    betas : array-like, optional
        Label shift severity levels. Default: np.linspace(0.0, 0.3, 8)
    
    Returns:
    --------
    dict, array, array, array
        Phase diagram matrices for each metric/shift type, and severity arrays
    """
    if alphas is None:
        alphas = np.linspace(0.0, 0.5, 8)
    if gammas is None:
        gammas = np.linspace(1.0, 4.0, 8)
    if betas is None:
        betas = np.linspace(0.0, 0.3, 8)
    
    # Initialize results matrices
    results = {
        'group_shift': {'dp': [], 'eo': [], 'ece_gap': []},
        'covariate_shift': {'dp': [], 'eo': [], 'ece_gap': []},
        'label_shift': {'dp': [], 'eo': [], 'ece_gap': []}
    }
    
    X_train, y_train, s_train = generate_data(seed=seed)  # Base data for training

    print("Generating phase diagram data...")
    
    # Group Shift
    print("  Group shift...", end='', flush=True)
    for severity in alphas:
        X_test, y_test, s_test = group_shift(severity=severity)
        y_pred, y_proba = train_classifier(X_train, y_train, X_test, seed=42)
        metrics = evaluate_shift(X_test, y_test, s_test, y_pred, y_proba)
        for metric in ['dp', 'eo', 'ece_gap']:
            results['group_shift'][metric].append(metrics[metric])
    print(" ✓")
    
    # Covariate Shift
    print("  Covariate shift...", end='', flush=True)
    for severity in gammas:
        X_test, y_test, s_test = covariate_shift(severity=severity, group='A')
        y_pred, y_proba = train_classifier(X_train, y_train, X_test, seed=42)
        metrics = evaluate_shift(X_test, y_test, s_test, y_pred, y_proba)
        for metric in ['dp', 'eo', 'ece_gap']:
            results['covariate_shift'][metric].append(metrics[metric])
    print(" ✓")
    
    # Label Shift
    print("  Label shift...", end='', flush=True)
    for severity in betas:
        X_test, y_test, s_test = label_shift(severity=severity)
        y_pred, y_proba = train_classifier(X_train, y_train, X_test, seed=42)
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


def run_sweep(seed=42):
    """
    Run experimental sweep across shift types and compute fairness metrics.
    
    This function evaluates how different shift scenarios affect fairness metrics
    (Demographic Parity, Equalized Odds, and ECE Gap).
    """
    print("\n" + "="*60)
    print("FAIRNESS METRICS EXPERIMENTAL SWEEP")
    print("="*60 + "\n")
    
    # Generate phase diagram data
    results, alphas, gammas, betas = generate_phase_diagram_data(seed=seed)
    
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
    run_sweep()