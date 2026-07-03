import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from glob import glob

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils import load_sweep_results


def _metrics_for_method(shift_results, method='baseline'):
    """Read schema-v2 method results while remaining compatible with v1 logs."""
    if method in shift_results:
        return shift_results[method]
    if method == 'baseline':
        return shift_results
    return None


def find_latest_log(log_dir='outputs/logs'):
    """
    Find the most recent log file in the logs directory.
    
    Parameters:
    -----------
    log_dir : str, optional
        Directory containing log files. Default: 'outputs/logs'
    
    Returns:
    --------
    str or None
        Path to the most recent log file, or None if no logs found
    """
    log_files = glob(os.path.join(log_dir, 'sweep_results_*.json'))
    if not log_files:
        return None
    # Sort by modification time and return the most recent
    return max(log_files, key=os.path.getmtime)


def plot_phase_diagrams(results, alphas, gammas, betas, save_dir='outputs'):
    """
    Plot phase diagrams (heatmaps) for DP, EO, and ECE gap across shift types.
    
    Creates a 3x3 grid of heatmaps showing all metrics and shift types.
    
    Parameters:
    -----------
    results : dict
        Output from generate_phase_diagram_data()
    alphas : array-like
        Group shift severity levels
    gammas : array-like
        Covariate shift severity levels
    betas : array-like
        Label shift severity levels
    save_dir : str, optional
        Directory to save figures. Default: 'outputs'
    """
    os.makedirs(save_dir, exist_ok=True)
    
    shift_types = ['group_shift', 'covariate_shift', 'label_shift']
    shift_labels = [
        'Group-Conditional Base-Rate Shift',
        'Covariate Shift',
        'Label Shift',
    ]
    severity_arrays = [alphas, gammas, betas]
    metric_types = ['dp', 'eo', 'ece_gap']
    metric_labels = ['Demographic Parity Difference', 'Equalized Odds Difference', 'ECE Gap']
    
    # Calculate vmax for each metric (across all shifts) for consistent scaling
    vmax_per_metric = {}
    for metric in metric_types:
        vmax_per_metric[metric] = max(
            max(_metrics_for_method(results[shift])[metric])
            for shift in shift_types
        )
    
    # Create a figure with 3 rows (shift types) x 3 columns (metrics)
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle('Phase Diagrams: Fairness Metrics Under Distribution Shifts', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    for row_idx, (shift_type, shift_label, severities) in enumerate(
        zip(shift_types, shift_labels, severity_arrays)
    ):
        for col_idx, (metric_type, metric_label) in enumerate(
            zip(metric_types, metric_labels)
        ):
            ax = axes[row_idx, col_idx]
            
            # Get metric data
            metric_data = _metrics_for_method(results[shift_type])[metric_type]
            metric_array = np.array(metric_data).reshape(1, -1)
            
            # Create heatmap with metric-specific vmax for consistent scaling across shifts
            sns.heatmap(
                metric_array,
                ax=ax,
                cmap='RdYlGn_r',  # Red (high unfairness) to Green (low unfairness)
                cbar=True,
                xticklabels=[f'{s:.2f}' for s in severities],
                yticklabels=[shift_label],
                vmin=0,
                vmax=vmax_per_metric[metric_type],  # Global max per metric for fair comparison
                annot=False, 
                cbar_kws={'label': metric_label}
            )
            
            ax.set_xlabel('Shift Severity', fontweight='bold')
            ax.set_title(f'{shift_label}\n{metric_label}', fontweight='bold', fontsize=11)
            
            # Rotate x labels for better readability
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(save_dir, 'phase_diagrams.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Phase diagrams saved to {output_path}")
    
    plt.close()


def plot_separate_heatmaps(results, alphas, gammas, betas, save_dir='outputs'):
    """
    Create cleaner separate heatmaps for each shift type showing all metrics.
    
    Parameters:
    -----------
    results : dict
        Output from generate_phase_diagram_data()
    alphas : array-like
        Group shift severity levels
    gammas : array-like
        Covariate shift severity levels
    betas : array-like
        Label shift severity levels
    save_dir : str, optional
        Directory to save figures. Default: 'outputs'
    """
    os.makedirs(save_dir, exist_ok=True)
    
    shift_types = ['group_shift', 'covariate_shift', 'label_shift']
    shift_labels = [
        'Group-Conditional Base-Rate Shift',
        'Covariate Shift',
        'Label Shift',
    ]
    severity_arrays = [alphas, gammas, betas]
    metric_types = ['dp', 'eo', 'ece_gap']
    metric_labels = ['Demographic Parity\nDifference', 'Equalized Odds\nDifference', 'ECE Gap']
    
    # Calculate vmax for each metric (across all shifts) for consistent scaling
    vmax_per_metric = {}
    for metric in metric_types:
        vmax_per_metric[metric] = max(
            max(_metrics_for_method(results[shift])[metric])
            for shift in shift_types
        )
    
    # Create separate figure for each shift type
    for shift_type, shift_label, severities in zip(shift_types, shift_labels, severity_arrays):
        fig, axes = plt.subplots(1, 3, figsize=(14, 3))
        fig.suptitle(f'Fairness Metrics Under {shift_label}', 
                     fontsize=14, fontweight='bold')
        
        for col_idx, (metric_type, metric_label) in enumerate(zip(metric_types, metric_labels)):
            ax = axes[col_idx]
            
            # Get metric data
            metric_data = np.array(
                _metrics_for_method(results[shift_type])[metric_type]
            ).reshape(1, -1)
            
            # Create heatmap with metric-specific vmax for consistent scaling
            sns.heatmap(
                metric_data,
                ax=ax,
                cmap='RdYlGn_r',
                cbar=True,
                xticklabels=[f'{s:.2f}' for s in severities],
                yticklabels=[shift_label],
                vmin=0,
                vmax=vmax_per_metric[metric_type],  # Global max per metric for fair comparison
                annot=False,  # Clean visualization without clutter
                cbar_kws={'label': 'Metric Value'}
            )
            
            ax.set_xlabel('Shift Severity', fontweight='bold')
            ax.set_title(metric_label, fontweight='bold')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Save figure
        filename = f'phase_diagram_{shift_type}.png'
        output_path = os.path.join(save_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"{shift_label} phase diagram saved to {output_path}")
        
        plt.close()


def plot_mitigation_comparison(results, alphas, gammas, betas,
                               save_dir='outputs'):
    """Compare baseline, KDE reweighting, and TPR threshold adjustment."""
    required = ('kde_reweighting', 'threshold_tpr')
    if any(
        _metrics_for_method(results['group_shift'], method) is None
        for method in required
    ):
        print('Skipping mitigation comparison: methods are absent from log.')
        return

    os.makedirs(save_dir, exist_ok=True)
    shifts = [
        ('group_shift', 'Group-Conditional Base-Rate Shift', alphas),
        ('covariate_shift', 'Covariate Shift', gammas),
        ('label_shift', 'Asymmetric Label Noise', betas),
    ]
    metrics = [
        ('dp', 'DP Difference'),
        ('eo', 'EO Difference'),
        ('tpr_gap', 'TPR Gap'),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(17, 12))
    for row, (shift_type, shift_label, severities) in enumerate(shifts):
        methods = {
            'Baseline': _metrics_for_method(results[shift_type], 'baseline'),
            'KDE reweighting': _metrics_for_method(
                results[shift_type], 'kde_reweighting'
            ),
            'TPR threshold': _metrics_for_method(
                results[shift_type], 'threshold_tpr'
            ),
        }
        for col, (metric, metric_label) in enumerate(metrics):
            ax = axes[row, col]
            for label, method_results in methods.items():
                ax.plot(severities, method_results[metric], marker='o', label=label)
            ax.set_title(f'{shift_label}: {metric_label}')
            ax.set_xlabel('Shift Severity')
            ax.set_ylabel(metric_label)
            ax.grid(alpha=0.25)
            ax.legend()
    fig.suptitle('Mitigation Comparison', fontweight='bold')
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    comparison_path = os.path.join(save_dir, 'mitigation_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Mitigation comparison saved to {comparison_path}')

    fig, axes = plt.subplots(3, 3, figsize=(17, 12))
    for row, (shift_type, shift_label, severities) in enumerate(shifts):
        baseline = _metrics_for_method(results[shift_type], 'baseline')
        alternatives = {
            'KDE - Baseline': _metrics_for_method(
                results[shift_type], 'kde_reweighting'
            ),
            'TPR threshold - Baseline': _metrics_for_method(
                results[shift_type], 'threshold_tpr'
            ),
        }
        for col, (metric, metric_label) in enumerate(metrics):
            ax = axes[row, col]
            ax.axhline(0, color='black', linewidth=1)
            for label, method_results in alternatives.items():
                delta = (
                    np.asarray(method_results[metric])
                    - np.asarray(baseline[metric])
                )
                ax.plot(severities, delta, marker='o', label=label)
            ax.set_title(f'{shift_label}: Δ {metric_label}')
            ax.set_xlabel('Shift Severity')
            ax.set_ylabel('Method - Baseline')
            ax.grid(alpha=0.25)
            ax.legend()
    fig.suptitle('Mitigation Change from Baseline (negative is fairer)',
                 fontweight='bold')
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    delta_path = os.path.join(save_dir, 'mitigation_delta.png')
    plt.savefig(delta_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Mitigation deltas saved to {delta_path}')


def plot_supplementary_retraining(results, alphas, gammas, betas,
                                  save_dir='outputs'):
    """Compare the frozen source classifier with target-distribution retraining."""
    if _metrics_for_method(results['group_shift'], 'target_retrained') is None:
        print('Skipping supplementary retraining: method is absent from log.')
        return

    shifts = [
        ('group_shift', 'Group-Conditional Base-Rate Shift', alphas),
        ('covariate_shift', 'Covariate Shift', gammas),
        ('label_shift', 'Asymmetric Label Noise', betas),
    ]
    metrics = [
        ('dp', 'DP Difference'),
        ('eo', 'EO Difference'),
        ('tpr_gap', 'TPR Gap'),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(17, 12))
    for row, (shift_type, shift_label, severities) in enumerate(shifts):
        baseline = _metrics_for_method(results[shift_type], 'baseline')
        retrained = _metrics_for_method(results[shift_type], 'target_retrained')
        for col, (metric, metric_label) in enumerate(metrics):
            ax = axes[row, col]
            ax.plot(severities, baseline[metric], marker='o', label='Frozen source')
            ax.plot(
                severities,
                retrained[metric],
                marker='s',
                linestyle='--',
                label='Target retrained',
            )
            ax.set_title(f'{shift_label}: {metric_label}')
            ax.set_xlabel('Shift Severity')
            ax.set_ylabel(metric_label)
            ax.grid(alpha=0.25)
            ax.legend()
    fig.suptitle('Supplementary: Target-Distribution Retraining',
                 fontweight='bold')
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    output_path = os.path.join(save_dir, 'supplementary_target_retraining.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Supplementary retraining plot saved to {output_path}')


def main(log_file=None):
    """
    Main function to plot phase diagrams from cached log results.
    
    Parameters:
    -----------
    log_file : str, optional
        Path to the log file to load. If None, uses the most recent log.
        If no log found, raises error.
    """
    print("\n" + "="*60)
    print("PHASE DIAGRAM VISUALIZATION (FROM CACHED LOGS)")
    print("="*60)
    
    # Find or use specified log file
    if log_file is None:
        log_file = find_latest_log(log_dir='outputs/logs')
        if log_file is None:
            print("\nERROR: No log files found in outputs/logs/")
            print("   Please run: python experiments/run_sweep.py")
            print("   to generate results first.\n")
            return
        print(f"\nUsing latest log: {log_file}")
    else:
        if not os.path.exists(log_file):
            print(f"\nERROR: Log file not found: {log_file}\n")
            return
        print(f"\nUsing log: {log_file}")
    
    # Load results from log
    print("Loading results from cache...")
    try:
        results, alphas, gammas, betas = load_sweep_results(log_file)
    except Exception as e:
        print(f"\nERROR: Could not load log file: {e}\n")
        return
    
    # Create figures directory if it doesn't exist
    fig_dir = 'outputs/figures'
    os.makedirs(fig_dir, exist_ok=True)
    
    # Plot combined phase diagrams (3x3 grid)
    print("\nGenerating combined phase diagrams...")
    plot_phase_diagrams(results, alphas, gammas, betas, save_dir=fig_dir)
    
    # Plot separate heatmaps (cleaner visualization)
    print("Generating separate shift-specific heatmaps...")
    plot_separate_heatmaps(results, alphas, gammas, betas, save_dir=fig_dir)

    print("Generating mitigation comparisons...")
    plot_mitigation_comparison(
        results, alphas, gammas, betas, save_dir=fig_dir
    )

    print("Generating supplementary target-retraining comparison...")
    plot_supplementary_retraining(
        results, alphas, gammas, betas, save_dir=fig_dir
    )
    
    print("\n" + "="*60)
    print("Phase diagram visualization complete!")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
