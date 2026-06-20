import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from glob import glob

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils import load_sweep_results


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
    shift_labels = ['Group Shift', 'Covariate Shift', 'Label Shift']
    severity_arrays = [alphas, gammas, betas]
    metric_types = ['dp', 'eo', 'ece_gap']
    metric_labels = ['Demographic Parity Difference', 'Equalized Odds Difference', 'ECE Gap']
    
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
            metric_data = results[shift_type][metric_type]
            metric_array = np.array(metric_data).reshape(1, -1)
            
            # Create heatmap
            sns.heatmap(
                metric_array,
                ax=ax,
                cmap='RdYlGn_r',  # Red (high unfairness) to Green (low unfairness)
                cbar=True,
                xticklabels=[f'{s:.2f}' for s in severities],
                yticklabels=[shift_label],
                vmin=0,
                vmax=1,  # Normalize to [0, 1] range
                annot=True,
                fmt='.3f',
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
    shift_labels = ['Group Shift', 'Covariate Shift', 'Label Shift']
    severity_arrays = [alphas, gammas, betas]
    metric_types = ['dp', 'eo', 'ece_gap']
    metric_labels = ['Demographic Parity\nDifference', 'Equalized Odds\nDifference', 'ECE Gap']
    
    # Create separate figure for each shift type
    for shift_type, shift_label, severities in zip(shift_types, shift_labels, severity_arrays):
        fig, axes = plt.subplots(1, 3, figsize=(14, 3))
        fig.suptitle(f'Fairness Metrics Under {shift_label}', 
                     fontsize=14, fontweight='bold')
        
        for col_idx, (metric_type, metric_label) in enumerate(zip(metric_types, metric_labels)):
            ax = axes[col_idx]
            
            # Get metric data
            metric_data = np.array(results[shift_type][metric_type]).reshape(1, -1)
            
            # Create heatmap
            sns.heatmap(
                metric_data,
                ax=ax,
                cmap='RdYlGn_r',
                cbar=True,
                xticklabels=[f'{s:.2f}' for s in severities],
                yticklabels=[shift_label],
                vmin=0,
                vmax=1,
                annot=True,
                fmt='.3f',
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
            print("\n❌ ERROR: No log files found in outputs/logs/")
            print("   Please run: python experiments/run_sweep.py")
            print("   to generate results first.\n")
            return
        print(f"\n✓ Using latest log: {log_file}")
    else:
        if not os.path.exists(log_file):
            print(f"\n❌ ERROR: Log file not found: {log_file}\n")
            return
        print(f"\n✓ Using log: {log_file}")
    
    # Load results from log
    print("Loading results from cache...")
    try:
        results, alphas, gammas, betas = load_sweep_results(log_file)
    except Exception as e:
        print(f"\n❌ ERROR: Could not load log file: {e}\n")
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
    
    print("\n" + "="*60)
    print("Phase diagram visualization complete!")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
