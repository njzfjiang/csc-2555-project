import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from glob import glob
from matplotlib.patches import Patch, Rectangle

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils import load_sweep_results


PHASE_METRICS = [
    ('dp', 'Demographic Parity Difference'),
    ('eo', 'Equalized Odds Difference'),
    ('tpr_gap', 'TPR Gap'),
    ('ece_gap', 'ECE Gap'),
]


def _metrics_for_method(shift_results, method='baseline'):
    """Read schema-v2 method results while remaining compatible with v1 logs."""
    if method in shift_results:
        return shift_results[method]
    if method == 'baseline':
        return shift_results
    return None


def _available_phase_metrics(results, shift_types):
    """Return metrics present in every baseline shift result.

    TPR gap was introduced in schema-v3 logs, so older cached sweeps retain the
    original three-metric phase-diagram layout instead of failing at plot time.
    """
    return [
        (metric, label)
        for metric, label in PHASE_METRICS
        if all(
            metric in _metrics_for_method(results[shift])
            for shift in shift_types
        )
    ]


def _method_available(results, shift_types, method):
    """Return whether a method is present for every requested shift type."""
    return all(
        _metrics_for_method(results[shift], method) is not None
        for shift in shift_types
    )


def _metric_std_for_method(shift_results, method, metric):
    """Return schema-v5 seed variability, or None for older logs."""
    values = (
        shift_results.get('metric_std', {})
        .get(method, {})
        .get(metric)
    )
    if values is None:
        return None
    return np.asarray(values, dtype=float)


def _plot_metric_curve(
    ax,
    severities,
    shift_results,
    method,
    metric,
    label,
    **plot_kwargs,
):
    """Plot a mean curve and a population-SD band when available."""
    method_results = _metrics_for_method(shift_results, method)
    mean = np.asarray(method_results[metric], dtype=float)
    line, = ax.plot(
        severities,
        mean,
        label=label,
        **plot_kwargs,
    )
    std = _metric_std_for_method(shift_results, method, metric)
    if std is not None:
        ax.fill_between(
            severities,
            np.clip(mean - std, 0.0, 1.0),
            np.clip(mean + std, 0.0, 1.0),
            color=line.get_color(),
            alpha=0.12,
            linewidth=0,
        )
    return line


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
    Plot phase diagrams for DP, EO, TPR gap, and ECE gap across shift types.
    
    Creates a grid of heatmaps showing all available metrics and shift types.
    
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
    phase_metrics = _available_phase_metrics(results, shift_types)
    metric_types = [metric for metric, _ in phase_metrics]
    metric_labels = [label for _, label in phase_metrics]
    
    # Calculate vmax for each metric (across all shifts) for consistent scaling
    vmax_per_metric = {}
    for metric in metric_types:
        vmax_per_metric[metric] = max(
            max(_metrics_for_method(results[shift])[metric])
            for shift in shift_types
        )
    
    # Create one row per shift type and one column per available metric.
    fig_width = 5 * len(metric_types) + 1
    fig, axes = plt.subplots(3, len(metric_types), figsize=(fig_width, 12))
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
    phase_metrics = _available_phase_metrics(results, shift_types)
    metric_types = [metric for metric, _ in phase_metrics]
    metric_labels = [label.replace(' Difference', '\nDifference')
                     for _, label in phase_metrics]
    
    # Calculate vmax for each metric (across all shifts) for consistent scaling
    vmax_per_metric = {}
    for metric in metric_types:
        vmax_per_metric[metric] = max(
            max(_metrics_for_method(results[shift])[metric])
            for shift in shift_types
        )
    
    # Create separate figure for each shift type
    for shift_type, shift_label, severities in zip(shift_types, shift_labels, severity_arrays):
        fig_width = 4.5 * len(metric_types) + 0.5
        fig, axes = plt.subplots(1, len(metric_types), figsize=(fig_width, 3))
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


def plot_danger_zone(results, save_dir='outputs'):
    """Plot metric disagreement and recorded-vs-clean label disagreement."""
    joint = results.get('joint_prior_label_shift')
    if joint is None:
        print('Skipping danger-zone plot: joint sweep is absent from log.')
        return

    os.makedirs(save_dir, exist_ok=True)
    alphas = np.asarray(joint['alphas'], dtype=float)
    betas = np.asarray(joint['betas'], dtype=float)
    delta_tpr = np.asarray(
        joint['metrics']['delta_tpr_gap_observed'], dtype=float
    )
    danger_mask = np.asarray(joint['danger_mask'], dtype=bool)
    recorded_minus_clean = -np.asarray(
        joint['metrics']['clean_minus_observed_tpr_gap'], dtype=float
    )
    label_disagreement_mask = np.asarray(
        joint['label_disagreement_mask'], dtype=bool
    )
    definition = joint['definition']

    color_limit = max(float(np.max(np.abs(delta_tpr))), 0.01)
    label_color_limit = max(
        float(np.max(np.abs(recorded_minus_clean))), 0.01
    )
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    ax = axes[0]
    sns.heatmap(
        delta_tpr,
        ax=ax,
        cmap='RdBu_r',
        center=0,
        vmin=-color_limit,
        vmax=color_limit,
        xticklabels=[f'{value:.2f}' for value in alphas],
        yticklabels=[f'{value:.2f}' for value in betas],
        cbar_kws={
            'label': r'$\Delta$ TPR gap using recorded $\tilde{Y}$'
                     ' (relative to no shift)'
        },
    )
    for beta_index, alpha_index in np.argwhere(danger_mask):
        ax.add_patch(
            Rectangle(
                (alpha_index, beta_index),
                1,
                1,
                fill=False,
                hatch='///',
                edgecolor='black',
                linewidth=0.4,
            )
        )

    consensus = 100 * float(definition['consensus_fraction'])
    ax.legend(
        handles=[
            Patch(
                facecolor='white',
                edgecolor='black',
                hatch='///',
                label=f'Danger zone (at least {consensus:.0f}% of seeds)',
            )
        ],
        loc='upper left',
        framealpha=0.95,
    )
    ax.set_xlabel(r'Group-conditioned prior-shift severity $\alpha$',
                  fontweight='bold')
    ax.set_ylabel(r'Asymmetric label-noise severity $\beta$',
                  fontweight='bold')
    ax.set_title(
        '(a) Metric-monitoring danger zone',
        fontweight='bold',
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.invert_yaxis()

    ax = axes[1]
    sns.heatmap(
        recorded_minus_clean,
        ax=ax,
        cmap='PuOr_r',
        center=0,
        vmin=-label_color_limit,
        vmax=label_color_limit,
        xticklabels=[f'{value:.2f}' for value in alphas],
        yticklabels=[f'{value:.2f}' for value in betas],
        cbar_kws={
            'label': r'TPR gap($\tilde{Y}$) $-$ TPR gap($Y^*$)'
        },
    )
    for beta_index, alpha_index in np.argwhere(label_disagreement_mask):
        ax.add_patch(
            Rectangle(
                (alpha_index, beta_index),
                1,
                1,
                fill=False,
                edgecolor='black',
                linewidth=1.0,
            )
        )
    ax.legend(
        handles=[
            Patch(
                facecolor='white',
                edgecolor='black',
                label=(
                    'Outlined: recorded/clean TPR gaps differ by at least '
                    f"{100 * float(definition['min_label_disagreement']):.0f} pp"
                ),
            )
        ],
        loc='upper left',
        framealpha=0.95,
    )
    ax.set_xlabel(r'Group-conditioned prior-shift severity $\alpha$',
                  fontweight='bold')
    ax.set_ylabel(r'Asymmetric label-noise severity $\beta$',
                  fontweight='bold')
    ax.set_title('(b) Recorded-label measurement distortion',
                 fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.invert_yaxis()

    fig.suptitle(
        'Joint Prior-Shift and Label-Noise Phase Diagram',
        fontsize=15,
        fontweight='bold',
    )
    plt.tight_layout(rect=(0, 0, 1, 0.95))

    output_path = os.path.join(save_dir, 'danger_zone_phase_diagram.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Danger-zone phase diagram saved to {output_path}')


def plot_target_data_access_comparison(results, gammas, save_dir='outputs'):
    """Compare target-data access regimes under the matched covariate shift."""
    shift_type = 'covariate_shift'
    methods = [
        ('baseline', 'Frozen source (source labels only)'),
        ('kde_reweighting', 'KDE (unlabeled target X)'),
        ('threshold_tpr', 'EO threshold (labeled target Y, S)'),
        ('target_retrained', 'Target ERM (labeled target X, Y)'),
    ]
    if any(
        _metrics_for_method(results[shift_type], method) is None
        for method, _ in methods
    ):
        print('Skipping access-regime comparison: methods are absent from log.')
        return

    os.makedirs(save_dir, exist_ok=True)
    metrics = [
        ('balanced_accuracy', 'Balanced Accuracy'),
        ('dp', 'DP Difference'),
        ('tpr_gap', 'TPR Gap'),
        ('eo', 'EO Difference'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for ax, (metric, metric_label) in zip(axes.flat, metrics):
        for method, label in methods:
            _plot_metric_curve(
                ax,
                gammas,
                results[shift_type],
                method,
                metric,
                marker='o',
                label=label,
            )
        ax.set_title(metric_label)
        ax.set_xlabel(r'Covariate-shift severity $\gamma$')
        ax.set_ylabel(metric_label)
        ax.grid(alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2)
    fig.suptitle(
        'Mitigation Comparison by Target-Data Access Regime\n'
        '(Pure Covariate Shift)',
        fontweight='bold',
    )
    plt.tight_layout(rect=(0, 0.1, 1, 0.94))
    output_path = os.path.join(
        save_dir, 'mitigation_access_regimes_covariate.png'
    )
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Access-regime comparison saved to {output_path}')


def plot_mitigation_comparison(results, alphas, gammas, betas,
                               save_dir='outputs'):
    """Compare mitigations and the diagnostic target-retraining reference."""
    shift_types = ('group_shift', 'covariate_shift', 'label_shift')
    required = ('kde_reweighting', 'threshold_tpr')
    if any(
        not _method_available(results, shift_types, method)
        for method in required
    ):
        print('Skipping mitigation comparison: methods are absent from log.')
        return

    include_target_retraining = _method_available(
        results, shift_types, 'target_retrained'
    )

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
            'Baseline': 'baseline',
            'KDE reweighting': 'kde_reweighting',
            'TPR threshold': 'threshold_tpr',
        }
        if include_target_retraining:
            methods['Target retraining (diagnostic)'] = 'target_retrained'
        for col, (metric, metric_label) in enumerate(metrics):
            ax = axes[row, col]
            for label, method in methods.items():
                _plot_metric_curve(
                    ax,
                    severities,
                    results[shift_type],
                    method,
                    metric,
                    label,
                    marker='o',
                )
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
        if include_target_retraining:
            alternatives['Target retraining - Baseline'] = _metrics_for_method(
                results[shift_type], 'target_retrained'
            )
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
        for col, (metric, metric_label) in enumerate(metrics):
            ax = axes[row, col]
            _plot_metric_curve(
                ax,
                severities,
                results[shift_type],
                'baseline',
                metric,
                'Frozen source',
                marker='o',
            )
            _plot_metric_curve(
                ax,
                severities,
                results[shift_type],
                'target_retrained',
                metric,
                'Target retrained',
                marker='s',
                linestyle='--',
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
    
    # Plot combined phase diagrams (3x4 for schema-v3 logs)
    print("\nGenerating combined phase diagrams...")
    plot_phase_diagrams(results, alphas, gammas, betas, save_dir=fig_dir)
    
    # Plot separate heatmaps (cleaner visualization)
    print("Generating separate shift-specific heatmaps...")
    plot_separate_heatmaps(results, alphas, gammas, betas, save_dir=fig_dir)

    print("Generating joint danger-zone phase diagram...")
    plot_danger_zone(results, save_dir=fig_dir)

    print("Generating target-data access-regime comparison...")
    plot_target_data_access_comparison(results, gammas, save_dir=fig_dir)

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
