import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy import stats
import seaborn as sns

# set all plots to times new roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 24

import matplotlib as mpl
from matplotlib.cm import get_cmap

mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=plt.cm.Set2.colors)  

def plot_training_time_vs_param_size(param_sizes, training_times, param_name, save_path=None):
    """
    Plots training time vs parameter size.

    Parameters:
    - param_sizes: List of parameter sizes (e.g., number of estimators).
    - training_times: List of training times corresponding to each parameter size.
    - param_name: Name of the parameter being varied (for labeling).
    - save_path: If provided, saves the plot to this path.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(param_sizes, training_times, marker='o')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(f'{param_name} Size (log scale)')
    plt.ylabel('Training Time (seconds, log scale)')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300,bbox_inches='tight')
        print(f'Plot saved to {save_path}')
    
    plt.show()
    
def plot_f1_vs_param_size(param_sizes, f1_scores, param_name, save_path=None):
    """
    Plots F1 score vs parameter size.

    Parameters:
    - param_sizes: List of parameter sizes (e.g., number of estimators).
    - f1_scores: List of F1 scores corresponding to each parameter size.
    - param_name: Name of the parameter being varied (for labeling).
    - save_path: If provided, saves the plot to this path.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(param_sizes, f1_scores, marker='o', color='orange')
    plt.xscale('log')
    plt.xlabel(f'{param_name} Size (log scale)')
    plt.ylabel('F1 Score')
    plt.ylim(0, 1)
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300,bbox_inches='tight')
        print(f'Plot saved to {save_path}')
    
    plt.show()


def plot_convergence_curve(param_sizes, precision_mean, precision_std, recall_mean, recall_std, 
                           f1_mean, f1_std, param_name='n_estimators', save_path=None, use_log_scale=False):
    """
    Plots convergence curves for F1, Precision, and Recall with standard deviation bands.
    
    Parameters:
    - param_sizes: List of parameter values
    - precision_mean, precision_std: Mean and std of precision scores
    - recall_mean, recall_std: Mean and std of recall scores
    - f1_mean, f1_std: Mean and std of F1 scores
    - param_name: Name of the parameter being varied
    - save_path: Path to save the figure
    - use_log_scale: Whether to use log scale for x-axis
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Convert param_sizes to numeric, handling 'auto' string
    param_sizes_numeric = []
    for val in param_sizes:
        if isinstance(val, str) and val == 'auto':
            # Use a placeholder value for 'auto' (e.g., dataset size or a large number)
            param_sizes_numeric.append(256)  # auto=256
        else:
            param_sizes_numeric.append(val)
    
    # Plot F1 score
    ax.plot(param_sizes_numeric, f1_mean, marker='o', label='F1 Score', linewidth=2, markersize=8)
    ax.fill_between(param_sizes_numeric, 
                     np.array(f1_mean) - np.array(f1_std), 
                     np.array(f1_mean) + np.array(f1_std), 
                     alpha=0.2)
    
    # Plot Precision
    ax.plot(param_sizes_numeric, precision_mean, marker='s', label='Precision', linewidth=2, markersize=8)
    ax.fill_between(param_sizes_numeric, 
                     np.array(precision_mean) - np.array(precision_std), 
                     np.array(precision_mean) + np.array(precision_std), 
                     alpha=0.2)
    
    # Plot Recall
    ax.plot(param_sizes_numeric, recall_mean, marker='^', label='Recall', linewidth=2, markersize=8)
    ax.fill_between(param_sizes_numeric, 
                     np.array(recall_mean) - np.array(recall_std), 
                     np.array(recall_mean) + np.array(recall_std), 
                     alpha=0.2)
    
    if use_log_scale:
        ax.set_xscale('log')
    
    # Set x-tick labels to show actual values (including 'auto')
    if 'auto' in [str(x) for x in param_sizes]:
        ax.set_xticks(param_sizes_numeric)
        ax.set_xticklabels(param_sizes, fontsize=18)
    
    if param_name == 'n_estimators':
        if use_log_scale:
            ax.set_xlabel('Number of Estimators (log scale)')
        else:
            ax.set_xlabel('Number of Estimators')
    elif param_name == 'max_samples':
        if use_log_scale:
            ax.set_xlabel('Maximum Samples per Tree (log scale)')
        else:
            ax.set_xlabel('Maximum Samples per Tree')
    else:   
        ax.set_xlabel(f'{param_name}')
    ax.set_ylabel('Score')
    # Dynamically set y-axis limits based on actual data range
    all_scores = np.concatenate([f1_mean, precision_mean, recall_mean])
    y_min = max(0, min(all_scores) - 0.05)  # Leave some margin, but don't go below 0
    y_max = min(1, max(all_scores) + 0.05)  # Leave some margin, but don't exceed 1
    ax.set_ylim([y_min, y_max])
    ax.legend(loc='best', fontsize=20)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Plot saved to {save_path}')
    
    plt.show()


def plot_stability_analysis(param_sizes, f1_mean, f1_std, param_name='n_estimators', save_path=None, use_log_scale=False):
    """
    Plots coefficient of variation (CV) to show stability/variance reduction.
    CV = std / mean
    
    Parameters:
    - param_sizes: List of parameter values
    - f1_mean: Mean F1 scores
    - f1_std: Std of F1 scores
    - param_name: Name of the parameter being varied
    - save_path: Path to save the figure
    - use_log_scale: Whether to use log scale for x-axis
    """
    # Calculate coefficient of variation
    cv = np.array(f1_std) / np.array(f1_mean)
    
    # Convert param_sizes to numeric, handling 'auto' string
    param_sizes_numeric = []
    for val in param_sizes:
        if isinstance(val, str) and val == 'auto':
            param_sizes_numeric.append(512)  # Assuming auto ~= 512 for shuttle dataset
        else:
            param_sizes_numeric.append(val)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(param_sizes_numeric, cv, marker='o', linewidth=2, markersize=8, color='crimson')
    
    if use_log_scale:
        ax.set_xscale('log')
    
    # Set x-tick labels to show actual values (including 'auto')
    if 'auto' in [str(x) for x in param_sizes]:
        ax.set_xticks(param_sizes_numeric)
        ax.set_xticklabels(param_sizes, fontsize=18)
    
    if param_name == 'n_estimators':
        ax.set_xlabel(f'Number of Estimators')
    elif param_name == 'max_samples':
        ax.set_xlabel('Maximum Samples per Tree')
    else:
        ax.set_xlabel(f'{param_name}')
    ax.set_ylabel('Coefficient of Variation (F1)')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Plot saved to {save_path}')
    
    plt.show()
    
    return cv


def plot_performance_vs_time(training_times, f1_mean, f1_std, param_sizes, 
                             param_name='n_estimators', save_path=None, use_log_scale=False):
    """
    Plots F1 score vs training time to identify the "sweet spot".
    
    Parameters:
    - training_times: List of training times
    - f1_mean: Mean F1 scores
    - f1_std: Std of F1 scores
    - param_sizes: List of parameter values (for annotations)
    - param_name: Name of the parameter being varied
    - save_path: Path to save the figure
    - use_log_scale: Whether to use log scale for x-axis
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot with error bars
    ax.errorbar(training_times, f1_mean, yerr=f1_std, 
                marker='o', linewidth=2, markersize=8, 
                capsize=5, capthick=2)
    
    # Annotate some key points
    for i in [0, len(param_sizes)//4, len(param_sizes)//2, -1]:
        ax.annotate(f'{param_name}={param_sizes[i]}', 
                   xy=(training_times[i], f1_mean[i]),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=16, alpha=0.7)
    
    if use_log_scale:
        ax.set_xscale('log')
    ax.set_xlabel('Training Time (seconds' + (' (log scale)' if use_log_scale else '') + ')')
    ax.set_ylabel('F1 Score')
    ax.set_ylim([min(f1_mean) - 0.05, 1.0])
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Plot saved to {save_path}')
    
    plt.show()


def create_summary_table(param_sizes, precision_mean, precision_std, recall_mean, recall_std,
                        f1_mean, f1_std, training_times, param_name='n_estimators', save_path=None):
    """
    Creates a summary table with key metrics.
    
    Parameters:
    - param_sizes: List of parameter values
    - precision_mean, precision_std: Mean and std of precision
    - recall_mean, recall_std: Mean and std of recall
    - f1_mean, f1_std: Mean and std of F1
    - training_times: Training times
    - param_name: Name of the parameter
    - save_path: Path to save CSV
    
    Returns:
    - DataFrame with summary statistics
    """
    cv = np.array(f1_std) / np.array(f1_mean)
    
    summary_df = pd.DataFrame({
        param_name: param_sizes,
        'Precision_Mean': precision_mean,
        'Precision_Std': precision_std,
        'Recall_Mean': recall_mean,
        'Recall_Std': recall_std,
        'F1_Mean': f1_mean,
        'F1_Std': f1_std,
        'F1_CV': cv,
        'Training_Time_s': training_times
    })
    
    if save_path:
        summary_df.to_csv(save_path, index=False)
        print(f'Summary table saved to {save_path}')
    
    return summary_df


def analyze_plateau_point(param_sizes, f1_mean, threshold=0.001):
    """
    Identifies where performance plateaus based on marginal improvement.
    
    Parameters:
    - param_sizes: List of parameter values
    - f1_mean: Mean F1 scores
    - threshold: Threshold for marginal improvement (default 0.1%)
    
    Returns:
    - plateau_idx: Index where plateau starts
    - plateau_value: Parameter value at plateau
    """
    f1_array = np.array(f1_mean)
    improvements = np.diff(f1_array) / f1_array[:-1]
    
    # Find first point where improvement is below threshold
    plateau_indices = np.where(improvements < threshold)[0]
    
    if len(plateau_indices) > 0:
        plateau_idx = plateau_indices[0] + 1  # +1 because diff reduces size by 1
        plateau_value = param_sizes[plateau_idx]
        return plateau_idx, plateau_value, improvements
    else:
        return None, None, improvements


def find_sweet_spot(training_times, f1_mean, param_sizes):
    """
    Finds the "sweet spot" - best F1 score per unit time.
    
    Parameters:
    - training_times: List of training times
    - f1_mean: Mean F1 scores
    - param_sizes: List of parameter values
    
    Returns:
    - sweet_spot_idx: Index of sweet spot
    - sweet_spot_value: Parameter value at sweet spot
    - efficiency: F1 score per second at each point
    """
    efficiency = np.array(f1_mean) / np.array(training_times)
    sweet_spot_idx = np.argmax(efficiency)
    sweet_spot_value = param_sizes[sweet_spot_idx]
    
    return sweet_spot_idx, sweet_spot_value, efficiency


def plot_training_time_scaling(param_sizes, training_times, param_name='max_samples', save_path=None):
    """
    Plots training time vs parameter size on log-log scale to analyze complexity.
    
    Parameters:
    - param_sizes: List of parameter values
    - training_times: List of training times
    - param_name: Name of the parameter being varied
    - save_path: Path to save the figure
    """
    # Convert param_sizes to numeric, handling 'auto' string
    param_sizes_numeric = []
    for val in param_sizes:
        if isinstance(val, str) and val == 'auto':
            param_sizes_numeric.append(512)
        else:
            param_sizes_numeric.append(val)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot on log-log scale
    ax.loglog(param_sizes_numeric, training_times, marker='o', linewidth=2, markersize=10, color='steelblue')
    
    # Fit a line to estimate slope (complexity)
    log_params = np.log(param_sizes_numeric)
    log_times = np.log(training_times)
    slope, intercept = np.polyfit(log_params, log_times, 1)
    
    # Plot fitted line
    fitted_times = np.exp(intercept) * np.array(param_sizes_numeric)**slope
    ax.loglog(param_sizes_numeric, fitted_times, '--', linewidth=2, color='red', 
             label=f'Slope = {slope:.2f}' + ('\n(Sublinear)' if slope < 1 else '\n(Linear)' if slope < 1.1 else '\n(Superlinear)'))
    
    # Set x-tick labels to show actual values (including 'auto')
    if 'auto' in [str(x) for x in param_sizes]:
        ax.set_xticks(param_sizes_numeric)
        ax.set_xticklabels(param_sizes, fontsize=18)
    
    if param_name == 'max_samples':
        ax.set_xlabel('Maximum Samples per Tree (log scale)')
    else:
        ax.set_xlabel(f'{param_name} (log scale)')
    ax.set_ylabel('Training Time (seconds, log scale)')
    ax.legend(loc='best', fontsize=20)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Plot saved to {save_path}')
    
    plt.show()
    
    return slope


def plot_precision_recall_tradeoff(param_sizes, precision_mean, precision_std, recall_mean, recall_std, 
                                   param_name='max_samples', save_path=None):
    """
    Plots Precision vs Recall tradeoff as parameter changes (swamping effect analysis).
    
    Parameters:
    - param_sizes: List of parameter values
    - precision_mean, precision_std: Mean and std of precision scores
    - recall_mean, recall_std: Mean and std of recall scores
    - param_name: Name of the parameter being varied
    - save_path: Path to save the figure
    """
    # Convert param_sizes to numeric for colormap
    param_sizes_numeric = []
    for val in param_sizes:
        if isinstance(val, str) and val == 'auto':
            param_sizes_numeric.append(512)
        else:
            param_sizes_numeric.append(val)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Create scatter plot with color gradient based on parameter size
    scatter = ax.scatter(recall_mean, precision_mean, c=param_sizes_numeric, 
                        s=200, cmap='viridis', edgecolors='black', linewidth=1.5)
    
    # Add error bars
    ax.errorbar(recall_mean, precision_mean, 
               xerr=recall_std, yerr=precision_std,
               fmt='none', ecolor='gray', alpha=0.5, capsize=4)
    
    # Annotate points with parameter values
    for i, param_val in enumerate(param_sizes):
        ax.annotate(f'{param_val}', 
                   xy=(recall_mean[i], precision_mean[i]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=14, alpha=0.8)
    
    # # Add diagonal line for F1 contours
    # recall_range = np.linspace(min(recall_mean)-0.05, max(recall_mean)+0.05, 100)
    # for f1 in [0.7, 0.8, 0.9, 0.95]:
    #     precision_contour = (f1 * recall_range) / (2 * recall_range - f1)
    #     precision_contour = np.clip(precision_contour, 0, 1)
    #     ax.plot(recall_range, precision_contour, '--', alpha=0.3, color='gray', linewidth=1)
    #     # Label the contour
    #     ax.text(recall_range[-1], precision_contour[-1], f'F1={f1}', 
    #            fontsize=12, alpha=0.5, va='bottom')
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim([min(recall_mean)-0.05, max(recall_mean)+0.05])
    ax.set_ylim([min(precision_mean)-0.05, max(precision_mean)+0.05])
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    if param_name == 'max_samples':
        cbar.set_label('Maximum Samples per Tree', fontsize=20)
    else:
        cbar.set_label(param_name, fontsize=20)
    
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Plot saved to {save_path}')
    
    plt.show()


def plot_bootstrap_stability_box_violin(df_results, save_path=None):
    """
    Creates box plot with violin overlay showing bootstrap vs no bootstrap stability.
    Includes statistical significance testing and effect size (Cohen's d).
    
    Note: Since we only have summary statistics (mean, std, min, max), we create 
    a visualization showing the distribution range rather than individual data points.
    
    Parameters:
    - df_results: DataFrame with bootstrap results containing mean, std, min, max
    - save_path: Path to save the figure
    """
    from scipy import stats
    import seaborn as sns
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Get data for both bootstrap settings
    bootstrap_true = df_results[df_results['param_value'] == True].iloc[0]
    bootstrap_false = df_results[df_results['param_value'] == False].iloc[0]
    
    # Create visualization showing mean with error bars and range
    positions = [0, 1]
    labels = ['Bootstrap=True', 'Bootstrap=False']
    
    means = [bootstrap_true['f1_score_mean'], bootstrap_false['f1_score_mean']]
    stds = [bootstrap_true['f1_score_std'], bootstrap_false['f1_score_std']]
    mins = [bootstrap_true['f1_score_min'], bootstrap_false['f1_score_min']]
    maxs = [bootstrap_true['f1_score_max'], bootstrap_false['f1_score_max']]
    
    # Plot range as boxes
    colors = [plt.cm.Set2.colors[0], plt.cm.Set2.colors[1]]
    for i, pos in enumerate(positions):
        # Draw range box
        height = maxs[i] - mins[i]
        rect = plt.Rectangle((pos - 0.25, mins[i]), 0.5, height, 
                            facecolor=colors[i], alpha=0.3, edgecolor=colors[i], linewidth=2)
        ax.add_patch(rect)
        
        # Draw mean line
        ax.plot([pos - 0.25, pos + 0.25], [means[i], means[i]], 
               color=colors[i], linewidth=3, label=labels[i] if i == 0 else None)
        
        # Draw std range
        ax.plot([pos, pos], [means[i] - stds[i], means[i] + stds[i]], 
               color='black', linewidth=2, alpha=0.7)
        ax.plot([pos - 0.1, pos + 0.1], [means[i] - stds[i], means[i] - stds[i]], 
               color='black', linewidth=2, alpha=0.7)
        ax.plot([pos - 0.1, pos + 0.1], [means[i] + stds[i], means[i] + stds[i]], 
               color='black', linewidth=2, alpha=0.7)
    
    # Calculate Cohen's d effect size
    mean_diff = means[0] - means[1]
    pooled_std = np.sqrt((stds[0]**2 + stds[1]**2) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std != 0 else 0
    
    # Add significance annotation based on overlap of confidence intervals
    # If std ranges don't overlap, likely significant
    ci_true = (means[0] - 2*stds[0], means[0] + 2*stds[0])
    ci_false = (means[1] - 2*stds[1], means[1] + 2*stds[1])
    
    overlap = not (ci_true[1] < ci_false[0] or ci_false[1] < ci_true[0])
    
    y_max = max(maxs)
    y_min = min(mins)
    y_range = y_max - y_min
    
    # Significance based on effect size and overlap
    if abs(cohens_d) > 0.8 and not overlap:
        sig_text = '***'
    elif abs(cohens_d) > 0.5:
        sig_text = '**'
    elif abs(cohens_d) > 0.2:
        sig_text = '*'
    else:
        sig_text = 'ns'
    
    ax.plot([0, 1], [y_max + 0.005, y_max + 0.005], 'k-', linewidth=1.5)
    ax.text(0.5, y_max + 0.007, sig_text, ha='center', fontsize=20)
    
    # Add effect size annotation
    ax.text(0.5, y_min - 0.01, f"Cohen's d = {cohens_d:.3f}", 
           ha='center', fontsize=18, style='italic')
    
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel('F1 Score')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xlim([-0.5, 1.5])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Plot saved to {save_path}')
    
    plt.show()
    
    return cohens_d, overlap


def plot_radar_chart_bootstrap(df_results, save_path=None):
    """
    Creates a radar chart comparing bootstrap vs no bootstrap across multiple metrics.
    Metrics: F1, Precision, Recall, AUC-PR, AUC-ROC
    
    Parameters:
    - df_results: DataFrame with bootstrap results 
    - save_path: Path to save the figure
    """
    from math import pi
    
    # Get metrics for both bootstrap settings
    bootstrap_true = df_results[df_results['param_value'] == True].iloc[0]
    bootstrap_false = df_results[df_results['param_value'] == False].iloc[0]
    
    # Define metrics and their values
    metrics = ['F1', 'Precision', 'Recall', 'AUC-PR', 'AUC-ROC']
    
    values_true = [
        bootstrap_true['f1_score_mean'],
        bootstrap_true['precision_mean'],
        bootstrap_true['recall_mean'],
        bootstrap_true['pr_auc_mean'],
        bootstrap_true['roc_auc_mean']
    ]
    
    values_false = [
        bootstrap_false['f1_score_mean'],
        bootstrap_false['precision_mean'],
        bootstrap_false['recall_mean'],
        bootstrap_false['pr_auc_mean'],
        bootstrap_false['roc_auc_mean']
    ]
    
    # Number of variables
    num_vars = len(metrics)
    
    # Compute angle for each axis
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    values_true += values_true[:1]  # Complete the circle
    values_false += values_false[:1]
    angles += angles[:1]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Plot data
    ax.plot(angles, values_true, 'o-', linewidth=2, label='Bootstrap=True', color=plt.cm.Set2.colors[0])
    ax.fill(angles, values_true, alpha=0.25, color=plt.cm.Set2.colors[0])
    
    ax.plot(angles, values_false, 'o-', linewidth=2, label='Bootstrap=False', color=plt.cm.Set2.colors[1])
    ax.fill(angles, values_false, alpha=0.25, color=plt.cm.Set2.colors[1])
    
    # Fix axis to go in the right order and start at 12 o'clock
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw axis lines for each angle and label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=20)
    
    # Set y-axis limits
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=16)
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=18)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Plot saved to {save_path}')
    
    plt.show()


def plot_bootstrap_contamination_interaction(df_interaction, save_path=None):
    """
    Creates grouped bar chart showing bootstrap × contamination interaction.
    
    Parameters:
    - df_interaction: DataFrame with bootstrap and contamination results
    - save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Get unique contamination values
    contamination_values = sorted(df_interaction['contamination'].unique())
    
    # Get F1 scores for each bootstrap setting
    f1_bootstrap_true = []
    f1_bootstrap_false = []
    
    for cont in contamination_values:
        df_cont = df_interaction[df_interaction['contamination'] == cont]
        f1_true = df_cont[df_cont['bootstrap'] == True]['f1_score_mean'].values[0]
        f1_false = df_cont[df_cont['bootstrap'] == False]['f1_score_mean'].values[0]
        f1_bootstrap_true.append(f1_true)
        f1_bootstrap_false.append(f1_false)
    
    # Set up bar positions
    x = np.arange(len(contamination_values))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x - width/2, f1_bootstrap_true, width, 
                   label='Bootstrap=True', color=plt.cm.Set2.colors[0])
    bars2 = ax.bar(x + width/2, f1_bootstrap_false, width,
                   label='Bootstrap=False', color=plt.cm.Set2.colors[1])
    
    # Customize plot
    ax.set_xlabel('Contamination')
    ax.set_ylabel('F1 Score')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{c:.2f}' for c in contamination_values])
    ax.legend(fontsize=20)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Plot saved to {save_path}')
    
    plt.show()



