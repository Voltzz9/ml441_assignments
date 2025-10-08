import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# set all plots to times new roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 24

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
                           f1_mean, f1_std, param_name='n_estimators', save_path=None):
    """
    Plots convergence curves for F1, Precision, and Recall with standard deviation bands.
    
    Parameters:
    - param_sizes: List of parameter values
    - precision_mean, precision_std: Mean and std of precision scores
    - recall_mean, recall_std: Mean and std of recall scores
    - f1_mean, f1_std: Mean and std of F1 scores
    - param_name: Name of the parameter being varied
    - save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot F1 score
    ax.plot(param_sizes, f1_mean, marker='o', label='F1 Score', linewidth=2, markersize=8)
    ax.fill_between(param_sizes, 
                     np.array(f1_mean) - np.array(f1_std), 
                     np.array(f1_mean) + np.array(f1_std), 
                     alpha=0.2)
    
    # Plot Precision
    ax.plot(param_sizes, precision_mean, marker='s', label='Precision', linewidth=2, markersize=8)
    ax.fill_between(param_sizes, 
                     np.array(precision_mean) - np.array(precision_std), 
                     np.array(precision_mean) + np.array(precision_std), 
                     alpha=0.2)
    
    # Plot Recall
    ax.plot(param_sizes, recall_mean, marker='^', label='Recall', linewidth=2, markersize=8)
    ax.fill_between(param_sizes, 
                     np.array(recall_mean) - np.array(recall_std), 
                     np.array(recall_mean) + np.array(recall_std), 
                     alpha=0.2)
    
    # ax.set_xscale('log')
    if param_name == 'n_estimators':
        ax.set_xlabel('Number of Estimators')
    else:   
        ax.set_xlabel(f'{param_name}')
    ax.set_ylabel('Score')
    ax.set_ylim([0.5, 1.0])
    ax.legend(loc='best', fontsize=20)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Plot saved to {save_path}')
    
    plt.show()


def plot_stability_analysis(param_sizes, f1_mean, f1_std, param_name='n_estimators', save_path=None):
    """
    Plots coefficient of variation (CV) to show stability/variance reduction.
    CV = std / mean
    
    Parameters:
    - param_sizes: List of parameter values
    - f1_mean: Mean F1 scores
    - f1_std: Std of F1 scores
    - param_name: Name of the parameter being varied
    - save_path: Path to save the figure
    """
    # Calculate coefficient of variation
    cv = np.array(f1_std) / np.array(f1_mean)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(param_sizes, cv, marker='o', linewidth=2, markersize=8, color='crimson')
    # ax.set_xscale('log')
    if param_name == 'n_estimators':
        ax.set_xlabel(f'Number of Estimators')
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
                             param_name='n_estimators', save_path=None):
    """
    Plots F1 score vs training time to identify the "sweet spot".
    
    Parameters:
    - training_times: List of training times
    - f1_mean: Mean F1 scores
    - f1_std: Std of F1 scores
    - param_sizes: List of parameter values (for annotations)
    - param_name: Name of the parameter being varied
    - save_path: Path to save the figure
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
    
    ax.set_xscale('log')
    ax.set_xlabel('Training Time (seconds, log scale)')
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


