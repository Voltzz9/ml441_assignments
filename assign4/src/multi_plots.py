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


def plot_multi_dataset_convergence(param_name, datasets=['shuttle', 'campaign', 'fraud'], 
                                   results_base_path='../results', save_path=None):
    """
    Plots convergence curves comparing different datasets for a given parameter.
    Shows F1, Precision, and Recall on the same plot with one line per dataset.
    
    Parameters:
    - param_name: Name of the parameter ('n_estimators', 'max_samples', 'contamination', 'max_features')
    - datasets: List of dataset names to compare
    - results_base_path: Base path to results directory
    - save_path: Path to save the figure
    """
    # Map parameter names to file names and folder names
    param_file_map = {
        'n_estimators': ('n_estimators', 'num_est_results_narrow.csv'),
        'max_samples': ('max_samples', 'num_samples_results.csv'),
        'contamination': ('contamination', 'contamination_results.csv'),
        'max_features': ('max_features', 'features_results.csv')
    }
    
    if param_name not in param_file_map:
        raise ValueError(f"Parameter {param_name} not supported. Choose from {list(param_file_map.keys())}")
    
    folder_name, file_name = param_file_map[param_name]
    
    # Create figure with 3 subplots (F1, Precision, Recall)
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    
    # Define colors for datasets
    colors = {'shuttle': '#1f77b4', 'campaign': '#ff7f0e', 'fraud': '#2ca02c'}
    markers = {'shuttle': 'o', 'campaign': 's', 'fraud': '^'}
    
    # Track if we should use log scale
    use_log_scale = param_name in ['n_estimators', 'max_samples']
    
    for dataset in datasets:
        # Load data
        file_path = os.path.join(results_base_path, dataset, folder_name, file_name)
        
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, skipping {dataset}")
            continue
            
        df = pd.read_csv(file_path)
        
        # Extract param values and metrics
        param_values = df['param_value'].values
        
        # Convert param_values to numeric, handling 'auto' string
        param_values_numeric = []
        for val in param_values:
            if val == 'auto':
                param_values_numeric.append(256)  # Default value for 'auto'
            else:
                param_values_numeric.append(float(val))
        
        f1_mean = df['f1_score_mean'].values
        f1_std = df['f1_score_std'].values
        precision_mean = df['precision_mean'].values
        precision_std = df['precision_std'].values
        recall_mean = df['recall_mean'].values
        recall_std = df['recall_std'].values
        
        # Plot F1 Score
        axes[0].plot(param_values_numeric, f1_mean, marker=markers[dataset], 
                    label=dataset.capitalize(), linewidth=2, markersize=8, 
                    color=colors[dataset])
        axes[0].fill_between(param_values_numeric, 
                            np.array(f1_mean) - np.array(f1_std), 
                            np.array(f1_mean) + np.array(f1_std), 
                            alpha=0.2, color=colors[dataset])
        
        # Plot Precision
        axes[1].plot(param_values_numeric, precision_mean, marker=markers[dataset], 
                    label=dataset.capitalize(), linewidth=2, markersize=8,
                    color=colors[dataset])
        axes[1].fill_between(param_values_numeric, 
                            np.array(precision_mean) - np.array(precision_std), 
                            np.array(precision_mean) + np.array(precision_std), 
                            alpha=0.2, color=colors[dataset])
        
        # Plot Recall
        axes[2].plot(param_values_numeric, recall_mean, marker=markers[dataset], 
                    label=dataset.capitalize(), linewidth=2, markersize=8,
                    color=colors[dataset])
        axes[2].fill_between(param_values_numeric, 
                            np.array(recall_mean) - np.array(recall_std), 
                            np.array(recall_mean) + np.array(recall_std), 
                            alpha=0.2, color=colors[dataset])
    
    # Configure subplots
    metric_names = ['F1 Score', 'Precision', 'Recall']
    for i, (ax, metric_name) in enumerate(zip(axes, metric_names)):
        if use_log_scale:
            ax.set_xscale('log')
        
        # Set labels
        if param_name == 'n_estimators':
            ax.set_xlabel('Number of Estimators' + (' (log scale)' if use_log_scale else ''))
        elif param_name == 'max_samples':
            ax.set_xlabel('Max Samples' + (' (log scale)' if use_log_scale else ''))
        elif param_name == 'contamination':
            ax.set_xlabel('Contamination')
        elif param_name == 'max_features':
            ax.set_xlabel('Max Features (proportion)')
        
        ax.set_ylabel(metric_name)
        ax.set_ylim([0, 1.05])
        ax.legend(loc='best', fontsize=18)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_multi_dataset_single_metric(param_name, metric='f1_score', datasets=['shuttle', 'campaign', 'fraud'],
                                     results_base_path='../results', save_path=None):
    """
    Plots a single metric (F1, Precision, or Recall) comparing different datasets.
    
    Parameters:
    - param_name: Name of the parameter ('n_estimators', 'max_samples', 'contamination', 'max_features')
    - metric: Metric to plot ('f1_score', 'precision', 'recall')
    - datasets: List of dataset names to compare
    - results_base_path: Base path to results directory
    - save_path: Path to save the figure
    """
    # Map parameter names to file names and folder names
    param_file_map = {
        'n_estimators': ('n_estimators', 'num_est_results_narrow.csv'),
        'max_samples': ('max_samples', 'num_samples_results.csv'),
        'contamination': ('contamination', 'contamination_results.csv'),
        'max_features': ('max_features', 'features_results.csv')
    }
    
    if param_name not in param_file_map:
        raise ValueError(f"Parameter {param_name} not supported. Choose from {list(param_file_map.keys())}")
    
    folder_name, file_name = param_file_map[param_name]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Define colors for datasets
    colors = {'shuttle': '#1f77b4', 'campaign': '#ff7f0e', 'fraud': '#2ca02c'}
    markers = {'shuttle': 'o', 'campaign': 's', 'fraud': '^'}
    
    # Track if we should use log scale
    use_log_scale = param_name in ['n_estimators', 'max_samples']
    
    for dataset in datasets:
        # Load data
        file_path = os.path.join(results_base_path, dataset, folder_name, file_name)
        
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, skipping {dataset}")
            continue
            
        df = pd.read_csv(file_path)
        
        # Extract param values and metrics
        param_values = df['param_value'].values
        
        # Convert param_values to numeric, handling 'auto' string
        param_values_numeric = []
        for val in param_values:
            if val == 'auto':
                param_values_numeric.append(256)  # Default value for 'auto'
            else:
                param_values_numeric.append(float(val))
        
        metric_mean = df[f'{metric}_mean'].values
        metric_std = df[f'{metric}_std'].values
        
        # Plot metric
        ax.plot(param_values_numeric, metric_mean, marker=markers[dataset], 
               label=dataset.capitalize(), linewidth=2, markersize=8, 
               color=colors[dataset])
        ax.fill_between(param_values_numeric, 
                       np.array(metric_mean) - np.array(metric_std), 
                       np.array(metric_mean) + np.array(metric_std), 
                       alpha=0.2, color=colors[dataset])
    
    # Configure plot
    if use_log_scale:
        ax.set_xscale('log')
    
    # Set labels
    if param_name == 'n_estimators':
        ax.set_xlabel('Number of Estimators' + (' (log scale)' if use_log_scale else ''))
    elif param_name == 'max_samples':
        ax.set_xlabel('Max Samples' + (' (log scale)' if use_log_scale else ''))
    elif param_name == 'contamination':
        ax.set_xlabel('Contamination')
    elif param_name == 'max_features':
        ax.set_xlabel('Max Features (proportion)')
    
    metric_display_names = {
        'f1_score': 'F1 Score',
        'precision': 'Precision',
        'recall': 'Recall'
    }
    ax.set_ylabel(metric_display_names.get(metric, metric))
    ax.set_ylim([0, 1.05])
    ax.legend(loc='best', fontsize=20)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_multi_dataset_training_time(param_name, datasets=['shuttle', 'campaign', 'fraud'],
                                     results_base_path='../results', save_path=None):
    """
    Compares training times across datasets for a given parameter.
    Useful for understanding computational requirements.
    """
    param_file_map = {
        'n_estimators': ('n_estimators', 'num_est_results_narrow.csv'),
        'max_samples': ('max_samples', 'num_samples_results.csv'),
        'contamination': ('contamination', 'contamination_results.csv'),
        'max_features': ('max_features', 'features_results.csv')
    }
    
    if param_name not in param_file_map:
        raise ValueError(f"Parameter {param_name} not supported.")
    
    folder_name, file_name = param_file_map[param_name]
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = {'shuttle': '#1f77b4', 'campaign': '#ff7f0e', 'fraud': '#2ca02c'}
    markers = {'shuttle': 'o', 'campaign': 's', 'fraud': '^'}
    use_log_scale = param_name in ['n_estimators', 'max_samples']
    
    for dataset in datasets:
        file_path = os.path.join(results_base_path, dataset, folder_name, file_name)
        
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, skipping {dataset}")
            continue
            
        df = pd.read_csv(file_path)
        param_values = df['param_value'].values
        
        param_values_numeric = []
        for val in param_values:
            if val == 'auto':
                param_values_numeric.append(256)
            else:
                param_values_numeric.append(float(val))
        
        training_times = df['training_time_mean'].values
        training_std = df['training_time_std'].values
        
        ax.plot(param_values_numeric, training_times, marker=markers[dataset], 
               label=dataset.capitalize(), linewidth=2, markersize=8, 
               color=colors[dataset])
        ax.fill_between(param_values_numeric, 
                       np.array(training_times) - np.array(training_std), 
                       np.array(training_times) + np.array(training_std), 
                       alpha=0.2, color=colors[dataset])
    
    if use_log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')
    
    param_labels = {
        'n_estimators': 'Number of Estimators',
        'max_samples': 'Max Samples',
        'contamination': 'Contamination',
        'max_features': 'Max Features (proportion)'
    }
    ax.set_xlabel(param_labels[param_name] + (' (log scale)' if use_log_scale else ''))
    ax.set_ylabel('Training Time (seconds' + (', log scale' if use_log_scale else '') + ')')
    ax.legend(loc='best', fontsize=20)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_multi_dataset_stability(param_name, datasets=['shuttle', 'campaign', 'fraud'],
                                results_base_path='../results', save_path=None):
    """
    Plots coefficient of variation (std/mean) for F1 scores across datasets.
    Shows which dataset-parameter combinations are most stable.
    """
    param_file_map = {
        'n_estimators': ('n_estimators', 'num_est_results_narrow.csv'),
        'max_samples': ('max_samples', 'num_samples_results.csv'),
        'contamination': ('contamination', 'contamination_results.csv'),
        'max_features': ('max_features', 'features_results.csv')
    }
    
    if param_name not in param_file_map:
        raise ValueError(f"Parameter {param_name} not supported.")
    
    folder_name, file_name = param_file_map[param_name]
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = {'shuttle': '#1f77b4', 'campaign': '#ff7f0e', 'fraud': '#2ca02c'}
    markers = {'shuttle': 'o', 'campaign': 's', 'fraud': '^'}
    use_log_scale = param_name in ['n_estimators', 'max_samples']
    
    for dataset in datasets:
        file_path = os.path.join(results_base_path, dataset, folder_name, file_name)
        
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, skipping {dataset}")
            continue
            
        df = pd.read_csv(file_path)
        param_values = df['param_value'].values
        
        param_values_numeric = []
        for val in param_values:
            if val == 'auto':
                param_values_numeric.append(256)
            else:
                param_values_numeric.append(float(val))
        
        f1_mean = df['f1_score_mean'].values
        f1_std = df['f1_score_std'].values
        cv = np.array(f1_std) / np.array(f1_mean)
        
        ax.plot(param_values_numeric, cv, marker=markers[dataset], 
               label=dataset.capitalize(), linewidth=2, markersize=8, 
               color=colors[dataset])
    
    if use_log_scale:
        ax.set_xscale('log')
    
    param_labels = {
        'n_estimators': 'Number of Estimators',
        'max_samples': 'Max Samples',
        'contamination': 'Contamination',
        'max_features': 'Max Features (proportion)'
    }
    ax.set_xlabel(param_labels[param_name] + (' (log scale)' if use_log_scale else ''))
    ax.set_ylabel('Coefficient of Variation (F1)')
    ax.legend(loc='best', fontsize=20)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_multi_dataset_pr_tradeoff(param_name, datasets=['shuttle', 'campaign', 'fraud'],
                                   results_base_path='../results', save_path=None):
    """
    Plots Precision vs Recall for each dataset as parameter changes.
    Different colors for datasets, trajectory shows parameter progression.
    """
    param_file_map = {
        'n_estimators': ('n_estimators', 'num_est_results_narrow.csv'),
        'max_samples': ('max_samples', 'num_samples_results.csv'),
        'contamination': ('contamination', 'contamination_results.csv'),
        'max_features': ('max_features', 'features_results.csv')
    }
    
    if param_name not in param_file_map:
        raise ValueError(f"Parameter {param_name} not supported.")
    
    folder_name, file_name = param_file_map[param_name]
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = {'shuttle': '#1f77b4', 'campaign': '#ff7f0e', 'fraud': '#2ca02c'}
    markers = {'shuttle': 'o', 'campaign': 's', 'fraud': '^'}
    
    for dataset in datasets:
        file_path = os.path.join(results_base_path, dataset, folder_name, file_name)
        
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, skipping {dataset}")
            continue
            
        df = pd.read_csv(file_path)
        
        precision_mean = df['precision_mean'].values
        precision_std = df['precision_std'].values
        recall_mean = df['recall_mean'].values
        recall_std = df['recall_std'].values
        
        # Plot path from low to high parameter values
        ax.plot(recall_mean, precision_mean, marker=markers[dataset], 
               label=dataset.capitalize(), linewidth=2, markersize=10, 
               color=colors[dataset], alpha=0.7)
        
        # Add error bars
        ax.errorbar(recall_mean, precision_mean, 
                   xerr=recall_std, yerr=precision_std,
                   fmt='none', ecolor=colors[dataset], alpha=0.3, capsize=3)
        
        # Mark start and end points
        ax.scatter(recall_mean[0], precision_mean[0], s=150, 
                  marker=markers[dataset], color=colors[dataset], 
                  edgecolors='black', linewidth=2, alpha=0.5, zorder=5)
        ax.scatter(recall_mean[-1], precision_mean[-1], s=250, 
                  marker=markers[dataset], color=colors[dataset], 
                  edgecolors='red', linewidth=2, zorder=5)
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    
    # Add legend for markers
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markersize=10, markeredgecolor='black', markeredgewidth=2, 
               label='Start (low param)', alpha=0.5),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markersize=13, markeredgecolor='red', markeredgewidth=2, 
               label='End (high param)')
    ]
    
    # Create legend with both dataset names and markers
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + legend_elements, loc='best', fontsize=16)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_multi_dataset_efficiency(param_name, datasets=['shuttle', 'campaign', 'fraud'],
                                  results_base_path='../results', save_path=None):
    """
    Plots F1 score vs training time for each dataset.
    Shows efficiency trade-offs (performance per unit time).
    """
    param_file_map = {
        'n_estimators': ('n_estimators', 'num_est_results_narrow.csv'),
        'max_samples': ('max_samples', 'num_samples_results.csv'),
        'contamination': ('contamination', 'contamination_results.csv'),
        'max_features': ('max_features', 'features_results.csv')
    }
    
    if param_name not in param_file_map:
        raise ValueError(f"Parameter {param_name} not supported.")
    
    folder_name, file_name = param_file_map[param_name]
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = {'shuttle': '#1f77b4', 'campaign': '#ff7f0e', 'fraud': '#2ca02c'}
    markers = {'shuttle': 'o', 'campaign': 's', 'fraud': '^'}
    
    for dataset in datasets:
        file_path = os.path.join(results_base_path, dataset, folder_name, file_name)
        
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, skipping {dataset}")
            continue
            
        df = pd.read_csv(file_path)
        
        training_times = df['training_time_mean'].values
        f1_mean = df['f1_score_mean'].values
        f1_std = df['f1_score_std'].values
        
        # Plot with error bars
        ax.errorbar(training_times, f1_mean, yerr=f1_std,
                   marker=markers[dataset], label=dataset.capitalize(), 
                   linewidth=2, markersize=10, color=colors[dataset],
                   capsize=5, capthick=2)
        
        # Mark start and end points
        ax.scatter(training_times[0], f1_mean[0], s=150, 
                  marker=markers[dataset], color=colors[dataset], 
                  edgecolors='black', linewidth=2, alpha=0.5, zorder=5)
        ax.scatter(training_times[-1], f1_mean[-1], s=250, 
                  marker=markers[dataset], color=colors[dataset], 
                  edgecolors='red', linewidth=2, zorder=5)
    
    ax.set_xlabel('Training Time (seconds)')
    ax.set_ylabel('F1 Score')
    ax.set_ylim([0, 1.05])
    ax.legend(loc='best', fontsize=20)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_parameter_sensitivity_heatmap(datasets=['shuttle', 'campaign', 'fraud'],
                                       results_base_path='../results', save_path=None):
    """
    Heatmap showing best F1 scores for each parameter-dataset combination.
    Helps identify which parameters matter most for each dataset.
    """
    param_file_map = {
        'n_estimators': ('n_estimators', 'num_est_results_narrow.csv'),
        'max_samples': ('max_samples', 'num_samples_results.csv'),
        'contamination': ('contamination', 'contamination_results.csv'),
        'max_features': ('max_features', 'features_results.csv')
    }
    
    # Collect best F1 scores for each parameter-dataset combination
    data = []
    
    for dataset in datasets:
        row_data = {'Dataset': dataset.capitalize()}
        
        for param_name, (folder_name, file_name) in param_file_map.items():
            file_path = os.path.join(results_base_path, dataset, folder_name, file_name)
            
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                best_f1 = df['f1_score_mean'].max()
                row_data[param_name] = best_f1
            else:
                row_data[param_name] = np.nan
        
        data.append(row_data)
    
    df_heatmap = pd.DataFrame(data)
    df_heatmap.set_index('Dataset', inplace=True)
    
    # Rename columns for better display
    df_heatmap.columns = ['n_estimators', 'max_samples', 'contamination', 'max_features']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.heatmap(df_heatmap, annot=True, fmt='.3f', cmap='YlGnBu', 
               cbar_kws={'label': 'Best F1 Score'}, ax=ax,
               vmin=0, vmax=1, linewidths=1, linecolor='white')
    
    ax.set_xlabel('Parameter', fontsize=22)
    ax.set_ylabel('Dataset', fontsize=22)
    ax.set_xticklabels(['n_estimators', 'max_samples', 'contamination', 'max_features'], 
                       rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_multi_dataset_improvement(param_name, datasets=['shuttle', 'campaign', 'fraud'],
                                   results_base_path='../results', save_path=None):
    """
    Plots relative improvement from minimum parameter value.
    Shows diminishing returns for each dataset.
    """
    param_file_map = {
        'n_estimators': ('n_estimators', 'num_est_results_narrow.csv'),
        'max_samples': ('max_samples', 'num_samples_results.csv'),
        'contamination': ('contamination', 'contamination_results.csv'),
        'max_features': ('max_features', 'features_results.csv')
    }
    
    if param_name not in param_file_map:
        raise ValueError(f"Parameter {param_name} not supported.")
    
    folder_name, file_name = param_file_map[param_name]
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = {'shuttle': '#1f77b4', 'campaign': '#ff7f0e', 'fraud': '#2ca02c'}
    markers = {'shuttle': 'o', 'campaign': 's', 'fraud': '^'}
    use_log_scale = param_name in ['n_estimators', 'max_samples']
    
    for dataset in datasets:
        file_path = os.path.join(results_base_path, dataset, folder_name, file_name)
        
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, skipping {dataset}")
            continue
            
        df = pd.read_csv(file_path)
        param_values = df['param_value'].values
        
        param_values_numeric = []
        for val in param_values:
            if val == 'auto':
                param_values_numeric.append(256)
            else:
                param_values_numeric.append(float(val))
        
        f1_mean = df['f1_score_mean'].values
        
        # Calculate relative improvement from baseline (first value)
        baseline = f1_mean[0]
        relative_improvement = ((f1_mean - baseline) / baseline) * 100
        
        ax.plot(param_values_numeric, relative_improvement, marker=markers[dataset], 
               label=dataset.capitalize(), linewidth=2, markersize=8, 
               color=colors[dataset])
    
    if use_log_scale:
        ax.set_xscale('log')
    
    param_labels = {
        'n_estimators': 'Number of Estimators',
        'max_samples': 'Max Samples',
        'contamination': 'Contamination',
        'max_features': 'Max Features (proportion)'
    }
    ax.set_xlabel(param_labels[param_name] + (' (log scale)' if use_log_scale else ''))
    ax.set_ylabel('Relative F1 Improvement (%)')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax.legend(loc='best', fontsize=20)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_statistical_comparison(param_name, datasets=['shuttle', 'campaign', 'fraud'],
                                results_base_path='../results', save_path=None):
    """
    Box plots comparing F1 score distributions across datasets.
    Shows confidence intervals using mean ± std.
    """
    param_file_map = {
        'n_estimators': ('n_estimators', 'num_est_results_narrow.csv'),
        'max_samples': ('max_samples', 'num_samples_results.csv'),
        'contamination': ('contamination', 'contamination_results.csv'),
        'max_features': ('max_features', 'features_results.csv')
    }
    
    if param_name not in param_file_map:
        raise ValueError(f"Parameter {param_name} not supported.")
    
    folder_name, file_name = param_file_map[param_name]
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = {'shuttle': '#1f77b4', 'campaign': '#ff7f0e', 'fraud': '#2ca02c'}
    
    box_data = []
    labels = []
    colors_list = []
    
    for dataset in datasets:
        file_path = os.path.join(results_base_path, dataset, folder_name, file_name)
        
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, skipping {dataset}")
            continue
            
        df = pd.read_csv(file_path)
        f1_mean = df['f1_score_mean'].values
        f1_std = df['f1_score_std'].values
        
        # Create approximate distribution using mean ± std
        simulated_data = []
        for mean, std in zip(f1_mean, f1_std):
            # Create 5 representative points around mean
            simulated_data.extend([
                mean - std,
                mean - 0.5*std,
                mean,
                mean + 0.5*std,
                mean + std
            ])
        
        box_data.append(simulated_data)
        labels.append(dataset.capitalize())
        colors_list.append(colors[dataset])
    
    bp = ax.boxplot(box_data, labels=labels, patch_artist=True,
                    showmeans=True, meanline=True)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    # Color other elements
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color='black', linewidth=1.5)
    
    ax.set_ylabel('F1 Score')
    ax.set_xlabel('Dataset')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')
    
    param_labels = {
        'n_estimators': 'n_estimators',
        'max_samples': 'max_samples',
        'contamination': 'contamination',
        'max_features': 'max_features'
    }
    ax.set_title(f'F1 Score Distribution across Datasets ({param_labels[param_name]})', 
                fontsize=20, pad=20)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_pareto_front_3d(param_name, datasets=['shuttle', 'campaign', 'fraud'],
                         results_base_path='../results', save_path=None):
    """
    3D scatter plot: F1 score vs Training Time vs Coefficient of Variation.
    Identifies Pareto-optimal configurations.
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    param_file_map = {
        'n_estimators': ('n_estimators', 'num_est_results_narrow.csv'),
        'max_samples': ('max_samples', 'num_samples_results.csv'),
        'contamination': ('contamination', 'contamination_results.csv'),
        'max_features': ('max_features', 'features_results.csv')
    }
    
    if param_name not in param_file_map:
        raise ValueError(f"Parameter {param_name} not supported.")
    
    folder_name, file_name = param_file_map[param_name]
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = {'shuttle': '#1f77b4', 'campaign': '#ff7f0e', 'fraud': '#2ca02c'}
    markers = {'shuttle': 'o', 'campaign': 's', 'fraud': '^'}
    
    for dataset in datasets:
        file_path = os.path.join(results_base_path, dataset, folder_name, file_name)
        
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, skipping {dataset}")
            continue
            
        df = pd.read_csv(file_path)
        
        f1_mean = df['f1_score_mean'].values
        f1_std = df['f1_score_std'].values
        training_times = df['training_time_mean'].values
        cv = f1_std / f1_mean
        
        ax.scatter(training_times, cv, f1_mean, 
                  marker=markers[dataset], s=150, 
                  color=colors[dataset], label=dataset.capitalize(),
                  alpha=0.7, edgecolors='black', linewidth=1)
        
        # Connect points with lines
        ax.plot(training_times, cv, f1_mean, 
               color=colors[dataset], alpha=0.3, linewidth=1)
    
    ax.set_xlabel('Training Time (s)', fontsize=18, labelpad=10)
    ax.set_ylabel('Coefficient of Variation', fontsize=18, labelpad=10)
    ax.set_zlabel('F1 Score', fontsize=18, labelpad=10)
    ax.legend(loc='best', fontsize=18)
    ax.grid(True, alpha=0.3)
    
    # Set better viewing angle
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()
