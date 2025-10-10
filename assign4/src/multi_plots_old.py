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
    use_log_scale = param_name in [ 'max_samples']
    
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
        raise ValueError(f"Parameter {param_name} not supported. Choose from {list(param_file_map.keys())}")
    
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
