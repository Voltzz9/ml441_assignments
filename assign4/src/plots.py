import matplotlib.pyplot as plt
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