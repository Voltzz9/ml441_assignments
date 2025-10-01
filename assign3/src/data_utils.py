import numpy as np
import pandas as pd
def scale_z_score(data):
    """Scale data using z-score normalization."""
    return (data - data.mean()) / data.std()

def scale_min_max(data, min_val=-1, max_val=1):
    """Scale data to a specified range [min_val, max_val]."""
    data_min = data.min()
    data_max = data.max()
    return (data - data_min) / (data_max - data_min) * (max_val - min_val) + min_val


# Function to generate synthetic data for regression
# Sine + Gaussian noise
def generate_synthetic_data(n_samples=500, noise_std=0.1, random_state=12):
    """Generate synthetic data for regression: y = sin(2 * pi * x) + noise."""
    np.random.seed(random_state)
    x = np.random.uniform(0, 1, n_samples)
    noise = np.random.normal(0, noise_std, n_samples)
    y = np.sin(2 * np.pi * x) + noise
    
    
    # save in csv in ../datasets/synfunc.csv
    
    df = pd.DataFrame({'x': x, 'y': y})
    df.to_csv('../datasets/synfunc.csv', index=False)