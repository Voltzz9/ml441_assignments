
def scale_z_score(data):
    """Scale data using z-score normalization."""
    return (data - data.mean()) / data.std()

def scale_min_max(data, min_val=-1, max_val=1):
    """Scale data to a specified range [min_val, max_val]."""
    data_min = data.min()
    data_max = data.max()
    return (data - data_min) / (data_max - data_min) * (max_val - min_val) + min_val
