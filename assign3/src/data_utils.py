
def scale_z_score(data):
    """Scale data using z-score normalization."""
    return (data - data.mean()) / data.std()
