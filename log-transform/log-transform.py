def log_transform(values):
    """
    Apply the log1p transformation to each value.
    """
    values=np.array(values)
    return np.log1p(values)