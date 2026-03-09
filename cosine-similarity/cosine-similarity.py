import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    up=np.dot(a,b)
    do=np.linalg.norm(a)*np.linalg.norm(b)
    return up/do if do!=0.0 else 0
    pass