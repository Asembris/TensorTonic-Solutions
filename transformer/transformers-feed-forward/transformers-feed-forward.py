import numpy as np

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Apply position-wise feed-forward network.
    """
    z1=np.matmul(x,W1)+b1
    a1=np.maximum(z1,0.0)
    z2=np.matmul(a1,W2)+b2
    return z2
    pass