import numpy as np

def identity_block(x, W1, W2):
    """
    Returns: np.ndarray of shape (batch, channels) with identity residual block output
    """
    x=np.array(x)
    W1=np.array(W1)
    W2=np.array(W2)
    x2= x @ W1.T
    x2=np.maximum(x2,0.0)
    x2=x2 @ W2.T + x
    x2=np.maximum(x2,0.0)
    return x2
    pass
