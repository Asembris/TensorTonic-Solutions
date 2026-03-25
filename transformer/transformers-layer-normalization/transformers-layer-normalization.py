import numpy as np

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.

    Args:
        x: Input array of shape (..., d_model)
        gamma: Scale parameter of shape (d_model,)
        beta: Shift parameter of shape (d_model,)
        eps: Small constant for numerical stability

    Returns:
        Normalized array of same shape as x
    """
    d_model=x.shape[-1]
    mean=np.sum(x,axis=-1,keepdims=True)/d_model
    var=np.sum((x-mean)**2,axis=-1,keepdims=True)/d_model
    x_n=(x-mean)/np.sqrt(var+eps)
    y_n=gamma*x_n+beta
    return y_n
    pass