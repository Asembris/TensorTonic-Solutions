import numpy as np

def batch_norm_forward(x: list, gamma: list, beta: list, eps: float = 1e-5) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as x.
    """
    x=np.array(x)
    gamma=np.array(gamma)
    beta=np.array(beta)
    axis = 0 if x.ndim == 2 else (0, 2, 3)
    mean=np.mean(x,axis=axis,keepdims=True)
    var=np.var(x,axis=axis,keepdims=True)
    x_hat=(x-mean)/np.sqrt(var+eps)
    if x.ndim == 4:
        gamma = gamma.reshape(1, -1, 1, 1)
        beta = beta.reshape(1, -1, 1, 1)
    y=gamma*x_hat+beta
    return y