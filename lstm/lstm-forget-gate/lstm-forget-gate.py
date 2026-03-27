import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def forget_gate(h_prev: np.ndarray, x_t: np.ndarray,
                W_f: np.ndarray, b_f: np.ndarray) -> np.ndarray:
    """Compute forget gate: f_t = sigmoid(W_f @ [h, x] + b_f)"""

    b_f=b_f.reshape((1,b_f.shape[0]))
    conc=np.concatenate((h_prev,x_t),axis=-1)
    z=conc @ W_f.T
    inv=z + b_f
    res=sigmoid(inv)
    return res
    pass