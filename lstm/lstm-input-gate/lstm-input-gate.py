import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def input_gate(h_prev: np.ndarray, x_t: np.ndarray,
               W_i: np.ndarray, b_i: np.ndarray,
               W_c: np.ndarray, b_c: np.ndarray) -> tuple:
    """Compute input gate and candidate memory."""
    b,input_dim=x_t.shape
    hidden_dim=h_prev.shape[1]
    conc=np.concatenate((h_prev,x_t),axis=-1)
    it=sigmoid(conc @ W_i.T + b_i)
    cand=np.tanh(conc @ W_c.T + b_c)
    return (it,cand)
    pass