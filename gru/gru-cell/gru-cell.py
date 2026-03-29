import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def gru_cell(x_t: np.ndarray, h_prev: np.ndarray,
             W_r: np.ndarray, W_z: np.ndarray, W_h: np.ndarray,
             b_r: np.ndarray, b_z: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    """Complete GRU cell forward pass."""
    conc=np.concatenate((h_prev,x_t),axis=-1)
    reset_gate=sigmoid( conc @ W_r.T + b_r)
    update_gate=sigmoid( conc @ W_z.T + b_z)
    h_candidate=np.tanh( np.concatenate((reset_gate*h_prev,x_t),axis=-1) @ W_h.T + b_h)
    h_t=update_gate*h_prev + (1-update_gate)*h_candidate
    return h_t
    pass