import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def lstm_cell(x_t: np.ndarray, h_prev: np.ndarray, C_prev: np.ndarray,
              W_f: np.ndarray, W_i: np.ndarray, W_c: np.ndarray, W_o: np.ndarray,
              b_f: np.ndarray, b_i: np.ndarray, b_c: np.ndarray, b_o: np.ndarray) -> tuple:
    """Complete LSTM cell forward pass."""
    conc=np.concatenate((h_prev,x_t),axis=-1)
    f=sigmoid(conc @ W_f.T  + b_f)
    i=sigmoid(conc @ W_i.T + b_i)
    o=sigmoid(conc @ W_o.T + b_o)
    cand=np.tanh(conc @ W_c.T + b_c)
    C_t=f*C_prev+i*cand
    h_t=o*np.tanh(C_t)
    return h_t,C_t
    pass