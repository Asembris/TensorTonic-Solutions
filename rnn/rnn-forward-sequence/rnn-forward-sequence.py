import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray,
                W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> tuple:
    """
    Forward pass through entire sequence.
    """
    b,seq_len,_=X.shape
    hidden_dim=h_0.shape[1]
    h_cur=h_0
    h_all=np.zeros((b,seq_len,hidden_dim))
    for i in range(seq_len):
        z=np.dot(X[:,i,:],W_xh.T)+np.dot(h_cur,W_hh)+b_h
        h_cur=np.tanh(z)
        h_all[:,i,:]=h_cur
    return (h_all,h_cur)
    pass