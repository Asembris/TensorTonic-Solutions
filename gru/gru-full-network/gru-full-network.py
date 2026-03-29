import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

class GRU:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim
        scale = np.sqrt(2.0 / (input_dim + hidden_dim))

        self.W_r = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_z = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_h = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.b_r = np.zeros(hidden_dim)
        self.b_z = np.zeros(hidden_dim)
        self.b_h = np.zeros(hidden_dim)

        self.W_y = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_y = np.zeros(output_dim)

    def forward(self, X: np.ndarray) -> tuple:
        """Forward pass. Returns (y, h_last)."""
        b,seq_len,input_dim=X.shape
        output_dim=self.W_y.shape[0]
        h_cur=np.zeros((b,self.hidden_dim))
        Y=np.zeros((b,seq_len,output_dim))
        for i in range(seq_len):
            conc=np.concatenate((h_cur,X[:,i,:]),axis=-1)
            reset_gate=sigmoid( conc @ self.W_r.T + self.b_r)
            update_gate=sigmoid( conc @ self.W_z.T + self.b_z)
            h_cand=np.tanh(np.concatenate((reset_gate*h_cur,X[:,i,:]),axis=-1) @ self.W_h.T + self.b_h)
            h_new=update_gate * h_cur+ (1-update_gate)*h_cand
            Y[:,i,:]= h_new @ self.W_y.T + self.b_y
            h_cur=h_new
        return (Y,h_cur)
        pass