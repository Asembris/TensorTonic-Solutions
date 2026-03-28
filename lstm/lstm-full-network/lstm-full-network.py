import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

class LSTM:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim
        scale = np.sqrt(2.0 / (input_dim + hidden_dim))

        self.W_f = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_i = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_c = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_o = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.b_f = np.zeros(hidden_dim)
        self.b_i = np.zeros(hidden_dim)
        self.b_c = np.zeros(hidden_dim)
        self.b_o = np.zeros(hidden_dim)

        self.W_y = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_y = np.zeros(output_dim)

    def forward(self, X: np.ndarray) -> tuple:
        """Forward pass. Returns (y, h_last, C_last)."""
        b,seq_len,_=X.shape
        output_dim=self.W_y.shape[0]
        h_cur,c_cur=np.zeros((b,self.hidden_dim)),np.zeros((b,self.hidden_dim))
        Y=np.zeros((b,seq_len,output_dim))
        for i in range(seq_len):
            conc=np.concatenate((h_cur,X[:,i,:]),axis=-1)
            forget_gate=sigmoid(conc @ self.W_f.T + self.b_f)
            input_gate=sigmoid(conc @ self.W_i.T  + self.b_i)
            output_gate=sigmoid(conc @ self.W_o.T  + self.b_o)
            candidate=conc @ self.W_c.T + self.b_c
            c_new=forget_gate*c_cur + input_gate*candidate
            h_new=output_gate*np.tanh(c_new)
            y_t=h_new @ self.W_y.T + self.b_y
            Y[:,i,:]=y_t
            h_cur=h_new
            c_cur=c_new
        return  (Y,h_cur,c_cur)
        pass