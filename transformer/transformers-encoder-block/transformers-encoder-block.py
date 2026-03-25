import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    d_model=x.shape[-1]
    mean=np.sum(x,axis=-1,keepdims=True)/d_model
    var=np.var(x,axis=-1,keepdims=True)
    xn=(x-mean)/np.sqrt(var+eps)
    yn=gamma*xn+beta
    return yn
    pass

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    res=np.zeros_like(Q)
    d_model=Q.shape[-1]
    dk=d_model//num_heads
    Q_p=np.split(np.matmul(Q,W_q),num_heads,axis=-1)
    K_p=np.split(np.matmul(K,W_k),num_heads,axis=-1)
    V_p=np.split(np.matmul(V,W_v),num_heads,axis=-1)
    for i in range(num_heads):
        att_scores=np.matmul(Q_p[i],K_p[i].transpose(0,2,1))/np.sqrt(dk)
        att_weights=softmax(att_scores,axis=-1)
        att_i=np.matmul(att_weights,V_p[i])
        res[:,:,i*dk:(i+1)*dk]=att_i
    output=np.matmul(res,W_o)
    return output
    pass

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    z1=np.matmul(x,W1)+b1
    a1=np.maximum(z1,0.0)
    z2=np.matmul(a1,W2)+b2
    return z2
    pass

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    att=multi_head_attention(x,x,x,W_q,W_k,W_v,W_o,num_heads)
    x1=layer_norm(att+x,gamma1,beta1)
    x2=feed_forward(x1,W1,b1,W2,b2)
    res=layer_norm(x2+x1,gamma2,beta2)
    return res
    pass