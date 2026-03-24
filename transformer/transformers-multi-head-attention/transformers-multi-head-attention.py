import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    d=K.shape[2]
    dk=d//num_heads
    res=np.zeros_like(Q)
    Q_p=np.split(np.matmul(Q,W_q),num_heads,axis=-1)
    K_p=np.split(np.matmul(K,W_k),num_heads,axis=-1)
    V_p=np.split(np.matmul(V,W_v),num_heads,axis=-1)
    for i in range(num_heads):
        att_score=np.matmul(Q_p[i],K_p[i].transpose(0,2,1))/np.sqrt(dk)
        att_weight=softmax(att_score)
        atti=np.matmul(att_weight,V_p[i])
        res[:,:,i*dk:(i+1)*dk]=atti
    out=np.matmul(res,W_o)
    return out
    pass