import numpy as np

def conv_block(x, W1, W2, Ws):
    """
    Returns: np.ndarray with sum of main path output and projected shortcut
    """
    x=np.array(x)
    W1=np.array(W1)
    W2=np.array(W2)
    Ws=np.array(Ws)
    shortcut= x @ Ws
    h= np.maximum(x @ W1,0.0)
    z= h @ W2
    res=z+shortcut
    res=np.maximum(res,0.0)
    return res
    pass
