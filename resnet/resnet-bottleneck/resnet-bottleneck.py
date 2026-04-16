import numpy as np

def bottleneck_block(x, W1, W2, W3, Ws):
    """
    Returns: np.ndarray with bottleneck residual block output (compress, process, expand + skip)
    """
    x=np.array(x)
    W1=np.array(W1)
    W2=np.array(W2)
    W3=np.array(W3)
    Ws=np.array(Ws)
    shortcut= x @ Ws

    x1=np.maximum(x @ W1,0.0)
    x2=np.maximum(x1 @ W2,0.0)
    x3=x2 @ W3
    res=np.maximum(x3+shortcut,0.0)
    return res
    pass
