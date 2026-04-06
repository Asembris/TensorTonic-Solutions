import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    x=np.asarray(x)
    res=np.maximum(x,0.0)
    return res
    pass