import numpy as np

def discriminator(x: np.ndarray) -> np.ndarray:
    """
    Classify inputs as real or fake.
    """
    W=np.random.randn(x.shape[1],1)
    b=np.random.randn(1,1)
    z=np.dot(x,W)+b
    a=1/(1+np.exp(-z))
    return a
    pass