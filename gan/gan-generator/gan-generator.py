import numpy as np

def generator(z: np.ndarray, output_dim: int) -> np.ndarray:
    """
    Generate fake data from noise vectors.
    """
    W1=np.random.randn(256, z.shape[1])
    b1=np.random.randn(256, 1)
    W2=np.random.randn(512, 256)
    b2=np.random.randn(512, 1)
    W3=np.random.randn(output_dim, 512)
    b3=np.random.randn(output_dim, 1)
    W=[W1,W2,W3]
    b=[b1,b2,b3]
    x=z.T
    for i in range(2):
        x=np.dot(W[i],x)+b[i]
        x=np.maximum(x,0.0)
    x=np.dot(W3,x)+b3
    x=np.tanh(x)
    return x.T
    
    pass