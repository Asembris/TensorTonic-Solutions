import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    x=np.array(x)
    if rng is None:
        rng=np.random.default_rng()
    mask=(rng.random(x.shape) > p).astype(float)
    mask/=(1-p)
    res=x*mask
    return (res,mask)
    pass