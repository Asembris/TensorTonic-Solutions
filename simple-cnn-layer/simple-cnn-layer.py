import numpy as np

def conv2d(x, W, b):
    """
    Simple 2D convolution layer forward pass.
    Valid padding, stride=1.
    """
    n,c_in,h,w=x.shape
    c_out,c_in_,kh,kw=W.shape
    h_out=h-kh+1
    w_out=w-kw+1
    y=np.zeros((n,c_out,h_out,w_out), dtype=float)
    for i in range(n):
        for j in range(c_out):
            for k in range(c_in):
                for u in range(kh):
                    for v in range(kw):
                        y[i,j]+=x[i,k,u:u+h_out,v:v+w_out]*W[j,k,u,v]
            y[i,j]+=b[j]
    return y                        
    pass