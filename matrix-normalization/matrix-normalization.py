import numpy as np

def matrix_normalization(matrix: list, axis=None, norm_type: str = "l2") -> np.ndarray:
    """
    Returns a NumPy array with the same shape as matrix.
    """
    # Write code here
    matrix=np.array(matrix,dtype=float)
    if norm_type=="l1":
        su=np.sum(np.abs(matrix),axis=axis,keepdims=True)
        su[su==0]=1
        res=matrix/su
        return res
    if norm_type=="l2":
        su=np.sqrt(np.sum(matrix*matrix,axis=axis,keepdims=True))
        su[su==0]=1
        res=matrix/su 
        return res 
    else:
        mx=np.max(np.abs(matrix),axis=axis,keepdims=True)
        mx[mx==0]=1
        res=matrix/mx
        return res
        