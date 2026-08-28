import math

def he_initialization(W: list, fan_in: int) -> list:
    """
    Returns the weights mapped to the He uniform range.
    """
    # Write code here
    W=np.array(W,dtype=np.float64)
    L=math.sqrt(6/fan_in)
    W*=2*L 
    W-=L 
    return W