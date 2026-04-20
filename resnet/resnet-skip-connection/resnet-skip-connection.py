import numpy as np

def compute_gradient_with_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Compute gradient flow through L layers WITH skip connections.
    Gradient at layer l = sum of paths through network
    """
    res=np.array(x)
    for grad in gradients_F:
        grad=np.array(grad)
        res=res+np.dot(res,grad)
    return res
        
    pass

def compute_gradient_without_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Compute gradient flow through L layers WITHOUT skip connections.
    """
    res=np.array(x)
    for grad in gradients_F:
        grad=np.array(grad)
        res=np.dot(res,grad)
    return res
    pass
