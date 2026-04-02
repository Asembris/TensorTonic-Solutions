import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """
    y_pred=np.asarray(y_pred)
    y_true=np.asarray(y_true)
    n=y_pred.shape[0]
    mse=np.sum((y_pred-y_true)**2,axis=-1)/n
    return mse
    pass
