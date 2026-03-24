import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    n=X.shape[0]
    w=np.zeros((X.shape[1],))
    b=0.0
    for i in range(steps):
        z=np.dot(X,w)+b
        a=_sigmoid(z)
        L=(-1.0/n)*np.sum(y*np.log(a)+(1-y)*np.log(1-a))
        w-=lr*np.dot(X.T,a-y)/n
        b-=lr*np.sum(a-y)/n
    return (w,b)
    pass