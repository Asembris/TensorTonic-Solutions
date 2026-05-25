import numpy as np

def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """
    y_train=np.array(y_train)
    X_test=np.array(X_test)
    labels,freq=np.unique(y_train,return_counts=True)
    index=np.argmax(freq)
    val=labels[index]
    res=np.full(X_test.shape,val)
    return res
    pass