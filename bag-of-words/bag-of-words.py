import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    d={}
    for e in tokens:
        d[e]=d.get(e,0)+1
    res=[d.get(e,0) for e in vocab]
    return np.array(res,dtype=int)
    pass