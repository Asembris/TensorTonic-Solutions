import numpy as np

def q_learning_update(Q, s, a, r, s_next, alpha, gamma):
    """
    Returns: updated Q-table Q_new
    """
    old_value=Q[s][a]
    best_next=max(Q[s_next])
    Q[s][a]=(1-alpha)*old_value+alpha*(r+gamma*best_next)
    return Q
    pass