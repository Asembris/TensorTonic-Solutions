def top_k_recommendations(scores, rated_indices, k):
    """
    Return indices of top-k unrated items by predicted score.
    """
    not_seen=[i for i,e in enumerate(scores) if i not in rated_indices]
    not_seen=sorted(not_seen,key=lambda x:-scores[x])
    not_seen=not_seen[:k]
    return not_seen