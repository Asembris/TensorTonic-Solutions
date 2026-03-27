def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    rel_in_rec=len([e for e in recommended[:k] if e in relevant])
    precision=rel_in_rec/k
    recall=rel_in_rec/len(relevant)
    return [precision,recall]