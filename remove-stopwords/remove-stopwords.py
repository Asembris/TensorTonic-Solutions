def remove_stopwords(tokens, stopwords):
    """
    Returns: list[str] - tokens with stopwords removed (preserve order)
    """
    res=[e for e in tokens if e not in stopwords]
    return res
    pass