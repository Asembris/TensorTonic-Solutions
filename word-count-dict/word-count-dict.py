def word_count_dict(sentences):
    """
    Returns: dict[str, int] - global word frequency across all sentences
    """
    d={}
    for sent in sentences:
        for word in sent:
            d[word]=d.get(word,0)+1
    return d
    pass