def frequency_encoding(values):
    """
    Replace each value with its frequency proportion.
    """
    n=len(values)
    d={}
    for e in values:
        d[e]=d.get(e,0)+1
    res=[d[e]/n for e in values]
    return res