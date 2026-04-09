def ordinal_encoding(values, ordering):
    """
    Encode categorical values using the provided ordering.
    """
    order={e:i for i,e in enumerate(ordering)}
    res=[order[e] for e in values]
    return res