# Remove Stopwords
def remove_stopwords(tokens, stopwords):
    """
    Removes stopwords from a list of tokens while preserving order.
    
    Args:
        tokens: list of strings to be filtered.
        stopwords: list of strings to be removed.
        
    Returns:
        list[str]: A new list containing only the non-stopword tokens.
    """
    # Converting the stopwords list to a set is a crucial optimization.
    # Checking if an item exists in a set takes O(1) time on average,
    # whereas checking a list takes O(n) time.
    stop_set = set(stopwords)
    
    # Use a list comprehension to build the new list, preserving order.
    filtered_tokens = [word for word in tokens if word not in stop_set]
    
    return filtered_tokens