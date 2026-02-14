def text_chunking(tokens, chunk_size, overlap):
    """
    Split tokens into fixed-size chunks with optional overlap.
    
    Args:
        tokens: List of strings (words/tokens)
        chunk_size: Maximum number of tokens per chunk
        overlap: Number of tokens to repeat from the previous chunk
        
    Returns:
        List of lists of tokens
    """
    if not tokens:
        return []

    # The step determines how far forward we jump after each chunk
    # e.g., if size is 10 and overlap is 2, we start the next chunk 8 tokens ahead
    step = chunk_size - overlap
    chunks = []
    
    # Iterate through the list starting from 0, jumping by 'step' each time
    for i in range(0, len(tokens), step):
        # Slice the tokens from current index to chunk_size
        chunk = tokens[i : i + chunk_size]
        chunks.append(chunk)
        
        # Stop once the end of the token list is reached or exceeded
        if i + chunk_size >= len(tokens):
            break
            
    return chunks