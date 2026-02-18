import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Pad or truncate a list of sequences to a uniform length.
    
    Args:
        seqs: List of lists of integers.
        pad_value: The value to use for padding.
        max_len: The target length. If None, uses the longest sequence length.
        
    Returns:
        np.ndarray of shape (N, L) and dtype int.
    """
    # Handle empty input
    if not seqs:
        return np.empty((0, 0), dtype=int)
    
    # Determine the target length (L)
    if max_len is None:
        max_len = max(len(s) for s in seqs) if seqs else 0
        
    num_sequences = len(seqs)
    
    # 1. Initialize result array filled with pad_value
    # Using np.full ensures every cell starts as the padding value
    padded_matrix = np.full((num_sequences, max_len), pad_value, dtype=int)
    
    # 2. Fill the matrix with sequence data
    for i, seq in enumerate(seqs):
        if not seq:
            continue
            
        # Truncate sequence if it exceeds max_len
        # Using min(len(seq), max_len) handles both truncation and short sequences
        length_to_copy = min(len(seq), max_len)
        
        # Copy the slice of the sequence into the matrix
        padded_matrix[i, :length_to_copy] = seq[:length_to_copy]
        
    return padded_matrix