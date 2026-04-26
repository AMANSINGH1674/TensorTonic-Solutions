import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Initialize the output matrix
    pe = np.zeros((seq_len, d_model), dtype=float)
    
    # 1. Create a column vector for positions: shape (seq_len, 1)
    position = np.arange(seq_len)[:, np.newaxis]
    
    # 2. Create a row vector for the frequency divisors: shape (1, ceil(d_model/2))
    # We generate the '2i' values: 0, 2, 4, ...
    two_i = np.arange(0, d_model, 2)
    div_term = 1.0 / (base ** (two_i / d_model))
    
    # 3. Calculate the angles via broadcasting: (seq_len, 1) * (1, ceil(d_model/2))
    # This creates a matrix of shape (seq_len, ceil(d_model/2))
    angles = position * div_term
    
    # 4. Apply sin to all even indices (0, 2, 4, ...)
    pe[:, 0::2] = np.sin(angles)
    
    # 5. Apply cos to all odd indices (1, 3, 5, ...)
    # If d_model is odd, pe[:, 1::2] expects one less column than pe[:, 0::2].
    # We slice angles[:, :d_model // 2] to ensure the shapes match perfectly.
    pe[:, 1::2] = np.cos(angles[:, :d_model // 2])
    
    return pe