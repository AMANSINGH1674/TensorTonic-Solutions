# Make Diagonal Matrix
import numpy as np

def make_diagonal(v):
    """
    Construct an n x n diagonal matrix from a 1D vector v.
    """
    # Convert input to a numpy array to handle lists correctly
    v = np.array(v)
    n = len(v)
    
    # Create an n x n matrix of zeros with the same data type as v
    diagonal_matrix = np.zeros((n, n), dtype=v.dtype)
    
    # Fill the diagonal elements
    # Since it's a square matrix, i == j is represented by (i, i)
    for i in range(n):
        diagonal_matrix[i, i] = v[i]
        
    return diagonal_matrix