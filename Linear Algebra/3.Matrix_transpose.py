# Matrix Transpose
import numpy as np

def matrix_transpose(A):
    """
    Transpose a matrix A manually.
    A is a 2D NumPy array of shape (N, M).
    """
    # Convert input to a numpy array
    A = np.array(A)
    N, M = A.shape
    
    # Create a new empty matrix of shape (M, N)
    # This ensures we handle rectangular matrices correctly
    transposed = np.zeros((M, N), dtype=A.dtype)
    
    # Manual indexing to swap (i, j) to (j, i)
    for i in range(N):
        for j in range(M):
            transposed[j, i] = A[i, j]
            
    return transposed