# Matrix Trace
import numpy as np

def matrix_trace(A):
    """
    Compute the trace of a square matrix A manually.
    A is a 2D NumPy array of shape (N, N).
    """
    # Ensure input is a numpy array
    A = np.array(A)
    
    # Validation: Ensure it is square (as per definition)
    rows, cols = A.shape
    if rows != cols:
        raise ValueError("Trace is only defined for square matrices.")
    
    # Initialize sum
    trace_sum = 0
    
    # Iterate through the diagonal elements using a single index i
    # A[i, i] always points to the diagonal element
    for i in range(rows):
        trace_sum += A[i, i]
        
    # Return as a scalar (int or float depending on input)
    return trace_sum