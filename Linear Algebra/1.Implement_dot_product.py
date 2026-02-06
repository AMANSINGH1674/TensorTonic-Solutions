# Implement Dot Product
import numpy as np

def dot_product(x, y):
    """
    Compute the dot product of two 1D arrays x and y.
    Must return a float.
    """
    # Convert inputs to numpy arrays
    x = np.array(x)
    y = np.array(y)
    
    # Validation: Must be 1D and have matching lengths
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Inputs must be 1D arrays.")
    if x.shape != y.shape:
        raise ValueError("Vectors must have the same length.")

    result = np.sum(x * y)
    
    return float(result)