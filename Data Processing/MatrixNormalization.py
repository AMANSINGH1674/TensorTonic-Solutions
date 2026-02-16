# Matrix Normalization
import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    """
    Normalize a 2D matrix along specified axis using L1, L2, or Max norm.
    """
    try:
        # 1. Input Validation and Conversion
        X = np.asanyarray(matrix, dtype=float)
        if X.ndim != 2:
            return None
        
        # 2. Calculate the Norm based on type
        if norm_type == 'l1':
            # Sum of absolute values
            norm = np.sum(np.abs(X), axis=axis, keepdims=True)
        elif norm_type == 'l2':
            # Square root of sum of squares
            norm = np.sqrt(np.sum(np.square(X), axis=axis, keepdims=True))
        elif norm_type == 'max':
            # Maximum absolute value
            norm = np.max(np.abs(X), axis=axis, keepdims=True)
        else:
            return None
            
        # 3. Normalize and Handle Zero Vectors
        # We use np.where or a simple addition of a tiny epsilon to avoid 0/0
        eps = 1e-12
        normalized_matrix = X / np.maximum(norm, eps)
        
        return normalized_matrix

    except (ValueError, TypeError):
        return None