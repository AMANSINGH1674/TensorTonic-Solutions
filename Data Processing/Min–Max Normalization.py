# Min–Max Normalization
import numpy as np

def minmax_scale(X, axis=0, eps=1e-12):
    """
    Scale X to [0,1]. If 2D and axis=0 (default), scale per column.
    
    Args:
        X: list or array (1D or 2D)
        axis: The axis along which to calculate min/max (0 for columns)
        eps: Small value to prevent division by zero
    Returns: 
        np.ndarray: Scaled data as floats
    """
    # Convert input to float array for calculation
    X_arr = np.asanyarray(X, dtype=float)
    
    # Calculate min and max along the specified axis
    # keepdims=True is critical for broadcasting subtraction/division
    x_min = np.min(X_arr, axis=axis, keepdims=True)
    x_max = np.max(X_arr, axis=axis, keepdims=True)
    
    # Calculate the range (denominator)
    diff = x_max - x_min
    
    # Prevent division by zero if all values in a column are the same
    # This effectively makes the result 0 for constant columns
    denominator = np.maximum(diff, eps)
    
    # Apply the formula: (x - min) / (max - min)
    X_scaled = (X_arr - x_min) / denominator
    
    return X_scaled
