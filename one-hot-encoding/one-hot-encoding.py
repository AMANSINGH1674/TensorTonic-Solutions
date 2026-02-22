import numpy as np

def one_hot(y, num_classes=None):
    """
    Convert integer labels into a one-hot encoded matrix.
    
    Args:
        y: array-like of shape (N,) containing non-negative integer labels.
        num_classes: The number of unique classes. If None, detected from labels.
        
    Returns:
        np.ndarray of shape (N, num_classes) with dtype float.
    """
    # 1. Convert input to a numpy array for processing
    y_arr = np.asanyarray(y, dtype=int)
    num_samples = y_arr.shape[0]
    
    # 2. Determine the number of classes (K)
    if num_classes is None:
        num_classes = int(np.max(y_arr) + 1) if num_samples > 0 else 0
        
    # 3. Validation: Ensure all labels are within the valid range [0, num_classes - 1]
    if np.any(y_arr >= num_classes) or np.any(y_arr < 0):
        raise ValueError("All labels must be non-negative and less than num_classes.")
        
    # 4. Initialize a matrix of zeros
    one_hot_matrix = np.zeros((num_samples, num_classes), dtype=float)
    
    # 5. Vectorized assignment using advanced indexing
    # np.arange(num_samples) creates row indices [0, 1, ..., N-1]
    # y_arr provides the column indices for the 1s
    one_hot_matrix[np.arange(num_samples), y_arr] = 1.0
    
    return one_hot_matrix