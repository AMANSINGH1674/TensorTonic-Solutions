import numpy as np

def confusion_matrix_norm(y_true, y_pred, num_classes=None, normalize='none'):
    """
    Compute K x K confusion matrix with optional normalization.
    """
    # 1. Convert inputs to NumPy arrays
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    
    # 2. Infer number of classes if not provided
    if num_classes is None:
        if len(y_true) == 0:
            num_classes = 0
        else:
            num_classes = max(np.max(y_true), np.max(y_pred)) + 1
            
    # Handle the edge case of empty arrays
    if len(y_true) == 0:
        return np.zeros((num_classes, num_classes), dtype=float if normalize != 'none' else int)
        
    # 3. Compute the raw confusion matrix using the 1D bincount trick
    # Index formula: i * K + j
    indices = y_true * num_classes + y_pred
    
    # Compute counts and reshape back to 2D (K x K)
    cm = np.bincount(indices, minlength=num_classes**2)
    cm = cm.reshape(num_classes, num_classes)
    
    # 4. Handle Normalization
    if normalize == 'none':
        return cm
        
    cm = cm.astype(float)
    
    if normalize == 'true':
        # Row-wise normalization: How many of the TRUE class X were predicted as Y?
        divisor = cm.sum(axis=1, keepdims=True)
    elif normalize == 'pred':
        # Column-wise normalization: Of all predicted as X, how many were TRULY Y?
        divisor = cm.sum(axis=0, keepdims=True)
    elif normalize == 'all':
        # Total normalization: What fraction of all samples fall into this cell?
        divisor = cm.sum()
    else:
        raise ValueError("Invalid normalization mode. Choose 'none', 'true', 'pred', or 'all'.")
        
    # Handle division by zero: Replace 0s in the divisor with 1s. 
    # Since the numerator will also be 0 in these cases, 0 / 1 = 0, which is exactly what we want.
    divisor = np.where(divisor == 0, 1.0, divisor)
    
    return cm / divisor