import numpy as np

def linear_regression_closed_form(X, y):
    """
    Compute the optimal weight vector using the normal equation.
    """
    # Convert inputs to numpy arrays to ensure we can use matrix operations
    X_np = np.array(X, dtype=float)
    y_np = np.array(y, dtype=float)
    
    # Compute the transpose of X
    X_T = X_np.T
    
    # Compute the product (X^T @ X)
    X_T_X = X_T @ X_np
    
    # Compute the inverse of (X^T @ X)
    X_T_X_inv = np.linalg.inv(X_T_X)
    
    # Compute the product (X^T @ y)
    X_T_y = X_T @ y_np
    
    # Compute the final weight vector w
    w = X_T_X_inv @ X_T_y
    
    # Return as a standard Python list (or it can remain a numpy array)
    return w.tolist()