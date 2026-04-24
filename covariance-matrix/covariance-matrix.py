import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    # Convert input to a numpy array
    X_np = np.asarray(X, dtype=float)
    
    # Check for invalid input: not 2D or less than 2 samples
    if X_np.ndim != 2 or X_np.shape[0] < 2:
        return None
        
    N, D = X_np.shape
    
    # Step 1: Center the data by subtracting the mean of each feature
    mu = np.mean(X_np, axis=0)
    X_centered = X_np - mu
    
    # Step 2: Compute the sample covariance matrix using matrix multiplication
    # We use (N - 1) for sample covariance
    Sigma = (X_centered.T @ X_centered) / (N - 1)
    
    return Sigma