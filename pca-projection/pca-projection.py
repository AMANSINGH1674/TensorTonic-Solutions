import numpy as np

def pca_projection(X, k):
    """
    Project data onto the top-k principal components.
    """
    # Convert input to a numpy array for vectorized operations
    X = np.array(X, dtype=float)
    n, d = X.shape
    
    # 1. Center the data by subtracting the mean of each feature
    X_mean = np.mean(X, axis=0)
    X_c = X - X_mean
    
    # 2. Compute the covariance matrix using sample covariance (n - 1)
    C = (X_c.T @ X_c) / (n - 1)
    
    # 3. Find the eigenvalues and eigenvectors
    # eigh is optimized for Hermitian/Symmetric matrices and returns real eigenvalues
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    
    # Sort the eigenvalues and their corresponding eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    # Extract the top-k principal components (eigenvectors)
    W = eigenvectors[:, :k]
    
    # Ensure deterministic eigenvector signs (matches standard libraries like Scikit-Learn)
    # This prevents arbitrary sign flips in the final projection
    for i in range(k):
        max_abs_idx = np.argmax(np.abs(W[:, i]))
        if W[max_abs_idx, i] < 0:
            W[:, i] = -W[:, i]
            
    # 4. Project the centered data onto the top-k components
    X_proj = X_c @ W
    
    # Return as an n x k list of floats
    return X_proj.tolist()