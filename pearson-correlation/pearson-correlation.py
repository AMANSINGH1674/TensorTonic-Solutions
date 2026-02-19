import numpy as np

def pearson_correlation(X):
    # Convert to numpy array and validate input
    X = np.array(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2:
        return None
    
    n, d = X.shape
    
    # 1. Center the data (subtract mean of each column)
    mean_X = np.mean(X, axis=0)
    X_centered = X - mean_X
    
    # 2. Compute Covariance Matrix 
    # Cov = (X_centered.T @ X_centered) / (n - 1)
    covariance_matrix = (X_centered.T @ X_centered) / (n - 1)
    
    # 3. Compute Standard Deviations
    # ddof=1 to match the (n-1) normalization used in covariance
    std_devs = np.std(X, axis=0, ddof=1)
    
    # 4. Create the denominator matrix (σi * σj)
    # Using np.outer(std, std) creates a matrix where entry (i, j) is σi * σj
    denominator = np.outer(std_devs, std_devs)
    
    # 5. Compute Correlation Matrix: R = Cov / (σi * σj)
    # Use np.divide to handle division and avoid warnings with zero variance
    with np.errstate(divide='ignore', invalid='ignore'):
        correlation_matrix = covariance_matrix / denominator
        
    # Ensure diagonals are 1.0 (unless variance is 0, which results in NaN)
    # The prompt mentions keeping diagonals as 1.0 if possible
    for i in range(d):
        if std_devs[i] > 0:
            correlation_matrix[i, i] = 1.0
            
    return correlation_matrix