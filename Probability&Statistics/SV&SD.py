# Sample Variance and Standard Deviation
import numpy as np

def sample_var_std(x):
    """
    Compute unbiased sample variance and standard deviation.
    
    Args:
        x: list or array of numeric data (n >= 2)
    Returns:
        tuple: (variance, standard_deviation) as floats
    """
    # Convert to numpy array for vectorized operations
    data = np.array(x)
    n = len(data)
    
    # 1. Compute the sample mean
    mean_x = np.mean(data)
    
    # 2. Compute Sample Variance (s^2) using Bessel's Correction (n - 1)
    # Formula: sum((xi - mean)^2) / (n - 1)
    # In NumPy, this is equivalent to np.var(data, ddof=1)
    variance = np.sum((data - mean_x) ** 2) / (n - 1)
    
    # 3. Compute Standard Deviation (s)
    std_dev = np.sqrt(variance)
    
    return float(variance), float(std_dev)