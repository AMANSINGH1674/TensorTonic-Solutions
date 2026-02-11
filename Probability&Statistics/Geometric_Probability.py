# Geometric Probability MF and mean
import numpy as np

def geometric_pmf_mean(k, p):
    """
    Compute Geometric PMF and Mean.
    
    Args:
        k: list or array of trial numbers (k >= 1)
        p: float success probability (0 < p <= 1)
    Returns:
        tuple: (pmf_array, mean_val)
    """
    # Convert k to a numpy array for vectorized calculation
    k_arr = np.array(k)
    
    # 1. Compute PMF: P(X=k) = (1-p)^(k-1) * p
    # This represents (k-1) failures followed by 1 success
    pmf = ((1 - p) ** (k_arr - 1)) * p
    
    # 2. Compute Theoretical Mean: E[X] = 1/p
    mean_val = 1.0 / p
    
    return pmf, float(mean_val)
