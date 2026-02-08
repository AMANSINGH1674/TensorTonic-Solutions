import numpy as np

def bernoulli_pmf_and_moments(x, p):
    """
    Compute Bernoulli PMF and distribution moments.
    
    Args:
        x: scalar, list, or array of 0s and 1s
        p: success probability (0 <= p <= 1)
    Returns:
        tuple: (pmf_array, mean, variance)
    """
    # Convert input to a numpy array for vectorization
    x_arr = np.array(x)
    
    # 1. Compute PMF: P(X=1) = p, P(X=0) = 1-p
    # Using np.where is efficient for large arrays
    pmf = np.where(x_arr == 1, p, 1 - p)
    
    # 2. Compute Mean (mu = p)
    mean_val = float(p)
    
    # 3. Compute Variance (sigma^2 = p * (1-p))
    var_val = float(p * (1 - p))
    
    return pmf, mean_val, var_val