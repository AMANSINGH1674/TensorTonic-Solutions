import numpy as np

def bootstrap_mean(x, n_bootstrap=1000, ci=0.95, rng=None):
    """
    Estimate the mean of a 1D dataset using bootstrap resampling.
    Returns: (boot_means, lower, upper)
    """
    # 1. Input Validation: Convert to a NumPy array of floats
    x = np.asarray(x, dtype=float)
    n = len(x)
    
    # 2. Handle random number generator initialization
    if rng is None:
        rng = np.random.default_rng()
        
    # 3. Generate Bootstrap Samples
    # We use a loop instead of a massive vectorized 2D matrix to prevent MemoryErrors 
    # when n and n_bootstrap are both large (e.g., 10,000 x 10,000 could consume ~800MB)
    boot_means = np.empty(n_bootstrap, dtype=float)
    
    for i in range(n_bootstrap):
        # Sample 'n' items with replacement by generating random indices
        indices = rng.integers(0, n, size=n)
        # Calculate and store the mean of this specific resample
        boot_means[i] = np.mean(x[indices])
        
    # 4. Calculate Confidence Interval
    # For a 95% CI (ci=0.95), alpha is 0.05. 
    # We want to trim the lowest 2.5% and the highest 2.5% of our bootstrap means.
    alpha = 1.0 - ci
    lower = np.quantile(boot_means, alpha / 2.0)
    upper = np.quantile(boot_means, 1.0 - (alpha / 2.0))
    
    return boot_means, float(lower), float(upper)