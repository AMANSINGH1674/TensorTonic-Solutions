import numpy as np
from scipy.special import comb

def binomial_pmf_cdf(n, p, k):
    """
    Compute Binomial PMF and CDF.
    
    Args:
        n: int - Number of trials
        p: float - Success probability (0 <= p <= 1)
        k: int - Number of successes (0 <= k <= n)
    Returns:
        tuple: (pmf, cdf) as scalar floats
    """
    # 1. Compute PMF using the formula: P(X=k) = (n choose k) * p^k * (1-p)^(n-k)
    # scipy.special.comb is used for numerical stability
    pmf_val = comb(n, k) * (p**k) * ((1 - p)**(n - k))
    
    # 2. Compute CDF: P(X <= k) = sum of PMF from i=0 to k
    # We iterate from 0 up to k (inclusive)
    cdf_val = sum(comb(n, i) * (p**i) * ((1 - p)**(n - i)) for i in range(k + 1))
    
    return float(pmf_val), float(cdf_val)