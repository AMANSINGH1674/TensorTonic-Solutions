import numpy as np

def poisson_pmf_cdf(lam, k):
    """
    Compute Poisson PMF and CDF.
    """
    # Edge case: k is 0
    if k == 0:
        pmf = np.exp(-lam)
        return float(pmf), float(pmf)
        
    # Create an array of i values from 0 to k
    i_vals = np.arange(k + 1)
    
    # 1. Compute log(i!) for all i from 0 to k without loops
    # log(i!) = sum(log(1) + log(2) + ... + log(i))
    log_terms = np.log(np.arange(1, k + 1))
    log_fact = np.zeros(k + 1)
    log_fact[1:] = np.cumsum(log_terms) # Vectorized cumulative sum
    
    # 2. Compute log PMF for all i from 0 to k simultaneously
    # log(P(X=i)) = -lam + i * log(lam) - log(i!)
    log_pmf_all = -lam + i_vals * np.log(lam) - log_fact
    
    # Convert back to standard probability space
    pmf_all = np.exp(log_pmf_all)
    
    # 3. Extract the PMF for exactly k (the last element)
    pmf = pmf_all[-1]
    
    # 4. The CDF is simply the sum of all PMF values from 0 to k
    cdf = np.sum(pmf_all)
    
    return float(pmf), float(cdf)