import numpy as np
from collections import Counter
from scipy.special import comb

# --- 1. Central Tendency ---
def mean_median_mode(x):
    data = np.array(x)
    mean_val = np.mean(data)
    median_val = np.median(data)
    
    counts = Counter(x)
    max_freq = max(counts.values())
    modes = [val for val, freq in counts.items() if freq == max_freq]
    mode_val = min(modes)
    
    return float(mean_val), float(median_val), float(mode_val)

# --- 2. Percentiles ---
def percentiles(x, q):
    data = np.array(x)
    percentiles_to_calc = np.array(q)
    # Uses linear interpolation by default: (n-1) * p
    return np.percentile(data, percentiles_to_calc, method='linear')

# --- 3. Bernoulli Distribution ---
def bernoulli_pmf_and_moments(x, p):
    x_arr = np.array(x)
    pmf = np.where(x_arr == 1, p, 1 - p)
    mean_val = float(p)
    var_val = float(p * (1 - p))
    return pmf, mean_val, var_val

# --- 4. Binomial Distribution ---
def binomial_pmf_cdf(n, p, k):
    # PMF: (n choose k) * p^k * (1-p)^(n-k)
    pmf_val = comb(n, k) * (p**k) * ((1 - p)**(n - k))
    # CDF: Sum of PMFs from 0 to k
    cdf_val = sum(comb(n, i) * (p**i) * ((1 - p)**(n - i)) for i in range(k + 1))
    return float(pmf_val), float(cdf_val)

# --- 5. Expected Value ---
def expected_value_discrete(x, p):
    x_arr = np.array(x)
    p_arr = np.array(p)
    if x_arr.shape != p_arr.shape:
        raise ValueError("Shapes must match.")
    if not np.allclose(np.sum(p_arr), 1.0, atol=1e-6):
        raise ValueError("Probabilities must sum to 1.")
    return float(np.sum(x_arr * p_arr))

# --- 6. Geometric Distribution ---
def geometric_pmf_mean(k, p):
    k_arr = np.array(k)
    pmf = ((1 - p) ** (k_arr - 1)) * p
    mean_val = 1.0 / p
    return pmf, float(mean_val)

# --- 7. Sample Variance & Std Dev ---
def sample_var_std(x):
    data = np.array(x)
    n = len(data)
    mean_x = np.mean(data)
    # Using n-1 for Bessel's Correction
    variance = np.sum((data - mean_x) ** 2) / (n - 1)
    std_dev = np.sqrt(variance)
    return float(variance), float(std_dev)

# --- 8. Min-Max Normalization ---
def minmax_scale(X, axis=0, eps=1e-12):
    X_arr = np.asanyarray(X, dtype=float)
    x_min = np.min(X_arr, axis=axis, keepdims=True)
    x_max = np.max(X_arr, axis=axis, keepdims=True)
    # Use eps to prevent division by zero in constant columns
    denominator = np.maximum(x_max - x_min, eps)
    return (X_arr - x_min) / denominator