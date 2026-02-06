import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Compute mean, median, and mode of a 1D numeric array.
    Returns: (mean, median, mode) as a tuple of floats.
    """
    # Convert input to a numpy array for processing
    data = np.array(x)
    
    # 1. Compute Mean
    mean_val = np.mean(data)
    
    # 2. Compute Median
    median_val = np.median(data)
    
    # 3. Compute Mode
    # Use Counter to get frequencies: {value: count}
    counts = Counter(x)
    max_freq = max(counts.values())
    
    # Find all values that have the maximum frequency
    modes = [val for val, freq in counts.items() if freq == max_freq]
    
    # Requirement: Smallest value if there's a tie
    mode_val = min(modes)
    
    # Ensure all returns are float type
    return float(mean_val), float(median_val), float(mode_val)