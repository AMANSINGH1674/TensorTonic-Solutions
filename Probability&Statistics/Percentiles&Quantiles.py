# Percentiles & Quantiles
import numpy as np

def percentiles(x, q):
    """
    Compute percentiles using linear interpolation.
    
    Args:
        x: list or array of numeric data
        q: list or array of percentile values (0-100)
    Returns:
        NumPy array of percentile values
    """
    # Convert inputs to numpy arrays for efficient processing
    data = np.array(x)
    percentiles_to_calc = np.array(q)
    
    # Linear interpolation requires the data to be sorted
    # np.percentile handles this internally, but good to keep in mind
    
    # Compute using the 'linear' method as required
    # This calculates the index as (n - 1) * p
    result = np.percentile(data, percentiles_to_calc, method='linear')
    
    return result