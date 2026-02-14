# Binning
import numpy as np

def binning(values, num_bins):
    """
    Assign each value to an equal-width bin.
    
    Args:
        values: List or array of numeric values
        num_bins: Number of intervals to create
    Returns:
        List of integers representing bin indices
    """
    if not values:
        return []
        
    x = np.array(values)
    min_val = np.min(x)
    max_val = np.max(x)
    
    # Handle edge case: all values are identical
    if min_val == max_val:
        return [0] * len(values)
    
    # Calculate width of each bin
    # width = (max - min) / num_bins
    width = (max_val - min_val) / num_bins
    
    # Calculate bin index: floor((x - min) / width)
    # Using np.floor to handle the vectorization
    bin_indices = np.floor((x - min_val) / width).astype(int)
    
    # Requirement: Clamp the maximum value to the last bin (num_bins - 1)
    # This is necessary because (max_val - min_val) / width equals exactly num_bins
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)
    
    return bin_indices.tolist()