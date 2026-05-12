import numpy as np

def expected_calibration_error(y_true, y_pred, n_bins):
    """
    Compute Expected Calibration Error (ECE).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n_samples = len(y_true)
    
    # 1. Determine which bin each prediction falls into.
    # Multiplying by n_bins maps [0, 1] to [0, n_bins]
    bin_indices = np.floor(y_pred * n_bins).astype(int)
    
    # 2. Handle the edge case where y_pred == 1.0. 
    # It would be assigned to index n_bins, which is out of bounds (0 to n_bins-1)
    bin_indices[bin_indices == n_bins] = n_bins - 1
    
    ece = 0.0
    
    # 3. Iterate through each bin to calculate its calibration error
    for m in range(n_bins):
        # Create a boolean mask for the current bin
        mask = bin_indices == m
        bin_size = np.sum(mask)
        
        # Skip empty bins (they don't contribute to ECE)
        if bin_size > 0:
            # Empirical accuracy (fraction of true positives)
            acc = np.mean(y_true[mask])
            
            # Average predicted confidence
            conf = np.mean(y_pred[mask])
            
            # Absolute difference weighted by the proportion of samples in this bin
            ece += (bin_size / n_samples) * np.abs(acc - conf)
            
    return float(ece)