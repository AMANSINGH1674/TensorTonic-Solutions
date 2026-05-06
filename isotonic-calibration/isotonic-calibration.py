import numpy as np

def calibrate_isotonic(cal_labels, cal_probs, new_probs):
    """
    Apply isotonic regression calibration using PAVA and linear interpolation.
    """
    # 1. Convert to numpy arrays for efficient operations
    cal_labels = np.array(cal_labels, dtype=float)
    cal_probs = np.array(cal_probs, dtype=float)
    new_probs = np.array(new_probs, dtype=float)
    
    # 2. Sort the calibration data by the predicted probability
    sort_idx = np.argsort(cal_probs)
    sorted_probs = cal_probs[sort_idx]
    sorted_labels = cal_labels[sort_idx]
    
    # 3. Fit Isotonic Regression using Pool Adjacent Violators Algorithm (PAVA)
    # We use a stack to keep track of blocks of merged points. 
    # Each item in the stack is a list: [sum_of_labels, count_of_items]
    stack = []
    
    for y in sorted_labels:
        cur_sum = y
        cur_count = 1
        
        # Check for monotonicity violation: 
        # If the current average is LESS than the previous block's average, they violate 
        # the non-decreasing rule and must be pooled (merged).
        while stack and (cur_sum / cur_count) < (stack[-1][0] / stack[-1][1]):
            prev_sum, prev_count = stack.pop()
            cur_sum += prev_sum
            cur_count += prev_count
            
        stack.append([cur_sum, cur_count])
        
    # Reconstruct the calibrated values array from the merged blocks
    calibrated_values = []
    for block_sum, count in stack:
        val = block_sum / count
        calibrated_values.extend([val] * count)
        
    calibrated_values = np.array(calibrated_values)
    
    # 4. Transform new predictions using linear interpolation
    # np.interp handles both the linear interpolation between points and 
    # the clamping of out-of-bounds predictions automatically.
    calibrated_new_probs = np.interp(new_probs, sorted_probs, calibrated_values)
    
    # Return as a standard python list
    return calibrated_new_probs.tolist()