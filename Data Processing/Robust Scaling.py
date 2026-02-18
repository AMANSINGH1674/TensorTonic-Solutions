# Robust Scaling
def get_median(sorted_list):
    n = len(sorted_list)
    if n == 0: return 0
    mid = n // 2
    if n % 2 == 1:
        return float(sorted_list[mid])
    else:
        return (sorted_list[mid - 1] + sorted_list[mid]) / 2.0

def robust_scaling(values):
    """
    Scale values using the median and Interquartile Range (IQR).
    """
    if not values:
        return []
    if len(values) == 1:
        return [0.0]

    # 1. Sort values to calculate quartiles
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    
    # 2. Compute Median of the full set
    median = get_median(sorted_vals)
    
    # 3. Determine halves for Q1 and Q3
    # If n is odd, the median element is excluded from both halves
    mid = n // 2
    lower_half = sorted_vals[:mid]
    upper_half = sorted_vals[mid + (1 if n % 2 == 1 else 0):]
    
    # 4. Compute Q1 (median of lower half) and Q3 (median of upper half)
    q1 = get_median(lower_half)
    q3 = get_median(upper_half)
    iqr = q3 - q1
    
    # 5. Apply the scaling formula
    # If IQR is 0, we only center the data (subtract median)
    if iqr == 0:
        return [float(x - median) for x in values]
    
    return [float(x - median) / iqr for x in values]