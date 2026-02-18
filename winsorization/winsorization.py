import math

def get_percentile(sorted_arr, p):
    """Compute percentile using linear interpolation."""
    n = len(sorted_arr)
    if n == 0: return 0.0
    if n == 1: return float(sorted_arr[0])
    
    # Calculate the virtual index k
    k = (n - 1) * p / 100.0
    floor_k = math.floor(k)
    ceil_k = math.ceil(k)
    
    if floor_k == ceil_k:
        return float(sorted_arr[int(k)])
    
    # Linear interpolation: arr[floor] + decimal_part * (arr[ceil] - arr[floor])
    weight = k - floor_k
    return sorted_arr[floor_k] + weight * (sorted_arr[ceil_k] - sorted_arr[floor_k])

def winsorize(values, lower_pct, upper_pct):
    """
    Clip values at the given percentile bounds using linear interpolation.
    """
    if not values:
        return []
    
    # 1. Sort values to find percentile boundaries
    sorted_vals = sorted(values)
    
    # 2. Compute the lower and upper bounds
    lower_bound = get_percentile(sorted_vals, lower_pct)
    upper_bound = get_percentile(sorted_vals, upper_pct)
    
    # 3. Clip the original values (preserve original order)
    # result = max(lower, min(upper, val))
    winsorized_values = []
    for val in values:
        if val < lower_bound:
            winsorized_values.append(float(lower_bound))
        elif val > upper_bound:
            winsorized_values.append(float(upper_bound))
        else:
            winsorized_values.append(float(val))
            
    return winsorized_values