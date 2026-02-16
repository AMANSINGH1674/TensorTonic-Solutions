# Linear Interpolation
def linear_interpolation(values):
    """
    Fill missing (None) values using linear interpolation.
    
    Args:
        values: List of numbers and None values. 
               First and last are guaranteed to be numbers.
    Returns:
        A new list of floats/integers with interpolated values.
    """
    # Create a copy to avoid modifying the original list
    result = list(values)
    n = len(result)
    
    i = 0
    while i < n:
        # If we find a None, we need to find the bounding known values
        if result[i] is None:
            left = i - 1
            v_left = result[left]
            
            # Look forward for the 'right' boundary
            right = i
            while right < n and result[right] is None:
                right += 1
            v_right = result[right]
            
            # Interpolate every index j in this gap
            # Formula: v_left + (j - left) / (right - left) * (v_right - v_left)
            denominator = right - left
            diff = v_right - v_left
            
            for j in range(i, right):
                result[j] = v_left + (j - left) / denominator * diff
                
            # Move the main pointer to the right boundary
            i = right
        else:
            i += 1
            
    return result