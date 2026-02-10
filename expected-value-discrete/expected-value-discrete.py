import numpy as np

def expected_value_discrete(x, p):
    """
    Compute the expected value of a discrete random variable.
    
    Returns: 
        float: The expected value E[X].
    Raises:
        ValueError: If probabilities do not sum to 1 or shapes do not match.
    """
    # Convert to NumPy arrays for vectorized multiplication
    x_arr = np.array(x)
    p_arr = np.array(p)
    
    # Check if shapes match
    if x_arr.shape != p_arr.shape:
        raise ValueError("The shapes of values (x) and probabilities (p) must match.")
    
    # Validate that probabilities sum to 1 (using a tolerance of 1e-6)
    if not np.allclose(np.sum(p_arr), 1.0, atol=1e-6):
        raise ValueError("Probabilities must sum to 1.")
    
    # Calculate E[X] = sum(xi * pi)
    expected_val = np.sum(x_arr * p_arr)
    
    return float(expected_val)