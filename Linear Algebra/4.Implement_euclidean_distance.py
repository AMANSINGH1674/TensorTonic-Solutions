# Implement Euclidean Distance
import numpy as np

def euclidean_distance(x, y):
    """
    Compute the Euclidean (L2) distance between vectors x and y.
    Must return a float.
    """
    # Convert to numpy arrays to ensure vectorization works
    x = np.array(x)
    y = np.array(y)
    
    # Calculate the sum of squared differences
    distance = np.sqrt(np.sum((x - y)**2))
    
    return float(distance)