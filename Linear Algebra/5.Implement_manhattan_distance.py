# Implement Manhattan Distance
import numpy as np

def manhattan_distance(x, y):
    """
    Compute the Manhattan (L1) distance between vectors x and y.
    Must return a float.
    """
    # Convert inputs to numpy arrays for vectorization
    x = np.array(x)
    y = np.array(y)
    
    # Calculate the sum of absolute differences
    # np.abs(x - y) computes |xi - yi| for all i simultaneously
    distance = np.sum(np.abs(x - y))
    
    return float(distance)