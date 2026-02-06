# Implement Cosine Similarity
import numpy as np

def cosine_similarity(a, b):
    """
    Compute the cosine similarity between two 1D arrays a and b.
    Must return a float.
    """
    # Convert inputs to numpy arrays
    a = np.array(a)
    b = np.array(b)
    
    # Calculate the dot product
    dot_val = np.dot(a, b)
    
    # Calculate the L2 norms (Euclidean norms)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    # Handle zero vectors: if either norm is 0, return 0.0
    if norm_a == 0 or norm_b == 0:
        return 0.0
        
    # Apply the formula
    similarity = dot_val / (norm_a * norm_b)
    
    return float(similarity)