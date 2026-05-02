import numpy as np

def clip_gradients(g, max_norm):
    """
    Clip gradients using global norm clipping.
    Preserves direction while capping maximum magnitude.
    """
    # Convert input to a float numpy array and create a copy to avoid in-place modification
    g_arr = np.array(g, dtype=float)
    
    # Edge case: If max_norm is non-positive, return the original gradients unchanged
    if max_norm <= 0:
        return g_arr
        
    # Compute the global L2 norm. 
    # We use .ravel() because np.linalg.norm can throw errors on arrays with >2 dimensions.
    # Flattening it guarantees we compute the true global norm across all elements.
    norm = np.linalg.norm(g_arr.ravel())
    
    # Edge case: If the norm is exactly 0, scaling would cause a division by zero.
    if norm == 0.0:
        return g_arr
        
    # If the gradient exceeds our speed limit, scale it down proportionally
    if norm > max_norm:
        scale_factor = max_norm / norm
        g_arr = g_arr * scale_factor
        
    return g_arr