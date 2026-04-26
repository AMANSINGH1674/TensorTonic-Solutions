import numpy as np

def matrix_inverse(A):
    """
    Returns the inverse of a square, non-singular matrix A.
    Returns None if the matrix is invalid, non-square, or singular.
    """
    # Ensure input is a NumPy array of floats to prevent integer division issues
    A = np.asarray(A, dtype=float)
    
    # 1. Validate that the input is exactly a 2D matrix
    if A.ndim != 2:
        return None
        
    # 2. Validate that the matrix is square (n x n)
    rows, cols = A.shape
    if rows != cols:
        return None
        
    # 3. Check for singularity using the determinant
    # We use a small epsilon (1e-10) instead of == 0 to account for floating-point errors
    det = np.linalg.det(A)
    if np.abs(det) < 1e-10:
        return None
        
    # 4. Compute and return the inverse matrix
    try:
        A_inv = np.linalg.inv(A)
        return A_inv
    except np.linalg.LinAlgError:
        # Fallback catch in case np.linalg.inv detects singularity differently
        return None