import numpy as np

def calculate_eigenvalues(matrix):
    """
    Calculate eigenvalues of a square matrix.
    """
    # 1. Input Validation: Convert to a NumPy array of floats
    try:
        A = np.array(matrix, dtype=float)
    except (ValueError, TypeError):
        # Catch cases where input contains non-numeric types
        return None
        
    # 2. Dimensionality Check: Ensure it is exactly a 2D matrix
    if A.ndim != 2:
        return None
        
    # 3. Square Matrix Check: Ensure rows == cols
    rows, cols = A.shape
    if rows != cols:
        return None
        
    # Handle the edge case of an empty square matrix
    if rows == 0:
        return np.array([])
        
    # 4. Compute Eigenvalues
    # np.linalg.eigvals is more efficient than np.linalg.eig if eigenvectors aren't needed
    eigenvalues = np.linalg.eigvals(A)
    
    # 5. Consistent Sorting
    # np.lexsort sorts by the last passed array first. 
    # Here, we sort primarily by the real part, and secondarily by the imaginary part.
    idx = np.lexsort((np.imag(eigenvalues), np.real(eigenvalues)))
    sorted_eigenvalues = eigenvalues[idx]
    
    return sorted_eigenvalues