import numpy as np

def r2_score(y_true, y_pred) -> float:
    """
    Compute R² (coefficient of determination) for 1D regression.
    Handle the constant-target edge case:
      - return 1.0 if predictions match exactly,
      - else 0.0.
    """
    # Ensure inputs are NumPy arrays for vectorized operations
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Handle the constant-target edge case
    if np.all(y_true == y_true[0]):
        if np.all(y_true == y_pred):
            return 1.0
        else:
            return 0.0
            
    # Calculate Residual Sum of Squares (SS_res)
    ss_res = np.sum((y_true - y_pred) ** 2)
    
    # Calculate Total Sum of Squares (SS_tot)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    # Compute and return the R² score
    return float(1 - (ss_res / ss_tot))

# --- Test Cases ---
if __name__ == "__main__":
    # Example 1: Close fit
    print("Example 1:", r2_score([3, 4, 5], [2.9, 4.1, 5.0]))
    # Output: 0.985...
    
    # Example 2: Constant target, perfect prediction
    print("Example 2:", r2_score([1, 1, 1], [1, 1, 1]))
    # Output: 1.0
    
    # Example 3: Constant target, imperfect prediction
    print("Example 3:", r2_score([1, 1, 1], [0, 2, 1]))
    # Output: 0.0