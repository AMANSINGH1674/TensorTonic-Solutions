import numpy as np

def zscore_standardize(X, axis=0, eps=1e-12):
    """
    Standardize X: (X - mean)/std.
    If 2D and axis=0, per column.
    Handles 1D and 2D arrays.
    Avoids divide-by-zero using eps.
    Returns float ndarray.
    """
    X = np.asarray(X, dtype=float)

    mean = np.mean(X, axis=axis, keepdims=True)
    std = np.std(X, axis=axis, keepdims=True)

    # Add eps for numerical stability (as required by test case)
    std = std + eps

    Z = (X - mean) / std

    return Z