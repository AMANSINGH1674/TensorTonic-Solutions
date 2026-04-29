import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    # Uses np.where to avoid exponential overflow for large negative numbers
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # 1. Input preparation
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1) # Ensure y is a 1D array of shape (N,)
    
    N, D = X.shape
    
    # 2. Parameter initialization
    w = np.zeros(D, dtype=float)
    b = 0.0
    
    # 3. Gradient Descent Training Loop
    for _ in range(steps):
        # Forward pass: compute linear combination and apply activation
        z = np.dot(X, w) + b
        p = _sigmoid(z)
        
        # Calculate error (Prediction - Target)
        error = p - y
        
        # Compute gradients
        # X.T is (D, N) and error is (N,). Dot product results in shape (D,)
        dw = (1.0 / N) * np.dot(X.T, error) 
        db = (1.0 / N) * np.sum(error)
        
        # Update parameters
        w -= lr * dw
        b -= lr * db
        
    return w, float(b)