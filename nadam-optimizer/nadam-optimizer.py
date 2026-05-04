import numpy as np

def nadam_step(w, m, v, grad, lr=0.002, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Perform one Nadam update step using Nesterov-accelerated adaptive moments.
    Returns: (w_new, m_new, v_new)
    """
    # 1. Convert inputs to NumPy arrays of floats for safe and fast vectorization
    w = np.asarray(w, dtype=float)
    m = np.asarray(m, dtype=float)
    v = np.asarray(v, dtype=float)
    grad = np.asarray(grad, dtype=float)
    
    # 2. Update First Moment (Exponential moving average of gradients)
    m_new = beta1 * m + (1.0 - beta1) * grad
    
    # 3. Update Second Moment (Exponential moving average of squared gradients)
    v_new = beta2 * v + (1.0 - beta2) * np.square(grad)
    
    # 4. Apply the Nesterov-Adjusted Update
    # The numerator applies the look-ahead momentum combined with the current gradient
    numerator = beta1 * m_new + (1.0 - beta1) * grad
    
    # The denominator scales the update inversely to the historical gradient magnitude
    denominator = np.sqrt(v_new) + eps
    
    # Calculate new weights
    w_new = w - lr * (numerator / denominator)
    
    return w_new, m_new, v_new