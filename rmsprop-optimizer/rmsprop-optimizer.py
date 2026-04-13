import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    """
    # 0. Ensure inputs are NumPy arrays for vectorized operations
    w, g, s = map(np.asarray, (w, g, s))
    
    # 1. Update Running Average of Squared Gradients
    # s_t = beta * s_{t-1} + (1 - beta) * g_t^2
    new_s = beta * s + (1 - beta) * (g**2)
    
    # 2. Parameter Update
    # w_t = w_{t-1} - (lr / sqrt(s_t + eps)) * g_t
    new_w = w - (lr / (np.sqrt(new_s) + eps)) * g
    
    return new_w, new_s