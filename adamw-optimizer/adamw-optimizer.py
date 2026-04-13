import numpy as np

def adamw_step(w, m, v, grad, lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.01, eps=1e-8):
    """
    Perform one AdamW update step using decoupled weight decay.
    """
    # 0. Ensure inputs are NumPy arrays (handles lists from test platforms)
    w, m, v, grad = map(np.asarray, (w, m, v, grad))
    
    # 1. Update First Moment (mt)
    # Moving average of the gradients
    new_m = beta1 * m + (1 - beta1) * grad
    
    # 2. Update Second Moment (vt)
    # Moving average of the squared gradients
    new_v = beta2 * v + (1 - beta2) * (grad**2)
    
    # 3. AdamW Parameter Update
    # Term A: Decoupled weight decay (directly on the weights)
    # Term B: Adaptive gradient update (using the moments)
    new_w = w - lr * (weight_decay * w) - lr * (new_m / (np.sqrt(new_v) + eps))
    
    return new_w, new_m, new_v

# --- Quick Verification ---
if __name__ == "__main__":
    w_init, m_init, v_init = [1.0, -2.0], [0, 0], [0, 0]
    grad = [0.3, -0.7]
    
    new_w, new_m, new_v = adamw_step(w_init, m_init, v_init, grad, lr=0.01, weight_decay=0.1)
    
    print(f"Updated Weights: {np.round(new_w, 3)}")
    # Expected Output: [0.967, -1.966]