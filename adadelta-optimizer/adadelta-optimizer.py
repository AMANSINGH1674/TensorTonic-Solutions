import numpy as np

def adadelta_step(w, grad, E_grad_sq, E_update_sq, rho=0.9, eps=1e-6):
    """
    Perform one AdaDelta update step.
    Returns: (new_w, new_E_grad_sq, new_E_update_sq)
    """
    # 1. Convert all inputs to NumPy arrays of floats for safe vectorization
    w = np.asarray(w, dtype=float)
    grad = np.asarray(grad, dtype=float)
    E_grad_sq = np.asarray(E_grad_sq, dtype=float)
    E_update_sq = np.asarray(E_update_sq, dtype=float)
    
    # Step 1: Update the running average of squared gradients
    new_E_grad_sq = rho * E_grad_sq + (1.0 - rho) * np.square(grad)
    
    # Step 2: Compute the parameter update (delta_w)
    # We use the RMS (Root Mean Square) of past updates divided by the RMS of current gradients.
    rms_update_prev = np.sqrt(E_update_sq + eps)
    rms_grad_curr = np.sqrt(new_E_grad_sq + eps)
    
    delta_w = - (rms_update_prev / rms_grad_curr) * grad
    
    # Step 3: Update the running average of squared parameter updates
    new_E_update_sq = rho * E_update_sq + (1.0 - rho) * np.square(delta_w)
    
    # Step 4: Apply the update to the parameters
    new_w = w + delta_w
    
    return new_w, new_E_grad_sq, new_E_update_sq