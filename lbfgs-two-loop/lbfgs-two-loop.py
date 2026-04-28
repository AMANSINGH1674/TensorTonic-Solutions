def _dot(a, b):
    """Dot product of two vectors."""
    return sum(x * y for x, y in zip(a, b))

def lbfgs_direction(grad, s_list, y_list):
    """
    Compute the L-BFGS search direction using the two-loop recursion.
    """
    m = len(s_list)
    
    # q starts as a copy of the gradient
    q = list(grad)
    
    # Store rhos and alphas computed during the backward loop 
    # to be reused in the forward loop
    alphas = [0.0] * m
    rhos = [0.0] * m
    
    # 1. Backward loop (from newest to oldest history)
    for i in range(m - 1, -1, -1):
        s_i = s_list[i]
        y_i = y_list[i]
        
        # Calculate and store rho_i = 1 / (y_i^T * s_i)
        rho = 1.0 / _dot(y_i, s_i)
        rhos[i] = rho
        
        # Calculate and store alpha_i = rho_i * (s_i^T * q)
        alpha = rho * _dot(s_i, q)
        alphas[i] = alpha
        
        # Update q = q - alpha_i * y_i
        q = [q_j - alpha * y_ij for q_j, y_ij in zip(q, y_i)]
        
    # 2. Initial scaling of the Hessian approximation (using the most recent pair)
    s_recent = s_list[-1]
    y_recent = y_list[-1]
    gamma = _dot(s_recent, y_recent) / _dot(y_recent, y_recent)
    
    # Scale q to initialize r
    r = [gamma * q_j for q_j in q]
    
    # 3. Forward loop (from oldest to newest history)
    for i in range(m):
        s_i = s_list[i]
        y_i = y_list[i]
        
        # Calculate beta_i = rho_i * (y_i^T * r)
        beta = rhos[i] * _dot(y_i, r)
        
        # Update r = r + s_i * (alpha_i - beta_i)
        factor = alphas[i] - beta
        r = [r_j + factor * s_ij for r_j, s_ij in zip(r, s_i)]
        
    # 4. Negate r to get the descent direction
    return [-r_j for r_j in r]