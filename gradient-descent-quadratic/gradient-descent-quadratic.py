def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Minimize f(x) = ax^2 + bx + c using vanilla gradient descent.
    """
    x = float(x0)
    
    for _ in range(steps):
        # Calculate the derivative at the current point
        grad = 2 * a * x + b
        
        # Update x: move opposite to the gradient
        x = x - lr * grad
        
    return float(x)