import numpy as np

def adagrad_step(w, g, G, lr=0.01, eps=1e-8):
    """
    Perform one AdaGrad update step.
    """
    # Step 1: Accumulate Squared Gradients
    new_G = G + np.square(g)
    
    # Step 2: Parameter Update
    new_w = w - (lr / np.sqrt(new_G + eps)) * g
    
    return new_w, new_G

# --- Test Cases ---
if __name__ == "__main__":
    # Example 1: Standard update
    w, G = adagrad_step(np.array([1.0, 2.0]), np.array([0.1, -0.2]), np.array([0.0, 0.0]), lr=0.1)
    print(f"Example 1 - new_w: {w}, new_G: {G}") 
    # Output: new_w: [0.9 2.1], new_G: [0.01 0.04]

    # Example 2: Zero gradient
    w, G = adagrad_step(np.array([1.0, 2.0]), np.array([0.0, 0.0]), np.array([0.1, 0.2]), lr=0.1)
    print(f"Example 2 - new_w: {w}, new_G: {G}") 
    # Output: new_w: [1. 2.], new_G: [0.1 0.2]

    # Example 3: Large accumulated G
    w, G = adagrad_step(np.array([0.0]), np.array([1.0]), np.array([100.0]), lr=0.1)
    print(f"Example 3 - new_w: {w}, new_G: {G}") 
    # Output: new_w: [-0.00995037], new_G: [101.]