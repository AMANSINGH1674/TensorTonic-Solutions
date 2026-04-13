import math

def cosine_annealing_schedule(base_lr, min_lr, total_steps, current_step):
    """
    Compute the learning rate using cosine annealing.
    """
    # Calculate the fraction of total steps completed
    progress = current_step / total_steps
    
    # Compute the cosine component
    cos_component = math.cos(math.pi * progress)
    
    # Apply the cosine annealing formula
    lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos_component)
    
    return float(lr)

# --- Test Cases ---
if __name__ == "__main__":
    # Example 1: Start of training (step 0)
    print(cosine_annealing_schedule(0.1, 0.0, 100, 0))  
    # Output: 0.1
    
    # Example 2: Halfway point (step 50)
    print(cosine_annealing_schedule(0.1, 0.0, 100, 50)) 
    # Output: 0.05
    
    # Example 3: End of training (step 100)
    print(cosine_annealing_schedule(0.1, 0.0, 100, 100)) 
    # Output: 0.0
    
    # Example 4: With a non-zero minimum learning rate
    print(cosine_annealing_schedule(0.1, 0.01, 100, 50)) 
    # Output: 0.055