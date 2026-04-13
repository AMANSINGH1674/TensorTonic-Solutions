def warmup_decay_schedule(base_lr, warmup_steps, total_steps, current_step):
    """
    Compute the learning rate at a given step using warmup + linear decay.
    """
    # Handle the warmup phase
    if warmup_steps > 0 and current_step < warmup_steps:
        return float(base_lr * (current_step / warmup_steps))
    
    # Handle the decay phase
    else:
        return float(base_lr * ((total_steps - current_step) / (total_steps - warmup_steps)))

# --- Test Cases ---
if __name__ == "__main__":
    # Example 1: Middle of warmup phase
    print(warmup_decay_schedule(0.1, 10, 100, 5))   # Output: 0.05
    
    # Example 2: Middle of decay phase
    print(warmup_decay_schedule(0.1, 10, 100, 55))  # Output: 0.05
    
    # Example 3: Peak learning rate (current_step == warmup_steps)
    print(warmup_decay_schedule(0.1, 10, 100, 10))  # Output: 0.1
    
    # Example 4: End of training (current_step == total_steps)
    print(warmup_decay_schedule(0.1, 10, 100, 100)) # Output: 0.0