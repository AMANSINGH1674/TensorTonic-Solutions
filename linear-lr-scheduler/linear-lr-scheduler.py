def linear_lr(step, total_steps, initial_lr, final_lr=0.0, warmup_steps=0) -> float:
    """
    Linear warmup (0 -> initial_lr) then linear decay (initial_lr -> final_lr).
    Steps are 0-based; clamp at final_lr after total_steps.
    """
    # 1. Post-training Phase: clamp exactly at final_lr
    if step >= total_steps:
        return float(final_lr)
        
    # 2. Warmup Phase: linearly scale up from 0 to initial_lr
    if step < warmup_steps:
        # Avoid division by zero by implicitly knowing warmup_steps > 0 if step < warmup_steps
        return float(step * (initial_lr / warmup_steps))
        
    # 3. Decay Phase: linearly scale down from initial_lr to final_lr
    # Calculate the remaining proportion of the decay phase (1.0 at start of decay, 0.0 at end)
    decay_ratio = (total_steps - step) / (total_steps - warmup_steps)
    
    # Interpolate between final_lr and initial_lr
    lr = final_lr + (initial_lr - final_lr) * decay_ratio
    
    return float(lr)