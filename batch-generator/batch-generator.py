import numpy as np

def batch_generator(X, y, batch_size, rng=None, drop_last=False):
    """
    Randomly shuffle a dataset and yield mini-batches (X_batch, y_batch).
    """
    # 1. Convert inputs to numpy arrays (this avoids modifying the originals)
    X_np = np.asarray(X)
    y_np = np.asarray(y)
    
    N = len(X_np)
    
    # 2. Create an array of indices and shuffle them
    indices = np.arange(N)
    if rng is not None:
        rng.shuffle(indices)
    else:
        np.random.shuffle(indices)
        
    # 3. Apply the shuffled indices to both X and y to keep them aligned
    X_shuffled = X_np[indices]
    y_shuffled = y_np[indices]
    
    # 4. Iterate over the dataset in steps of batch_size
    for i in range(0, N, batch_size):
        end_idx = i + batch_size
        
        # 5. Handle the drop_last constraint
        if drop_last and end_idx > N:
            break
            
        # 6. Yield the contiguous slices as a tuple
        yield X_shuffled[i:end_idx], y_shuffled[i:end_idx]