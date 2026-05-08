import numpy as np

def mean_average_precision(y_true_list, y_score_list, k=None):
    """
    Compute Mean Average Precision (mAP) for multiple retrieval queries.
    """
    ap_per_query = []
    
    for y_true, y_score in zip(y_true_list, y_score_list):
        # Convert to numpy arrays
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        
        # R is the TOTAL relevant items in the entire query.
        # This is the strict denominator, even if k is provided.
        R = np.sum(y_true)
        
        if R == 0:
            ap_per_query.append(0.0)
            continue
            
        # Sort items by their predicted score in descending order
        # We use a stable sort mechanism to prevent random swaps on score ties
        sort_idx = np.argsort(y_score)[::-1]
        y_true_sorted = y_true[sort_idx]
        
        # Apply ranking cutoff if k is provided
        if k is not None:
            y_true_sorted = y_true_sorted[:k]
            
        # Compute the rank sequence: [1, 2, 3, ..., N]
        ranks = np.arange(1, len(y_true_sorted) + 1)
        
        # Compute cumulative hits
        cumulative_hits = np.cumsum(y_true_sorted)
        
        # Compute precision at each rank
        precisions = cumulative_hits / ranks
        
        # Mask out precisions at ranks where the item was not relevant
        relevant_precisions = precisions * y_true_sorted
        
        # Average the precisions using the strict R denominator
        ap = np.sum(relevant_precisions) / R
        ap_per_query.append(float(ap))
        
    map_value = np.mean(ap_per_query) if ap_per_query else 0.0
    
    return float(map_value), ap_per_query