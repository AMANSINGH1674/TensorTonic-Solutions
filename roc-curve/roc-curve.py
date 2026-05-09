import numpy as np

def roc_curve(y_true, y_score):
    """
    Compute ROC curve from binary labels and scores.
    Returns: (fpr, tpr, thresholds)
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    
    # 1. Sort by descending score
    # We use mergesort for stable sorting
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score_sorted = y_score[desc_score_indices]
    y_true_sorted = y_true[desc_score_indices]
    
    # 2. Find indices where the score changes to handle ties correctly.
    # np.diff will be non-zero just before the value changes.
    distinct_value_indices = np.where(np.diff(y_score_sorted))[0]
    # The very last index must always be included as the final threshold
    threshold_idxs = np.r_[distinct_value_indices, y_true_sorted.size - 1]
    
    # 3. Compute Cumulative True Positives
    # By taking the cumsum of y_true, we get the rolling count of TPs.
    # We slice it using threshold_idxs to only keep counts at the boundary of score changes.
    tps = np.cumsum(y_true_sorted)[threshold_idxs]
    
    # 4. Compute Cumulative False Positives
    # Total items evaluated up to an index is `index + 1`.
    # FPs = (Total Evaluated) - (True Positives)
    fps = (threshold_idxs + 1) - tps
    
    # 5. Insert the starting point (FPR=0, TPR=0, Threshold=inf)
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    thresholds = np.r_[np.inf, y_score_sorted[threshold_idxs]]
    
    # 6. Normalize counts to True Positive Rate (TPR) and False Positive Rate (FPR)
    total_p = tps[-1]
    total_n = fps[-1]
    
    # Handle edge cases to avoid division by zero if a class is entirely missing
    tpr = tps / total_p if total_p > 0 else np.zeros_like(tps, dtype=float)
    fpr = fps / total_n if total_n > 0 else np.zeros_like(fps, dtype=float)
    
    return fpr, tpr, thresholds