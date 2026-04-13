def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    Calculates total TP, FP, and FN across all classes.
    """
    # 1. Count True Positives (global matches)
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    
    # 2. Total samples
    total = len(y_true)
    if total == 0:
        return 0.0
    
    # 3. In single-label multiclass, every error is both a FP and a FN
    # So: Total FP = Total FN = (Total Samples - Total TP)
    errors = total - tp
    fp = errors
    fn = errors
    
    # 4. Apply the Micro-F1 formula
    # F1 = (2 * TP) / (2 * TP + FP + FN)
    denominator = (2 * tp + fp + fn)
    
    return float((2 * tp) / denominator)
