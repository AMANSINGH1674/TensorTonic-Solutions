import numpy as np

def classification_metrics(y_true, y_pred, average="micro", pos_label=1):
    """
    Compute accuracy, precision, recall, F1 for single-label classification.
    Averages: 'micro' | 'macro' | 'weighted' | 'binary' (uses pos_label).
    Return dict with float values.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # 1. Global Accuracy
    accuracy = float(np.mean(y_true == y_pred))
    
    # 2. Micro Averaging
    # For single-label multi-class, global micro P, R, and F1 are mathematically equal to accuracy.
    if average == "micro":
        return {
            "accuracy": accuracy,
            "precision": accuracy,
            "recall": accuracy,
            "f1": accuracy
        }
        
    # 3. Binary Averaging
    if average == "binary":
        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
        fn = np.sum((y_true == pos_label) & (y_pred != pos_label))
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        
        return {
            "accuracy": accuracy,
            "precision": float(p),
            "recall": float(r),
            "f1": float(f)
        }
        
    # 4. Macro and Weighted Averaging
    if average in ["macro", "weighted"]:
        # Extract all unique classes present in either true or predicted arrays
        classes, indices = np.unique(np.concatenate((y_true, y_pred)), return_inverse=True)
        K = len(classes)
        N = len(y_true)
        
        # Map original arbitrary labels to a contiguous 0..K-1 range for fast binning
        y_true_mapped = indices[:N]
        y_pred_mapped = indices[N:]
        
        # Build the Confusion Matrix using 1D bincounting (highly optimized)
        cm = np.bincount(y_true_mapped * K + y_pred_mapped, minlength=K*K).reshape(K, K)
        
        # Extract components per class
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        support = cm.sum(axis=1) # Actual true occurrences of each class
        
        # Safely compute per-class metrics without throwing DivisionByZero warnings
        with np.errstate(divide='ignore', invalid='ignore'):
            precision = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
            recall = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
            f1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0.0)
            
        if average == "macro":
            return {
                "accuracy": accuracy,
                "precision": float(np.mean(precision)),
                "recall": float(np.mean(recall)),
                "f1": float(np.mean(f1))
            }
            
        elif average == "weighted":
            total_support = support.sum()
            return {
                "accuracy": accuracy,
                "precision": float(np.average(precision, weights=support)) if total_support > 0 else 0.0,
                "recall": float(np.average(recall, weights=support)) if total_support > 0 else 0.0,
                "f1": float(np.average(f1, weights=support)) if total_support > 0 else 0.0
            }
            
    raise ValueError("Invalid average mode. Choose from 'micro', 'macro', 'weighted', or 'binary'.")