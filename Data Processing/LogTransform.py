import math

def log_transform(values):
    """
    Apply the log1p transformation: ln(1 + x).
    """
    # math.log1p is numerically more stable for very small values of x
    return [math.log1p(v) for v in values]