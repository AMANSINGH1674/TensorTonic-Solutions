import numpy as np

def chi2_independence(C):
    """
    Compute chi-square test statistic and expected frequencies.
    """
    # 1. Convert the input to a float numpy array for safe division
    observed = np.array(C, dtype=float)
    
    # 2. Calculate the row totals, column totals, and the grand total
    row_totals = np.sum(observed, axis=1)
    col_totals = np.sum(observed, axis=0)
    grand_total = np.sum(observed)
    
    # 3. Compute Expected frequencies matrix via the outer product
    # E_ij = (row_i * col_j) / total
    expected = np.outer(row_totals, col_totals) / grand_total
    
    # 4. Compute the Chi-Square test statistic
    # chi^2 = sum( (O - E)^2 / E )
    chi2 = np.sum((observed - expected) ** 2 / expected)
    
    return float(chi2), expected