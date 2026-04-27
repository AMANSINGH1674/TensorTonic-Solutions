import numpy as np

def t_test_one_sample(x, mu0):
    """
    Compute one-sample t-statistic.
    """
    # 1. Convert the input to a float numpy array
    x_arr = np.array(x, dtype=float)
    
    # Get the sample size
    n = len(x_arr)
    
    # 2. Calculate the sample mean
    x_bar = np.mean(x_arr)
    
    # 3. Calculate the sample standard deviation
    # ddof=1 applies Bessel's correction (dividing by n - 1 instead of n)
    s = np.std(x_arr, ddof=1)
    
    # 4. Calculate the standard error of the mean
    se = s / np.sqrt(n)
    
    # 5. Compute the final t-statistic
    t_stat = (x_bar - mu0) / se
    
    return float(t_stat)