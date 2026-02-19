def min_max_scaling(data):
    if not data or not data[0]:
        return []
    
    n_rows = len(data)
    n_cols = len(data[0])
    
    # Initialize a result matrix with the same dimensions
    scaled_data = [[0.0] * n_cols for _ in range(n_rows)]
    
    for j in range(n_cols):
        # Extract the column
        column = [data[i][j] for i in range(n_rows)]
        
        col_min = min(column)
        col_max = max(column)
        col_range = col_max - col_min
        
        for i in range(n_rows):
            if col_range == 0:
                # If all values are the same, scaled value is 0.0
                scaled_data[i][j] = 0.0
            else:
                # Apply linear scaling
                scaled_data[i][j] = float((data[i][j] - col_min) / col_range)
                
    return scaled_data