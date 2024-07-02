import pandas as pd
import numpy as np



# Function to find the minimum NaN sequence length in each column
def min_nan_sequence_length(column):
    nan_lengths = []
    current_length = 0
    
    for value in column:
        if pd.isna(value):
            current_length += 1
        else:
            if current_length > 0:
                nan_lengths.append(current_length)
                current_length = 0
    
    # If the column ends with NaNs, add the last sequence length
    if current_length > 0:
        nan_lengths.append(current_length)
    
    # Return the minimum NaN sequence length
    return min(nan_lengths) if nan_lengths else 0

# Function to compress NaN subsequences in each column
def compress_nan_sequences(column, min_nan_seq):
    new_column = []
    i = 0
    while i < len(column):
        if pd.isna(column.iloc[i]):
            # Remove min_nan_seq entries after the starting NaN value
            for j in range(min_nan_seq):
                if i < len(column):
                    i += 1
                else:
                    break
            while i < len(column)-1:
                new_column.append(column.iloc[i])
                i+=1
                if not pd.isna(column.iloc[i]):
                    break
        else:
            new_column.append(column.iloc[i])
            i += 1
    
    # Append NaNs for remaining entries at the end of the column
    new_column.extend([np.nan] * (len(column) - len(new_column)))
    
    return pd.Series(new_column)

def logger_nan_cleaner(log_table):
    """ only apply to datasets with regular nan occurrence patterns. 
        Per column each nan sequence should be the same length and there should be the same number of nan sequences in each column.
        
        Example usage:
              A    B     C      D\n
        0   NaN  NaN   NaN  100.0\n
        1   NaN  NaN   NaN  200.0\n
        2   NaN  NaN  40.0    NaN\n
        3  28.0  5.0   NaN    NaN\n
        4   NaN  NaN   NaN  400.0\n
        5   NaN  NaN   NaN  500.0\n
        6   NaN  NaN  50.0    NaN\n
        7   4.0  6.0   NaN    NaN\n
        8   NaN  NaN   NaN  400.0\n
        9   NaN  NaN   NaN  500.0\n
        min nan sequence length: 2
        length difference: 4
            A    B     C      D
        0  28.0  5.0  40.0  100.0\n
        1  28.0  5.0  40.0  200.0\n
        2  28.0  5.0  40.0  400.0\n
        3   4.0  6.0  50.0  500.0\n
        4   4.0  6.0  50.0  400.0\n
        5   4.0  6.0  50.0  500.0\n
    """
    min_nan_seq = log_table.apply(min_nan_sequence_length).min()
    print(f"min nan sequence length: {min_nan_seq}")
    df_compressed = log_table.apply(lambda x: compress_nan_sequences(x, min_nan_seq), axis=0)
    df_compressed = df_compressed.dropna(how='all')
    df_compressed = df_compressed.interpolate(method='nearest',axis=0).ffill(axis=0).bfill(axis=0)
    print(f"length difference: {len(log_table) - len(df_compressed)}")
    return df_compressed

if __name__ == "__main__":
    # Example DataFrame
    data = {
        'A': [np.nan, np.nan, np.nan, 28,     np.nan, np.nan, np.nan, 4,     np.nan, np.nan],
        'B': [np.nan, np.nan, np.nan, 5,      np.nan, np.nan, np.nan, 6,     np.nan, np.nan],
        'C': [np.nan, np.nan, 40,     np.nan, np.nan, np.nan, 50,    np.nan, np.nan, np.nan],
        'D': [100,    200,    np.nan, np.nan, 400,    500,   np.nan, np.nan, 400,    500,  ]
    }

    df = pd.DataFrame(data)
    print(df)
    df = logger_nan_cleaner(df)
    print(df)