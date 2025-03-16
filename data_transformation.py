import pandas as pd
from sklearn.preprocessing import StandardScaler


def remove_duplicates(df, subset):
    """
    Remove duplicates from a DataFrame based on a subset of columns.

    Args:
    df: DataFrame
    subset: list of str, the columns based on which to check for duplicates

    Returns:
    DataFrame without duplicates
    """
    duplicates = df.duplicated(subset=subset, keep=False)
    if duplicates.any():
        print(f"Found duplicates: {df[duplicates]}")
        df = df.drop_duplicates(subset=subset, keep='first')
    return df


def pivot_and_fill(df, index, columns, values):
    pivot_df = df.pivot(index=index, columns=columns, values=values)
    pivot_df = pivot_df.ffill(axis=1)  # Use ffill instead of fillna(method='ffill')
    pivot_df = pivot_df.bfill(axis=1)  # Use bfill instead of fillna(method='bfill')
    return pivot_df

def scale_data(df):
    """
    Normalize the data using StandardScaler.

    Args:
    df: DataFrame, data to be scaled

    Returns:
    DataFrame of the scaled data
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df.fillna(0))  # Just in case, fill any remaining NaNs with 0
    scaled_df = pd.DataFrame(scaled_data, index=df.index, columns=df.columns)
    return scaled_df
