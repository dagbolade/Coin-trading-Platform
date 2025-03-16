import pandas as pd


# Define the functions for data preprocessing
def convert_to_datetime(df, date_column):
    df[date_column] = pd.to_datetime(df[date_column])
    return df


def check_missing_values(df):
    missing_values = df.isnull().sum()
    return missing_values


def fill_missing_values(df, method='ffill'):
    df.fillna(method=method, inplace=True)
    return df
