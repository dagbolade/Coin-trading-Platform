import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def calculate_daily_returns(crypto_data):
    """
    Calculate daily returns for cryptocurrencies.

    Args:
    crypto_data (DataFrame): Dataframe containing cryptocurrency price data.

    Returns:
    DataFrame: Daily returns of cryptocurrencies.
    """
    daily_returns = crypto_data.pivot_table(index='Date', columns='Ticker', values='Close').pct_change(fill_method=None).dropna()
    return daily_returns


def calculate_correlation_matrix(daily_returns):
    """
    Calculate the correlation matrix from daily returns data.

    Args:
    daily_returns (DataFrame): Dataframe containing daily returns of cryptocurrencies.

    Returns:
    DataFrame: Correlation matrix.
    """
    correlation_matrix = daily_returns.corr()
    return correlation_matrix



def find_top_correlations(correlation_matrix, selected_tickers, top_n=4):
    """
    Find the top positively and negatively correlated cryptocurrencies for the selected tickers.

    Args:
    correlation_matrix (DataFrame): Correlation matrix of cryptocurrencies.
    selected_tickers (list): List of selected cryptocurrency tickers.
    top_n (int): Number of top correlated cryptocurrencies to return.

    Returns:
    dict: Dictionary with top positively and negatively correlated cryptocurrencies for each selected ticker.
    """
    top_correlated = {}
    for ticker in selected_tickers:
        correlations = correlation_matrix[ticker].drop(ticker)  # Drop the self-correlation
        top_positive = correlations.nlargest(top_n)
        top_negative = correlations.nsmallest(top_n)
        top_correlated[ticker] = {
            'Positive': top_positive,
            'Negative': top_negative
        }
    return top_correlated



def plot_correlation_heatmap(correlation_matrix, title='Correlation Matrix'):
    """
    Plot a heatmap for the correlation matrix.

    Args:
    correlation_matrix (DataFrame): Correlation matrix to plot.
    title (str): Title for the heatmap.
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.05)
    plt.title(title)
    plt.show()

