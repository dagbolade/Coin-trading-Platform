import pandas as pd
import numpy as np


def calculate_average_gain_loss(data, period=14):
    delta = data.diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    average_gain = up.rolling(window=period).mean().fillna(0)
    average_loss = abs(down.rolling(window=period).mean()).fillna(0)
    return average_gain, average_loss


def calculate_rs(average_gain, average_loss):
    rs = average_gain / average_loss
    return rs


def calculate_rsi(rs):
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_sma_ema_rsi(data, window_sma=10, window_ema=10, window_rsi=14):
    """
    Calculate Simple Moving Average (SMA), Exponential Moving Average (EMA), and Relative Strength Index (RSI).

    Parameters:
    - data (pd.DataFrame): DataFrame containing the 'Close' prices.
    - window_sma (int): Window size for SMA. Default is 10.
    - window_ema (int): Window size for EMA. Default is 10.
    - window_rsi (int): Window size for RSI. Default is 14.

    Returns:
    - data (pd.DataFrame): DataFrame with added SMA, EMA, and RSI columns.
    """
    # Calculate SMA
    data['SMA'] = data['Close'].rolling(window=window_sma).mean().fillna(0)

    # Calculate EMA
    data['EMA'] = data['Close'].ewm(span=window_ema, adjust=False).mean().fillna(0)

    # Calculate average gain and average loss for RSI
    average_gain, average_loss = calculate_average_gain_loss(data['Close'], period=window_rsi)

    # Calculate the Relative Strength (RS)
    rs = calculate_rs(average_gain, average_loss)

    # Calculate the RSI
    data['RSI'] = calculate_rsi(rs).fillna(0)

    return data
