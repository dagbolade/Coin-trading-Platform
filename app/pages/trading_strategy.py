# app/pages/trading_strategy.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import datetime
from trading_metrics import calculate_sharpe_ratio, calculate_sortino_ratio, calculate_max_drawdown

def calculate_ema(data, window):
    return data.ewm(span=window, adjust=False).mean()

def generate_trading_signals(data, strategy):
    if strategy == 'Simple Moving Average (SMA) Crossover':
        # User input for selecting the short and long window periods
        short_window = st.number_input('Enter the short window period for SMA:', min_value=1, max_value=100, value=10)
        long_window = st.number_input('Enter the long window period for SMA:', min_value=50, max_value=500, value=50)

        # Calculate the short and long moving averages
        data['SMA_Short'] = data['Close'].rolling(window=short_window, min_periods=1).mean()
        data['SMA_Long'] = data['Close'].rolling(window=long_window, min_periods=1).mean()

        # Generate trading signals based on the crossover strategy
        data['Signal'] = np.where(data['SMA_Short'] > data['SMA_Long'], 'Buy', 'Sell')
        data['Signal'] = np.where(data['SMA_Short'] == data['SMA_Long'], 'Hold', data['Signal'])

    elif strategy == 'Exponential Moving Average (EMA) Crossover':
        # User input for selecting the short and long window periods
        short_window = st.number_input('Enter the short window period for EMA:', min_value=1, max_value=50, value=10)
        long_window = st.number_input('Enter the long window period for EMA:', min_value=50, max_value=200, value=50)

        # Calculate the short and long exponential moving averages
        data['EMA_Short'] = calculate_ema(data['Close'], window=short_window)
        data['EMA_Long'] = calculate_ema(data['Close'], window=long_window)

        # Generate trading signals based on the crossover strategy
        data['Signal'] = np.where(data['EMA_Short'] > data['EMA_Long'], 'Buy', 'Sell')
        data['Signal'] = np.where(data['EMA_Short'] == data['EMA_Long'], 'Hold', data['Signal'])

    return data

def plot_strategy_results(data, ticker, strategy):
    # Calculate the daily returns
    data['Return'] = data['Close'].pct_change()

    # Calculate the strategy returns
    data['Strategy_Return'] = data['Return'] * (data['Signal'].shift(1) == 'Buy').astype(int)

    # Calculate the cumulative returns
    data['Cumulative_Returns'] = (1 + data['Strategy_Return']).cumprod()

    # Display the trading signals and strategy returns
    st.write("Trading Signals and Strategy Returns:")
    st.dataframe(data[['Close', 'Signal', 'Return', 'Strategy_Return', 'Cumulative_Returns']])

    # Plot the strategy returns
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
    fig.add_trace(go.Scatter(x=data.index, y=data['Cumulative_Returns'], mode='lines', name='Cumulative Returns'))
    fig.update_layout(title=f"{ticker} - {strategy}", xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)

    # Stop loss and take profit
    stop_loss = st.number_input('Enter the Stop Loss Percentage:', min_value=0.0, max_value=100.0, value=5.0, step=0.1)
    take_profit = st.number_input('Enter the Take Profit Percentage:', min_value=0.0, max_value=100.0, value=5.0, step=0.1)

    # Calculate the stop loss and take profit levels based on the entry price
    data.loc[data['Signal'] == 'Buy', 'Entry_Price'] = data['Close']
    data['Stop_Loss'] = data['Entry_Price'] * (1 - stop_loss / 100)
    data['Take_Profit'] = data['Entry_Price'] * (1 + take_profit / 100)

    # Display the stop loss and take profit levels
    st.write("Stop Loss and Take Profit Levels:")
    st.dataframe(data[['Close', 'Entry_Price', 'Stop_Loss', 'Take_Profit']])

    # Plot the stop loss and take profit levels
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
    fig.add_trace(go.Scatter(x=data[data['Signal'] == 'Buy'].index, y=data[data['Signal'] == 'Buy']['Entry_Price'], mode='markers', name='Entry Price', marker_symbol='triangle-up'))
    fig.add_trace(go.Scatter(x=data.index, y=data['Stop_Loss'], mode='lines', name='Stop Loss', line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=data.index, y=data['Take_Profit'], mode='lines', name='Take Profit', line=dict(color='green', dash='dash')))
    fig.update_layout(title=f"{ticker} - Stop Loss and Take Profit Levels", xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)

    # Performance metrics
    sharpe_ratio = calculate_sharpe_ratio(data['Strategy_Return'])
    sortino_ratio = calculate_sortino_ratio(data['Strategy_Return'])
    max_drawdown = calculate_max_drawdown(data['Strategy_Return'])

    st.write("Performance Metrics:")
    st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    st.write(f"Sortino Ratio: {sortino_ratio:.2f}")
    st.write(f"Maximum Drawdown: {max_drawdown:.2%}")


def trading_strategy_page():
    if 'selected_cryptos_full' in st.session_state:
        selected_cryptos_full = st.session_state['selected_cryptos_full']
        unique_tickers = selected_cryptos_full['Ticker'].unique()
        ticker = st.selectbox('Select a Cryptocurrency', unique_tickers)

        # ensure the 'Date' column is a datetime object
        if 'Date' in selected_cryptos_full.columns:
            selected_cryptos_full['Date'] = pd.to_datetime(selected_cryptos_full['Date'])
            selected_cryptos_full.set_index('Date', inplace=True)

        # User input for selecting the date range
        start_date = pd.to_datetime(st.date_input('Start Date', selected_cryptos_full.index.min().date()))
        end_date = pd.to_datetime(st.date_input('End Date', selected_cryptos_full.index.max().date()))

        # Filter the data based on the selected date range
        crypto_data = selected_cryptos_full.loc[(selected_cryptos_full['Ticker'] == ticker) &
                                                 (selected_cryptos_full.index >= start_date) &
                                                 (selected_cryptos_full.index <= end_date)]

        # User input for selecting the trading strategy
        strategy = st.selectbox('Select Trading Strategy', ['Simple Moving Average (SMA) Crossover',
                                                            'Exponential Moving Average (EMA) Crossover'])

        # Generate trading signals and apply the selected strategy
        strategy_data = generate_trading_signals(crypto_data, strategy)

        # Plot the strategy results
        plot_strategy_results(strategy_data, ticker, strategy)

    else:
        st.error("Please load and preprocess the cryptocurrency data first.")