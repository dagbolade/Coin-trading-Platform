import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import mplfinance as mpf
import plotly.graph_objects as go


def plot_time_series(data, ticker):
    # If 'Date' is not in the columns, it means it's already the index
    if 'Date' not in data.columns:
        series = data[data['Ticker'] == ticker]['Close']
    else:
        # If 'Date' is a column, then set it as the index
        series = data[data['Ticker'] == ticker].set_index('Date')['Close']

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(series.index, series, label='Close Price')
    ax.set_title('Time Series Plot')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()

    return fig


def plot_interval_change(data, ticker, interval='M'):
    # Check if 'Date' is in the columns and set it as the index if necessary
    if 'Date' in data.columns:
        data = data.set_index('Date')

    # Calculate the mean price for the given interval
    data_interval = data[data['Ticker'] == ticker]['Close'].resample(interval).mean()

    # Calculate the percentage change
    pct_change = data_interval.pct_change()

    # Plotting
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(pct_change.index, pct_change, label='Percentage Change')
    ax.set_title(f'Change in Distribution Over {interval} Intervals for {ticker}')
    ax.set_xlabel(f'{interval} Interval')
    ax.set_ylabel('Percentage Change')
    ax.legend()

    return fig


def plot_rolling_statistics(data, ticker, window=30):
    # Extract data for the given ticker
    ticker_data = data[data['Ticker'] == ticker]

    # Check if 'Date' is in the columns and set it as the index if necessary
    if 'Date' in ticker_data.columns:
        ticker_data = ticker_data.set_index('Date')

    # Select the 'Close' prices and calculate rolling statistics
    series = ticker_data['Close']
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()

    # Plotting
    plt.figure(figsize=(14, 7))
    series.plot(label='Original', color='blue')
    rolling_mean.plot(label=f'Rolling Mean ({window} days)', color='red')
    rolling_std.plot(label=f'Rolling Std Dev ({window} days)', color='green')
    plt.title(f'Rolling Mean & Standard Deviation for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    fig = plt.gcf()  # Getting the current figure
    plt.close()  # Close the plot to prevent it from displaying in non-Streamlit environments
    return fig
    #plt.show()


# def plot_distribution(data, ticker):
#     # calculate daily returns
#     data['Daily Return'] = data[data['Ticker'] == ticker]['Close'].pct_change()
#     returns = data[data['Ticker'] == ticker]['Daily Return'].dropna()
#     sns.histplot(returns, bins=50, kde=True)
#     plt.title(f'Distribution of Daily Returns for {ticker}')
#     plt.xlabel('Daily Return')
#     plt.ylabel('Frequency')
#     plt.legend()
#     fig = plt.gcf()  # Getting the current figure
#     plt.close()  # Close the plot to prevent it from displaying in non-Streamlit environments
#     return fig
#plt.show()


def plot_boxplot(data, ticker):
    subset = data[data['Ticker'] == ticker]
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=subset[['High', 'Low', 'Open', 'Close']])
    plt.title(f'{ticker} Price Distribution')
    plt.ylabel('Price')
    fig = plt.gcf()  # Getting the current figure
    plt.close()  # Close the plot to prevent it from displaying in non-Streamlit environments
    return fig
    #plt.show()


def plot_candlestick(data, ticker):
    ohlc = data[data['Ticker'] == ticker][['Open', 'High', 'Low', 'Close', 'Volume']]
    fig = go.Figure(data=[go.Candlestick(x=ohlc.index,
                                         open=ohlc['Open'], high=ohlc['High'],
                                         low=ohlc['Low'], close=ohlc['Close'])])
    fig.update_layout(title=f'Candlestick Chart for {ticker}', xaxis_title='Date', yaxis_title='Price')
    return fig

    #plt.show()


def plot_volatility_clustering(data, ticker):
    #calculate daily returns
    data['Daily Return'] = data[data['Ticker'] == ticker]['Close'].pct_change()
    # Check if 'Date' is in the index, use it directly; otherwise, set it as the index
    if 'Date' not in data.columns:
        # 'Date' is already the index
        dates = data[data['Ticker'] == ticker].index
        volatilities = data[data['Ticker'] == ticker]['Daily Return'].rolling(window=7).std()
    else:
        # 'Date' is a column, set it as index
        data = data.set_index('Date')
        dates = data[data['Ticker'] == ticker].index
        volatilities = data[data['Ticker'] == ticker]['Daily Return'].rolling(window=7).std()

    plt.figure(figsize=(14, 7))
    plt.plot(dates, volatilities, label='7-Day Rolling Std Dev')
    plt.title(f'Volatility Clustering in {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.legend()
    plt.grid(True)
    fig = plt.gcf()  # Getting the current figure
    plt.close()  # Close the plot to prevent it from displaying in non-Streamlit environments
    return fig
    #plt.show()


def plot_kde_of_closes(data, selected_cryptos):
    plt.figure(figsize=(14, 7))
    for ticker in selected_cryptos:
        sns.kdeplot(data[data['Ticker'] == ticker]['Close'], fill=True, color="r", label="Density")
    plt.title('KDE of Close Prices')
    plt.axvline(data[data['Ticker'] == ticker]['Close'].mean(), color='r', linestyle='--', label='Mean')
    plt.axvline(data[data['Ticker'] == ticker]['Close'].median(), color='g', linestyle='-', label='Median')
    plt.axvline(data[data['Ticker'] == ticker]['Close'].mode()[0], color='b', linestyle=':', label='Mode')
    plt.xlabel('Price')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    fig = plt.gcf()  # Getting the current figure
    plt.close()  # Close the plot to prevent it from displaying in non-Streamlit environments
    return fig


def plot_candlestick_with_signals_and_ma(data, ticker):
    # Filter for the specific ticker
    crypto_subset = data[data['Ticker'] == ticker]

    if crypto_subset.empty:
        print(f"No data available for {ticker}")
        return None

    # Create buy and sell signals based on some criteria
    buy_signals = crypto_subset['Close'] > crypto_subset['Open']  # Example condition for buying
    sell_signals = crypto_subset['Open'] > crypto_subset['Close']  # Example condition for selling

    fig = go.Figure()

    # Add candlestick plot
    fig.add_trace(go.Candlestick(
        x=crypto_subset.index,
        open=crypto_subset['Open'],
        high=crypto_subset['High'],
        low=crypto_subset['Low'],
        close=crypto_subset['Close'],
        name=f'{ticker} Candlesticks'))

    # Plot buy signals
    fig.add_trace(go.Scatter(
        x=crypto_subset.index[buy_signals],
        y=crypto_subset['Close'][buy_signals],
        mode='markers',
        marker_symbol='triangle-up',
        marker_color='green',
        marker_size=10,
        name='Buy Signal'))

    # Plot sell signals
    fig.add_trace(go.Scatter(
        x=crypto_subset.index[sell_signals],
        y=crypto_subset['Close'][sell_signals],
        mode='markers',
        marker_symbol='triangle-down',
        marker_color='red',
        marker_size=10,
        name='Sell Signal'))

    # Add moving averages
    windows = [7, 14, 30]
    colors = ['magenta', 'blue', 'black']  # Different color for each MA
    for window, color in zip(windows, colors):
        ma = crypto_subset['Close'].rolling(window=window).mean()
        fig.add_trace(go.Scatter(
            x=crypto_subset.index,
            y=ma,
            mode='lines',
            line=dict(color=color, width=2),
            name=f'{window}-Day MA'))

    # Update layout
    fig.update_layout(
        title=f'{ticker} Trading Signals and Moving Averages',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        autosize=False,
        width=1000,  # Specify the width of the figure
        height=600,  # Specify the height of the figure
        margin=dict(l=50, r=50, b=100, t=100, pad=4)
    )

    return fig
