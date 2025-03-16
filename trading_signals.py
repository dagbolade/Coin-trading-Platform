import numpy as np
import pandas as pd


def calculate_rsi(data, periods=14):
    close_prices = pd.Series(data)
    delta = close_prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=periods).mean()
    avg_loss = loss.rolling(window=periods).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def generate_lstm_trading_signals(predicted_prices_df, last_known_price, rsi_period=14, ma_short_period=50,
                                  ma_long_period=200):
    # Extract the predicted prices from the DataFrame
    predicted_prices = predicted_prices_df['Predicted_Close'].values

    # Calculate RSI on predicted prices
    predicted_prices_df['RSI'] = calculate_rsi(predicted_prices, periods=rsi_period)

    # Calculate short-term and long-term moving averages on predicted prices
    predicted_prices_df['MA_Short'] = predicted_prices_df['Predicted_Close'].rolling(window=ma_short_period).mean()
    predicted_prices_df['MA_Long'] = predicted_prices_df['Predicted_Close'].rolling(window=ma_long_period).mean()

    # Generate signals
    signals = []
    for i in range(len(predicted_prices_df)):
        if i == 0:
            signals.append(None)
        else:
            if (predicted_prices_df['RSI'].iloc[i] > 70) and (
                    predicted_prices_df['Predicted_Close'].iloc[i] < predicted_prices_df['Predicted_Close'].iloc[
                i - 1]):
                signals.append('Sell')
            elif (predicted_prices_df['RSI'].iloc[i] < 30) and (
                    predicted_prices_df['Predicted_Close'].iloc[i] > predicted_prices_df['Predicted_Close'].iloc[
                i - 1]):
                signals.append('Buy')
            elif predicted_prices_df['MA_Short'].iloc[i] > predicted_prices_df['MA_Long'].iloc[i]:
                signals.append('Buy')
            elif predicted_prices_df['MA_Short'].iloc[i] < predicted_prices_df['MA_Long'].iloc[i]:
                signals.append('Sell')
            else:
                signals.append('Hold')

    predicted_prices_df['Signal'] = signals

    return predicted_prices_df

def generate_trading_signals(predicted_prices, last_known_price, rsi_period=7, ma_short_period=10, ma_long_period=100,
                             price_threshold=0.01):
    predicted_prices_df = pd.DataFrame({'Predicted_Close': predicted_prices}, index=pd.date_range(start='today', periods=len(predicted_prices), freq='D'))
    predicted_prices_df['RSI'] = calculate_rsi(predicted_prices_df['Predicted_Close'], periods=rsi_period)
    predicted_prices_df['MA_Short'] = predicted_prices_df['Predicted_Close'].rolling(window=ma_short_period).mean()
    predicted_prices_df['MA_Long'] = predicted_prices_df['Predicted_Close'].rolling(window=ma_long_period).mean()
    predicted_prices_df['Price_Change'] = predicted_prices_df['Predicted_Close'].pct_change()

    signals = []
    for i in range(len(predicted_prices_df)):
        if i == 0:
            signals.append(None)
        else:
            if (predicted_prices_df['RSI'].iloc[i] > 70) and (
                    predicted_prices_df['Price_Change'].iloc[i] < -price_threshold):
                signals.append('Sell')
            elif (predicted_prices_df['RSI'].iloc[i] < 30) and (
                    predicted_prices_df['Price_Change'].iloc[i] > price_threshold):
                signals.append('Buy')
            elif predicted_prices_df['MA_Short'].iloc[i] > predicted_prices_df['MA_Long'].iloc[i] and \
                    predicted_prices_df['Predicted_Close'].iloc[i] > predicted_prices_df['Predicted_Close'].iloc[i - 1]:
                signals.append('Buy')
            elif predicted_prices_df['MA_Short'].iloc[i] < predicted_prices_df['MA_Long'].iloc[i] and \
                    predicted_prices_df['Predicted_Close'].iloc[i] < predicted_prices_df['Predicted_Close'].iloc[i - 1]:
                signals.append('Sell')
            elif predicted_prices_df['Predicted_Close'].iloc[i] < predicted_prices_df['Predicted_Close'].iloc[i - 1]:
                signals.append('Sell')
            else:
                signals.append('Hold')

    predicted_prices_df['Signal'] = signals
    return predicted_prices_df


# using last known price to generate signals


# using rsi and moving averages to generate signals
def generate_prophet1_trading_signals(forecast, rsi_period=14, ma_short_period=7, ma_long_period=30,
                                      price_threshold=0.01):
    # Calculate RSI on forecasted 'yhat' values
    forecast['RSI'] = calculate_rsi(forecast['yhat'], periods=rsi_period)

    # Calculate short-term and long-term moving averages on forecasted 'yhat' values
    forecast['MA_Short'] = forecast['yhat'].rolling(window=ma_short_period).mean()
    forecast['MA_Long'] = forecast['yhat'].rolling(window=ma_long_period).mean()

    # Calculate the percentage change in forecasted prices
    forecast['Price_Change'] = forecast['yhat'].pct_change()

    # Generate signals
    forecast['Signal'] = np.where(
        (forecast['RSI'] < 30) & (forecast['Price_Change'] > price_threshold) & (
                    forecast['MA_Short'] > forecast['MA_Long']), 'Buy',
        np.where(
            (forecast['RSI'] > 70) & (forecast['Price_Change'] < -price_threshold) & (
                        forecast['MA_Short'] < forecast['MA_Long']), 'Sell',
            np.where(
                forecast['yhat'] < forecast['yhat'].shift(1), 'Sell',
                'Hold'
            )
        )
    )

    return forecast


import plotly.graph_objects as go


def plot_forecast_with_signals2(forecast):
    # Create a figure
    fig = go.Figure()

    # Plot the historical forecast line
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        mode='lines',
        name='Forecast'
    ))

    # Plot buy signals
    fig.add_trace(go.Scatter(
        x=forecast[forecast['Signal'] == 'Buy']['ds'],
        y=forecast[forecast['Signal'] == 'Buy']['yhat'],
        mode='markers',
        name='Buy Signal',
        marker=dict(color='green', size=10, symbol='triangle-up')
    ))

    # Plot sell signals
    fig.add_trace(go.Scatter(
        x=forecast[forecast['Signal'] == 'Sell']['ds'],
        y=forecast[forecast['Signal'] == 'Sell']['yhat'],
        mode='markers',
        name='Sell Signal',
        marker=dict(color='red', size=10, symbol='triangle-down')
    ))

    # Plot MA Short
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['MA_Short'],
        mode='lines',
        name='MA Short',
        line=dict(color='orange', width=2)
    ))

    # Plot MA Long
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['MA_Long'],
        mode='lines',
        name='MA Long',
        line=dict(color='purple', width=2)
    ))

    # Update layout to include titles and axis labels as necessary
    fig.update_layout(
        title='Forecast with Trading Signals and Moving Averages',
        xaxis_title='Date',
        yaxis_title='Price',
        showlegend=True
    )

    return fig

def generate_arima_trading_signals(forecast_df, last_known_price, rsi_period=14, threshold=0.01):
    """
    Generate trading signals based on ARIMA forecasted prices.

    Parameters:
    forecast_df (pd.DataFrame): A DataFrame with 'Date' and 'Forecast' columns.
    last_known_price (float): The last known price of the asset.
    rsi_period (int): The period for calculating RSI.
    threshold (float): The threshold for determining signals.

    Returns:
    pd.DataFrame: The input DataFrame with additional 'RSI', 'Change', and 'Signal' columns.
    """
    # Calculate RSI on forecasted prices
    forecast_df['RSI'] = calculate_rsi(forecast_df['Forecast'], periods=rsi_period)

    # Calculate the percentage change between the forecast and the last known price
    forecast_df['Change'] = (forecast_df['Forecast'] - last_known_price) / last_known_price

    # Determine signals based on RSI, change threshold, and previous price comparison
    forecast_df['Signal'] = np.where(
        (forecast_df['RSI'] < 30) & (forecast_df['Change'] > threshold), 'Buy',
        np.where(
            (forecast_df['RSI'] > 70) & (forecast_df['Change'] < -threshold), 'Sell',
            np.where(
                forecast_df['Forecast'] < forecast_df['Forecast'].shift(1), 'Sell',
                'Hold'
            )
        )
    )

    return forecast_df



def plot_arima_forecast_with_signals(arima_signals_df):
    """
    Plot ARIMA forecast with trading signals.

    Parameters:
    arima_signals_df (pd.DataFrame): The DataFrame with 'Date', 'Forecast', and 'Signal' columns.

    Returns:
    plotly.graph_objs._figure.Figure: The plotly figure object for the forecast with signals.
    """
    # Create the figure
    fig = go.Figure()

    # Add line for the forecast
    fig.add_trace(go.Scatter(x=arima_signals_df['Date'], y=arima_signals_df['Forecast'],
                             mode='lines', name='Forecast'))

    # Add markers for buy signals
    buy_signals = arima_signals_df[arima_signals_df['Signal'] == 'Buy']
    fig.add_trace(go.Scatter(x=buy_signals['Date'], y=buy_signals['Forecast'],
                             mode='markers', name='Buy Signal',
                             marker=dict(color='green', size=10, symbol='triangle-up')))

    # Add markers for sell signals
    sell_signals = arima_signals_df[arima_signals_df['Signal'] == 'Sell']
    fig.add_trace(go.Scatter(x=sell_signals['Date'], y=sell_signals['Forecast'],
                             mode='markers', name='Sell Signal',
                             marker=dict(color='red', size=10, symbol='triangle-down')))

    # Add markers for hold signals if needed
    # hold_signals = arima_signals_df[arima_signals_df['Signal'] == 'Hold']
    # fig.add_trace(go.Scatter(x=hold_signals['Date'], y=hold_signals['Forecast'],
    #                          mode='markers', name='Hold Signal', marker=dict(color='blue', size=10, symbol='circle')))

    # Update layout for readability
    fig.update_layout(title='ARIMA Forecast with Trading Signals',
                      xaxis_title='Date', yaxis_title='Price',
                      showlegend=True)

    return fig
