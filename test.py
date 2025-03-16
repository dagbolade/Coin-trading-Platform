import pickle

import h5py
import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostRegressor
from keras._tf_keras import keras
from matplotlib import pyplot as plt
from pmdarima import auto_arima
from scipy.stats import stats, norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score
from sklearn.model_selection import train_test_split
from scipy.signal import find_peaks

from Data_downolader import CryptoDataDownloader
from arima import predict_arima, fit_arima_model, find_best_arima

from bi_lstm_model import build_bi_lstm_model, train_bi_lstm_model
from clustering import select_cryptos_closest_to_centroids, plot_clusters, add_cluster_labels, apply_kmeans
from correlation import calculate_correlation_matrix, find_top_correlations, calculate_daily_returns
from data_preprocessing import convert_to_datetime
from data_transformation import scale_data, pivot_and_fill, remove_duplicates
from dimensionality_reduction import plot_explained_variance, apply_pca
from eda import plot_time_series, plot_rolling_statistics, plot_boxplot, plot_candlestick, \
    plot_volatility_clustering, plot_kde_of_closes, plot_candlestick_with_signals_and_ma
from feature_engineering import calculate_sma_ema_rsi
from lstm_model import prepare_lstm_data, build_lstm_model, train_lstm_model
from prophet_model import prepare_data_for_prophet, train_prophet_model, plot_forecast
from trading_metrics import calculate_sharpe_ratio, calculate_max_drawdown, calculate_sortino_ratio
from trading_signals import generate_trading_signals, \
    generate_prophet1_trading_signals, plot_forecast_with_signals2, generate_arima_trading_signals, \
    plot_arima_forecast_with_signals, generate_lstm_trading_signals

from xgboost_model import train_xgboost_model, preprocess_data, \
    forecast_xgboost

# Set up the sidebar menu
menu_options = [
    "Overview", "About", "Data Preprocessing", "Exploratory Data Analysis",
    "Correlation Analysis", "Prediction", "Highest Return Prediction", "Trading Strategy", "News"
]
menu_choice = st.sidebar.radio("Menu", menu_options)


def show_overview():
    st.header("Overview")
    st.write("Welcome to the Cryptocurrency Analysis Dashboard!")


def show_about():
    st.header("About")
    st.write("This section provides information about the app and its functionalities.")


def data_preprocessing():
    st.header("Data Preprocessing")
    # Include all your data preprocessing code here
    st.title('Cryptocurrency Data Downloader')

    # Cache the data downloader function to prevent unnecessary downloads
    @st.cache_data
    def download_data(tickers, start_date, end_date, interval):
        downloader = CryptoDataDownloader(tickers, start_date=start_date, end_date=end_date, interval=interval)
        return downloader.download_data()

    # Sidebar for user inputs
    st.sidebar.header('Download Settings')
    all_cryptos = [
        'BNB-USD', 'ADA-USD', 'XRP-USD', 'DOGE-USD', 'DOT-USD', 'UNI-USD', 'BTC-USD',
        'BCH-USD', 'LTC-USD', 'BTC-USD', 'ETH-USD', 'LINK-USD', 'XLM-USD', 'USDC-USD', 'USDT-USD', 'VET-USD', 'ETC-USD',
        'FIL-USD', 'TRX-USD', 'EOS-USD', 'THETA-USD', 'XMR-USD', 'NEO-USD', 'AAVE-USD', 'ATOM-USD', 'WIF-USD', 'BONK'
                                                                                                               '-USD',
        'SHIB-USD', 'SOL-USD',
        'DGB-USD', 'CHZ-USD', 'ENJ-USD', 'MANA-USD', 'BAT-USD', 'SAND-USD'
    ]

    # Use top_30_cryptos as default in multiselect
    selected_tickers = st.sidebar.multiselect(
        'Select Cryptocurrencies, 30 cryptos have been added by default',
        options=all_cryptos,
        default=all_cryptos
    )

    start_date = st.sidebar.date_input('Start Date', value=pd.to_datetime('2022-01-01'))
    end_date = st.sidebar.date_input('End Date', value=pd.to_datetime('today'))
    interval = st.sidebar.selectbox('Select Interval', options=['1d', '1wk', '1mo'], index=0)
    download_button = st.sidebar.button('Download Data')

    # Initiate data downloading
    if download_button and selected_tickers:
        with st.spinner('Downloading data...'):
            st.session_state.crypto_data = download_data(selected_tickers, start_date, end_date, interval)
            st.success('Data downloaded successfully!')
            st.write("Data Sample:")
            st.dataframe(st.session_state.crypto_data.head())
    if 'crypto_data' in st.session_state and not st.session_state.crypto_data.empty:
        st.header("Data Preprocessing and Transformation")

        # If 'Date' is in the index, reset index to make 'Date' a column
        if 'Date' in st.session_state.crypto_data.index.names:
            st.session_state.crypto_data.reset_index(inplace=True)
            st.write("'Date' column reset from index.")

        # Check again after resetting index
        if 'Date' not in st.session_state.crypto_data.columns:
            st.error("'Date' column is missing from the data after reset. Please check the data source.")
            st.stop()

        st.session_state.crypto_data = remove_duplicates(st.session_state.crypto_data, subset=['Ticker', 'Date'])
        st.write("Duplicates removed.")

        processed_data = pivot_and_fill(st.session_state.crypto_data, index='Ticker', columns='Date', values='Close')
        st.write("Data pivoted and filled.")
        st.dataframe(processed_data.head())
        scaled_data = scale_data(processed_data)
        st.write("Data scaled.")
        reduced_data_df, pca = apply_pca(scaled_data, n_components=10)
        st.write("PCA applied with 10 components.")
        st.dataframe(reduced_data_df.head())

        # Display explained variance plot
        fig = plot_explained_variance(pca, n_components=10)
        st.pyplot(fig)

        # Perform clustering
        clusters, kmeans = apply_kmeans(reduced_data_df, n_clusters=4)
        processed_data = add_cluster_labels(processed_data, clusters)
        st.write("K-means clustering applied with 4 clusters.")
        st.dataframe(processed_data.head())

        # Show the counts of cryptocurrencies in each cluster
        st.write("Cluster Distribution:")
        st.write(processed_data['Cluster'].value_counts())

        # Visualize clusters
        fig = plot_clusters(reduced_data_df.values, clusters)
        st.pyplot(fig)

        # Automatically select cryptos closest to centroids
        selected_cryptos = select_cryptos_closest_to_centroids(reduced_data_df, clusters, kmeans.cluster_centers_)
        st.write("Cryptocurrencies closest to each cluster centroid:")
        st.dataframe(selected_cryptos)

        # Convert 'Date' column to datetime and set as index
        st.session_state.crypto_data = convert_to_datetime(st.session_state.crypto_data, 'Date').set_index('Date')

        # Drop every other column on selected cryptos, except the Ticker and cluster
        selected_cryptos = selected_cryptos[['Cluster']]
        st.write("Selected Cryptos:")
        st.write(selected_cryptos)

        # Merge the selected cryptocurrencies with the original data to get their full details including Date
        selected_cryptos_full = st.session_state.crypto_data.merge(selected_cryptos, left_on='Ticker', right_index=True)

        # Make sure the data is sorted by Date and its a column not an index
        selected_cryptos_full = selected_cryptos_full.reset_index().sort_values('Date')
        st.write("Selected Cryptos Full Details:")
        st.write(selected_cryptos_full)

        # Calculate daily returns for all cryptocurrencies
        daily_returns = calculate_daily_returns(st.session_state.crypto_data)

        # Apply the feature engineering to the selected cryptos
        selected_cryptos_full = calculate_sma_ema_rsi(selected_cryptos_full)
        st.write("Selected Cryptos Full Details with Feature Engineering:")
        st.write(selected_cryptos_full)

        # covert to datetime and set as index in selected cryptos full
        selected_cryptos_full = convert_to_datetime(selected_cryptos_full, 'Date').set_index('Date')
        st.write("Selected Cryptos Full Details with Feature Engineering:")
        st.write(selected_cryptos_full)

        # Store in session state
        st.session_state['selected_cryptos'] = selected_cryptos
        st.session_state['daily_returns'] = daily_returns
        st.session_state['selected_cryptos_full'] = selected_cryptos_full

        st.write("Preprocessing completed. Data stored in session state.")
    else:
        st.info("Please download data to proceed with transformation and analysis.")


def exploratory_data_analysis():
    st.header("Exploratory Data Analysis")
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # Ensure the required data is available in the session state
    if 'selected_cryptos_full' in st.session_state:
        selected_cryptos_full = st.session_state['selected_cryptos_full']

        # Convert 'Date' if it's not already the index
        if 'Date' in selected_cryptos_full.columns:
            selected_cryptos_full['Date'] = pd.to_datetime(selected_cryptos_full['Date'])
            selected_cryptos_full.set_index('Date', inplace=True)

        # Create a dropdown menu to select a cryptocurrency
        unique_tickers = selected_cryptos_full['Ticker'].unique()
        selected_ticker = st.selectbox('Select a Cryptocurrency', unique_tickers)

        # Filter data for the selected cryptocurrency
        ticker_data = selected_cryptos_full[selected_cryptos_full['Ticker'] == selected_ticker]

        # Display plots for the selected cryptocurrency
        if not ticker_data.empty:
            # describe the data
            st.write(f"Descriptive statistics for {selected_ticker}:")
            st.write(ticker_data.describe())

            fig = plot_time_series(ticker_data, selected_ticker)
            st.pyplot(fig)

            fig = plot_rolling_statistics(ticker_data, selected_ticker)
            st.pyplot(fig)

            # fig = plot_distribution(ticker_data, selected_ticker)
            # st.pyplot(fig)

            fig = plot_boxplot(ticker_data, selected_ticker)
            st.pyplot(fig)

            fig = plot_candlestick(ticker_data, selected_ticker)
            st.plotly_chart(fig)

            fig = plot_volatility_clustering(ticker_data, selected_ticker)
            st.pyplot(fig)

            fig = plot_kde_of_closes(ticker_data, [selected_ticker])
            st.pyplot(fig)

            fig = plot_candlestick_with_signals_and_ma(ticker_data, selected_ticker)
            st.plotly_chart(fig)
        else:
            st.error("No data available for the selected cryptocurrency.")
    else:
        st.info("Please perform data preprocessing to generate and select cryptocurrencies for analysis.")


def correlation_analysis():
    if 'selected_cryptos' in st.session_state and 'daily_returns' in st.session_state:
        selected_cryptos = st.session_state['selected_cryptos']
        daily_returns = st.session_state['daily_returns']

        st.write("Selected Cryptocurrencies:", selected_cryptos)
        st.write("Daily Returns Sample:", daily_returns.head())

        # Perform correlation analysis using these
        correlation_matrix = calculate_correlation_matrix(daily_returns)
        top_correlations = find_top_correlations(correlation_matrix, selected_cryptos.index.tolist())

        # Display top correlations
        for ticker, correlation_data in top_correlations.items():
            st.write(f"Top correlations for {ticker}")
            st.write("Positive Correlations:", correlation_data['Positive'])
            st.write("Negative Correlations:", correlation_data['Negative'])
    else:
        st.error("Data not available. Please run the preprocessing first.")


import plotly.graph_objs as go


# Feature engineering with lagged features
def create_lagged_features(data, lag_periods):
    lagged_data = data.copy()
    for lag in lag_periods:
        lagged_data[f'Close_lag_{lag}'] = lagged_data['Close'].shift(lag)
    lagged_data.dropna(inplace=True)
    return lagged_data


def buy_sell_analysis(selected_tickers, predicted_prices, days_to_predict, investment_amount):
    for ticker in selected_tickers:
        st.subheader(f"Predicted Prices for {ticker}")
        predicted_prices_df = pd.DataFrame({"Date": pd.date_range(start=pd.Timestamp.today(), periods=days_to_predict),
                                            "Predicted_Price": predicted_prices})
        st.write(predicted_prices_df)

        # Calculate potential profit or loss
        buy_price = predicted_prices_df["Predicted_Price"].min()
        sell_price = predicted_prices_df["Predicted_Price"].max()
        potential_profit = (sell_price - buy_price) * (investment_amount / buy_price)
        potential_loss = (buy_price - sell_price) * (investment_amount / buy_price)

        st.write(f"Investing ${investment_amount:.2f} in {ticker}:")
        if potential_profit > 0:
            st.write(f"Potential Profit: ${potential_profit:.2f}")
            st.write(
                f"Best Time to Buy: {predicted_prices_df[predicted_prices_df['Predicted_Price'] == buy_price]['Date'].values[0]}")
        else:
            st.write(f"Potential Loss: ${potential_loss:.2f}")
            st.write(
                f"Best Time to Sell: {predicted_prices_df[predicted_prices_df['Predicted_Price'] == sell_price]['Date'].values[0]}")

        st.write("---")

    st.success("Analysis Complete!")


def prediction():
    st.header("Prediction")

    if 'selected_cryptos_full' in st.session_state:
        selected_cryptos_full = st.session_state['selected_cryptos_full']

        # Allowing user to select a model
        model_choice = st.selectbox('Select Prediction Model',
                                    ['LSTM', 'Prophet', 'BI-LSTM', 'ARIMA', 'RandomForest', 'CatBoost'])

        # User input for selecting cryptocurrency
        ticker = st.selectbox('Select a Cryptocurrency for Prediction:', selected_cryptos_full['Ticker'].unique())
        days_to_predict = st.number_input('Enter the number of days to predict:', min_value=1, max_value=365, value=30)

        # Investment amount input
        investment_amount = st.number_input("Enter your investment amount (e.g., $100):", min_value=1.0, value=100.0,
                                            step=1.0)

        # Filter data for the selected cryptocurrency
        crypto_data = selected_cryptos_full[selected_cryptos_full['Ticker'] == ticker]

        if st.button('Predict'):
            with st.spinner(f'Training {model_choice} model and making predictions...'):
                if model_choice == 'LSTM':

                    X, y, scaler = prepare_lstm_data(crypto_data, 'Close', sequence_length=60)
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

                    # reset index to make 'Date' a column
                    crypto_data = crypto_data.reset_index()

                    ls_model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
                    ls_model, history = train_lstm_model(ls_model, X_train, y_train, X_test, y_test)

                    # reset index to make 'Date' a column
                    crypto_data = crypto_data.reset_index()
                    # Ensure the crypto_data index is a DatetimeIndex
                    crypto_data.index = pd.to_datetime(crypto_data['Date'])

                    # Predict using the last available data to generate predictions for multiple days ahead
                    predictions = []
                    current_sequence = X_test[-1].reshape(1, X_test.shape[1], X_test.shape[2])

                    for _ in range(days_to_predict):  # Number of days you want to predict
                        # Predict the next step and get the last predicted price
                        current_prediction = ls_model.predict(current_sequence)
                        last_predicted_price = scaler.inverse_transform(current_prediction).flatten()[0]
                        predictions.append(last_predicted_price)

                        # Update the sequence for the next prediction
                        current_sequence = np.roll(current_sequence, -1, axis=1)
                        current_sequence[0, -1, :] = current_prediction.flatten()

                    predicted_prices = np.array(predictions)
                    st.session_state['lstm_model'] = predicted_prices

                    # Get the last known price from the historical data
                    last_known_price = scaler.inverse_transform(X_test[-1:, -1, 0].reshape(-1, 1)).flatten()[0]

                    # Calculate the prediction dates starting from the day after the last date in the index
                    prediction_dates = pd.date_range(start=crypto_data['Date'].iloc[-1] + pd.Timedelta(days=1),
                                                     periods=days_to_predict, freq='D')

                    # Generate a DataFrame with predictions and the corresponding dates
                    predicted_prices_df = pd.DataFrame({'Predicted_Close': predicted_prices}, index=prediction_dates)

                    # Generate trading signals based on the predicted prices
                    predicted_signals = generate_lstm_trading_signals(predicted_prices_df, last_known_price)

                    # Concatenate historical close prices with predicted prices
                    combined_data = pd.concat([crypto_data[['Close']], predicted_prices_df['Predicted_Close']])

                    # Display predicted prices and signals
                    st.write(f"Predicted prices for the next {days_to_predict} days with LSTM:")
                    st.dataframe(predicted_signals[['Predicted_Close', 'Signal']])

                    # Plot the historical and predicted prices
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=crypto_data['Date'], y=crypto_data['Close'], mode='lines',
                                             name='Historical Close'))
                    fig.add_trace(
                        go.Scatter(x=prediction_dates, y=predicted_prices, mode='lines', name='Predicted Close'))
                    fig.update_layout(title=f"{ticker} - Historical vs Predicted Close Prices",
                                      xaxis_title='Date', yaxis_title='Price')
                    st.plotly_chart(fig, use_container_width=True)

                    # show trading signals on the plot
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=crypto_data['Date'], y=crypto_data['Close'], mode='lines',
                                             name='Historical Close'))
                    fig.add_trace(
                        go.Scatter(x=prediction_dates, y=predicted_prices, mode='lines', name='Predicted Close'))
                    fig.add_trace(
                        go.Scatter(x=prediction_dates, y=predicted_prices, mode='markers', name='Predicted Close',
                                   marker=dict(color=predicted_signals['Signal'].map(
                                       {'Buy': 'green', 'Sell': 'red', 'Hold': 'blue'}))))
                    fig.add_hline(y=predicted_prices.max(), line_dash="dot", line_color="green",
                                  annotation_text=f"Predicted High: ${predicted_prices.max():.2f}",
                                  annotation_position="bottom right")
                    fig.add_hline(y=predicted_prices.min(), line_dash="dot", line_color="red",
                                  annotation_text=f"Predicted Low: ${predicted_prices.min():.2f}",
                                  annotation_position="top right")

                    fig.update_layout(title=f"{ticker} - Historical vs Predicted Close Prices",
                                      xaxis_title='Date', yaxis_title='Price')
                    st.plotly_chart(fig, use_container_width=True)

                    #  display the last day od the prediction only
                    st.write(f"The predicted price for day {days_to_predict} is ${predicted_prices[-1]:.2f}")

                    # display the predicted high and low as output
                    st.write(f"Predicted High and Low for day {days_to_predict}:")
                    st.write(f"Predicted High: ${predicted_prices.max():.2f}")
                    st.write(f"Predicted Low: ${predicted_prices.min():.2f}")

                    # determine the signals and give advice
                    if predicted_signals['Signal'].str.contains('Buy').any():
                        st.success(f"🟢 Buy Advice: Consider buying {ticker}.")
                    elif predicted_signals['Signal'].str.contains('Sell').any():
                        st.error(f"🔴 Sell Advice: Consider selling {ticker}.")
                    else:
                        st.info(f"🔵 Hold Advice: Maintain your position in {ticker}.")

                    # store prediced prices in session state and predicted signals
                    st.session_state['predicted_prices'] = predicted_prices
                    st.session_state['predicted_signals'] = predicted_signals

                    buy_sell_analysis([ticker], predicted_prices, days_to_predict, investment_amount)
                    # Save the LSTM model to a pickle file
                    ls_model.save('lstm_model.h5', overwrite=True)


                elif model_choice == 'Prophet':

                    df_prophet = prepare_data_for_prophet(crypto_data, ticker)
                    model = train_prophet_model(df_prophet)
                    future = model.make_future_dataframe(periods=days_to_predict)
                    forecast = model.predict(future)
                    predicted_prices = forecast['yhat'].values[-days_to_predict:]

                    # Get the last known price from historical data
                    last_known_price = df_prophet['y'].iloc[-1]

                    # save the forecast in session state
                    st.session_state['prophet_model'] = model

                    signals = generate_prophet1_trading_signals(forecast)

                    # Display the forecast and signals
                    st.write("Prophet forecast and trading signals for the next {} days:".format(days_to_predict))
                    st.write(forecast[['ds', 'yhat']].tail(days_to_predict))
                    st.write(signals[['ds', 'RSI', 'MA_Short', 'MA_Long', 'Signal']].tail(days_to_predict))

                    # Optionally, plot the results
                    fig = plot_forecast_with_signals2(signals)
                    st.info("""
                        **Trading Signals Explained:**

                        - **Buy Signal**: Indicated by the Short Moving Average (MA Short) crossing above the Long Moving Average (MA Long), suggesting the asset may be entering an uptrend.

                        - **Sell Signal**: Given when the MA Short crosses below the MA Long, indicating a potential downtrend.

                        - **Hold**: No crossovers and the price is relatively stable within the averages, suggesting to maintain the current position without making new trades.

                        _Note: These signals are used in technical analysis but do not guarantee future performance and should not be the only factor considered when making investment decisions.
                        So in other words, this is not financial advice. Do your own research before investing.
                    """)

                    st.plotly_chart(fig, use_container_width=True)

                    # display the last day od the prediction only
                    st.write(f"The predicted price for day {days_to_predict} is ${forecast['yhat'].iloc[-1]:.2f}")

                    # Determine the latest signal and give trading advice
                    latest_signal = signals['Signal'].iloc[-1]
                    if latest_signal == 'Buy':
                        st.success(f"🟢 Buy Advice: Consider buying {ticker}.")
                    elif latest_signal == 'Sell':
                        st.error(f"🔴 Sell Advice: Consider selling {ticker}.")
                    else:
                        st.info(f"🔵 Hold Advice: Maintain your position in {ticker}.")
                    # save the forecast in session state and the signals
                    st.session_state['forecast'] = forecast
                    st.session_state['signals'] = signals

                    buy_sell_analysis([ticker], predicted_prices, days_to_predict, investment_amount)

                    # Save the Prophet model to a pickle file
                    with open('prophet_model.pkl', 'wb') as file:
                        pickle.dump(model, file)

                elif model_choice == 'BI-LSTM':

                    X, y, scaler = prepare_lstm_data(crypto_data, 'Close', sequence_length=60)
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

                    bi_model = build_bi_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
                    bi_model, history = train_bi_lstm_model(bi_model, X_train, y_train, X_test, y_test)

                    # reset index to make 'Date' a column
                    crypto_data = crypto_data.reset_index()
                    # Ensure the crypto_data index is a DatetimeIndex
                    crypto_data.index = pd.to_datetime(crypto_data['Date'])

                    # Predict using the last available data to generate predictions for multiple days ahead
                    predictions = []
                    current_sequence = X_test[-1:]

                    for _ in range(days_to_predict):  # Number of days you want to predict
                        # Predict the next step and get the last predicted price
                        current_prediction = bi_model.predict(current_sequence)
                        last_predicted_price = scaler.inverse_transform(current_prediction).flatten()[0]
                        predictions.append(last_predicted_price)

                        # Update the sequence with the last predicted price
                        current_sequence = np.roll(current_sequence, -1)
                        current_sequence[:, -1, :] = current_prediction

                    predicted_prices = np.array(predictions)


                    # Get the last known price from the historical data
                    last_known_price = scaler.inverse_transform(X_test[-1:, -1, 0].reshape(-1, 1)).flatten()[0]

                    # Calculate the prediction dates starting from the day after the last date in the index
                    prediction_dates = pd.date_range(start=crypto_data['Date'].iloc[-1] + pd.Timedelta(days=1),
                                                     periods=days_to_predict, freq='D')

                    # Generate a DataFrame with predictions and the corresponding dates
                    predicted_prices_df = pd.DataFrame({'Predicted_Close': predicted_prices}, index=prediction_dates)

                    # Generate trading signals based on the predicted prices
                    predicted_signals = generate_lstm_trading_signals(predicted_prices_df, last_known_price)

                    # Concatenate historical close prices with predicted prices
                    combined_data = pd.concat([crypto_data[['Close']], predicted_prices_df['Predicted_Close']])

                    # Display predicted prices and signals
                    st.write(f"Predicted prices for the next {days_to_predict} days with Bi-LSTM:")
                    st.dataframe(predicted_signals[['Predicted_Close', 'Signal']])

                    # Plot the historical and predicted prices
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=crypto_data['Date'], y=crypto_data['Close'], mode='lines',
                                             name='Historical Close'))
                    fig.add_trace(
                        go.Scatter(x=prediction_dates, y=predicted_prices, mode='lines', name='Predicted Close'))
                    fig.update_layout(title=f"{ticker} - Historical vs Predicted Close Prices",
                                      xaxis_title='Date', yaxis_title='Price')
                    st.plotly_chart(fig, use_container_width=True)

                    # show trading signals on the plot
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=crypto_data['Date'], y=crypto_data['Close'], mode='lines',
                                             name='Historical Close'))
                    fig.add_trace(
                        go.Scatter(x=prediction_dates, y=predicted_prices, mode='lines', name='Predicted Close'))
                    fig.add_trace(
                        go.Scatter(x=prediction_dates, y=predicted_prices, mode='markers', name='Predicted Close',
                                   marker=dict(color=predicted_signals['Signal'].map(
                                       {'Buy': 'green', 'Sell': 'red', 'Hold': 'blue'}))))
                    fig.add_hline(y=predicted_prices.max(), line_dash="dot", line_color="green",
                                  annotation_text=f"Predicted High: ${predicted_prices.max():.2f}",
                                  annotation_position="bottom right")
                    fig.add_hline(y=predicted_prices.min(), line_dash="dot", line_color="red",
                                  annotation_text=f"Predicted Low: ${predicted_prices.min():.2f}",
                                  annotation_position="top right")

                    fig.update_layout(title=f"{ticker} - Historical vs Predicted Close Prices",
                                      xaxis_title='Date', yaxis_title='Price')
                    st.plotly_chart(fig, use_container_width=True)

                    #  display the last day od the prediction only
                    st.write(f"The predicted price for day {days_to_predict} is ${predicted_prices[-1]:.2f}")

                    # display the predicted high and low as output
                    st.write(f"Predicted High and Low for day {days_to_predict}:")
                    st.write(f"Predicted High: ${predicted_prices.max():.2f}")
                    st.write(f"Predicted Low: ${predicted_prices.min():.2f}")

                    # determine the signals and give advice
                    if predicted_signals['Signal'].str.contains('Buy').any():
                        st.success(f"🟢 Buy Advice: Consider buying {ticker}.")
                    elif predicted_signals['Signal'].str.contains('Sell').any():
                        st.error(f"🔴 Sell Advice: Consider selling {ticker}.")
                    else:
                        st.info(f"🔵 Hold Advice: Maintain your position in {ticker}.")

                    buy_sell_analysis([ticker], predicted_prices, days_to_predict, investment_amount)
                    # Save the Bi-LSTM model to a pickle file
                    bi_model.save('bi_lstm_model.h5', overwrite=True)

                elif model_choice == 'ARIMA':
                    # reset index to make 'Date' a column
                    time_series_data = crypto_data.reset_index()

                    # Ensure the time series is indexed properly and is a Series
                    time_series_data.index = pd.to_datetime(time_series_data['Date'])
                    time_series_data = time_series_data['Close']
                    print(time_series_data.head())

                    # Ensure the time series is indexed properly and is a Series
                    time_series_data.index = pd.to_datetime(time_series_data.index)

                    # Find the best ARIMA model
                    auto_model = find_best_arima(time_series_data, seasonal=False)

                    # Fit the ARIMA model
                    model_fit = fit_arima_model(time_series_data, auto_model.order)

                    # Predict future prices
                    forecast = predict_arima(model_fit, days_to_predict)
                    # Generate forecast index starting from the last known date in the series
                    last_known_date = time_series_data.index.max()
                    forecast_index = pd.date_range(start=last_known_date + pd.Timedelta(days=1),
                                                   periods=days_to_predict, freq='D')
                    # Create forecast DataFrame
                    forecast_df = pd.DataFrame({'Date': forecast_index, 'Forecast': forecast})

                    # save the forecast in session state
                    st.session_state['arima_model'] = model_fit

                    # Use the last known price for generating signals
                    last_known_price = time_series_data.iloc[-1]

                    # Generate ARIMA trading signals
                    arima_signals_df = generate_arima_trading_signals(forecast_df, last_known_price)

                    # Display the predicted prices and signals
                    st.write(f"ARIMA forecast and signals for the next {days_to_predict} days:")
                    st.dataframe(arima_signals_df[['Date', 'Forecast', 'Signal']])

                    # Plot the forecast with signals
                    fig = plot_arima_forecast_with_signals(arima_signals_df)

                    st.plotly_chart(fig, use_container_width=True)

                    # display the last day od the prediction only
                    st.write(f"The predicted price for day {days_to_predict} is ${forecast[-1]:.2f}")

                    # Determine the latest signal and give trading advice
                    latest_signal = arima_signals_df['Signal'].iloc[-1]
                    if latest_signal == 'Buy':
                        st.success(f"🟢 Buy Advice: Consider buying {ticker}.")
                    elif latest_signal == 'Sell':
                        st.error(f"🔴 Sell Advice: Consider selling {ticker}.")
                    else:
                        st.info(f"🔵 Hold Advice: Maintain your position in {ticker}.")

                    # display a predictive time for possible high and low of a cryptocurrency
                    st.write(f"Predicted High and Low for day {days_to_predict}:")
                    st.write(f"Predicted High: ${forecast_df['Forecast'].max():.2f}")
                    st.write(f"Predicted Low: ${forecast_df['Forecast'].min():.2f}")

                    # show the predicted high and low on the plot
                    fig.add_hline(y=forecast_df['Forecast'].max(), line_dash="dot", line_color="green",
                                  annotation_text=f"Predicted High: ${forecast_df['Forecast'].max():.2f}",
                                  annotation_position="bottom right")
                    fig.add_hline(y=forecast_df['Forecast'].min(), line_dash="dot", line_color="red",
                                  annotation_text=f"Predicted Low: ${forecast_df['Forecast'].min():.2f}",
                                  annotation_position="top right")
                    st.plotly_chart(fig, use_container_width=True)

                    predicted_prices = forecast_df['Forecast'].values

                    buy_sell_analysis([ticker], predicted_prices, days_to_predict, investment_amount)

                    # Save the ARIMA model to a pickle file
                    with open('arima_model.pkl', 'wb') as file:
                        pickle.dump(model_fit, file)


                elif model_choice == 'RandomForest':
                    from sklearn.ensemble import RandomForestRegressor

                    features = ['Open', 'High', 'Low', 'Volume', 'SMA', 'EMA', 'RSI']
                    target = 'Close'
                    preprocessed_data = selected_cryptos_full[selected_cryptos_full['Ticker'] == ticker]

                    # Create lagged features
                    lag_periods = [1, 2, 3, 4, 5]  # Adjust the lag periods as needed
                    lagged_data = create_lagged_features(preprocessed_data, lag_periods)

                    # Update the features list to include lagged features
                    features += [f'Close_lag_{lag}' for lag in lag_periods]

                    X = lagged_data[features]
                    y = lagged_data[target]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

                    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                    rf_model.fit(X_train, y_train)

                    future_data = lagged_data[features].tail(days_to_predict)
                    future_predictions = rf_model.predict(future_data)

                    # Display the predicted prices for the next days
                    st.write(f"Predicted prices for the next {days_to_predict} days with Random Forest:")
                    for i, price in enumerate(future_predictions, start=1):
                        st.write(f"Day {i}: {price:.2f}")

                    predicted_prices = np.array(future_predictions)
                    # Reset the index and rename the 'index' column to 'Date'
                    preprocessed_data.reset_index(inplace=True)
                    preprocessed_data.rename(columns={'index': 'Date'}, inplace=True)
                    # Calculate the prediction dates starting from the day after the last date in the data
                    last_date = preprocessed_data['Date'].iloc[-1]
                    prediction_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_to_predict)
                    # Create a DataFrame with the predicted prices and corresponding dates
                    predicted_prices_df = pd.DataFrame({'Date': prediction_dates, 'Predicted_Price': predicted_prices})
                    # Display the predicted prices DataFrame
                    st.write("Predicted Prices:")
                    st.dataframe(predicted_prices_df)
                    # Rename the 'Predicted_Price' column to 'Predicted_Close'
                    predicted_prices_df = predicted_prices_df.rename(columns={'Predicted_Price': 'Predicted_Close'})
                    # Generate trading signals based on the predicted prices
                    trading_signals = generate_trading_signals(predicted_prices, preprocessed_data[target].iloc[-1])
                    # Display the trading signals
                    st.write("Trading Signals:")
                    st.dataframe(trading_signals)
                    # Plot the predicted prices
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(x=preprocessed_data['Date'], y=preprocessed_data[target], name='Actual Price'))
                    fig.add_trace(go.Scatter(x=prediction_dates, y=predicted_prices, name='Predicted Price'))
                    fig.update_layout(title=f"{ticker} - Actual vs Predicted Prices", xaxis_title="Date",
                                      yaxis_title="Price")
                    st.plotly_chart(fig)
                    # Display the predicted high and low prices
                    st.write(f"Predicted High Price: {predicted_prices.max():.2f}")
                    st.write(f"Predicted Low Price: {predicted_prices.min():.2f}")
                    # Provide investment advice based on the predicted prices and trading signals
                    if trading_signals['Signal'].iloc[-1] == 'Buy':
                        st.success(
                            f"Investment Advice: Consider buying {ticker} based on the predicted price increase.")
                    elif trading_signals['Signal'].iloc[-1] == 'Sell':
                        st.warning(
                            f"Investment Advice: Consider selling {ticker} based on the predicted price decrease.")
                    else:
                        st.info(f"Investment Advice: Hold {ticker} as no significant price change is predicted.")

                    buy_sell_analysis([ticker], predicted_prices, days_to_predict, investment_amount)

                    # Save the Random Forest model to a pickle file
                    with open('random_forest_model.pkl', 'wb') as file:
                        pickle.dump(rf_model, file)


                elif model_choice == 'CatBoost':
                    from catboost import CatBoostRegressor
                    features = ['Open', 'High', 'Low', 'Volume', 'SMA', 'EMA', 'RSI']
                    target = 'Close'
                    preprocessed_data = selected_cryptos_full[selected_cryptos_full['Ticker'] == ticker]
                    # Create lagged features
                    lag_periods = [1, 2, 3, 4, 5]  # Adjust the lag periods as needed
                    lagged_data = create_lagged_features(preprocessed_data, lag_periods)

                    # Update the features list to include lagged features
                    features += [f'Close_lag_{lag}' for lag in lag_periods]

                    X = lagged_data[features]
                    y = lagged_data[target]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

                    cat_model = CatBoostRegressor(iterations=100, learning_rate=0.1, random_seed=42)
                    cat_model.fit(X_train, y_train)

                    future_data = lagged_data[features].tail(days_to_predict)
                    future_predictions = cat_model.predict(future_data)

                    # Display the predicted prices for the next days
                    st.write(f"Predicted prices for the next {days_to_predict} days with CatBoost:")
                    for i, price in enumerate(future_predictions, start=1):
                        st.write(f"Day {i}: {price:.2f}")

                    predicted_prices = np.array(future_predictions)
                    # Reset the index and rename the 'index' column to 'Date'
                    preprocessed_data.reset_index(inplace=True)
                    preprocessed_data.rename(columns={'index': 'Date'}, inplace=True)

                    preprocessed_data['Date'] = pd.to_datetime(preprocessed_data['Date'])

                    # Calculate the prediction dates starting from the day after the last date in the data
                    last_date = preprocessed_data['Date'].iloc[-1]
                    prediction_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_to_predict)
                    # Create a DataFrame with the predicted prices and corresponding dates
                    predicted_prices_df = pd.DataFrame({'Date': prediction_dates, 'Predicted_Price': predicted_prices})
                    # Display the predicted prices DataFrame
                    st.write("Predicted Prices:")
                    st.dataframe(predicted_prices_df)
                    # Rename the 'Predicted_Price' column to 'Predicted_Close'
                    predicted_prices_df = predicted_prices_df.rename(columns={'Predicted_Price': 'Predicted_Close'})
                    # Generate trading signals based on the predicted prices
                    trading_signals = generate_trading_signals(predicted_prices, preprocessed_data[target].iloc[-1])
                    # Display the trading signals
                    st.write("Trading Signals:")
                    st.dataframe(trading_signals)
                    # Plot the predicted prices
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(x=preprocessed_data['Date'], y=preprocessed_data[target], name='Actual Price'))
                    fig.add_trace(go.Scatter(x=prediction_dates, y=predicted_prices, name='Predicted Price'))
                    fig.update_layout(title=f"{ticker} - Actual vs Predicted Prices", xaxis_title="Date",
                                      yaxis_title="Price")
                    st.plotly_chart(fig)
                    # Display the predicted high and low prices
                    st.write(f"Predicted High Price: {predicted_prices.max():.2f}")
                    st.write(f"Predicted Low Price: {predicted_prices.min():.2f}")
                    # Provide investment advice based on the predicted prices and trading signals
                    if trading_signals['Signal'].iloc[-1] == 'Buy':
                        st.success(
                            f"Investment Advice: Consider buying {ticker} based on the predicted price increase.")
                    elif trading_signals['Signal'].iloc[-1] == 'Sell':
                        st.warning(
                            f"Investment Advice: Consider selling {ticker} based on the predicted price decrease.")
                    else:
                        st.info(f"Investment Advice: Hold {ticker} as no significant price change is predicted.")

                    # Call the best_time_to_buy function
                    buy_sell_analysis([ticker], predicted_prices, days_to_predict, investment_amount)

                    # Save the CatBoost model to a pickle file
                    with open('catboost_model.pkl', 'wb') as file:
                        pickle.dump(cat_model, file)


    else:
        st.error("Please ensure the cryptocurrency data is loaded and preprocessed.")


# News

import requests

# Set your API key here
API_KEY = 'pub_3971050fd96dbc2e856ed4aa8aeaa79c531bb'


def fetch_news(api_key, query):
    url = f'https://newsdata.io/api/1/news?apikey={api_key}&q={query}&language=en'
    response = requests.get(url)
    if response.status_code == 200:
        news_data = response.json()
        return news_data['results'] if 'results' in news_data else []
    else:
        st.error(f'Failed to fetch news: {response.status_code}')
        return []


def display_news(articles):
    if articles:
        for article in articles:
            st.subheader(article['title'] if 'title' in article else 'No Title')
            st.write(article['content'] if 'content' in article else 'No Content Available')
            if 'link' in article and article['link']:
                st.markdown(f"[Read more]({article['link']})", unsafe_allow_html=True)
            st.write("---")  # For a separator line
    else:
        st.error('No news articles found.')


def news_page(api_key):
    st.info("This page allows you to search for news articles related to cryptocurrencies.")
    search_query = st.text_input("Search for news on any topic:")
    search_button = st.button('Search')

    if search_button and search_query:
        articles = fetch_news(api_key, search_query)
        display_news(articles)
    elif search_button:
        st.error("Please enter a search term.")

    # Check if selected cryptos are available
    if 'selected_cryptos_full' in st.session_state:
        selected_cryptos_full = st.session_state['selected_cryptos_full']
        unique_tickers = selected_cryptos_full['Ticker'].unique()
        st.info("You can also select a cryptocurrency to fetch news related to it.")
        ticker = st.selectbox('Or select a Cryptocurrency for news:', unique_tickers, index=0)
        fetch_button = st.button(f"Fetch news for {ticker}")

        if fetch_button:
            articles = fetch_news(api_key, ticker)
            display_news(articles)
    else:
        st.error("Cryptocurrency data is not loaded. Please load the data to proceed.")


def highest_return_prediction():
    st.header("Highest Return Prediction")

    if 'selected_cryptos_full' in st.session_state:
        selected_cryptos_full = st.session_state['selected_cryptos_full']

        # Allowing user to select a model
        model_choice = st.selectbox('Select Prediction Model',
                                    ['LSTM', 'Prophet', 'BI-LSTM', 'ARIMA', 'RandomForest', 'CatBoost'])

        # User input for selecting cryptocurrencies
        selected_tickers = st.multiselect('Select Cryptocurrencies for Prediction:',
                                          selected_cryptos_full['Ticker'].unique())
        days_to_predict = st.number_input('Enter the number of days to predict:', min_value=1, max_value=365, value=30)

        if st.button('Predict Highest Return'):
            with st.spinner(f'Training {model_choice} model and making predictions...'):
                predicted_returns = {}

                for ticker in selected_tickers:
                    # Filter data for the selected cryptocurrency
                    crypto_data = selected_cryptos_full[selected_cryptos_full['Ticker'] == ticker]

                    if model_choice == 'LSTM':

                        X, y, scaler = prepare_lstm_data(crypto_data, 'Close', sequence_length=60)
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

                        model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
                        model, history = train_lstm_model(model, X_train, y_train, X_test, y_test)

                        # reset index to make 'Date' a column
                        crypto_data = crypto_data.reset_index()
                        # Ensure the crypto_data index is a DatetimeIndex
                        crypto_data.index = pd.to_datetime(crypto_data['Date'])

                        # Predict using the last available data to generate predictions for multiple days ahead
                        predictions = []
                        current_sequence = X_test[-1:]

                        for _ in range(days_to_predict):  # Number of days you want to predict
                            # Predict the next step and get the last predicted price
                            current_prediction = model.predict(current_sequence)
                            last_predicted_price = scaler.inverse_transform(current_prediction).flatten()[0]
                            predictions.append(last_predicted_price)

                            # Update the sequence with the last predicted price
                            current_sequence = np.roll(current_sequence, -1)
                            current_sequence[:, -1, :] = current_prediction

                        predicted_prices = np.array(predictions)

                    elif model_choice == 'Prophet':

                        df_prophet = prepare_data_for_prophet(crypto_data, ticker)
                        model = train_prophet_model(df_prophet)
                        future = model.make_future_dataframe(periods=days_to_predict)
                        forecast = model.predict(future)

                        predicted_prices = forecast['yhat'].values

                    elif model_choice == 'BI-LSTM':

                        X, y, scaler = prepare_lstm_data(crypto_data, 'Close', sequence_length=60)
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

                        bi_model = build_bi_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
                        bi_model, history = train_bi_lstm_model(bi_model, X_train, y_train, X_test, y_test)

                        # reset index to make 'Date' a column
                        crypto_data = crypto_data.reset_index()
                        # Ensure the crypto_data index is a DatetimeIndex
                        crypto_data.index = pd.to_datetime(crypto_data['Date'])

                        # Predict using the last available data to generate predictions for multiple days ahead
                        predictions = []
                        current_sequence = X_test[-1:]

                        for _ in range(days_to_predict):  # Number of days you want to predict
                            # Predict the next step and get the last predicted price
                            current_prediction = bi_model.predict(current_sequence)
                            last_predicted_price = scaler.inverse_transform(current_prediction).flatten()[0]
                            predictions.append(last_predicted_price)

                            # Update the sequence with the last predicted price
                            current_sequence = np.roll(current_sequence, -1)
                            current_sequence[:, -1, :] = current_prediction
                        predicted_prices = np.array(predictions)



                    elif model_choice == 'ARIMA':

                        time_series_data = crypto_data.reset_index()
                        time_series_data.index = pd.to_datetime(time_series_data['Date'])
                        time_series_data = time_series_data['Close']
                        time_series_data.index = pd.to_datetime(time_series_data.index)

                        auto_model = find_best_arima(time_series_data, seasonal=False)
                        model_fit = fit_arima_model(time_series_data, auto_model.order)
                        forecast = predict_arima(model_fit, days_to_predict)
                        predicted_prices = forecast

                    elif model_choice == 'RandomForest':
                        features = ['Open', 'High', 'Low', 'Volume', 'SMA', 'EMA', 'RSI']
                        target = 'Close'
                        preprocessed_data = selected_cryptos_full[selected_cryptos_full['Ticker'] == ticker]

                        lag_periods = [1, 2, 3, 4, 5]
                        lagged_data = create_lagged_features(preprocessed_data, lag_periods)
                        features += [f'Close_lag_{lag}' for lag in lag_periods]

                        X = lagged_data[features]
                        y = lagged_data[target]
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

                        model = RandomForestRegressor(n_estimators=100, random_state=42)
                        model.fit(X_train, y_train)
                        future_data = lagged_data[features].tail(days_to_predict)
                        future_predictions = model.predict(future_data)
                        predicted_prices = np.array(future_predictions)

                    elif model_choice == 'CatBoost':
                        features = ['Open', 'High', 'Low', 'Volume', 'SMA', 'EMA', 'RSI']
                        target = 'Close'
                        preprocessed_data = selected_cryptos_full[selected_cryptos_full['Ticker'] == ticker]

                        lag_periods = [1, 2, 3, 4, 5]
                        lagged_data = create_lagged_features(preprocessed_data, lag_periods)
                        features += [f'Close_lag_{lag}' for lag in lag_periods]

                        X = lagged_data[features]
                        y = lagged_data[target]
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

                        model = CatBoostRegressor(iterations=100, learning_rate=0.1, random_seed=42)
                        model.fit(X_train, y_train)
                        future_data = lagged_data[features].tail(days_to_predict)
                        future_predictions = model.predict(future_data)
                        predicted_prices = np.array(future_predictions)

                    # Calculate the predicted return for the selected cryptocurrency
                    predicted_return = (predicted_prices[-1] - predicted_prices[0]) / predicted_prices[0] * 100
                    predicted_returns[ticker] = predicted_return

                # Find the cryptocurrency with the highest predicted return
                best_crypto = max(predicted_returns, key=predicted_returns.get)
                best_return = predicted_returns[best_crypto]

                st.write(f"Based on the {model_choice} model predictions for the next {days_to_predict} days:")
                st.write(f"The cryptocurrency with the highest predicted return is: {best_crypto}")
                st.write(f"Predicted return for {best_crypto}: {best_return:.2f}%")

                # Display predicted returns for all selected cryptocurrencies
                st.write("Predicted returns for all selected cryptocurrencies:")
                for ticker, return_percentage in predicted_returns.items():
                    st.write(f"{ticker}: {return_percentage:.2f}%")

    else:
        st.error("Please ensure the cryptocurrency data is loaded and preprocessed.")


# calculate the trading metrics

#trading strategy using cross-over strategy and set stop loss and take profit allowing user to input the values


def calculate_ema(data, window):
    return data.ewm(span=window, adjust=False).mean()


def trading_strategy():
    st.header("Trading Strategy")

    if 'selected_cryptos_full' in st.session_state:
        selected_cryptos_full = st.session_state['selected_cryptos_full']
        unique_tickers = selected_cryptos_full['Ticker'].unique()
        ticker = st.selectbox('Select a Cryptocurrency', unique_tickers)

        # Convert the index to datetime format
        selected_cryptos_full = selected_cryptos_full.reset_index()

        # Convert the 'Date' column to datetime format and filter out invalid dates
        selected_cryptos_full['Date'] = pd.to_datetime(selected_cryptos_full['Date'], errors='coerce')
        selected_cryptos_full = selected_cryptos_full[selected_cryptos_full['Date'].notnull()]

        # User input for selecting the date range
        start_date = st.date_input('Select start date', selected_cryptos_full['Date'].min())
        end_date = st.date_input('Select end date', selected_cryptos_full['Date'].max())

        # Convert the start_date and end_date to datetime format
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        # Filter the data based on the selected date range
        crypto_data = selected_cryptos_full[(selected_cryptos_full['Ticker'] == ticker) &
                                            (selected_cryptos_full['Date'] >= start_date) &
                                            (selected_cryptos_full['Date'] <= end_date)]

        # User input for selecting the trading strategy
        strategy = st.selectbox('Select Trading Strategy', ['Simple Moving Average (SMA) Crossover',
                                                            'Exponential Moving Average (EMA) Crossover'])

        if strategy == 'Simple Moving Average (SMA) Crossover':
            # User input for selecting the short and long window periods
            short_window = st.number_input('Enter the short window period for SMA:', min_value=1, max_value=100,
                                           value=10)
            long_window = st.number_input('Enter the long window period for SMA:', min_value=50, max_value=500,
                                          value=50)

            # Calculate the short and long moving averages
            crypto_data['SMA_Short'] = crypto_data['Close'].rolling(window=short_window, min_periods=1).mean()
            crypto_data['SMA_Long'] = crypto_data['Close'].rolling(window=long_window, min_periods=1).mean()

            # Generate trading signals based on the crossover strategy
            crypto_data['Signal'] = np.where(crypto_data['SMA_Short'] > crypto_data['SMA_Long'], 'Buy', 'Sell')
            crypto_data['Signal'] = np.where(crypto_data['SMA_Short'] == crypto_data['SMA_Long'], 'Hold',
                                             crypto_data['Signal'])

        elif strategy == 'Exponential Moving Average (EMA) Crossover':
            # User input for selecting the short and long window periods
            short_window = st.number_input('Enter the short window period for EMA:', min_value=1, max_value=50,
                                           value=10)
            long_window = st.number_input('Enter the long window period for EMA:', min_value=50, max_value=200,
                                          value=50)

            # Calculate the short and long exponential moving averages
            crypto_data['EMA_Short'] = calculate_ema(crypto_data['Close'], window=short_window)
            crypto_data['EMA_Long'] = calculate_ema(crypto_data['Close'], window=long_window)

            # Generate trading signals based on the crossover strategy
            crypto_data['Signal'] = np.where(crypto_data['EMA_Short'] > crypto_data['EMA_Long'], 'Buy', 'Sell')
            crypto_data['Signal'] = np.where(crypto_data['EMA_Short'] == crypto_data['EMA_Long'], 'Hold',
                                             crypto_data['Signal'])

        # Calculate the daily returns
        crypto_data['Return'] = crypto_data['Close'].pct_change()

        # Calculate the strategy returns
        crypto_data['Strategy_Return'] = crypto_data['Return'] * (crypto_data['Signal'].shift(1) == 'Buy').astype(int)

        # Calculate the cumulative returns
        crypto_data['Cumulative_Returns'] = (1 + crypto_data['Strategy_Return']).cumprod()

        # Display the trading signals and strategy returns
        st.write("Trading Signals and Strategy Returns:")
        st.dataframe(crypto_data[['Date', 'Close', 'Signal', 'Return', 'Strategy_Return', 'Cumulative_Returns']])

        # Plot the strategy returns
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=crypto_data['Date'], y=crypto_data['Close'], mode='lines', name='Close Price'))
        fig.add_trace(go.Scatter(x=crypto_data['Date'], y=crypto_data['Cumulative_Returns'], mode='lines',
                                 name='Cumulative Returns'))
        fig.update_layout(title=f"{ticker} - {strategy}", xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig)

        # Stop loss and take profit
        stop_loss = st.number_input('Enter the Stop Loss Percentage:', min_value=0.0, max_value=100.0, value=5.0,
                                    step=0.1)
        take_profit = st.number_input('Enter the Take Profit Percentage:', min_value=0.0, max_value=100.0, value=5.0,
                                      step=0.1)

        # Calculate the stop loss and take profit levels
        crypto_data['Stop_Loss'] = crypto_data['Close'] * (1 - stop_loss / 100)
        crypto_data['Take_Profit'] = crypto_data['Close'] * (1 + take_profit / 100)

        # Display the stop loss and take profit levels
        st.write("Stop Loss and Take Profit Levels:")
        st.dataframe(crypto_data[['Date', 'Close', 'Stop_Loss', 'Take_Profit']])

        # Plot the stop loss and take profit levels
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=crypto_data['Date'], y=crypto_data['Close'], mode='lines', name='Close Price'))
        fig.add_trace(go.Scatter(x=crypto_data['Date'], y=crypto_data['Stop_Loss'], mode='lines', name='Stop Loss',
                                 line=dict(color='red', dash='dash')))
        fig.add_trace(go.Scatter(x=crypto_data['Date'], y=crypto_data['Take_Profit'], mode='lines', name='Take Profit',
                                 line=dict(color='green', dash='dash')))
        fig.update_layout(title=f"{ticker} - Stop Loss and Take Profit Levels", xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig)

        # Performance metrics
        sharpe_ratio = calculate_sharpe_ratio(crypto_data['Strategy_Return'])
        sortino_ratio = calculate_sortino_ratio(crypto_data['Strategy_Return'])
        max_drawdown = calculate_max_drawdown(crypto_data['Strategy_Return'])

        st.write("Performance Metrics:")
        st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        st.write(f"Sortino Ratio: {sortino_ratio:.2f}")
        st.write(f"Maximum Drawdown: {max_drawdown:.2%}")

        # User input for selecting the prediction method
        prediction_method = st.selectbox('Select Prediction Method', ['Use Trading Strategy', 'Use Trained Model'])

        if prediction_method == 'Use Trading Strategy':
            # User input for the number of days to predict
            predict_days = st.number_input('Enter the number of days to predict:', min_value=1, value=30, step=1)

            # Generate future dates for prediction
            last_date = crypto_data['Date'].max()
            future_dates = pd.date_range(start=last_date, periods=predict_days + 1, freq='D')[1:]

            # Create a new DataFrame for future predictions
            future_data = pd.DataFrame({'Date': future_dates})
            future_data = pd.concat([crypto_data.tail(1), future_data], ignore_index=True)

            # Calculate the moving averages for future dates
            if strategy == 'Simple Moving Average (SMA) Crossover':
                future_data['SMA_Short'] = future_data['Close'].rolling(window=short_window, min_periods=1).mean()
                future_data['SMA_Long'] = future_data['Close'].rolling(window=long_window, min_periods=1).mean()
                future_data['Signal'] = np.where(future_data['SMA_Short'] > future_data['SMA_Long'], 'Buy', 'Sell')
                future_data['Signal'] = np.where(future_data['SMA_Short'] == future_data['SMA_Long'], 'Hold',
                                                 future_data['Signal'])
            elif strategy == 'Exponential Moving Average (EMA) Crossover':
                future_data['EMA_Short'] = future_data['Close'].ewm(span=short_window, adjust=False).mean()
                future_data['EMA_Long'] = future_data['Close'].ewm(span=long_window, adjust=False).mean()
                future_data['Signal'] = np.where(future_data['EMA_Short'] > future_data['EMA_Long'], 'Buy', 'Sell')
                future_data['Signal'] = np.where(future_data['EMA_Short'] == future_data['EMA_Long'], 'Hold',
                                                 future_data['Signal'])

            # predict the future prices
            future_data['Return'] = future_data['Close'].pct_change()
            future_data['Strategy_Return'] = future_data['Return'] * (future_data['Signal'].shift(1) == 'Buy').astype(
                int)
            future_data['Cumulative_Returns'] = (1 + future_data['Strategy_Return']).cumprod()

            # Display the future predictions
            st.write("Future Predictions:")
            st.dataframe(future_data[['Date', 'Close', 'Signal', 'Return', 'Strategy_Return', 'Cumulative_Returns']])

            # Plot the future predictions with interactive technical analysis
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=crypto_data['Date'], y=crypto_data['Close'], mode='lines', name='Close Price'))
            fig.add_trace(go.Scatter(x=future_data['Date'], y=future_data['Close'], mode='lines', name='Future Price'))
            fig.update_layout(title=f"{ticker} - Future Predictions with {strategy}", xaxis_title='Date',
                              yaxis_title='Price')

            # Add interactive tools for technical analysis
            fig.update_layout(
                dragmode='drawline',
                newshape=dict(line_color='yellow'),
                modebar=dict(orientation='v'),
                xaxis=dict(rangeslider=dict(visible=True))
            )

            st.plotly_chart(fig)

    else:
        st.error("Please load and preprocess the cryptocurrency data first.")


# Conditional navigation based on sidebar choice
if menu_choice == "Overview":
    show_overview()
elif menu_choice == "About":
    show_about()
elif menu_choice == "Data Preprocessing":
    data_preprocessing()
elif menu_choice == "Exploratory Data Analysis":
    exploratory_data_analysis()
elif menu_choice == "Correlation Analysis":
    correlation_analysis()
elif menu_choice == "Prediction":
    prediction()
elif menu_choice == "Highest Return Prediction":
    highest_return_prediction()
elif menu_choice == "Trading Strategy":
    trading_strategy()
elif menu_choice == "News":
    news_page(API_KEY)
