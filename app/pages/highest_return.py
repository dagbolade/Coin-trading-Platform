import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import streamlit as st
from arima import find_best_arima, fit_arima_model, predict_arima
from bi_lstm_model import build_bi_lstm_model, train_bi_lstm_model
from lstm_model import prepare_lstm_data, build_lstm_model, train_lstm_model
from prophet_model import train_prophet_model, prepare_data_for_prophet



def create_lagged_features(data, lag_periods):
    lagged_data = data.copy()
    for lag in lag_periods:
        lagged_data[f'Close_lag_{lag}'] = lagged_data['Close'].shift(lag)
    lagged_data.dropna(inplace=True)
    return lagged_data

def highest_return_page():
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