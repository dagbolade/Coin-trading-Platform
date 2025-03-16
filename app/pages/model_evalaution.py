# app/pages/model_evaluation.py
import catboost
import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from keras.src.saving import load_model
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from app.pages.prediction import create_lagged_features
from lstm_model import prepare_lstm_data, build_lstm_model, train_lstm_model, evaluate_lstm_model
from prophet_model import prepare_data_for_prophet, train_prophet_model, evaluate_prophet_model
from bi_lstm_model import build_bi_lstm_model, train_bi_lstm_model, prepare_bi_lstm_data, evaluate_bi_lstm_model
from model_evaluation import calculate_arima_metrics, calculate_random_forest_metrics, calculate_catboost_metrics
from sklearn.ensemble import RandomForestRegressor

import arima
from random_forest_model import train_random_forest


def model_evaluation_page():
    st.header("Model Evaluation and Comparison")

    if 'selected_cryptos_full' in st.session_state:
        selected_cryptos_full = st.session_state['selected_cryptos_full']

        # User input for selecting cryptocurrencies
        selected_tickers = st.multiselect('Select Cryptocurrencies for Evaluation:',
                                          selected_cryptos_full['Ticker'].unique())

        # User input for selecting models to evaluate
        model_options = ['LSTM', 'Prophet', 'ARIMA', 'Random Forest', 'CatBoost']
        selected_models = st.multiselect('Select Models to Evaluate:', model_options)

        if st.button('Evaluate Models'):
            with st.spinner('Evaluating models...'):
                evaluation_results = {}

                for ticker in selected_tickers:
                    # Filter data for the selected cryptocurrency
                    crypto_data = selected_cryptos_full[selected_cryptos_full['Ticker'] == ticker]

                    # Split data into training and test sets
                    train_size = int(len(crypto_data) * 0.8)
                    train_data = crypto_data[:train_size]
                    test_data = crypto_data[train_size:]

                    evaluation_results[ticker] = {}

                    # Evaluate selected models
                    if 'LSTM' in selected_models:
                        try:
                            lstm_model = load_model('lstm_model.h5')
                            X, y, scaler = prepare_lstm_data(test_data, 'Close', sequence_length=60)
                            X_test, _, y_test, _ = train_test_split(X, y, test_size=0.2, shuffle=False)
                            lstm_mse, lstm_mae, lstm_rmse, lstm_r2 = evaluate_lstm_model(lstm_model, X_test, y_test,
                                                                                         scaler)
                            evaluation_results[ticker]['LSTM'] = {'MSE': lstm_mse, 'MAE': lstm_mae, 'RMSE': lstm_rmse,
                                                                  'R2': lstm_r2}
                        except Exception as e:
                            st.warning(f"Error loading LSTM model: {e}. Skipping evaluation.")


                    if 'Prophet' in selected_models:
                        df_prophet = prepare_data_for_prophet(train_data, ticker)
                        prophet_model = train_prophet_model(df_prophet)
                        future = prophet_model.make_future_dataframe(periods=len(test_data))
                        forecast = prophet_model.predict(future)
                        prophet_metrics = evaluate_prophet_model(df_prophet, forecast)
                        evaluation_results[ticker]['Prophet'] = prophet_metrics

                    if 'ARIMA' in selected_models:
                        time_series_data = train_data.reset_index()['Close']
                        time_series_data.index = pd.to_datetime(train_data.index)
                        auto_model = arima.find_best_arima(time_series_data, seasonal=False)
                        model_fit = arima.fit_arima_model(time_series_data, auto_model.order)
                        predictions = arima.predict_arima(model_fit, len(test_data))
                        arima_metrics = calculate_arima_metrics(test_data['Close'].values, predictions)
                        evaluation_results[ticker]['ARIMA'] = arima_metrics

                    if 'Random Forest' in selected_models:
                        features = ['Open', 'High', 'Low', 'Volume', 'SMA', 'EMA', 'RSI']
                        target = 'Close'
                        preprocessed_data = train_data

                        lag_periods = [1, 2, 3, 4, 5]
                        lagged_data = create_lagged_features(preprocessed_data, lag_periods)
                        features += [f'Close_lag_{lag}' for lag in lag_periods]

                        X = lagged_data[features]
                        y = lagged_data[target]
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

                        rf_model = train_random_forest(X_train, y_train)
                        y_pred = rf_model.predict(X_test)
                        rf_metrics = calculate_random_forest_metrics(y_test, y_pred)
                        evaluation_results[ticker]['Random Forest'] = rf_metrics

                    if 'CatBoost' in selected_models:
                        features = ['Open', 'High', 'Low', 'Volume', 'SMA', 'EMA', 'RSI']
                        target = 'Close'
                        preprocessed_data = train_data

                        lag_periods = [1, 2, 3, 4, 5]
                        lagged_data = create_lagged_features(preprocessed_data, lag_periods)
                        features += [f'Close_lag_{lag}' for lag in lag_periods]

                        X = lagged_data[features]
                        y = lagged_data[target]
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

                        cat_model = CatBoostRegressor(iterations=100, learning_rate=0.1, depth=6, loss_function='RMSE')
                        cat_model.fit(X_train, y_train, verbose=False)
                        y_pred = cat_model.predict(X_test)
                        cat_metrics = calculate_catboost_metrics(y_test, y_pred)
                        evaluation_results[ticker]['CatBoost'] = cat_metrics

                st.success('Evaluation completed!')

                # Display the evaluation metrics for each cryptocurrency
                for ticker in evaluation_results:
                    st.subheader(f"Evaluation Metrics for {ticker}")

                    # Display the evaluation metrics in a DataFrame
                    df_metrics = pd.DataFrame.from_dict(evaluation_results[ticker], orient='index')
                    df_metrics = df_metrics.applymap(lambda x: round(x, 4) if isinstance(x, (int, float)) else x)
                    st.write(df_metrics)

                    # Plot the evaluation metrics as a bar chart
                    metrics = ['RMSE', 'MSE', 'MAE', 'R2']
                    models = list(evaluation_results[ticker].keys())

                    metric_data = {}
                    for metric in metrics:
                        metric_data[metric] = [
                            evaluation_results[ticker][model][metric] if metric in evaluation_results[ticker][
                                model] else 0 for model in models]

                    df_chart = pd.DataFrame(metric_data, index=models)
                    st.bar_chart(df_chart)
    else:
        st.warning('Please select cryptocurrencies from the Data page first.')
# app/pages/prediction.py