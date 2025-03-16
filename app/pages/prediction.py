# app/pages/prediction.py

import streamlit as st
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import plotly.graph_objects as go
from keras.src.metrics.accuracy_metrics import accuracy
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

import arima
import lstm_model
import prophet_model
from arima import predict_arima
from bi_lstm_model import train_bi_lstm_model, build_bi_lstm_model
from lstm_model import train_lstm_model
from trading_signals import generate_trading_signals, plot_arima_forecast_with_signals, generate_arima_trading_signals, \
    generate_lstm_trading_signals, plot_forecast_with_signals2, generate_prophet1_trading_signals


# creating lagged features for catboost and random forest
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
    st.write(
        "Please note that this analysis is based on the predicted prices and does not guarantee future performance.")


def prediction_page():
    st.header("Prediction for Cryptocurrencies")
    st.info("""The 4 main models used for prediction are: LSTM, Prophet, Random Forest, and CatBoost, while the other trained 
    models are ARIMA and BI-LSTM. """)

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
                    # callling the lstm model
                    X, y, scaler = lstm_model.prepare_lstm_data(crypto_data, 'Close', sequence_length=60)
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

                    # reset index to make 'Date' a column
                    crypto_data = crypto_data.reset_index()

                    ls_model = lstm_model.build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
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

                    buy_sell_analysis([ticker], predicted_prices, days_to_predict, investment_amount)
                    # Show the confidence of the model in percentage using r2

                    actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
                    predicted_prices_test = scaler.inverse_transform(ls_model.predict(X_test)).flatten()
                    r2 = r2_score(actual_prices, predicted_prices_test)
                    st.write(f"Model Confidence Level : {r2 * 100:.2f}%")

                elif model_choice == 'Prophet':
                    # Prepare the data for Prophet
                    df_prophet = prophet_model.prepare_data_for_prophet(crypto_data, ticker)
                    model = prophet_model.train_prophet_model(df_prophet)
                    future = model.make_future_dataframe(periods=days_to_predict)
                    forecast = model.predict(future)
                    predicted_prices = forecast['yhat'].values[-days_to_predict:]
                    # Get the last known price from historical data
                    last_known_price = df_prophet['y'].iloc[-1]
                    # Save the forecast in session state
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
                    # Display the last day of the prediction only
                    st.write(f"The predicted price for day {days_to_predict} is ${forecast['yhat'].iloc[-1]:.2f}")
                    # Determine the latest signal and give trading advice
                    latest_signal = signals['Signal'].iloc[-1]
                    if latest_signal == 'Buy':
                        st.success(f"🟢 Buy Advice: Consider buying {ticker}.")
                    elif latest_signal == 'Sell':
                        st.error(f"🔴 Sell Advice: Consider selling {ticker}.")
                    else:
                        st.info(f"🔵 Hold Advice: Maintain your position in {ticker}.")
                    # Save the forecast and signals in session state
                    st.session_state['forecast'] = forecast
                    st.session_state['signals'] = signals
                    buy_sell_analysis([ticker], predicted_prices, days_to_predict, investment_amount)
                    # Get the common dates between df_prophet and forecast
                    common_dates = pd.Index(np.intersect1d(df_prophet['ds'], forecast['ds']))
                    # Get the actual values for the common dates
                    actual_values = df_prophet.set_index('ds').loc[common_dates, 'y'].values
                    # Get the predicted values for the common dates
                    predicted_values = forecast.set_index('ds').loc[common_dates, 'yhat'].values
                    # Calculate R-squared (R²)
                    r2 = r2_score(actual_values, predicted_values)
                    st.write(f"Model Confidence Level : {r2 * 100:.2f}%")

                elif model_choice == 'BI-LSTM':
                    # preparing the data
                    X, y, scaler = lstm_model.prepare_lstm_data(crypto_data, 'Close', sequence_length=60)
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

                    st.session_state['bi_lstm_model'] = bi_model

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
                    # Show the confidence of the model in percentage
                    actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
                    predicted_prices_test = scaler.inverse_transform(bi_model.predict(X_test)).flatten()
                    r2 = r2_score(actual_prices, predicted_prices_test)
                    st.write(f"Model Confidence Level : {r2 * 100:.2f}%")

                elif model_choice == 'ARIMA':
                    # reset index to make 'Date' a column
                    time_series_data = crypto_data.reset_index()

                    # Ensure the time series is indexed properly and is a Series
                    time_series_data.index = pd.to_datetime(time_series_data['Date'])
                    time_series_data = time_series_data['Close']

                    # Find the best ARIMA model
                    auto_model = arima.find_best_arima(time_series_data, seasonal=False)

                    # Fit the ARIMA model
                    model_fit = arima.fit_arima_model(time_series_data, auto_model.order)

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


                elif model_choice == 'RandomForest':
                    features = ['Open', 'High', 'Low', 'Volume', 'SMA', 'EMA', 'RSI']
                    target = 'Close'
                    preprocessed_data = selected_cryptos_full[selected_cryptos_full['Ticker'] == ticker]

                    # Create lagged features
                    lag_periods = [1, 2, 3, 4, 5]
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
                    accuracy = rf_model.score(X_test, y_test)
                    st.write(f"Model Accuracy: {accuracy:.2f}%")

                elif model_choice == 'CatBoost':
                    features = ['Open', 'High', 'Low', 'Volume', 'SMA', 'EMA', 'RSI']
                    target = 'Close'
                    preprocessed_data = selected_cryptos_full[selected_cryptos_full['Ticker'] == ticker]

                    # Create lagged features
                    lag_periods = [1, 2, 3, 4, 5]
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

                    buy_sell_analysis([ticker], predicted_prices, days_to_predict, investment_amount)
                    # Show the accuracy of the model
                    accuracy = cat_model.score(X_test, y_test)
                    st.write(f"Model Accuracy: {accuracy:.2f}%")


    else:
        st.error("Please ensure the cryptocurrency data is loaded and preprocessed.")
