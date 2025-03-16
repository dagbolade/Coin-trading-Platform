import logging

import pandas as pd
from matplotlib import pyplot as plt

import bi_lstm_model
from Data_loader import load_data
from data_preprocessing import convert_to_datetime, check_missing_values, fill_missing_values
from data_transformation import remove_duplicates, pivot_and_fill, scale_data
from dimensionality_reduction import apply_pca, plot_explained_variance
from clustering import apply_kmeans, plot_clusters, plot_cluster_distribution, add_cluster_labels, \
    select_cryptos_closest_to_centroids
from correlation import calculate_daily_returns, calculate_correlation_matrix, find_top_correlations, \
    plot_correlation_heatmap
from eda import plot_time_series, plot_rolling_statistics, plot_boxplot, plot_candlestick, \
    plot_volatility_clustering, plot_kde_of_closes
from prophet_model import prepare_data_for_prophet, train_prophet_model, make_predictions, plot_forecast, \
    evaluate_prophet_model, plot_performance_metrics
from lstm_model import prepare_lstm_data, build_lstm_model, train_lstm_model, evaluate_lstm_model, plot_history, \
    plot_predictions
from sklearn.model_selection import train_test_split
from feature_engineering import calculate_sma_ema_rsi
from random_forest_model import prepare_data, train_random_forest, predict_and_evaluate, plot_rf_predictions, \
    train_custom_model_for_BTC
from svr_module import preprocess_data, train_svr, evaluate_svr, plot_svr_predictions
from bi_lstm_model import prepare_bi_lstm_data, \
    plot_bi_predictions, build_bi_lstm_model, train_bi_lstm_model, evaluate_bi_lstm_model

#from xgboost_model import preprocess_data, split_data, train_xgboost_model, plot_xg_predictions


# Define the file path for the CSV data
file_path = 'top_39_cryptos_historical_data.csv'

# Load the data
crypto_data = load_data(file_path)

# If data is loaded successfully, proceed with displaying the head or any other operations
if crypto_data is not None:
    print(crypto_data.tail())

# Convert 'Date' column to datetime
crypto_data = convert_to_datetime(crypto_data, 'Date')

# Check for missing values
missing_values = check_missing_values(crypto_data)
print(missing_values)

# Fill missing values if needed
crypto_data = fill_missing_values(crypto_data, method='ffill')

# Remove duplicates
crypto_data = remove_duplicates(crypto_data, subset=['Ticker', 'Date'])

# Pivot and fill the data
pivot_data = pivot_and_fill(crypto_data, index='Ticker', columns='Date', values='Close')

# Check for remaining missing values
remaining_na = pivot_data.isnull().sum().sum()
print(f'Remaining missing values: {remaining_na}')

# Normalize the data
scaled_data_df = scale_data(pivot_data)
print(scaled_data_df.head(20))

# Use the function with your data
reduced_data_df, pca = apply_pca(scaled_data_df)  # Capture both the DataFrame and PCA object
print(reduced_data_df.head(10))

# Now use the PCA object for plotting
plot_explained_variance(pca, n_components=10)  # Plot the explained variance

# Apply K-Means clustering
clusters, kmeans = apply_kmeans(reduced_data_df, n_clusters=4)  # Now captures both clusters and the kmeans object

# Add cluster labels to the original pivoted data
pivot_data = add_cluster_labels(pivot_data, clusters)

# Visualize the clusters
plot_clusters(reduced_data_df.values, clusters)  # Ensure to pass numpy array if needed

# Plot the distribution of clusters
plot_cluster_distribution(pivot_data)

# Select the cryptocurrencies closest to the cluster centroids
selected_cryptos = select_cryptos_closest_to_centroids(reduced_data_df, clusters, kmeans.cluster_centers_)

print(selected_cryptos)

# Convert 'Date' column to datetime and set as index
crypto_data = convert_to_datetime(crypto_data, 'Date').set_index('Date')

# drop everyother column  on selected cryptos, except the Ticker nd cluster
selected_cryptos = selected_cryptos[['Cluster']]
print(selected_cryptos)

# Merge the selected cryptocurrencies with the original data to get their full details including Date
selected_cryptos_full = crypto_data.merge(selected_cryptos, left_on='Ticker', right_index=True)

# make sure the data is sorted by Date and its a column not an index
selected_cryptos_full = selected_cryptos_full.reset_index().sort_values('Date')

# Display the selected cryptocurrencies with their full details
print(selected_cryptos_full)

# Calculate daily returns for all cryptocurrencies
daily_returns = calculate_daily_returns(crypto_data)

# apply the feature engineering to the selected cryptos
selected_cryptos_full = calculate_sma_ema_rsi(selected_cryptos_full)

print(selected_cryptos_full)

# Calculate the correlation matrix for all cryptocurrencies
correlation_matrix = calculate_correlation_matrix(daily_returns)

selected_cryptos = selected_cryptos.index.tolist()

# Find top correlations for the selected cryptocurrencies
top_correlations = find_top_correlations(correlation_matrix, selected_cryptos)

# Printing top correlations along with their values
for ticker, correlation_data in top_correlations.items():
    print(f"Top positive correlations for {ticker}:")
    for other_ticker, value in correlation_data['Positive'].items():
        print(f"{other_ticker}: {value}")
    print(f"\nTop negative correlations for {ticker}:")
    for other_ticker, value in correlation_data['Negative'].items():
        print(f"{other_ticker}: {value}")
    print("\n")

# Optionally, plot the correlation matrix heatmap
#plot_correlation_heatmap(correlation_matrix)

# Plot time series, rolling statistics, distributions, candlestick charts, and volatility clustering for each
# selected crypto
selected_cryptos_full['Date'] = pd.to_datetime(selected_cryptos_full['Date'])
selected_cryptos_full.set_index('Date', inplace=True)

for ticker in selected_cryptos_full['Ticker'].unique():
    plot_time_series(selected_cryptos_full, ticker)
    plot_rolling_statistics(selected_cryptos_full, ticker)

    plot_boxplot(selected_cryptos_full, ticker)
    plot_candlestick(selected_cryptos_full, ticker)
    plot_volatility_clustering(selected_cryptos_full, ticker)
    plot_kde_of_closes(selected_cryptos_full, [ticker])

# 1st model- Prophet

# for ticker in selected_cryptos_full['Ticker'].unique():
#     df_prophet = prepare_data_for_prophet(crypto_data, ticker)
#     model = train_prophet_model(df_prophet)
#     combined = make_predictions(model, df_prophet, periods=15)
#     metrics = evaluate_prophet_model(df_prophet, combined)
#
#     print(f"Metrics for {ticker}: {metrics}")
#     plot_forecast(combined)

# 2nd model- LSTM
# Dictionary to store trained models and their evaluation metrics
# trained_models = {}
# evaluation_metrics = {}
#
# # Main processing loop
# for ticker, df in selected_cryptos_full.groupby('Ticker'):
#     # Prepare data
#     X, y, scaler = prepare_lstm_data(df, column='Close', sequence_length=60)
#
#     # Split data into training and test set
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
#
#     # Build LSTM model
#     model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
#
#     # Train model
#     model, history = train_lstm_model(model, X_train, y_train, X_test, y_test)
#
#     # Evaluate model
#     mse, mae, rmse, r2 = evaluate_lstm_model(model, X_test, y_test, scaler)
#
#     # Store results
#     trained_models[ticker] = model
#     evaluation_metrics[ticker] = {
#         'MSE': mse,
#         'MAE': mae,
#         'RMSE': rmse,
#         'R2': r2
#     }
#
#     # Plot training history and predictions
#     plot_history(history, ticker)
#     y_pred = model.predict(X_test)
#     y_pred_inv = scaler.inverse_transform(y_pred)
#     y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
#     plot_predictions(y_test_inv, y_pred_inv, ticker)
#
#     # Display evaluation metrics
#     print(f"Evaluation Metrics for {ticker}:")
#     print("Mean Squared Error:", mse)
#     print("Mean Absolute Error:", mae)
#     print("Root Mean Squared Error:", rmse)
#     print("R-squared:", r2)

# XGBoost model

# Process and model each cryptocurrency
# for ticker in selected_cryptos_full['Ticker'].unique():
#     print(f"Processing {ticker}")
#     df = selected_cryptos_full[selected_cryptos_full['Ticker'] == ticker]
#
#     features = ['Open', 'High', 'Low', 'Volume']
#     target = 'Close'
#
#     # Preprocess the data
#     data_preprocessed = preprocess_data(df, features, target)
#
#     # Split the data into training and testing sets
#     train, test = split_data(data_preprocessed)
#     X_train, y_train = train[features + ['Daily_Return']], train[target]
#     X_test, y_test = test[features + ['Daily_Return']], test[target]
#
#     # Train the model
#     model, preds, rmse, r2 = train_xgboost_model(X_train, y_train, X_test, y_test)
#
#     # Output the performance metrics
#     print(f'Ticker: {ticker}, RMSE: {rmse}, R-squared: {r2}')
#
#     # Plot the predictions
#     plot_predictions(test.index, y_test, preds, ticker)

# 4th model- Random Forest
# for ticker in selected_cryptos_full['Ticker'].unique():
#     print(f'Processing {ticker}...')
#     data = selected_cryptos_full[selected_cryptos_full['Ticker'] == ticker]
#
#     if ticker == 'BTC-USD':
#         print("Applying custom feature engineering for BTC...")
#         data = create_btc_features(data)
#
#     X_train, X_test, y_train, y_test = prepare_data(data, ['Adj Close', 'Ticker'], 'Close')
#
#     if ticker == 'BTC-USD':
#         print("Training custom model for BTC...")
#         model = train_custom_model_for_BTC(X_train, y_train)
#     else:
#         model = train_random_forest(X_train, y_train)
#
#     mse, rmse, mae, rsquared, y_pred = predict_and_evaluate(model, X_test, y_test)
#     print(f"Metrics for {ticker}: MSE={mse}, RMSE={rmse}, MAE={mae}, R-squared={rsquared}")
#     plot_rf_predictions(y_test, y_pred, ticker)

# 5th model- SVR
# svr_models = {}
# evaluation_metrics = {}
#
# for ticker, df in selected_cryptos_full.groupby('Ticker'):
#     features_scaled, scaled_Close, feature_scaler, target_scaler = preprocess_data(df)
#     X_train, X_test, y_train, y_test = train_test_split(features_scaled, scaled_Close, test_size=0.2, shuffle=False)
#
#     # Train the SVR model
#     svr = train_svr(X_train, y_train)
#
#     # Evaluate the model
#     mse, rmse, y_pred, r2 = evaluate_svr(svr, X_test, y_test, target_scaler)
#     evaluation_metrics[ticker] = {'MSE': mse, 'RMSE': rmse, 'R2': r2}
#
#     # Print metrics
#     print(f"Evaluation Metrics for {ticker}: MSE = {mse}, RMSE = {rmse}, R2 = {r2}")
#
#     # Adjust the actual y_test for plotting
#     y_test_array = y_test.values.reshape(-1, 1)  # Adjust here
#     y_test_actual = target_scaler.inverse_transform(y_test_array).flatten()
#
#     # Plot the actual vs predicted prices
#     plot_svr_predictions(y_test_actual, y_pred, ticker)

# 6th model- Bi-LSTM

#Dictionary to store trained models and their evaluation metrics
trained_models = {}
evaluation_metrics = {}

#Main processing loop


# for ticker, df in selected_cryptos_full.groupby('Ticker'):
#     # Prepare data
#     X, y, scaler = prepare_lstm_data(df, column='Close', sequence_length=60)
#
#     # Split data into training and test set
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
#
#     # Build LSTM model
#     model = build_bi_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
#
#     # Train model
#     model, history = train_bi_lstm_model(model, X_train, y_train, X_test, y_test)
#
#     # Evaluate model
#     mse, mae, rmse, r2 = evaluate_bi_lstm_model(model, X_test, y_test, scaler)
#
#     # Store results
#     trained_models[ticker] = model
#     evaluation_metrics[ticker] = {
#         'MSE': mse,
#         'MAE': mae,
#         'RMSE': rmse,
#         'R2': r2
#     }
#
#     # Plot training history and predictions
#     plot_history(history, ticker)
#     y_pred = model.predict(X_test)
#     y_pred_inv = scaler.inverse_transform(y_pred)
#     y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
#     plot_bi_predictions(y_test_inv, y_pred_inv, ticker)
#
#     # Display evaluation metrics
#     print(f"Evaluation Metrics for {ticker}:")
#     print("Mean Squared Error:", mse)
#     print("Mean Absolute Error:", mae)
#     print("Root Mean Squared Error:", rmse)
#     print("R-squared:", r2)





