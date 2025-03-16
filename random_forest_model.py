from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def prepare_data(data, drop_columns, target, test_size=0.2, shuffle=False):
    # Print columns for debugging
    print("Columns before dropping:", data.columns)

    # Check and drop the target column if it exists
    if target in data.columns:
        y = data[target]
        X = data.drop(columns=[target], errors='ignore')  # Safely ignore if not present
    else:
        raise ValueError(f"Target column '{target}' not found in the data.")

    # Drop specified non-target columns, safely ignoring if they are not present
    X = X.drop(columns=drop_columns, errors='ignore')

    # Ensure only numeric columns are retained
    X = X.select_dtypes(include=[np.number])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=shuffle)
    return X_train, X_test, y_train, y_test


def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    rf_model.fit(X_train, y_train)
    return rf_model


def predict_and_evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    rsquared = r2_score(y_test, y_pred)
    return mse, rmse, mae, rsquared, y_pred


def train_custom_model_for_BTC(X_train, y_train):
    rf_model = RandomForestRegressor(n_estimators=500, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model


def plot_rf_predictions(y_test, y_pred, ticker):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.index, y_test, label='Actual')
    plt.plot(y_test.index, y_pred, label='Predicted')
    plt.title(f'{ticker} - Actual vs Predicted Closing Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


def predict_future_prices(model, historical_features, days_to_predict):
    # Get the most recent features to predict the next day
    recent_features = historical_features.iloc[-1].values.reshape(1, -1)

    future_predictions = []
    for day in range(days_to_predict):
        # Predict the next day's price
        next_day_prediction = model.predict(recent_features)[0]
        future_predictions.append(next_day_prediction)

        # Update the recent_features to include the prediction (This step depends on how your features are structured)
        # For example, if your features include lagged prices, you would update the lags here

        # This is a placeholder for updating the features, you'll need to implement this based on your actual features
        # recent_features = update_features_with_prediction(recent_features, next_day_prediction)

    # Generate future dates
    last_date = historical_features.index.max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict + 1)]

    # Create a DataFrame with future dates and predictions
    predictions_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': future_predictions})
    predictions_df.set_index('Date', inplace=True)

    return predictions_df

