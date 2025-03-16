# model_evaluation.py

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score


def evaluate_lstm_model(model, X_test, y_test, scaler):
    """
    Evaluates the performance of an LSTM model.

    Args:
        model (keras.models.Model): The trained LSTM model.
        X_test (numpy.ndarray): The features of the test data.
        y_test (numpy.ndarray): The true target values of the test data.
        scaler (sklearn.preprocessing.MinMaxScaler): The scaler used for normalization.

    Returns:
        tuple: A tuple containing the MSE, MAE, RMSE, and R² values.
    """
    y_pred = model.predict(X_test)
    y_pred_inv = scaler.inverse_transform(y_pred)  # Inverse transform to get actual price
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_inv, y_pred_inv)
    return mse, mae, rmse, r2


def evaluate_bi_lstm_model(bi_model, X_test, y_test, scaler):
    """
    Evaluates the performance of a Bi-LSTM model.

    Args:
        bi_model (keras.models.Model): The trained Bi-LSTM model.
        X_test (numpy.ndarray): The features of the test data.
        y_test (numpy.ndarray): The true target values of the test data.
        scaler (sklearn.preprocessing.MinMaxScaler): The scaler used for normalization.

    Returns:
        tuple: A tuple containing the MSE, MAE, RMSE, and R² values.
    """
    y_pred = bi_model.predict(X_test)
    y_pred_inv = scaler.inverse_transform(y_pred)  # Inverse transform to get actual price
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_inv, y_pred_inv)
    return mse, mae, rmse, r2


def evaluate_prophet_model(df_prophet, combined):
    """
    Evaluates the performance of a Prophet model.

    Args:
        df_prophet (pandas.DataFrame): The original Prophet data.
        combined (pandas.DataFrame): The combined actual and forecast data.

    Returns:
        dict: A dictionary containing the MSE, RMSE, and R² values.
    """
    if 'ds' not in df_prophet.columns:
        # set 'ds' as column
        df_prophet.reset_index(inplace=True)
    if 'ds' not in combined.columns:
        # set 'ds' as column
        combined.reset_index(inplace=True)

    last_date_in_actual = df_prophet['ds'].max()
    historical_forecast = combined[(combined['ds'] <= last_date_in_actual)]
    actuals = df_prophet[df_prophet['ds'] <= last_date_in_actual]['y']
    mse = mean_squared_error(actuals, historical_forecast['yhat'])
    mae = mean_absolute_error(actuals, historical_forecast['yhat'])
    rmse = np.sqrt(mse)
    r2 = r2_score(actuals, historical_forecast['yhat'])
    metrics = {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    }
    return metrics


def calculate_arima_metrics(y_true, y_pred):
    """
    Calculates evaluation metrics for ARIMA models.

    Args:
        y_true (numpy.ndarray): The true target values.
        y_pred (numpy.ndarray): The predicted values.

    Returns:
        dict: A dictionary containing the calculated evaluation metrics.
    """
    # Calculate root mean squared error (RMSE)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Calculate mean squared error (MSE)
    mse = mean_squared_error(y_true, y_pred)

    # Calculate mean absolute error (MAE)
    mae = mean_absolute_error(y_true, y_pred)

    # Calculate R-squared (R²)
    r2 = r2_score(y_true, y_pred)

    metrics = {
        'RMSE': rmse,
        'MSE': mse,
        'MAE': mae,
        'R2': r2
    }

    return metrics


def calculate_random_forest_metrics(y_true, y_pred):
    """
    Calculates evaluation metrics for Random Forest models.

    Args:
        y_true (numpy.ndarray): The true target values.
        y_pred (numpy.ndarray): The predicted values.

    Returns:
        dict: A dictionary containing the calculated evaluation metrics.
    """
    # Calculate root mean squared error (RMSE)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Calculate mean squared error (MSE)
    mse = mean_squared_error(y_true, y_pred)

    # Calculate mean absolute error (MAE)
    mae = mean_absolute_error(y_true, y_pred)

    # Calculate R-squared (R²)
    r2 = r2_score(y_true, y_pred)

    metrics = {
        'RMSE': rmse,
        'MSE': mse,
        'MAE': mae,
        'R2': r2
    }

    return metrics


def calculate_catboost_metrics(y_true, y_pred):
    """
    Calculates evaluation metrics for CatBoost models.

    Args:
        y_true (numpy.ndarray): The true target values.
        y_pred (numpy.ndarray): The predicted values.

    Returns:
        dict: A dictionary containing the calculated evaluation metrics.
    """
    # Calculate root mean squared error (RMSE)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Calculate mean squared error (MSE)
    mse = mean_squared_error(y_true, y_pred)

    # Calculate mean absolute error (MAE)
    mae = mean_absolute_error(y_true, y_pred)

    # Calculate R-squared (R²)
    r2 = r2_score(y_true, y_pred)

    metrics = {
        'RMSE': rmse,
        'MSE': mse,
        'MAE': mae,
        'R2': r2
    }

    return metrics
