from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

# Function to find best ARIMA model
def find_best_arima(series, seasonal=False):
    auto_model = auto_arima(series, seasonal=seasonal, stepwise=True,
                            suppress_warnings=True, error_action="ignore", max_order=None, trace=True)
    return auto_model

# Function to fit ARIMA model
def fit_arima_model(series, order):
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    return model_fit

# Function to predict future prices using ARIMA
def predict_arima(model_fit, steps):

    # Predict future values
    forecast_values = model_fit.forecast(steps=steps)
    return forecast_values
