from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np


def prepare_data_for_prophet(data, ticker):
    if 'Date' in data.columns:
        crypto_subset = data[data['Ticker'] == ticker][['Date', 'Close']]
    else:
        crypto_subset = data[data['Ticker'] == ticker].reset_index()
        crypto_subset.rename(columns={'index': 'Date'}, inplace=True)
    df_prophet = crypto_subset.rename(columns={'Date': 'ds', 'Close': 'y'})
    return df_prophet


def train_prophet_model(df_prophet):
    prophet_model = Prophet(daily_seasonality=True)
    prophet_model.fit(df_prophet)
    return prophet_model


def make_predictions(prophet_model, df_prophet, periods=7):
    future = prophet_model.make_future_dataframe(periods=periods)
    forecast = prophet_model.predict(future)
    # Merge actual and forecast for plotting
    forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    forecast.set_index('ds', inplace=True)
    df_prophet.set_index('ds', inplace=True)
    combined = df_prophet.join(forecast, how='outer')
    return combined


def evaluate_prophet_model(df_prophet, combined):
    if 'ds' not in df_prophet.columns:
        # set 'ds' as column
        df_prophet.reset_index(inplace=True)
        #if 'ds' not in combined.columns:
        # set 'ds' as column
        combined.reset_index(inplace=True)

    last_date_in_actual = df_prophet['ds'].max()
    historical_forecast = combined[(combined['ds'] <= last_date_in_actual)]
    actuals = df_prophet[df_prophet['ds'] <= last_date_in_actual]['y']
    mse = mean_squared_error(actuals, historical_forecast['yhat'])
    mae = mean_absolute_error(actuals, historical_forecast['yhat'])
    rmse = np.sqrt(mse)
    r_squared = r2_score(actuals, historical_forecast['yhat'])

    metrics = {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r_squared
    }
    return metrics


import plotly.graph_objs as go


def plot_forecast(historical, forecast, days_to_predict):
    # Find the start of the forecast period
    forecast_start = historical['ds'].max()

    # Create traces for the plot
    historical_trace = go.Scatter(
        x=historical['ds'],
        y=historical['y'],
        mode='lines',
        name='Historical'
    )

    forecast_trace = go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        mode='lines',
        name='Forecast',
        line=dict(color='red'),
    )

    # Traces for the confidence interval, which is optional
    lower_band = go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_lower'],
        line=dict(width=0),
        mode='lines',
        name='Lower Confidence Interval',
        showlegend=False
    )

    upper_band = go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_upper'],
        line=dict(width=0),
        mode='lines',
        fill='tonexty',  # this fills the area between the confidence intervals
        fillcolor='rgba(255, 0, 0, 0.3)',  # semi-transparent red fill
        name='Upper Confidence Interval',
        showlegend=False
    )

    # Combine all traces
    data = [historical_trace, lower_band, upper_band, forecast_trace]

    # Set up layout
    layout = go.Layout(
        title='Actual vs Forecast Prices',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Price'),
        showlegend=True
    )

    fig = go.Figure(data=data, layout=layout)
    return fig


def plot_performance_metrics(df_cv):
    fig = plot_cross_validation_metric(df_cv, metric='mape')
    plt.show()
