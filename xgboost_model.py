import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score


def preprocess_data(data, features, target):

    X = data[features]
    y = data[target]
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    return X_scaled, y, scaler


def train_xgboost_model(X_train, y_train):
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        colsample_bytree=0.3,
        learning_rate=0.1,
        max_depth=5,
        alpha=10,
        n_estimators=100
    )
    model.fit(X_train, y_train)
    return model


def forecast_xgboost(model, scaler, last_known_features, days):
    # Placeholder to collect predictions
    future_predictions = []

    # Simulate feature set for the number of days
    features_for_prediction = last_known_features.copy()

    for _ in range(days):
        # Predict using reshaped features
        current_pred = model.predict(features_for_prediction)

        # Store the prediction
        future_predictions.append(current_pred[0])


        features_for_prediction = np.array([current_pred[0], current_pred[0], current_pred[0], last_known_features[-1]])

    # Reshape predictions to match the scaler's expected input
    predictions_array = np.array(future_predictions).reshape(-1, 1)

    # Inverse transform predictions
    inverse_scaled_predictions = scaler.inverse_transform(predictions_array).flatten()

    return inverse_scaled_predictions
