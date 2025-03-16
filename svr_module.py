from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np


def preprocess_data(df, window=3):
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df['3d_MA'] = df['Close'].rolling(window=window).mean().bfill()

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    features_scaled = feature_scaler.fit_transform(df[['Close', '3d_MA']])
    df['scaled_Close'] = target_scaler.fit_transform(df[['Close']])
    return features_scaled, df['scaled_Close'], feature_scaler, target_scaler


def train_svr(X_train, y_train):
    svr = SVR(kernel='rbf')
    svr.fit(X_train, y_train)
    return svr


def evaluate_svr(model, X_test, y_test, target_scaler):
    y_pred_scaled = model.predict(X_test)
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, np.sqrt(mse), y_pred, r2


def plot_svr_predictions(y_test, y_pred, ticker):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title(f'{ticker} - Actual vs Predicted Prices')
    plt.legend()
    plt.show()
