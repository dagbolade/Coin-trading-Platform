from keras.src.layers import Bidirectional
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


def prepare_bi_lstm_data(df, column='Close', sequence_length=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[[column]])

    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i - sequence_length:i])
        y.append(scaled_data[i])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshape for LSTM [samples, time steps, features]

    return X, y, scaler


def build_bi_lstm_model(input_shape):
    bi_model = Sequential()

    # Add a Bidirectional LSTM layer
    bi_model.add(Bidirectional(LSTM(units=50, return_sequences=True,
                                    kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                                    bias_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
                               input_shape=input_shape))

    # Optionally add more LSTM layers
    bi_model.add(Bidirectional(LSTM(units=50, return_sequences=False)))
    bi_model.add(Dropout(0.3))

    # Output layer
    bi_model.add(Dense(units=1))

    # Compile the model
    bi_model.compile(optimizer='adam', loss='mean_squared_error')
    return bi_model


def train_bi_lstm_model(bi_model, X_train, y_train, X_test, y_test, ):
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = bi_model.fit(X_train, y_train, epochs=100, batch_size=64,
                           validation_data=(X_test, y_test), verbose=1,
                           callbacks=[early_stopping])

    return bi_model, history


def evaluate_bi_lstm_model(bi_model, X_test, y_test, scaler):
    y_pred = bi_model.predict(X_test)
    y_pred_inv = scaler.inverse_transform(y_pred)  # Inverse transform to get actual price
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    mse = mean_squared_error(y_test_inv, y_pred_inv)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_inv, y_pred_inv)

    return mse, mae, rmse, r2


# save the model
def save_bi_lstm_model(bi_model, model_path):
    bi_model.save(model_path)


def plot_history(history, ticker):
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.title(f'{ticker} - Model Training and Validation Loss')
    plt.legend()
    plt.show()


def plot_bi_predictions(y_test_inv, y_pred_inv, ticker):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_inv, label='Actual Price')
    plt.plot(y_pred_inv, label='Predicted Price')
    plt.title(f'{ticker} - Actual vs Predicted Closing Prices')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
