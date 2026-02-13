import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

def lstm_forecast(df, forecast_days=30, lookback=60):
    prices = df["Close"].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)

    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i])
        y.append(scaled[i])

    X, y = np.array(X), np.array(y)

    model = Sequential([
        Input(shape=(X.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    last_window = scaled[-lookback:]
    future = []

    for _ in range(forecast_days):
        pred = model.predict(last_window.reshape(1, lookback, 1), verbose=0)
        future.append(pred[0, 0])
        last_window = np.vstack([last_window[1:], pred])

    future_prices = scaler.inverse_transform(np.array(future).reshape(-1, 1)).flatten()

    future_dates = pd.date_range(
        start=df.index[-1] + pd.Timedelta(days=1),
        periods=forecast_days
    )

    return pd.DataFrame({
        "Date": future_dates,
        "LSTM_Forecast": future_prices
    })
