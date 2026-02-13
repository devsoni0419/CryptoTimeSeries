import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def lstm_forecast(df, forecast_days=30):
    data = df[["Close"]].copy()
    data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
    data = data.dropna()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X = []
    y = []

    lookback = 60

    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i])
        y.append(scaled[i])

    X = np.array(X)
    y = np.array(y)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")

    model.fit(X, y, epochs=3, batch_size=32, verbose=0)

    last_seq = scaled[-lookback:]
    preds = []

    for _ in range(forecast_days):
        pred = model.predict(last_seq.reshape(1, lookback, 1), verbose=0)
        preds.append(pred[0][0])
        last_seq = np.vstack([last_seq[1:], pred])

    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

    future_dates = pd.date_range(
        start=data.index[-1] + pd.Timedelta(days=1),
        periods=forecast_days,
        freq="D"
    )

    return pd.DataFrame({
        "Date": future_dates,
        "LSTM_Forecast": preds
    })
