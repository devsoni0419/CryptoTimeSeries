import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

st.set_page_config(page_title="Time Series Forecast Dashboard", layout="wide")

DATA_PATH = "data/stock_data.csv"
FORECAST_DAYS = 30
LOOKBACK = 60

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    return df[["Close"]].dropna()

df = load_data()

st.title("📈 Time Series Forecast Dashboard")

st.subheader("Historical Price Trend")
fig_hist = go.Figure()
fig_hist.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close"))
fig_hist.update_layout(template="plotly_dark", xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig_hist, use_container_width=True)

st.subheader("ARIMA Forecast")

recent = df["Close"].iloc[-200:].astype(float)

arima = ARIMA(recent, order=(5, 1, 0))
arima_model = arima.fit()

arima_forecast = arima_model.forecast(FORECAST_DAYS)

arima_dates = pd.date_range(
    start=recent.index[-1] + pd.Timedelta(days=1),
    periods=FORECAST_DAYS,
    freq="D"
)

arima_df = pd.DataFrame(
    {"ARIMA Forecast": arima_forecast},
    index=arima_dates
)

st.line_chart(arima_df)

st.subheader("SARIMA Forecast")
sarima = SARIMAX(recent, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
sarima_forecast = sarima.forecast(FORECAST_DAYS)

fig_sarima = go.Figure()
fig_sarima.add_trace(go.Scatter(x=recent.index, y=recent, name="Recent Actual"))
fig_sarima.add_trace(go.Scatter(x=arima_dates, y=sarima_forecast, name="SARIMA Forecast"))
fig_sarima.update_layout(template="plotly_dark")
st.plotly_chart(fig_sarima, use_container_width=True)

st.subheader("Prophet Forecast")
prophet_df = df.reset_index().rename(columns={"Date": "ds", "Close": "y"})
model = Prophet()
model.fit(prophet_df)
future = model.make_future_dataframe(periods=FORECAST_DAYS)
forecast = model.predict(future)

fig_prophet = go.Figure()
fig_prophet.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Actual"))
fig_prophet.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="Prophet Forecast"))
fig_prophet.update_layout(template="plotly_dark")
st.plotly_chart(fig_prophet, use_container_width=True)

st.subheader("LSTM Forecast")
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[["Close"]])

X, y = [], []
for i in range(LOOKBACK, len(scaled)):
    X.append(scaled[i - LOOKBACK:i])
    y.append(scaled[i])

X, y = np.array(X), np.array(y)

model = Sequential([
    Input(shape=(X.shape[1], 1)),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.fit(X, y, epochs=3, batch_size=32, verbose=0)

last_seq = scaled[-LOOKBACK:]
future_preds = []

for _ in range(FORECAST_DAYS):
    pred = model.predict(last_seq.reshape(1, LOOKBACK, 1), verbose=0)
    future_preds.append(pred[0, 0])
    last_seq = np.append(last_seq[1:], pred, axis=0)

lstm_forecast = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()

fig_lstm = go.Figure()
fig_lstm.add_trace(go.Scatter(x=recent.index, y=recent, name="Recent Actual"))
fig_lstm.add_trace(go.Scatter(x=arima_dates, y=lstm_forecast, name="LSTM Forecast"))
fig_lstm.update_layout(template="plotly_dark")
st.plotly_chart(fig_lstm, use_container_width=True)
