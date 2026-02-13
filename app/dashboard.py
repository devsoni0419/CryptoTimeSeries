import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sys
import os

sys.path.append(os.path.abspath(""))

from src.arima_model import arima_forecast
from src.sarima_model import sarima_forecast
from src.prophet_model import prophet_forecast
from src.lstm_model import lstm_forecast


st.set_page_config(page_title="Time Series Forecast Dashboard", layout="wide")
st.title("📈 Time Series Forecast Dashboard")


DATA_PATH = "data/processed_stock_data.csv"
df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
df.set_index("Date", inplace=True)

df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
df = df.dropna().sort_index().asfreq("D")

df["Return"] = df["Close"].pct_change()
df = df.dropna()

series = df["Return"].astype(float)

FORECAST_DAYS = 30

train = series[:-FORECAST_DAYS]
test = series[-FORECAST_DAYS:]

future_dates = pd.date_range(
    start=test.index[0],
    periods=FORECAST_DAYS,
    freq="D"
)

def metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse
    }


st.subheader("Historical Returns")

fig = go.Figure()
fig.add_trace(go.Scatter(x=series.index, y=series, name="Returns"))
st.plotly_chart(fig, use_container_width=True)


st.subheader("Naive Baseline")

naive_preds = np.repeat(train.iloc[-1], FORECAST_DAYS)
naive_metrics = metrics(test.values, naive_preds)
st.write(naive_metrics)


st.subheader("ARIMA Forecast")

arima_preds = arima_forecast(train, steps=FORECAST_DAYS)
arima_metrics = metrics(test.values, arima_preds)

fig = go.Figure()
fig.add_trace(go.Scatter(x=train.index[-90:], y=train[-90:], name="Train"))
fig.add_trace(go.Scatter(x=future_dates, y=arima_preds, name="ARIMA"))
st.plotly_chart(fig, use_container_width=True)

st.write(arima_metrics)


st.subheader("SARIMA Forecast")

sarima_preds = sarima_forecast(train, steps=FORECAST_DAYS)
sarima_metrics = metrics(test.values, sarima_preds)

fig = go.Figure()
fig.add_trace(go.Scatter(x=train.index[-90:], y=train[-90:], name="Train"))
fig.add_trace(go.Scatter(x=future_dates, y=sarima_preds, name="SARIMA"))
st.plotly_chart(fig, use_container_width=True)

st.write(sarima_metrics)


st.subheader("Prophet Forecast")

prophet_df = prophet_forecast(df, forecast_days=FORECAST_DAYS)
prophet_preds = prophet_df["yhat"].values
prophet_metrics = metrics(test.values, prophet_preds)

fig = go.Figure()
fig.add_trace(go.Scatter(x=train.index[-90:], y=train[-90:], name="Train"))
fig.add_trace(go.Scatter(x=future_dates, y=prophet_preds, name="Prophet"))
st.plotly_chart(fig, use_container_width=True)

st.write(prophet_metrics)


st.subheader("LSTM Forecast")

lstm_df = lstm_forecast(df, forecast_days=FORECAST_DAYS)
lstm_preds = lstm_df["LSTM_Forecast"].values
lstm_metrics = metrics(test.values, lstm_preds)

fig = go.Figure()
fig.add_trace(go.Scatter(x=train.index[-90:], y=train[-90:], name="Train"))
fig.add_trace(go.Scatter(x=future_dates, y=lstm_preds, name="LSTM"))
st.plotly_chart(fig, use_container_width=True)

st.write(lstm_metrics)


st.info(
    "Forecasts are evaluated on returns. "
    "Financial time series often resemble random walks."
)
