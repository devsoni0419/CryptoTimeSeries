import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
import sys
import os

sys.path.append(os.path.abspath(""))
from src.arima_model import arima_forecast
from src.sarima_model import sarima_forecast
from src.prophet_model import prophet_forecast
from src.lstm_model import lstm_forecast

st.set_page_config(
    page_title="Time Series Forecast Dashboard",
    layout="wide"
)

st.title("📈 Time Series Forecast Dashboard")


DATA_PATH = "data/processed_stock_data.csv"
df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
df.set_index("Date", inplace=True)

close_series = df["Close"]


st.subheader("Historical Price Trend")

fig_hist = go.Figure()
fig_hist.add_trace(
    go.Scatter(
        x=close_series.index,
        y=close_series.values,
        name="Close Price",
        line=dict(width=2)
    )
)
fig_hist.update_layout(height=400)
st.plotly_chart(fig_hist, use_container_width=True)


FORECAST_DAYS = 30
last_date = close_series.index[-1]
future_dates = pd.date_range(
    start=last_date + timedelta(days=1),
    periods=FORECAST_DAYS,
    freq="B"
)


st.subheader("ARIMA Forecast")

arima_preds = arima_forecast(close_series, steps=FORECAST_DAYS)

fig_arima = go.Figure()
fig_arima.add_trace(go.Scatter(
    x=close_series.index[-90:],
    y=close_series[-90:],
    name="Recent Actual"
))
fig_arima.add_trace(go.Scatter(
    x=future_dates,
    y=arima_preds,
    name="ARIMA Forecast"
))
fig_arima.update_layout(height=350)
st.plotly_chart(fig_arima, use_container_width=True)


st.subheader("SARIMA Forecast")

sarima_preds = sarima_forecast(close_series, steps=FORECAST_DAYS)

fig_sarima = go.Figure()
fig_sarima.add_trace(go.Scatter(
    x=close_series.index[-90:],
    y=close_series[-90:],
    name="Recent Actual"
))
fig_sarima.add_trace(go.Scatter(
    x=future_dates,
    y=sarima_preds,
    name="SARIMA Forecast"
))
fig_sarima.update_layout(height=350)
st.plotly_chart(fig_sarima, use_container_width=True)


st.subheader("Prophet Forecast")

prophet_df = prophet_forecast(df, forecast_days=FORECAST_DAYS)

fig_prophet = go.Figure()
fig_prophet.add_trace(go.Scatter(
    x=close_series.index[-90:],
    y=close_series[-90:],
    name="Recent Actual"
))
fig_prophet.add_trace(go.Scatter(
    x=prophet_df["ds"],
    y=prophet_df["yhat"],
    name="Prophet Forecast"
))
fig_prophet.update_layout(height=350)
st.plotly_chart(fig_prophet, use_container_width=True)

st.subheader("LSTM Forecast")

lstm_df = lstm_forecast(df, forecast_days=FORECAST_DAYS)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df.index[-100:],
    y=df["Close"].iloc[-100:],
    name="Recent Actual",
    line=dict(color="lightblue")
))

fig.add_trace(go.Scatter(
    x=lstm_df["Date"],
    y=lstm_df["LSTM_Forecast"],
    name="LSTM Forecast",
    line=dict(color="orange")
))

fig.update_layout(
    template="plotly_dark",
    xaxis_title="Date",
    yaxis_title="Price"
)

st.plotly_chart(fig, use_container_width=True)

st.info(
    "LSTM forecast is a demonstration output. "
    "Recursive multi-step prediction may accumulate error."
)
