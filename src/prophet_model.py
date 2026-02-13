from prophet import Prophet
import pandas as pd

def prophet_forecast(df, forecast_days):
    prophet_df = df.reset_index()[["Date", "Close"]]
    prophet_df.columns = ["ds", "y"]

    model = Prophet(daily_seasonality=True)
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)

    return forecast[["ds", "yhat"]].tail(forecast_days)
