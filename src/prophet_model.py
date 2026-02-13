import pandas as pd
from prophet import Prophet

def prophet_forecast(df, forecast_days=30):
    data = df.reset_index()[["Date", "Return"]]
    data.columns = ["ds", "y"]
    data = data.dropna()

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=False
    )
    model.fit(data)

    future = model.make_future_dataframe(periods=forecast_days, freq="D")
    forecast = model.predict(future)

    return forecast[["ds", "yhat"]].tail(forecast_days)
