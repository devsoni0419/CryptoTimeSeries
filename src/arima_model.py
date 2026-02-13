import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def arima_forecast(series, steps=30):
    series = pd.to_numeric(series, errors="coerce").dropna()
    model = ARIMA(series, order=(5, 1, 0))
    fitted = model.fit()
    forecast = fitted.forecast(steps=steps)
    return forecast.values
