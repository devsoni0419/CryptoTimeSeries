import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def arima_forecast(series, steps=30):
    series = pd.to_numeric(series, errors="coerce").dropna()
    model = ARIMA(series, order=(1, 0, 1))
    fitted = model.fit()
    return fitted.forecast(steps=steps).values
