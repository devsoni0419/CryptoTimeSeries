import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def sarima_forecast(series, steps=30):
    series = pd.to_numeric(series, errors="coerce").dropna()
    model = SARIMAX(
        series,
        order=(1, 0, 1),
        seasonal_order=(1, 0, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    fitted = model.fit(disp=False)
    return fitted.forecast(steps=steps).values
