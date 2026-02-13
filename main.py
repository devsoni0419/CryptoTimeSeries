from src.data_loader import load_processed_data
from src.arima_model import arima_forecast
from src.sarima_model import sarima_forecast
from src.prophet_model import prophet_forecast
from src.lstm_model import lstm_forecast

df = load_processed_data()
series = df["Close"]

print("ARIMA:", arima_forecast(series).head())
print("SARIMA:", sarima_forecast(series).head())
print("Prophet:\n", prophet_forecast(df).head())
print("LSTM next price:", lstm_forecast(series))
