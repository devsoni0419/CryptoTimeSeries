import yfinance as yf
import pandas as pd
import os

def download_stock_data(
    ticker="AAPL",
    start="2015-01-01",
    end=None,
    save_path="data/stock_data.csv"
):
    os.makedirs("data", exist_ok=True)

    data = yf.download(ticker, start=start, end=end)

    if data.empty:
        raise RuntimeError("No data downloaded. Check ticker or internet connection.")

    data.reset_index(inplace=True)

    data = data[["Date", "Close"]]

    data.to_csv(save_path, index=False)
    print(f"✅ Data saved to {save_path}")

if __name__ == "__main__":
    download_stock_data()
