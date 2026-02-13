import pandas as pd

def preprocess_stock_data(
    input_path="data/stock_data.csv",
    output_path="data/processed_stock_data.csv"
):
    df = pd.read_csv(input_path)

    df = df.dropna(subset=["Date", "Close"])

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    df = df.dropna()

    df = df.sort_values("Date")

    df = df[["Date", "Close"]]

    df.to_csv(output_path, index=False)

    return df
