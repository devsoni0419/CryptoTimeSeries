import pandas as pd

def load_processed_data(path="data/processed_stock_data.csv"):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    return df
