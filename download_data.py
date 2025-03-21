import os
from datetime import datetime

import yfinance as yf
from pandas import DataFrame


def main():
    stocks: list[str] = ["AAPL", "MSFT", "AMZN", "GOOG", "TSLA", "NVDA", "META"]
    crypto: list[str] = ["BTC-USD", "ETH-USD", "DOGE-USD", "ADA-USD", "SOL-USD"]
    forex: list[str] = ["EURUSD=X", "GBPUSD=X", "JPY=X", "EURJPY=X", "EURGBP=X", "CNY=X"]

    if not os.path.exists("data"):
        print("Downloading…")
        save_data(stocks, "stocks")
        save_data(crypto, "crypto")
        save_data(forex, "forex")
        print("Done.")
    else:
        print("Data already exists!")


def save_data(data: list[str], category: str) -> None:
    folder: str = f"data/{category}"
    if not os.path.exists(folder):
        os.makedirs(folder)

    today: str = datetime.today().strftime("%Y-%m-%d")
    for item in data:
        # Using "today" to actually download until yesterday so the market is closed.
        df: DataFrame = yf.download(item, end=today, progress=False)

        # DATA CLEANING
        # Remove "Close" column ("Adj Close" is better), but renaming
        # "Adj Close" to "Close" because training would fail otherwise.
        df.columns = ["Adj Close", "Close", "High", "Low", "Open", "Volume"]
        df.drop(columns=["Close"], inplace=True)
        df.rename(columns={"Adj Close": "Close"}, inplace=True)

        if category == "forex":  # "Volume" is always zero in forex data
            df.drop(columns=["Volume"], inplace=True)

        df.to_csv(f"{folder}/{item.lower()}.csv")


if __name__ == "__main__":
    main()
