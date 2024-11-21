import os
import pandas as pd
import yfinance as yf


def main():
    stocks = ["AAPL", "MSFT", "AMZN", "GOOG", "TSLA", "NVDA", "META"]
    crypto = ["BTC-USD", "ETH-USD", "DOGE-USD", "ADA-USD", "SOL-USD"]
    forex = ["EURUSD=X", "GBPUSD=X", "JPY=X", "EURJPY=X", "EURGBP=X", "CNY=X"]

    if not os.path.exists("data"):
        save_data(stocks, "stocks")
        save_data(crypto, "crypto")
        save_data(forex, "forex")


def save_data(data, name):
    folder = f"data/{name}"

    if not os.path.exists(folder):
        os.makedirs(folder)

    for item in data:
        csv = yf.download(item, period="max")
        csv.to_csv(f"{folder}/{item.lower()}.csv")

        # Clean data
        df = pd.read_csv(f"{folder}/{item.lower()}.csv")
        df.drop([0, 1], inplace=True)
        df.rename(columns={"Price": "Date"}, inplace=True)
        if name == "forex":
            df.drop("Volume", axis=1, inplace=True)
        df.to_csv(f"{folder}/{item.lower()}.csv", index=False)


if __name__ == "__main__":
    main()
