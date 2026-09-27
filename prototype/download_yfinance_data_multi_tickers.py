import yfinance as yf
import pandas as pd
import os

# version
VERSION = "v1"

# set target data
PERIOD = '5y'
TICKERS = ["GOOG", "NVDA", "MSFT", "AMZN", "AVGO", "TSLA", "META", "LLY", "MU", "JPM",
          "WMT", "AMD", "V", "XOM", "MA", "INTC", "ABBV", "CSCO", "BAC", "ORCL"]
FEATURES = ["Open", "High", "Low", "Close", "Volume"]
FILE_NAME = f"./outputs/stock_data_{len(TICKERS)}_ohlcv_{PERIOD}_{VERSION}.csv"

# create directories not exist
os.makedirs("./outputs", exist_ok=True)

# download data
data = yf.download(TICKERS, period=PERIOD)

# convert yfinance multi-index to long format
data = (data.stack(level='Ticker', future_stack=True).reset_index())

# remove data with missing values
data = data.dropna(subset=FEATURES)

# sort by ticker, then date
data = data.sort_values(['Ticker', 'Date']).reset_index(drop=True)

# save to csv file
data.to_csv(FILE_NAME, index=False)

# data info
print('---------')
print(f"Data (head):\n{data.head()}")
print('---------')
print(f"Data (tail):\n{data.tail()}")
print('---------')
print(f"Ticker: {data['Ticker'].unique()}")
print('---------')
print(f"Shape: {data.shape}")   