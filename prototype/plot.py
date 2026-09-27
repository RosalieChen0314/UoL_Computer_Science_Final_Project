from matplotlib import pyplot as plt
import pandas as pd

# version
VERSION = "v1"

# configuration
RESULTS_FILE = f"./outputs/multi_20_ticker_ohlcv_test_predictions_{VERSION}.csv"
FEATURES = ["Open", "High", "Low", "Close", "Volume"]

# load prediction results
results = pd.read_csv(RESULTS_FILE)
results["Date"] = pd.to_datetime(results["Date"])

# sort results data (confirm)
results = (results.sort_values(["Ticker", "Date"]).reset_index(drop=True))

# plot actual and predicted values for each ticker and feature
for ticker in results["Ticker"].unique():
    ticker_results = results[results["Ticker"] == ticker].copy()
    for feature in FEATURES:
        plt.figure(figsize=(12, 6))
        plt.plot(ticker_results["Date"], ticker_results[f"Actual_{feature}"], label=f"Actual {feature}")
        plt.plot(ticker_results["Date"], ticker_results[f"Predicted_{feature}"], label=f"Predicted {feature}")
        plt.xlabel("Date")
        plt.ylabel(feature)
        plt.title(f"{ticker}: Actual and Predicted {feature}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"./plots/{VERSION}_{ticker}_actual_and_predicted_{feature.lower()}.jpg")
        plt.close()