import numpy as np
import pandas as pd
from pathlib import Path
from dqn_environment import TradingEnvironment, ACTION_HOLD, ACTION_BUY, ACTION_SELL

# version
VERSION = "v1-DQN_variant_4"
# tickers
TICKERS = ["GOOG", "NVDA", "MSFT", "AMZN", "AVGO", "TSLA", "META", "LLY", "MU", "JPM", "WMT", "AMD", "V", "XOM", "MA", "INTC", "ABBV", "CSCO", "BAC", "ORCL"]

# files
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
DQN_PREDICTIONS_FILE = (OUTPUT_DIR / f"../outputs/dqn_test_predictions_{VERSION}.csv")
RESULTS_FILE = (OUTPUT_DIR / f"../outputs/dqn_test_results_{VERSION}.csv")
SUMMARY_FILE = (OUTPUT_DIR / f"../outputs/dqn_test_summary_{VERSION}.csv")

# evaluation configurations
INITIAL_CASH = 10000.0
TRANSACTION_COST = 0.0
ACTION_NAMES = {ACTION_HOLD: "Hold", ACTION_BUY: "Buy", ACTION_SELL: "Sell"}

# load DQN predictions
prediction_data = pd.read_csv(DQN_PREDICTIONS_FILE)
prediction_data["Date"] = pd.to_datetime(prediction_data["Date"])

# keep selected tickers and sort by ticker/date
prediction_data = prediction_data[prediction_data["Ticker"].isin(TICKERS)].copy()
prediction_data = prediction_data.sort_values(["Ticker", "Date"]).reset_index(drop=True)

# store detailed and summary evaluation results
trading_results = []
summary_results = []

# evaluate each ticker independently
for ticker in TICKERS:
    ticker_data = prediction_data[prediction_data["Ticker"] == ticker].sort_values("Date").reset_index(drop=True)

    if len(ticker_data) < 2:
        print(f"Skipping {ticker}: not enough DQN predictions.")
        continue

    # create trading environment for evaluation
    env = TradingEnvironment(ticker=ticker, 
                             prices=ticker_data["Actual_Close"].values, 
                             predicted_close=ticker_data["Predicted_Close"].values, 
                             dates=ticker_data["Date"].values, initial_cash=INITIAL_CASH, transaction_cost=TRANSACTION_COST)
    env.reset()
    done = False

    action_counts = {ACTION_HOLD: 0, ACTION_BUY: 0, ACTION_SELL: 0}

    # baseline
    first_price = float(ticker_data.loc[0, "Actual_Close"])
    last_price = float(ticker_data.loc[len(ticker_data) - 1, "Actual_Close"])

    baseline_shares = int(INITIAL_CASH // first_price)
    baseline_cash = INITIAL_CASH - baseline_shares * first_price
    buy_hold_final = baseline_cash + baseline_shares * last_price

    step = 0

    # evaluation loop
    while not done and step < len(ticker_data):
        row = ticker_data.loc[step]
        current_date = row["Date"]
        actual_close = float(row["Actual_Close"])
        predicted_close = float(row["Predicted_Close"])
        action = int(row["Action_ID"])
        action_counts[action] += 1

        # execute previously predicted DQN action
        state_next, reward, done, info = env.step(action)
        # save detailed evaluation result
        trading_results.append({"Ticker": ticker, "Date": current_date, "Actual_Close": actual_close, "Predicted_Close": predicted_close, "Action": ACTION_NAMES[action], "Q_Hold": float(row["Q_Hold"]), "Q_Buy": float(row["Q_Buy"]), "Q_Sell": float(row["Q_Sell"]), "Reward": reward, "Cash": info["cash"], "Stock": info["stock"], "Portfolio_Value": info["portfolio_value"]})

        step += 1

    # calculate final portfolio performance
    final_portfolio_value = env.get_portfolio_value()
    dqn_return = final_portfolio_value / INITIAL_CASH - 1.0
    buy_hold_return = buy_hold_final / INITIAL_CASH - 1.0
    excess_return = dqn_return - buy_hold_return

    # save summary
    summary_results.append({
        "Ticker": ticker, 
        "Initial_Value": INITIAL_CASH, 
        "Final_Portfolio_Value": final_portfolio_value, 
        "DQN_Return": dqn_return, 
        "Buy_Hold_Final_Value": buy_hold_final, 
        "Buy_Hold_Return": buy_hold_return, 
        "Excess_Return": excess_return, 
        "Hold_Count": action_counts[ACTION_HOLD], 
        "Buy_Count": action_counts[ACTION_BUY], 
        "Sell_Count": action_counts[ACTION_SELL], 
        "Total_Trades": env.total_trades
        })
    print("---------")
    print(f"{ticker:5s}")
    print(f"DQN: {dqn_return * 100:7.2f}%")
    print(f"Buy/Hold: {buy_hold_return * 100:7.2f}%")
    print(f"Excess: {excess_return * 100:7.2f}%")

# save detailed trading results
trading_results_df = pd.DataFrame(trading_results)
trading_results_df.to_csv(RESULTS_FILE, index=False)
# save evaluation summary
summary_results_df = pd.DataFrame(summary_results)
summary_results_df.to_csv(SUMMARY_FILE, index=False)

# evaluation
if len(summary_results_df) > 0:
    average_dqn_return = summary_results_df["DQN_Return"].mean()
    average_buy_hold_return = summary_results_df["Buy_Hold_Return"].mean()
    average_excess_return = summary_results_df["Excess_Return"].mean()
    outperform_count = (summary_results_df["Excess_Return"] > 0).sum()
    print("---------")
    print(f"Average DQN return: {average_dqn_return * 100:.2f}%")
    print(f"Average buy-and-hold return: {average_buy_hold_return * 100:.2f}%")
    print(f"Average excess return: {average_excess_return * 100:.2f}%")
    print(f"Tickers outperforming Buy and Hold: {outperform_count}/{len(summary_results_df)}")