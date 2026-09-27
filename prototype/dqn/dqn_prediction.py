import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from .dqn_environment import TradingEnvironment, ACTION_HOLD, ACTION_BUY, ACTION_SELL

# version
VERSION = "v1-DQN_variant_4"

# tickers
TICKERS = ["GOOG", "NVDA", "MSFT", "AMZN", "AVGO", "TSLA", "META", "LLY", "MU", "JPM", "WMT", "AMD", "V", "XOM", "MA", "INTC", "ABBV", "CSCO", "BAC", "ORCL"]

# files
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
PREDICTIONS_FILE = (OUTPUT_DIR /f"../outputs/multi_{len(TICKERS)}_ticker_ohlcv_test_predictions_v1.csv")
MODEL_FILE = (OUTPUT_DIR /f"../models/dqn_multi_ticker_model_{VERSION}.keras")
DQN_PREDICTIONS_FILE = (OUTPUT_DIR /f"../outputs/dqn_test_predictions_{VERSION}.csv")

# prediction configurations
INITIAL_CASH = 10000
TRANSACTION_COST = 0
# actions
ACTION_NAMES = {ACTION_HOLD: "Hold", ACTION_BUY: "Buy", ACTION_SELL: "Sell"}

# function for loading dqn model
def load_dqn_model():
    model = tf.keras.models.load_model(MODEL_FILE, compile=False)
    return model

# function to predict one DQN action for chatbot
def predict_dqn_action(model, previous_close, current_close, predicted_close, cash=INITIAL_CASH, stock=0):
    # state (same as TradingEnvironment.get_state())
    # LSTM predicted return
    predicted_return = (predicted_close / current_close) - 1
    # previous actual market return
    recent_return = (current_close / previous_close) - 1
    # current stock value
    stock_value = stock * current_close
    # current portfolio value
    portfolio_value = cash + stock_value
    if portfolio_value <= 0:
        portfolio_value = 0.00000001
    # normalize cash and holdings
    cash_ratio = cash / portfolio_value
    stock_ratio = stock_value / portfolio_value

    # DQN state
    state = np.array([predicted_return, recent_return, cash_ratio, stock_ratio], dtype=np.float32)
    # batch (expand dimention (4,) --> (1, 4))
    state_batch = np.expand_dims(state, axis=0)
    # predict q values
    q_values = model.predict(state_batch, verbose=0)[0]
    # select action wigh highest q value
    action_id = int(np.argmax(q_values))
    return {"action_id": action_id,
            "action": ACTION_NAMES[action_id],
            "q_hold": float(q_values[ACTION_HOLD]),
            "q_buy": float(q_values[ACTION_BUY]),
            "q_sell": float(q_values[ACTION_SELL]),
            "state": state}

def generate_test_predictions():
    # load LSTM test predictions
    test_data = pd.read_csv(PREDICTIONS_FILE)
    test_data["Date"] = pd.to_datetime(test_data["Date"])
    # keep selected tickers and sort by ticker/date
    test_data = test_data[test_data["Ticker"].isin(TICKERS)].copy()
    test_data = test_data.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    # load trained DQN model
    model = tf.keras.models.load_model(MODEL_FILE, compile=False)
    # store DQN predictions
    dqn_predictions = []

    # generate predictions for each ticker
    for ticker in TICKERS:
        ticker_data = test_data[test_data["Ticker"] == ticker].sort_values("Date").reset_index(drop=True)
        if len(ticker_data) < 3:
            print(f"Skipping {ticker}: not enough test data.")
            continue

        # create trading environment for this ticker
        env = TradingEnvironment(ticker=ticker, 
                                prices=ticker_data["Actual_Close"].values, 
                                predicted_close=ticker_data["Predicted_Close"].values, 
                                dates=ticker_data["Date"].values, 
                                initial_cash=INITIAL_CASH, 
                                transaction_cost=TRANSACTION_COST)
        state = env.reset()
        done = False
        action_counts = {ACTION_HOLD: 0, ACTION_BUY: 0, ACTION_SELL: 0}

        # DQN prediction loop
        while not done:
            current_step = env.current_step
            current_date = ticker_data.loc[current_step, "Date"]
            actual_close = float(ticker_data.loc[current_step, "Actual_Close"])
            predicted_close = float(ticker_data.loc[current_step, "Predicted_Close"])

            # predict Q-values for the current state
            state_batch = np.expand_dims(state, axis=0)
            q_values = model.predict(state_batch, verbose=0)

            # choose action with the highest Q-value
            action = int(np.argmax(q_values[0]))
            action_counts[action] += 1

            # execute action to obtain the next state
            state_next, reward, done, info = env.step(action)

            # save DQN prediction
            dqn_predictions.append({"Ticker": ticker, 
                                    "Date": current_date, 
                                    "Actual_Close": actual_close, 
                                    "Predicted_Close": predicted_close, 
                                    "Action": ACTION_NAMES[action], 
                                    "Action_ID": action, 
                                    "Q_Hold": float(q_values[0, ACTION_HOLD]), 
                                    "Q_Buy": float(q_values[0, ACTION_BUY]), 
                                    "Q_Sell": float(q_values[0, ACTION_SELL])})
            state = state_next
        print(f"{ticker:5s}:  Hold: {action_counts[ACTION_HOLD]:3d} Buy: {action_counts[ACTION_BUY]:3d} Sell: {action_counts[ACTION_SELL]:3d}")

    # save DQN predictions
    dqn_predictions_df = pd.DataFrame(dqn_predictions)
    dqn_predictions_df.to_csv(DQN_PREDICTIONS_FILE, index=False)

if __name__ == "__main__":
    generate_test_predictions()