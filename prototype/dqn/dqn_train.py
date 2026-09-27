import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from dqn_environment import (TradingEnvironment,NUM_ACTIONS)
from dqn_agent import (DQNAgent)
from pathlib import Path

# version
VERSION = "v1-DQN_variant_4"

# set seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# tickers
TICKERS = ["GOOG", "NVDA", "MSFT", "AMZN", "AVGO", "TSLA", "META", "LLY", "MU", "JPM",
          "WMT", "AMD", "V", "XOM", "MA", "INTC", "ABBV", "CSCO", "BAC", "ORCL"]

# files
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
PREDICTIONS_FILE = (OUTPUT_DIR / f"../outputs/multi_{len(TICKERS)}_ticker_ohlcv_validation_predictions_v1.csv")
MODEL_FILE = (OUTPUT_DIR / f"../models/dqn_multi_ticker_model_{VERSION}.keras")
TRAINING_HISTORY_FILE = (OUTPUT_DIR / f"../outputs/dqn_training_history_{VERSION}.csv")

# training configurations
STATE_SIZE = 4          # number of features in each environment state
TRAINING_ROUNDS = 10    # number of training rounds over all ticker environments
INITIAL_CASH = 10000    # initial cash balance for each trading episode
TRANSACTION_COST = 0    # default: 0    # transaction cost applied to each trade

SAVE_INTERVAL = 5               # save the DQN model every x completed episodes
UPDATE_AFTER_ACTIONS = 4        # train the DQN using experience replay every x environment steps
UPDATE_TARGET_NETWORK = 1000    # update the target Q-network every 1000 environment steps

GAMMA = 0.95                # default: 0.95      # discount factor for future rewards
EPSILON = 1.0               # epsilon greedy parameter
EPSILON_MIN = 0.05          # minimum epsilon greedy parameter
EPSILON_DECAY = 0.995       # default:  0.995             # decay rate for epsilon greedy parameter
LEARNING_RATE = 0.001       # default: 0.001       # learning rate for the optimizer
REPLAY_BUFFER_SIZE = 20000  # size of the replay buffer
BATCH_SIZE = 64             # size of the minibatch for training taken from replay buffer


# load validation predictions
results = pd.read_csv(PREDICTIONS_FILE)
results["Date"] = pd.to_datetime(results["Date"])

# check if all tickers are present in the predictions file and sort by ticker/date
training_data = results[results["Ticker"].isin(TICKERS)].copy()
training_data = (training_data.sort_values(["Ticker", "Date"]).reset_index(drop=True))

# create separate trading environments for each ticker
environments = {}
for ticker in TICKERS:
    ticker_data = (training_data[training_data["Ticker"] == ticker].sort_values("Date").reset_index(drop=True))
    # minimum 3 data points are required for the environment to work (previous day price, current day price, predicted price)
    if len(ticker_data) < 3:
        print(f"Skipping {ticker}: not enough data.")
        continue
    # trading environment for this ticker
    env = TradingEnvironment(
        ticker=ticker,
        prices=ticker_data["Actual_Close"].values,
        predicted_close=ticker_data["Predicted_Close"].values,
        dates=ticker_data["Date"].values,
        initial_cash=INITIAL_CASH,
        transaction_cost=(TRANSACTION_COST)
    )
    environments[ticker] = env
print("Environment tickers:",list(environments.keys()))

# create DQN agent (one shared agent for all tickers)
agent = DQNAgent(state_size=STATE_SIZE, num_actions=NUM_ACTIONS, gamma=GAMMA, epsilon=EPSILON, epsilon_min=EPSILON_MIN, epsilon_decay=EPSILON_DECAY, learning_rate=LEARNING_RATE, replay_buffer_size=REPLAY_BUFFER_SIZE)
agent.model.summary()

# training history to store results for each episode
training_history = []
episode_count = 0
frame_count = 0

# training loop
for training_round in range(TRAINING_ROUNDS):
    print("---------")
    print(f"Training round {training_round + 1} / {TRAINING_ROUNDS}")

    # random ticker order
    ticker_order = list(environments.keys())
    random.shuffle(ticker_order)

    # each ticker --> one episode
    for ticker in ticker_order:
        env = environments[ticker]
        state = env.reset()
        done = False
        episode_reward = 0
        losses = []
        action_counts = {0: 0, 1: 0, 2: 0}

        while not done:
            frame_count += 1
            # choose action
            action = agent.act(state, training=True)
            action_counts[action] += 1
            # execute action in ticker environment
            state_next, reward, done, info = env.step(action)
            # store transition in replay buffer
            agent.remember(state, action, reward, state_next, done)
            # train DQN periodically using experience replay
            if frame_count % UPDATE_AFTER_ACTIONS == 0:
                loss = agent.replay(batch_size=BATCH_SIZE)
                if loss is not None:
                    losses.append(loss)
            # update target Q-network periodically
            if frame_count % UPDATE_TARGET_NETWORK == 0:
                agent.update_target_model()
            state = state_next
            episode_reward += reward
        episode_count += 1

        # decay epsilon after each episode
        agent.decay_epsilon()

        final_portfolio_value = env.get_portfolio_value()
        average_loss = np.mean(losses) if losses else np.nan

        print(
            f"{ticker:5s} "
            f"Portfolio: {final_portfolio_value:,.2f} "
            f"Reward: {episode_reward:.5f} "
            f"Epsilon: {agent.epsilon:.4f}"
        )

        # save training history
        training_history.append({
            "Training_Round": training_round + 1,
            "Episode": episode_count,
            "Ticker": ticker,
            "Episode_Reward": episode_reward,
            "Final_Portfolio_Value": final_portfolio_value,
            "Average_Loss": average_loss,
            "Epsilon": agent.epsilon,
            "Hold_Count": action_counts[0],
            "Buy_Count": action_counts[1],
            "Sell_Count": action_counts[2],
            "Total_Trades": env.total_trades
        })

        # save model periodically
        if episode_count % SAVE_INTERVAL == 0:
            agent.model.save(MODEL_FILE)

# update target model weights
agent.update_target_model()
# save model 
agent.model.save(MODEL_FILE)
# save training history
training_history_df = pd.DataFrame(training_history)
training_history_df.to_csv(TRAINING_HISTORY_FILE,index=False)
