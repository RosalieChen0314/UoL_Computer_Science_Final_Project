import os
import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import joblib

# version

VERSION = "v1"

# training configurations
TICKERS = ["GOOG", "NVDA", "MSFT", "AMZN", "AVGO", "TSLA", "META", "LLY", "MU", "JPM",
          "WMT", "AMD", "V", "XOM", "MA", "INTC", "ABBV", "CSCO", "BAC", "ORCL"]
PERIOD = '5y'
WINDOW_SIZE = 60
FEATURES = ["Open", "High", "Low", "Close", "Volume"]

# input file name
INPUT_FILE = f"./outputs/stock_data_{len(TICKERS)}_ohlcv_{PERIOD}_{VERSION}.csv"


# output file names
MODEL_FILE = f"./models/lstm_ohlcv_multi_tickers_model_{VERSION}.keras"
SCALER_FILE = f"./models/ohlcv_multi_tickers_scaler_{VERSION}.save"
VALIDATION_RESULTS_FILE = f"./outputs/multi_{len(TICKERS)}_ticker_ohlcv_validation_predictions_{VERSION}.csv"
TEST_RESULTS_FILE = f"./outputs/multi_{len(TICKERS)}_ticker_ohlcv_test_predictions_{VERSION}.csv"

# train/validation/test split ratios
TRAIN_RATIO = 0.7
VALIDATION_RATIO = 0.15
# TEST_RATIO = 0.15

# create directories not exist
os.makedirs("./models", exist_ok=True)
os.makedirs("./outputs", exist_ok=True)

# load data
data = pd.read_csv(INPUT_FILE)
data["Date"] = pd.to_datetime(data["Date"])

# keep rows while its ticker appears in the ticker list
data = data[data["Ticker"].isin(TICKERS)].copy()

# sort data with Ticker, Date
data = data.sort_values(["Ticker", "Date"]).reset_index(drop=True)

# splitting configuration for train/val/test data (chronological) of each ticker
split_indices = {}
training_rows = []

# process each ticker to determine the split indices for train/val/test sets
for ticker, ticker_data in data.groupby("Ticker"):
    ticker_data = (ticker_data.sort_values("Date").reset_index(drop=True))
    n = len(ticker_data)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VALIDATION_RATIO))
    split_indices[ticker] = {"train_end": train_end, "val_end": val_end}
    training_rows.append(ticker_data.iloc[:train_end][FEATURES])

# combine dataframes for each ticker
training_rows = pd.concat(training_rows, ignore_index=True)
# fit scaler apply to training data
scaler = StandardScaler()
scaler.fit(training_rows[FEATURES].values)

# create train/val/test datasets (60-day windows)
X_train, y_train, X_val, y_val, X_test, y_test = [], [], [], [], [], []
train_info, val_info, test_info = [], [], []

# process each ticker
for ticker, ticker_data in data.groupby("Ticker"):
    ticker_data = (ticker_data.sort_values("Date").reset_index(drop=True))
    n = len(ticker_data)
    # extract the split settings for ticker
    train_end = split_indices[ticker]["train_end"]
    val_end = split_indices[ticker]["val_end"]
    # scale 
    scaled_values = scaler.transform(ticker_data[FEATURES].values)
    # 60-day windows
    for i in range(WINDOW_SIZE, n):
        X = scaled_values[i - WINDOW_SIZE : i]
        # target y
        y = scaled_values[i]
        target_data = ticker_data.loc[i, "Date"]
        # training dataset
        if i < train_end:
            X_train.append(X)
            y_train.append(y)
            train_info.append({"Ticker": ticker, "Date": target_data})
        # validation dataset
        elif i < val_end:
            X_val.append(X)
            y_val.append(y)
            val_info.append({"Ticker": ticker, "Date": target_data})
        # test dataset
        else:
            X_test.append(X)
            y_test.append(y)
            test_info.append({"Ticker": ticker, "Date": target_data})

# convert to numpy arrays
X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)
X_val = np.array(X_val, dtype=np.float32)
y_val = np.array(y_val, dtype=np.float32)
X_test = np.array(X_test, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32)

# dataset shapes 
print("---------")
print("Dataset:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_val:", X_val.shape)
print("y_val:", y_val.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

# build model (LSTM)
model = keras.models.Sequential([
    keras.layers.Input(shape=(WINDOW_SIZE, len(FEATURES))),
    keras.layers.LSTM(64, return_sequences=True),   # default 64
    keras.layers.LSTM(64, return_sequences=False),  # default 64
    keras.layers.Dense(128, activation="relu"),     # default 128
    keras.layers.Dropout(0.5),                      # default 0.5
    # outputs: Open, High, Low, Close, Volume
    keras.layers.Dense(len(FEATURES))])
model.summary()

# compile
model.compile(optimizer="adam", loss="mae", 
              metrics=[keras.metrics.MeanSquaredError(name="mse"),
                       keras.metrics.RootMeanSquaredError(name="rmse")])  

# train model
history = model.fit(X_train, y_train, 
                    validation_data=(X_val, y_val),
                    epochs=20,          # default 20
                    batch_size=32,      # default 32
                    verbose=1
                    )

# test evaluation
test_mae, test_mse, test_rmse = model.evaluate(X_test, y_test, verbose=1)
print("---------")
print("Test results:")
print("MAE:", test_mae)
print("MSE:", test_mse)
print("RMSE:", test_rmse)

# validation prediction (scaled)
validation_predictions_scaled = model.predict(X_val, verbose=0)
# validation predictions (original units)
validation_predictions = scaler.inverse_transform(validation_predictions_scaled)
# validation actual values (original units)
validation_actual_values = scaler.inverse_transform(y_val)
# validation ticker/date information
val_info_df = pd.DataFrame(val_info)
# validation results (compare prediction/actual values)
validation_results = pd.DataFrame({
    "Ticker": val_info_df["Ticker"],
    "Date": val_info_df["Date"],
    "Actual_Open": validation_actual_values[:, 0],
    "Predicted_Open": validation_predictions[:, 0],
    "Actual_High": validation_actual_values[:, 1],
    "Predicted_High": validation_predictions[:, 1],
    "Actual_Low": validation_actual_values[:, 2],
    "Predicted_Low": validation_predictions[:, 2],
    "Actual_Close": validation_actual_values[:, 3],
    "Predicted_Close": validation_predictions[:, 3],
    "Actual_Volume":validation_actual_values[:, 4],
    "Predicted_Volume": validation_predictions[:, 4]})
# sort results by ticker and date
validation_results = (validation_results.sort_values(["Ticker", "Date"]).reset_index(drop=True))
# save validation results
validation_results.to_csv(VALIDATION_RESULTS_FILE, index=False)

# test prediction (scaled)
predictions_scaled = model.predict(X_test, verbose=0)
# test prediction (original units)
predictions = scaler.inverse_transform(predictions_scaled)
# test (actual, original units)
actual_values = scaler.inverse_transform(y_test)
# test results (compare prediction/autual values)
test_info_df = pd.DataFrame(test_info)
results = pd.DataFrame({
    "Ticker": test_info_df["Ticker"],
    "Date": test_info_df["Date"],
    "Actual_Open": actual_values[:, 0],
    "Predicted_Open": predictions[:, 0],
    "Actual_High": actual_values[:, 1],
    "Predicted_High": predictions[:, 1],
    "Actual_Low": actual_values[:, 2],
    "Predicted_Low": predictions[:, 2],
    "Actual_Close": actual_values[:, 3],
    "Predicted_Close": predictions[:, 3],
    "Actual_Volume": actual_values[:, 4],
    "Predicted_Volume": predictions[:, 4]
    })
# save test results
results.to_csv(TEST_RESULTS_FILE, index=False)

# save scaler
joblib.dump(scaler, SCALER_FILE)
# save model
model.save(MODEL_FILE)

