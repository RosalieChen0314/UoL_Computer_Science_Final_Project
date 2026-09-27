import numpy as np
import yfinance as yf
import tensorflow as tf


WINDOW_SIZE = 60
FEATURES = ["Open", "High", "Low", "Close", "Volume"]

# load model 
def load_model(model_name):
    return tf.keras.models.load_model(model_name, compile=False)

# get last 60 CLOSE prices
def get_last_60_close(stock):
    df = yf.download(stock, period="6mo")
    df = df[["Close"]].dropna()
    return df.tail(WINDOW_SIZE).values

# get last 60 ohlcv
def get_last_60_ohlcv(stock):
    df = yf.download(stock, period="6mo")
    df = df[FEATURES].dropna()
    return df.tail(WINDOW_SIZE).values

# predict (only closing price)
def predict_price(model, data, scaler):
    X = np.array(data)
    # normalize data
    X_scaled = scaler.transform(X)
    
    # predict
    # LSTM expects (samples, timesteps, features)
    X_scaled = X_scaled.reshape(1, WINDOW_SIZE, 1)
    pred_scaled = model.predict(X_scaled, verbose=0)[0][0]
    
    # convert normalized prediction data back to price
    pred_real = scaler.inverse_transform([[pred_scaled]])[0][0]

    return pred_real

# predict all features: "Open", "High", "Low", "Close", "Volume"
def predict_ohlcv(model, data, scaler):
    data = np.array(data) 
    # check data shape
    if data.shape != (WINDOW_SIZE, len(FEATURES)):
        raise ValueError(f"Unexpected input shape {data.shape}. (Expect: ({WINDOW_SIZE}, {len(FEATURES)})")
    
    # scale data
    data_scaled = scaler.transform(data)
    
    # predict
    # LSTM expects (samples, timesteps, features)
    X_scaled = data_scaled.reshape(1, WINDOW_SIZE, len(FEATURES))
    pred_scaled = model.predict(X_scaled, verbose=0)
    
    # convert values back to real units
    pred_real = scaler.inverse_transform(pred_scaled)[0]
    
    prediction_real = {
        "Open": float(pred_real[0]),
        "High": float(pred_real[1]),
        "Low": float(pred_real[2]),
        "Close": float(pred_real[3]),
        "Volume": float(pred_real[4]),
    }
    
    return prediction_real