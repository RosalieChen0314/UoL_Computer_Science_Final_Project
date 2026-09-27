from flask import Flask, render_template, request, jsonify
import joblib
from predict_util import load_model
from chatbot import get_bot_response
from pathlib import Path

# version
VERSION = "v1"

app = Flask(__name__)

# load model
MODEL_DIR = Path(__file__).resolve().parent / "models"
lstm_model = load_model(MODEL_DIR / f"lstm_ohlcv_multi_tickers_model_{VERSION}.keras")
scaler = joblib.load(MODEL_DIR / f"ohlcv_multi_tickers_scaler_{VERSION}.save")
dqn_model = load_model(MODEL_DIR / f"dqn_multi_ticker_model_{VERSION}.keras")

# render index.html
@app.route("/")
def home():
    return render_template("index.html")

# chat
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "")
    bot_response = get_bot_response(user_input, lstm_model, scaler, dqn_model)
    print(bot_response)
    bot_response_json = jsonify({"response": bot_response})
    return bot_response_json


if __name__ == "__main__":
    app.run(debug=True)