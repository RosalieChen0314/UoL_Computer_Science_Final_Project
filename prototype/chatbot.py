from predict_util import get_last_60_ohlcv, predict_ohlcv
from dqn.dqn_prediction import predict_dqn_action
import yfinance as yf
import pandas as pd
import re
import random
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# download NLTK resources for NER
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')

# intents from chatbot_intents.json
LSTM_INTENTS = {"prediction_close", "prediction_open", "prediction_high", "prediction_low", "prediction_volume", "prediction_all"}
DQN_INTENTS = {"action"}
FINANCIAL_INTENTS = LSTM_INTENTS | DQN_INTENTS
# actions
ACTION_NAMES = {0: "HOLD", 1: "BUY", 2: "SELL"}

# load chatbot intents
def read_json(file):
    try:
        df = pd.read_json(file)
        return df["intents"]
    except FileNotFoundError as e:
        print(e)
        return []
def load_files():
    return read_json("./chatbot_intents.json")

# get intents
# allowed intents: LSTM_INTENTS/ DQN_INTENTS
def get_intent(user_input, allowed_intents):
    intents = load_files()
    normalized_input = user_input.lower()
    for intent in intents:
        if intent["tag"] not in allowed_intents:
            continue
        for pattern in intent["patterns"]:
            pattern_matches = re.search(r"\b" + re.escape(pattern) + r"\b", normalized_input, re.IGNORECASE)
            if pattern_matches:
                return intent
    return None

# Named Entity Recognition (organizations) --> extract company's name from user input
def recognize_organization(text):
    sentences = sent_tokenize(text)
    for sentence in sentences:
        tokens = word_tokenize(sentence)
        tagged = nltk.pos_tag(tokens)
        chunks = nltk.ne_chunk(tagged)
        for entity in chunks:
            if hasattr(entity, "label"):
                if entity.label() == "ORGANIZATION":
                    organization = " ".join(word for word, tag in entity.leaves())
                    return organization
    return None

# extract explicit ticker from user input
def extract_explicit_ticker(text):
    pattern = re.findall(r"\b[A-Z]{1,5}(?:\.[A-Z])?\b", text)
    for p in pattern:
        try:
            data = yf.Ticker(p).history(period="5d")
            if not data.empty:
                return p
        except Exception:
            continue
    return None

# search company's ticker from company name
def search_yfinance(company):
    try:
        search = yf.Search(company, max_results=10, news_count=0)
        for result in search.quotes:
            symbol = result.get("symbol")
            quote_type = result.get("quoteType")
            if symbol and quote_type == "EQUITY":
                print("Yahoo Finance match:", company, symbol)
                return symbol
    except Exception as e:
        print("Yahoo Finance search error:", e)
    return None

# extract company query
# user: "google predict" --> company's name should be extracted
def extract_company_query(user_input):
    query = user_input
    # remove phrases used for intent detection
    intent_words = ["what should i do", "what to do",
        "trading action", "opening price", "open price", "closing price", "close price", "highest price", "high price", "lowest price", "low price", "trading volume", 
        "recommendation","prediction", "predict", "forecast", "overview", "action", "recommend", "suggestion", "suggest", "buy", "sell", "hold",
        "trade", "opening", "closing", "highest", "lowest", "volume", "price", "open", "close", "high", "low"]
    # longest phrases first
    intent_words.sort(key=len, reverse=True)
    for word in intent_words:
        query = re.sub(r"\b" + re.escape(word) + r"\b", " ", query, flags=re.IGNORECASE)
    # remove common words
    filler_words = ["for", "of", "the", "stock", "please", "me", "tell", "give", "can", "you"]
    for word in filler_words:
        query = re.sub(r"\b" + re.escape(word) + r"\b", " ", query, flags=re.IGNORECASE)
    # remove extra spaces
    query = " ".join(query.split())
    return query

# extract stock (explicit ticker/company's name(NER)/yfinance query search)
def extract_stock(user_input):
    # user input includes explicit ticker
    ticker = extract_explicit_ticker(user_input)
    if ticker:
        print(f"Ticker: {ticker}")
        return ticker

    # user input includes company's name --> NER
    company = recognize_organization(user_input)
    if company:
        print(f"Company dectected (NER): {company}")
        ticker = search_yfinance(company)
        if ticker:
            print(f"Ticker: {ticker}")
            return ticker

    # extract company (search phrase)
    company_query = extract_company_query(user_input)
    print(f"Companry search query: {company_query}")
    if not company_query:
        return None

    # yfinance search query
    ticker = search_yfinance(company_query)
    print(f"Ticker: {ticker}")
    return ticker

def generate_response(user_input):
    normalized_input = user_input.lower()
    intents = load_files()
    for intent in intents:
        tag = intent["tag"]
        if tag in FINANCIAL_INTENTS:
            continue
        for pattern in intent["patterns"]:
            if re.search(r"\b" + re.escape(pattern) + r"\b", normalized_input, re.IGNORECASE):
                return random.choice(intent["responses"])
    return None

# LSTM prediction handler
def financial_prediction_handler(user_input, model, scaler):
    # check if user input matches LSTM prediction intents
    prediction_intent = get_intent(user_input, LSTM_INTENTS)
    if prediction_intent is None:
        return None
    
    # identify ticker
    stock = extract_stock(user_input)
    if stock is None:
        return "Please provide a company name (with capital letter) or ticker."
    
    # get historical data and make prediction
    try:
        data = get_last_60_ohlcv(stock)
    except Exception as e:
        print("Stock data eoor:", e)
        return f"Error occurs when retriving market data for {stock}."
    
    # LSTM predict
    try:
        prediction = predict_ohlcv(model, data, scaler)
    except Exception as e:
        print("LSTM prediction error", e)
        return f"Error occurs when generating prediction results for {stock}."

    # generate response
    # get responses from matched intent
    responses = prediction_intent["responses"]
    if not responses:
        return None
    
    # choose one response randomly
    res = random.choice(responses)
    
    # fill response placeholders
    res_filled = res.format(stock=stock, Open=prediction["Open"], Close=prediction["Close"], High=prediction["High"], 
                            Low=prediction["Low"], Volume=prediction["Volume"])
    
    return res_filled

# DQN action handler
def action_handler(user_input, lstm_model, scaler, dqn_model):
    # identify action intent
    action_intent = get_intent(user_input, DQN_INTENTS)
    if action_intent is None:
        return None
    # get ticker
    stock = extract_stock(user_input)
    if stock is None:
        return "No stock identified. Please provide a company name or ticker."
    # get historical data 
    try:
        data = get_last_60_ohlcv(stock)
    except Exception as e:
        print("Stock data error:", e)
        return f"Error occurs when retriving market data for {stock}."
    # LSTM predict: next closing price
    try:
        prediction = predict_ohlcv(lstm_model, data, scaler)
        prediction_close = float(prediction["Close"])

        # Open=0, High=1, Low=2, Close=3, Volume=4
        current_close = float(data[-1, 3])
        previous_close = float(data[-2, 3])
    except Exception as e:
        print("Stock:", stock)
        print("LSTM prediction error :", e)
        return f"Error occurs when generating prediction results for {stock}."

    # DQN predict: action
    try:
        result = predict_dqn_action(model=dqn_model, previous_close=previous_close, current_close=current_close, predicted_close=prediction_close)
        # check q values
        print("--------- DQN ---------")
        print("Previous close:", previous_close)
        print("Current close:", current_close)
        print("Predicted close:", prediction_close)
        print("DQN state:", result["state"])
        print("Q Hold:", result["q_hold"])
        print("Q Buy:", result["q_buy"])
        print("Q Sell:", result["q_sell"])
        print("Action:", result["action"])
        print("------------------------")
    except Exception as e:
        print("DQN prediction error:", e)
        return f"Error occurs when generating trading action for {stock}."

    # chatbot response
    response = random.choice(action_intent["responses"])
    return response.format(stock=stock, action=result["action"], current_close=current_close, predicted_close=prediction_close)

def get_bot_response(user_input, lstm_model, scaler, dqn_model):
    # dqn action request
    response = action_handler(user_input, lstm_model, scaler, dqn_model)
    if response:
        return response
    # request LSTM prediction
    response = financial_prediction_handler(user_input, lstm_model, scaler)
    if response:
        return response
    # if user input does not contain financial prediction keywords but contain chatbot intents
    response = generate_response(user_input)
    if response:
        return response
    # no matching financial keywords nor chatbot intents
    return "I don't understand. Please rephrase it."
