# Financial Advisor Bot 
## Steps to run the chatbot
### 1. Set up environment  
#### 1.0. Install the required libraries/packages:
~~~
pip install -r requirements.txt
~~~
#### 1.1. (Optional) Error might occurred when importing keras (optree)  
Solution:

~~~
pip uninstall optree -y  
conda install -c conda-forge optree
~~~
### 2. Check the models are existed
Confirm that in the models folder there are two models and a scaler:
1. dqn_multi_ticker_model_{version}.keras
2. lstm_ohlcv_multi_tickers_model_{version}.keras
3. ohlcv_multi_tickers_scaler_{version}.save
### 3. Run chatbot
#### 3.1. Run Flask API
~~~
python app.py
~~~
#### 3.2. Open the link in the broser or terminal
http://127.0.0.1:5000


## Steps to train the models
### 1. Set up environment  
#### 1.0. Install the required libraries/packages:
~~~
pip install -r requirements.txt
~~~
#### 1.1. (Optional) Error might occurred when importing keras (optree)  
Solution:

~~~
pip uninstall optree -y  
conda install -c conda-forge optree
~~~

### 2. Download history stock market data from yfiance
~~~
python download_yfinance_data_multi_tickers.py
~~~
The data is saved in outputs folder.

### 3. Train the LSTM model
~~~
python train_model_multi_features_multi_ticker.py
~~~

### 4. Train the DQN agent
~~~
python dqn/dqn_train.py
~~~

### 5. Generate DQN predictions
~~~
python -m dqn.dqn_prediction
~~~

### 6. Evaluate DQN decisions
~~~
python dqn/dqn_evaluation.py
~~~

## Steps to plot the relevant data
### 1. plot the LSTM model predicted and actual values for each tickers' OHLCV
~~~
python plot.py
~~~

## Reference list for coding (documentations/blog posts/tutorial videos)
1. [StandardScaler in Python: Scikit-learn Guide & Examples](#https://www.digitalocean.com/community/tutorials/standardscaler-function-in-python)
2. [Deep Q-Networks (DQN) in Finance](#https://leomercanti.medium.com/deep-q-networks-dqn-in-finance-ca31cd92aecc)
3. [Train a Deep Q Network with TF-Agents](#https://www.tensorflow.org/agents/tutorials/2_environments_tutorial)
4. [DQN Agent](#https://deepwiki.com/zcg-joker/Stock_Trading_AI_Based_RL/4.1-dqn-agent)
5. [Deep Learning Trading Strategy from the beginning to the production](#https://www.youtube.com/watch?v=XPWY0x-PpMI&t=4s)
[(Github)](#https://github.com/CloseToAlgoTrading/CodeFromVideo/tree/master/Episode_16_DQN)
6. [DQN Variants Implementation Hands-on Practical](#https://apxml.com/courses/advanced-reinforcement-learning/chapter-2-deep-q-networks/dqn-implementation-practice)