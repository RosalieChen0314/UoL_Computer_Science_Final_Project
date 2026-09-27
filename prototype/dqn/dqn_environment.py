import numpy as np

# actions
ACTION_HOLD = 0
ACTION_BUY = 1
ACTION_SELL = 2
NUM_ACTIONS = 3

class TradingEnvironment:
    def __init__(self, ticker, prices, predicted_close, dates=None, initial_cash=10000.0, transaction_cost=0.0):
        self.ticker = ticker
        self.prices = np.asarray(prices, dtype=np.float32)
        self.predicted_close = np.asarray(predicted_close, dtype=np.float32)

        if dates is not None:
            self.dates = np.asarray(dates)
        else:
            self.dates = np.arange(len(self.prices))

        self.initial_cash = float(initial_cash)
        self.transaction_cost = float(transaction_cost)
        self.reset()


    # reset the environment to the initial state
    def reset(self):
        # use previous day's price --> start from index 1
        self.current_step = 1
        self.cash = self.initial_cash
        self.stock = 0
        self.previous_portfolio_value = (self.initial_cash)
        self.total_trades = 0
        return self._get_state()

    # state
    def _get_state(self):
        current_price = self.prices[self.current_step]
        previous_price = self.prices[self.current_step - 1]
        predicted_price = self.predicted_close[self.current_step]

        # LSTM predicted return
        predicted_return = (predicted_price / current_price - 1)
        # previous actual market return
        recent_return = (current_price / previous_price - 1)
        # current stock value
        stock_value = (self.stock * current_price)
        # current total portfolio value
        portfolio_value = (self.cash + stock_value)

        if portfolio_value <= 0:
            portfolio_value = 0.00000001 # zero will cause problems in operations

        # normalize cash and holdings
        cash_ratio = (self.cash / portfolio_value)
        stock_ratio = (stock_value / portfolio_value)
        state = np.array([predicted_return, recent_return, cash_ratio, stock_ratio], dtype=np.float32)
        return state

    # execute action
    def step(self, action):
        current_price = float(self.prices[self.current_step])
        trade_executed = False

        # BUY --> buy one share
        if action == ACTION_BUY:
            total_cost = (current_price * (1 + self.transaction_cost))
            if self.cash >= total_cost:
                self.stock += 1
                self.cash -= total_cost
                self.total_trades += 1
                trade_executed = True

        # SELL --> sell one share
        elif action == ACTION_SELL:
            if self.stock > 0:
                total_sales = (current_price * (1 - self.transaction_cost))
                self.stock -= 1
                self.cash += total_sales
                self.total_trades += 1
                trade_executed = True

        # HOLD --> do nothing
        elif action == ACTION_HOLD:
            pass

        else:
            raise ValueError(f"Unknown action: {action}")


        # move to the next step (next trading day)
        self.current_step += 1
        # define when the episode is done  
        done = (self.current_step>= len(self.prices) - 1)
        next_price = float(self.prices[self.current_step])

        # update portfolio value
        portfolio_value = (self.cash + self.stock * next_price)

        # reward --> % of portfolio change
        reward = (portfolio_value - self.previous_portfolio_value) / max(self.previous_portfolio_value, 0.00000001)

        # update previous portfolio value
        self.previous_portfolio_value = (portfolio_value)
        next_state = self._get_state()

        # info of the current step
        info = {
            "ticker": self.ticker,
            "date": self.dates[self.current_step],
            "price": next_price,
            "cash": self.cash,
            "stock": self.stock,
            "portfolio_value": portfolio_value,
            "trade_executed": trade_executed}

        return (next_state, float(reward), done, info)

    # get current portfolio value
    def get_portfolio_value(self):
        current_price = float(self.prices[self.current_step])
        return (self.cash + self.stock * current_price)