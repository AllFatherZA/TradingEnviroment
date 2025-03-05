import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from collections import deque
import os
import math
from utils import TradingGraph  # Ensure this module is properly imported

class CustomEnv:
    def __init__(self, df, drawdown_p, crystalball, stoploss, lot_size, account_size, decimals, spread, multiple, df_normalized, 
                 conversion=1, initial_balance=5000, lookback_window_size=100, render_range=100, show_reward=False, show_indicators=False):
        self.df = df.reset_index()
        self.initial_balance = initial_balance
        self.lookback_window_size = lookback_window_size
        self.render_range = render_range
        self.show_reward = show_reward
        self.show_indicators = show_indicators
        self.df_total_steps = len(self.df) - 1
        self.orders_history = deque(maxlen=self.lookback_window_size)
        self.market_history = deque(maxlen=self.lookback_window_size)
        self.df_normalized = df_normalized
        self.columns = list(self.df.columns[5:])
        
        # Trading parameters
        self.decimals = decimals
        self.lot_size = lot_size
        self.account_size = account_size
        self.spread = spread
        self.multiple = multiple
        self.conversion = conversion
        self.stoploss = stoploss
        self.drawdown_p = drawdown_p
        self.crystalball = crystalball
        
        # State variables
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.positions_held = 0
        self.buys = 0
        self.sells = 0
        self.holding = False
        self.hold_value = 0
        self.order_buy = False
        self.order_sell = False
        self.pnl = 0
        self.trades_analyst = [0]
        self.accurate = 0
        self.trades = deque(maxlen=self.render_range)

    def reset(self, env_steps_size=4320):
        self.visualization = TradingGraph(render_range=self.render_range, show_reward=self.show_reward, show_indicators=self.show_indicators)
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.positions_held = 0
        self.buys = 0
        self.sells = 0
        self.holding = False
        self.hold_value = 0
        self.order_buy = False
        self.order_sell = False
        self.pnl = 0
        self.trades_analyst = [0]
        self.accurate = 0
        self.trades.clear()

        if env_steps_size > 0:
            self.start_step = self.df_total_steps - env_steps_size
            self.end_step = self.start_step + env_steps_size
        else:
            self.start_step = self.lookback_window_size
            self.end_step = self.df_total_steps

        self.current_step = self.start_step

        for i in reversed(range(self.lookback_window_size)):
            current_step = self.current_step - i
            self.orders_history.append([self.balance, self.net_worth, self.buys, self.sells, self.positions_held])
            self.market_history.append([self.df.loc[current_step, column] for column in self.columns])

        return self.df[self.current_step - self.crystalball:self.current_step]

    def next_observation(self):
        return self.df.loc[(self.current_step - self.crystalball):self.current_step, 'Open':'CloseDiff']

    def step(self, action):
        current_price = self.df.loc[self.current_step, 'Close']
        date = self.df.loc[self.current_step, 'Date']
        high = self.df.loc[self.current_step, 'High']
        low = self.df.loc[self.current_step, 'Low']

        if self.order_buy:
            pnl = self.buy_pnl_index(current_price)
            if pnl < -self.stoploss:
                self.balance += pnl
                self.pnl += pnl
                self.order_buy = False
                self.holding = False
                self.trades.append({'Date': date, 'High': high, 'Low': low, 'total': self.buys, 'type': "close", 'current_price': current_price})
                self.trades_analyst.append(pnl)

        if self.order_sell:
            pnl = self.sell_pnl_index(current_price)
            if pnl < -self.stoploss:
                self.balance += pnl
                self.pnl += pnl
                self.order_sell = False
                self.holding = False
                self.trades.append({'Date': date, 'High': high, 'Low': low, 'total': self.sells, 'type': "close", 'current_price': current_price})
                self.trades_analyst.append(pnl)

        if action == 1 and self.balance > self.initial_balance * self.drawdown_p:
            if not self.order_sell and not self.holding:
                self.buys += 1
                self.trades.append({'Date': date, 'High': high, 'Low': low, 'total': self.buys, 'type': "buy", 'current_price': current_price})
                self.holding = True
                self.hold_value = current_price
                self.order_buy = True

        if action == -1 and self.balance > self.initial_balance * self.drawdown_p:
            if not self.order_buy and not self.holding:
                self.sells += 1
                self.trades.append({'Date': date, 'High': high, 'Low': low, 'total': self.sells, 'type': "sell", 'current_price': current_price})
                self.holding = True
                self.hold_value = current_price
                self.order_sell = True

        if self.order_buy:
            pnl = self.buy_pnl_index(current_price)
            self.net_worth = self.balance + pnl
        if self.order_sell:
            pnl = self.sell_pnl_index(current_price)
            self.net_worth = self.balance + pnl

        self.orders_history.append([self.balance, self.net_worth, self.buys, self.sells, self.positions_held])
        self.current_step += 1

        done = self.net_worth <= self.initial_balance * self.drawdown_p
        obs = self.next_observation()

        return obs, done

    def render(self, visualize=False):
        if visualize:
            return self.visualization.render(self.df.loc[self.current_step], self.net_worth, self.trades)

    def buy_pnl_index(self, current_price):
        return round(self.conversion * (current_price - self.hold_value - self.spread) * self.lot_size * self.multiple, 2)

    def sell_pnl_index(self, current_price):
        return round(self.conversion * (self.hold_value - current_price - self.spread) * self.lot_size * self.multiple, 2)


def random_games(env, visualize, test_episodes=1, comment=""):
    days = 0
    step = 10 / 1440

    for episode in range(test_episodes):
        state = env.reset()
        while True:
            env.render(visualize)
            action = get_action(state)
            state, done = env.step(action)
            os.system('cls')
            days += step
            print(f"Accuracy %: {env.accurate / env.episode_orders}, \nNet Worth: {env.balance}, \nAverage Profit per Trade: {env.pnl / env.episode_orders}, \nOrders: {env.episode_orders}, \nMax Profit: {max(env.trades_analyst)}, \nMax Loss: {min(env.trades_analyst)}, \nTotal Accurate Trades: {env.accurate}, \nDays: {days}, \nProfit %: {(env.pnl / env.initial_balance) * 100}")
            if done or env.current_step == env.end_step:
                break


def get_action(df):
    future_days = 1
    df[f'{future_days}_Day_Price_Forecast'] = df['CloseDiff'].shift(-future_days)
    df = df[['CloseDiff', f'{future_days}_Day_Price_Forecast']]
    X = np.array(df[['CloseDiff']])
    X = X[:df.shape[0] - future_days]
    y = np.array(df[f'{future_days}_Day_Price_Forecast'])
    y = y[:-future_days]

    poly = PolynomialFeatures(degree=4)
    X_poly = poly.fit_transform(X)
    lin_reg = LinearRegression()
    lin_reg.fit(X_poly, y)

    svr_rbf_confidence = lin_reg.score(X_poly, y)
    print(f"SVR Accuracy: {svr_rbf_confidence}")

    y_current = np.array(df['CloseDiff'].iloc[-1])
    prediction = 1 if 0 > y_current else -1 if 0 < y_current else 0
    print(f"Prediction: {prediction}")

    return prediction


# Load data and initialize environment
df = pd.read_csv('../US30(10).csv')
df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
df['CloseDiff'] = df['Close'].diff()
df = df[3000:]

env = CustomEnv(df=df, crystalball=1000, drawdown_p=0.95, df_normalized=df, lot_size=0.02, account_size=1, decimals=1, spread=6, multiple=1, conversion=17.99, stoploss=60)
random_games(env, visualize=True)