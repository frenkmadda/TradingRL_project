import numpy as np
import pandas as pd

from gym_anytrading.envs.trading_env import TradingEnv, Actions, Positions
from gymnasium.envs.registration import register


class CryptoEnv(TradingEnv):
    def __init__(self, df, window_size, frame_bound, render_mode=None):
        assert len(frame_bound) == 2

        self.frame_bound = frame_bound
        self.current_price = None
        self.total_profit = 0
        self.open_price = None

        super().__init__(df, window_size, render_mode)

    def _process_data(self):
        # Extracts the 'Close' prices from the data frame.
        # Computes the price differences and combines them with the prices to form signal features.
        prices = self.df.loc[:, "Close"].to_numpy()
        prices = prices[self.frame_bound[0] - self.window_size : self.frame_bound[1]]

        diff = np.diff(prices)
        signal_features = np.column_stack((prices, diff))

        return prices, signal_features

    def _calculate_reward(self, action):
        # Calculates the reward based on the action taken (Buy or Sell) and the current position (Long or Short).
        # The reward is the difference between the current price and the open price.
        step_reward = 0
        self.current_price = self.prices[self._current_tick]

        if action == Actions.Buy and self._position == Positions.Short:
            step_reward = self.current_price - self.open_price
        elif action == Actions.Sell and self._position == Positions.Long:
            step_reward = self.open_price - self.current_price

        return step_reward

    def _update_profit(self, action):
        # Updates the total profit based on the action taken (Buy or Sell) and the current position (Long or Short).
        # The total profit is the sum of the differences between the current price and the open price.
        if action == Actions.Buy and self._position == Positions.Short:
            self.total_profit += self.current_price - self.open_price
        elif action == Actions.Sell and self._position == Positions.Long:
            self.total_profit += self.open_price - self.current_price

    def max_possible_profit(self):
        # Calculates the maximum possible profit by finding the difference between the maximum and minimum prices within
        # the frame bounds
        prices = self.df.loc[:, "Close"].to_numpy()
        prices = prices[self.frame_bound[0] : self.frame_bound[1]]

        return np.max(prices) - np.min(prices)


c_df = pd.read_csv("data/crypto/btc-usd.csv")
register(
    id="crypto-v0",
    entry_point="crypto_env:CryptoEnv",
    kwargs={"df": c_df, "window_size": 30, "frame_bound": (30, len(c_df))},
)
