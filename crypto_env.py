import numpy as np
import pandas as pd

from gym_anytrading.envs.trading_env import TradingEnv, Actions, Positions
from gymnasium.envs.registration import register


class CryptoEnv(TradingEnv):
    def __init__(self, df, window_size, frame_bound, trade_fee=0.001, render_mode=None):
        assert len(frame_bound) == 2

        self.frame_bound = frame_bound
        self.trade_fee = trade_fee  # fee of 0.1% per trade
        self.current_price = None
        self.total_profit = 0
        self.open_price = None
        self._money_spent = 0

        super().__init__(df, window_size, render_mode)

    def _process_data(self):
        # Extracts the 'Close' prices from the data frame.
        # Computes the price differences and combines them with the prices to form signal features.
        prices = self.df.loc[:, "Close"].to_numpy()

        # prices[self.frame_bound[0] - self.window_size]  # validate index
        prices = prices[self.frame_bound[0] - self.window_size : self.frame_bound[1]]

        diff = np.insert(np.diff(prices), 0, 0)
        signal_features = np.column_stack((prices, diff))

        return prices.astype(np.float32), signal_features.astype(np.float32)

    def _calculate_reward(self, action):
        step_reward = 0

        if action == Actions.Hold.value:
            step_reward += -50  # Penalità per evitare inattività

        trade = ((action == Actions.Buy.value and self._position == Positions.Short) or
                 (action == Actions.Sell.value and self._position == Positions.Long))

        if trade:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]
            price_diff = current_price - last_trade_price

            if self._position == Positions.Short:
                step_reward += -price_diff
            elif self._position == Positions.Long:
                step_reward += price_diff

        return step_reward

    def _update_profit(self, action):
        if action == Actions.Hold.value:
            return  # Nessuna modifica al profitto se si mantiene la posizione

        trade = ((action == Actions.Buy.value and self._position == Positions.Short) or
                 (action == Actions.Sell.value and self._position == Positions.Long))

        if trade or self._truncated:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]

            if self._position == Positions.Long:
                shares = (self._total_profit * (1 - self.trade_fee)) / last_trade_price
                self._total_profit = (shares * (1 - self.trade_fee)) * current_price
                self._money_spent += shares * last_trade_price
            elif self._position == Positions.Short:
                shares = (self._total_profit * (1 - self.trade_fee)) / current_price
                self._total_profit = (shares * (1 - self.trade_fee)) * last_trade_price
                self._money_spent += shares * current_price

    def max_possible_profit(self):
        current_tick = self._start_tick
        last_trade_tick = current_tick - 1
        profit = 1.0

        while current_tick <= self._end_tick:
            if self.prices[current_tick] < self.prices[current_tick - 1]:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] < self.prices[current_tick - 1]):
                    current_tick += 1

                current_price = self.prices[current_tick - 1]
                last_trade_price = self.prices[last_trade_tick]
                shares = profit / current_price
                profit = shares * last_trade_price * (1 - self.trade_fee)
            else:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] >= self.prices[current_tick - 1]):
                    current_tick += 1

                current_price = self.prices[current_tick - 1]
                last_trade_price = self.prices[last_trade_tick]
                shares = profit / last_trade_price
                profit = shares * current_price * (1 - self.trade_fee)

            last_trade_tick = current_tick - 1

        return profit

    def get_total_profit(self):
        return self._total_profit

    def get_money_spent(self):
        return self._money_spent


c_df = pd.read_csv("data/crypto/btc-usd.csv")
register(
    id="crypto-v0",
    entry_point="crypto_env:CryptoEnv",
    kwargs={"df": c_df, "window_size": 30, "frame_bound": (30, len(c_df))},
)
