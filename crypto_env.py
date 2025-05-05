import numpy as np
import pandas as pd

from trading_env import TradingEnv, Actions, Positions
from gymnasium.envs.registration import register
default_budget = 10000


class CryptoEnv(TradingEnv):
    def __init__(self, df, window_size, frame_bound, trade_fee=0.001, render_mode=None):
        assert len(frame_bound) == 2

        self.frame_bound = frame_bound
        self.trade_fee = trade_fee  # fee of 0.1% per trade
        self.current_price = None
        self.total_profit = 0
        self.open_price = None
        self._money_spent = 0

        # Wallet
        self.budget = default_budget
        self._crypto_holdings = 0

        super().__init__(df, window_size, render_mode)

    def reset(self, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)

        # Reset wallet variables
        self.budget = default_budget
        self._crypto_holdings = 0

        return observation, info

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
            hold_penalty = self.prices[self._last_trade_tick] * 0.001  # Penalità per inattività
            if self._last_trade_tick is not None:
                last_trade_price = self.prices[self._last_trade_tick]
                hold_penalty = last_trade_price * 0.0001  # 0.01% del prezzo dell'ultimo trade
            step_reward -= hold_penalty

        trade = ((action == Actions.Buy.value and self._position == Positions.Short) or
                 (action == Actions.Sell.value and self._position == Positions.Long))

        if trade:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]
            price_diff = current_price - last_trade_price
            percent_change = price_diff / last_trade_price

            if self._position == Positions.Short:
                step_reward -= percent_change
            elif self._position == Positions.Long:
                step_reward += percent_change

            # Ricompensa positiva per una vendita con profitto
            if action == Actions.Sell.value and percent_change > 0:
                step_reward += percent_change * 10  # Ricompensa proporzionale al guadagno

        # Incorporare profitto e ROI nella ricompensa
        profit = self.get_total_profit()
        roi = (profit - 1) * 100
        step_reward += profit + roi

        return step_reward


    def _update_profit(self, action):
        if action == Actions.Hold.value:
            return  # Nessuna modifica al profitto se si mantiene la posizione

        trade = ((action == Actions.Buy.value and self._position == Positions.Short) or
                 (action == Actions.Sell.value and self._position == Positions.Long))

        if trade or self._truncated:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]

            if action == Actions.Buy.value:
                # Calculate how much crypto we can buy with current budget
                shares = (self.budget * (1 - self.trade_fee)) / current_price
                self._crypto_holdings += shares
                self.budget = 0  # All budget used for buying
                self._money_spent += shares * current_price

            elif action == Actions.Sell.value:
                # Calculate cash received from selling all crypto
                cash_received = self._crypto_holdings * current_price * (1 - self.trade_fee)
                self.budget += cash_received
                self._crypto_holdings = 0  # All crypto sold

            # Update total profit as the wallet value
            self._total_profit = self.get_wallet_value()/default_budget

    def get_wallet_value(self):
        """Returns the total value of the wallet (budget + crypto)"""
        if self._current_tick is None:
            return default_budget

        current_price = self.prices[self._current_tick]
        return self.budget + (self._crypto_holdings * current_price)

    def get_wallet_info(self):
        """Returns detailed information about the wallet"""
        if self._current_tick is None:
            return {
                'budget': self.budget,
                'crypto_holdings': 0,
                'crypto_value': 0,
                'total_value': self.budget
            }

        current_price = self.prices[self._current_tick]
        return {
            'budget': self.budget,
            'crypto_holdings': self._crypto_holdings,
            'crypto_value': self._crypto_holdings * current_price,
            'total_value': self.budget + (self._crypto_holdings * current_price)
        }


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

    def get_current_price(self):
        if self._current_tick is None:
            return None
        return self.prices[self._current_tick]

    def get_budget(self):
        return self.budget


    def _get_info(self):
        info = super()._get_info()
        wallet_info = self.get_wallet_info()
        info.update(wallet_info)
        return info


# c_df = pd.read_csv("data/crypto/btc-usd.csv")
# register(
#     id="crypto-v0",
#     entry_point="crypto_env:CryptoEnv",
#     kwargs={"df": c_df, "window_size": 30, "frame_bound": (30, len(c_df))},
# )

