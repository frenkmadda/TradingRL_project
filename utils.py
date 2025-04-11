import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm


def train_model(model, total_learning_timesteps=1_000_000):
    # Addestra il modello e restituisce l'ambiente vettorializzato
    model.learn(total_timesteps=total_learning_timesteps, callback=ProgressBar(100))
    return model.get_env()


def test_model(model, env, total_num_episodes=50):
    rewards = []
    portfolio_histories = []
    buy_sell_markers = []

    for episode in tqdm(range(total_num_episodes), desc="Testing"):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        portfolio = []
        buys = []
        sells = []
        holdings = []

        while not done:
            if model is not None:
                action, _ = model.predict(obs)
            else:
                action = env.action_space.sample()

            step_result = env.step(action)
            if len(step_result) == 4:
                obs, reward, done, info = step_result
            else:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated

            total_reward += reward

            # Recupero del valore del portafoglio da info
            wallet_value = None
            if isinstance(info, list) and len(info) > 0:
                wallet_value = info[0].get("total_value", None)
            elif isinstance(info, dict):
                wallet_value = info.get("total_value", None)

            if wallet_value is not None:
                portfolio.append(wallet_value)

            current_price = env.get_current_price()

            if isinstance(current_price, list):
                current_price = current_price[0]

            budget = env.get_budget()
            if isinstance(budget, list):
                budget = budget[0]

            if action == 1:  # Buy
                buys.append(len(portfolio) - 1)
                print(f"BUY at ${current_price:.2f}, Portfolio Value: ${wallet_value:.2f}, Budget: ${budget:.2f}")
            elif action == 0:  # Sell
                sells.append(len(portfolio) - 1)
                print(f"SELL at ${current_price:.2f}, Portfolio Value: ${wallet_value:.2f}, Budget: ${budget:.2f}")
            elif action == 2:  # Hold
                holdings.append(len(portfolio) - 1)
                print(f"HOLD at ${current_price:.2f}, Portfolio Value: ${wallet_value:.2f}, Budget: ${budget:.2f}")

        rewards.append(total_reward)
        portfolio_histories.append(portfolio)

        if action == 1 or action == 0:
            buy_sell_markers.append((buys, sells))

    return rewards, info, portfolio_histories, buys, sells, holdings




def train_test_model(model, train_env, test_env, seed=69, total_learning_timesteps=1_000_000, total_num_episodes=50, train=True):
    # Imposta il seed
    train_env.reset(seed=seed)
    train_env = train_model(model, total_learning_timesteps)

    # Se richiesto, addestra il modello e usa l'ambiente vettorializzato; altrimenti, usa gym_env
    # if train and model is not None:
    #     train_env = train_model(model, total_learning_timesteps)
    # else:
    #     env = train_env

    rewards, info, portfolio_histories, buys, sells, holdings = test_model(model, test_env, total_num_episodes)

    train_env.close()
    test_env.close()
    return rewards, info, portfolio_histories, buys, sells, holdings



def get_results(reward_over_episodes, model_name, print_results=False):
    avg_reward = np.mean(reward_over_episodes)
    min_reward = np.min(reward_over_episodes)
    max_reward = np.max(reward_over_episodes)

    if print_results:
        print(f"\nResults for {model_name} model:")
        print(f"Minimum reward: {min_reward:.3f}")
        print(f"Maximum reward: {max_reward:.3f}")
        print(f"Average reward: {avg_reward:.3f}\n")

    return min_reward, max_reward, avg_reward


# Progress bar for model.learn()
class ProgressBar(BaseCallback):
    def __init__(self, check_freq: int, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq

    def _on_training_start(self) -> None:
        self.progress_bar = tqdm(
            total=self.model._total_timesteps, desc="model.learn()"
        )

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            self.progress_bar.update(self.check_freq)
        return True

    def _on_training_end(self) -> None:
        self.progress_bar.close()
