import utils
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import A2C, PPO, DQN
from crypto_env import CryptoEnv
import gym_anytrading
import gymnasium as gym
import quantstats as qs

from stable_baselines3 import A2C, PPO, DQN
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def plot_rewards(rewards, model_name, total_num_episodes):
    '''
    Function to plot the distribuition of the rewards
    :param rewards: the rewards obtained from the model
    :param model_name: the model chosen for the training, namely PPO, A2C or DQN
    :param total_num_episodes: the total number of episodes
    '''

    plot_settings = {}
    plot_data = {"x": [i for i in range(1, total_num_episodes + 1)]}
    plot_data[f"{model_name}_rewards"] = rewards
    plot_settings[f"{model_name}_rewards"] = {"label": model_name}
    rewards_data = pd.DataFrame(plot_data)
    plt.figure(figsize=(12, 6))

    for key in plot_data:
        if key == "x":
            continue
        plt.plot("x", key, data=rewards_data, linewidth=1, label=plot_settings[key]["label"])

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"Rewards per Episode for {model_name}")
    plt.legend()
    # plt.show()
    filename = f'grafici/rewards_{model_name.lower()}.png'
    plt.savefig(filename)
    plt.close()


def plot_portfolio_during_evaluation(portfolio_values, buy_idx, sell_idx, hold_idx, model_name, roi):
    plt.figure(figsize=(14, 7))
    plt.plot(portfolio_values, label='Portfolio Value')

    for b in buy_idx:
        plt.scatter(b, portfolio_values[b], color='green', marker='^', s=100, label='Buy' if b == buy_idx[0] else "")
    for s in sell_idx:
        # if s > 0 and portfolio_values[s] < portfolio_values[s - 1]:  # Verifica se c'è effettivamente una vendita
        plt.scatter(s, portfolio_values[s], color='red', marker='v', s=100, label='Sell' if s == sell_idx[0] else "")
    for h in hold_idx:
        plt.scatter(h, portfolio_values[h], color='blue', marker='o', s=30, label='Hold' if h == hold_idx[0] else "")

    plt.title(f'Portfolio Value During Evaluation ({model_name}) (ROI: {roi:.2f}%)')
    plt.xlabel('Trading Days')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True, which='both', linestyle='-', linewidth=0.5)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)

    filename = f'grafici/eval_portfolio_{model_name.lower()}.png'
    plt.savefig(filename, dpi=300)
    print(f"[Salvato grafico in {filename}]")
    plt.close()


def train_and_get_rewards(model_name, train_env, test_env, seed, total_learning_timesteps, total_num_episodes):
    '''
    Function used to train the models
    :param model_name: The model chosen between A2C, PPO and DQN
    :param env: The gym environment created for the training
    :param seed: namely the seed
    :param total_learning_timesteps:
    :param total_num_episodes:
    '''

    print(f"Training {model_name} model…")

    if model_name == "PPO":
        model = PPO("MlpPolicy", train_env)
    elif model_name == "DQN":
        model = DQN("MlpPolicy", train_env,
                    learning_rate=0.0005,
                    buffer_size=50000,
                    learning_starts=1000,
                    batch_size=32,
                    tau=1.0,
                    gamma=0.99,
                    train_freq=4,
                    target_update_interval=1000,
                    exploration_fraction=0.1,
                    exploration_final_eps=0.05,
                    verbose=0,
                    device="cuda")
    elif model_name == "A2C":
        model = A2C("MlpPolicy", train_env,
                    learning_rate=0.0003,
                    gamma=0.95,
                    n_steps=5,
                    vf_coef=0.5,
                    ent_coef=0.01,
                    max_grad_norm=0.5,
                    verbose=0)
    else:
        model = None

    rewards, info, portfolio_histories, buys, sells, holdings = utils.train_test_model(
        model, train_env, test_env, seed, total_learning_timesteps, total_num_episodes)

    # Calcolo ROI
    if isinstance(info, list):
        profit = info[0]["total_profit"]
    else:
        profit = info["total_profit"]

    roi = (profit - 1) * 100

    # Plot e salvataggio
    plot_rewards(rewards, model_name, total_num_episodes)
    plot_portfolio_during_evaluation(
        portfolio_values=portfolio_histories[0],
        buy_idx=buys,
        sell_idx=sells,
        hold_idx=holdings,
        # buy_idx=buy_sell_markers[0][0],
        # sell_idx=buy_sell_markers[0][1],
        model_name=model_name,
        roi=roi
    )


if __name__ == "__main__":
    dataset_path = "data/crypto/ada-usd.csv"
    dataset_type = "crypto-v0"  # "stocks-v0", "forex-v0", "crypto-v0"

    df = pd.read_csv(
        dataset_path,
        header=0,
        parse_dates=["Date"],
        index_col="Date",
    )
    df.head()

    # split the dataset into train and test
    train_size = int(len(df) * 0.85)
    test_size = len(df) - train_size
    train_df = df[:train_size]
    test_df = df[train_size:]


    seed = 69  # Nice
    total_num_episodes = 100
    total_learning_timesteps = 1_000

    window_size = 15
    end_index = len(df)

    train_env = CryptoEnv(
        df=train_df,
        window_size=window_size,
        frame_bound=(window_size, end_index),
    )

    test_env = CryptoEnv(
        df=test_df,
        window_size=window_size,
        frame_bound=(window_size, end_index),
    )

    # Using the function in train_and_test.py
    train_and_get_rewards("DQN", train_env, test_env, seed, total_learning_timesteps, total_num_episodes)
    train_and_get_rewards("PPO", train_env, test_env, seed, total_learning_timesteps, total_num_episodes)
    train_and_get_rewards("A2C", train_env, test_env, seed, total_learning_timesteps, total_num_episodes)
