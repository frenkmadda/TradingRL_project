import utils
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import A2C, PPO, DQN


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
    plt.show()


def train_and_get_rewards(model_name,  env, seed, total_learning_timesteps, total_num_episodes):
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
        model = PPO("MlpPolicy", env)
    elif model_name == "DQN":
        model = DQN("MlpPolicy", env,
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
        model = A2C("MlpPolicy", env,
                    learning_rate=0.0003,
                    gamma=0.95,
                    n_steps=5,
                    vf_coef=0.5,
                    ent_coef=0.01,
                    max_grad_norm=0.5,
                    verbose=0)
    else:
        model = None

    rewards, info = utils.train_test_model(model, env, seed, total_learning_timesteps, total_num_episodes)
    _, _, avg_res = utils.get_results(rewards, model_name, print_results=True)

    profit = info[0]["total_profit"]
    # money_spent = env.unwrapped.get_money_spent()
    # money_left = env.unwrapped.get_wallet_value()
    # roi = 100 * (money_left - money_spent) / money_spent
    roi = (profit - 1) * 100

    print(f"Total Profit = {profit:.8f}")
    # print(f"Total Money Spent = {money_spent:.2f}")
    # print(f"Money Left = {money_left:.2f}")
    print(f"ROI = {roi:.2f}%")
    plot_rewards(rewards, model_name, total_num_episodes)

