import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm


def train_test_model(model, gym_env, seed=69, total_learning_timesteps=1_000_000, total_num_episodes=50):
    obs = gym_env.reset(seed=seed)
    vec_env = None

    if model is not None:
        model.learn(total_timesteps=total_learning_timesteps, callback=ProgressBar(100))
        # model.learn(total_timesteps=total_learning_timesteps, progress_bar=True)
        # ImportError: You must install tqdm and rich in order to use the progress bar callback.
        # It is included if you install stable-baselines with the extra packages: `pip install stable-baselines3[extra]`

        vec_env = model.get_env()
        obs = vec_env.reset()

    reward_over_episodes = []
    tbar = tqdm(range(total_num_episodes))

    for episode in tbar:
        if vec_env:
            obs = vec_env.reset()
        else:
            obs, info = gym_env.reset()

        total_reward = 0
        done = False

        while not done:
            if model is not None:
                action, _states = model.predict(obs)
                obs, current_reward, done, info = vec_env.step(action)
            else:  # random
                action = gym_env.action_space.sample()
                obs, current_reward, terminated, truncated, info = gym_env.step(action)
                done = terminated or truncated

            total_reward += current_reward

        reward_over_episodes.append(total_reward)
        if episode % 10 == 0:
            avg_reward = np.mean(reward_over_episodes)
            tbar.set_description(f"Episode: {episode}, Avg. Reward: {avg_reward:.3f}")
            tbar.update()

    tbar.close()
    gym_env.close()
    return reward_over_episodes, info


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