import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm


def train_model(model, total_learning_timesteps=1_000_000):
    # Addestra il modello e restituisce l'ambiente vettorializzato
    model.learn(total_timesteps=total_learning_timesteps, callback=ProgressBar(100))
    return model.get_env()


def test_model(model, env, total_num_episodes=50):
    rewards = []
    # Usa la stessa interfaccia per entrambi i casi
    for episode in tqdm(range(total_num_episodes), desc="Testing"):
        obs = env.reset()
        total_reward = 0
        done = False
        while not done:
            # Se il modello è stato addestrato, predici l'azione, altrimenti agisci casualmente
            if model is not None:
                action, _ = model.predict(obs)
            else:
                action = env.action_space.sample()

            # Normalizza la chiamata a step() per gestire sia ambienti vettorializzati che non
            step_result = env.step(action)
            if len(step_result) == 4:
                obs, reward, done, info = step_result
            else:  # Gestione per ambienti che restituiscono terminated e truncated
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated

            total_reward += reward

        rewards.append(total_reward)
    return rewards, info


def train_test_model(model, gym_env, seed=69, total_learning_timesteps=1_000_000, total_num_episodes=50, train=True):
    # Imposta il seed
    gym_env.reset(seed=seed)

    # Se richiesto, addestra il modello e usa l'ambiente vettorializzato; altrimenti, usa gym_env
    if train and model is not None:
        env = train_model(model, total_learning_timesteps)
    else:
        env = gym_env

    rewards, info = test_model(model, env, total_num_episodes)

    gym_env.close()
    return rewards, info


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
