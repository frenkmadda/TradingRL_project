import numpy as np
import pandas as pd
import random
from enum import Enum
import matplotlib.pyplot as plt


class Actions(Enum):
    Sell = 0
    Buy = 1
    Hold = 2


class QLearningAgent:
    def __init__(self, price_bins=10, volume_bins=10, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.01,
                 epsilon_decay=0.995):
        self.price_bins = np.linspace(-5, 5, price_bins)  # Intervalli di variazione del prezzo
        self.volume_bins = np.linspace(-5, 5, volume_bins)  # Intervalli di variazione del volume
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.q_table = np.zeros((price_bins, volume_bins, len(Actions)))  # Stato discretizzato

    def discretize_state(self, price_change, volume_change):
        price_idx = min(max(np.digitize(price_change, self.price_bins) - 1, 0), len(self.price_bins) - 1)
        volume_idx = min(max(np.digitize(volume_change, self.volume_bins) - 1, 0), len(self.volume_bins) - 1)
        return (price_idx, volume_idx)

    def get_action(self, state, explore=True):
        if explore and np.random.rand() < self.epsilon:
            return random.choice(list(Actions))  # Esplorazione
        return Actions(np.argmax(self.q_table[state]))  # Sfruttamento

    def get_action_for_position(self, state, has_position, explore=True):
        """Get an action that's valid for the current position"""
        if explore and np.random.rand() < self.epsilon:
            if has_position:
                # If we have a position, we can either Sell or Hold
                return random.choice([Actions.Sell, Actions.Hold])
            else:
                # If we don't have a position, we can either Buy or Hold
                return random.choice([Actions.Buy, Actions.Hold])
        else:
            # Copy the Q-values for this state
            q_values = self.q_table[state].copy()

            # Mask invalid actions
            if has_position:
                q_values[Actions.Buy.value] = float('-inf')  # Can't buy if already holding
            else:
                q_values[Actions.Sell.value] = float('-inf')  # Can't sell if not holding

            return Actions(np.argmax(q_values))

    def update_q_table(self, state, action, reward, next_state, done):
        best_next_action = np.argmax(self.q_table[next_state])
        target = reward + (self.gamma * self.q_table[next_state][best_next_action] * (not done))
        self.q_table[state][action.value] += self.alpha * (target - self.q_table[state][action.value])

        # Decadimento epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def calculate_roi(initial_value, final_value):
    return ((final_value - initial_value) / initial_value) * 100


def training(df, agent, total_episodes=100, initial_balance=10000, train_split=0.9, commission=0.001):
    train_size = int(len(df) * train_split)
    df_train = df.iloc[:train_size]
    df_test = df.iloc[train_size:]

    all_rewards = []
    training_roi = []

    print("\n--- Training Phase ---")
    for episode in range(total_episodes):
        balance = initial_balance
        held = 0
        total_reward = 0
        has_position = False

        # Track portfolio value over time
        portfolio_values = []

        for i in range(len(df_train) - 1):
            state = agent.discretize_state(df_train.iloc[i]['price_change'], df_train.iloc[i]['volume_change'])

            # Get action that's valid for current position
            action = agent.get_action_for_position(state, has_position)

            current_price = df_train.iloc[i]['Close']
            next_price = df_train.iloc[i + 1]['Close']

            prev_portfolio_value = balance + (held * current_price)
            portfolio_values.append(prev_portfolio_value)

            # Execute action
            if action == Actions.Buy and not has_position:
                # Use all balance to buy
                held = (balance * (1 - commission)) / current_price
                balance = 0
                has_position = True
            elif action == Actions.Sell and has_position:
                # Sell all holdings
                balance += held * current_price * (1 - commission)
                held = 0
                has_position = False

            # Calculate reward
            profit = (balance + held * next_price) / initial_balance
            roi = (profit - 1) * 100

            reward = 0
            if action == Actions.Hold:
                reward -= current_price * 0.0001

            price_diff = next_price - current_price
            percent_change = price_diff / current_price

            if action == Actions.Buy and not has_position:
                reward += percent_change
            elif action == Actions.Sell and has_position:
                reward += percent_change

            reward += profit + roi
            total_reward += reward

            next_state = agent.discretize_state(df_train.iloc[i + 1]['price_change'],
                                                df_train.iloc[i + 1]['volume_change'])
            done = (i == len(df_train) - 2)
            agent.update_q_table(state, action, reward, next_state, done)

        final_train_value = balance + (held * df_train.iloc[-1]['Close'])
        roi_train = calculate_roi(initial_balance, final_train_value)

        all_rewards.append(total_reward)
        training_roi.append(roi_train)

        if episode % 10 == 0 or episode == total_episodes - 1:
            print(
                f"Episode {episode + 1}: Reward = {total_reward:.2f}, ROI = {roi_train:.2f}%, Portfolio Value = {final_train_value:.2f}")

    # Plot training progress
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(all_rewards)
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')

    plt.subplot(1, 2, 2)
    plt.plot(training_roi)
    plt.title('Training ROI')
    plt.xlabel('Episode')
    plt.ylabel('ROI (%)')
    plt.savefig('grafici/training_progress.png', dpi=300)

    # Analyze Q-table after training
    print("\nQ-table Analysis:")
    action_preferences = {a.name: 0 for a in Actions}
    for p in range(len(agent.price_bins)):
        for v in range(len(agent.volume_bins)):
            best_action = Actions(np.argmax(agent.q_table[p, v]))
            action_preferences[best_action.name] += 1

    total_states = len(agent.price_bins) * len(agent.volume_bins)
    for action, count in action_preferences.items():
        print(f"States preferring {action}: {count} ({count / total_states * 100:.2f}%)")

    # Evaluation phase
    print("\n--- Evaluation Phase ---")
    balance = initial_balance
    held = 0
    has_position = False
    total_reward = 0
    actions_taken = {"Buy": 0, "Sell": 0, "Hold": 0}

    # Track portfolio value during evaluation
    eval_portfolio_values = []
    eval_dates = df_test.index.tolist()

    buy_dates = []
    sell_dates = []
    holdings = []

    for i in range(len(df_test) - 1):
        state = agent.discretize_state(df_test.iloc[i]['price_change'], df_test.iloc[i]['volume_change'])

        # Get appropriate action based on position, no random exploration
        action = agent.get_action_for_position(state, has_position, explore=False)

        # Track action counts
        actions_taken[action.name] += 1

        current_price = df_test.iloc[i]['Close']
        next_price = df_test.iloc[i + 1]['Close']

        # Store portfolio value
        current_portfolio = balance + (held * current_price)
        eval_portfolio_values.append(current_portfolio)

        # Print state info periodically
        if i % 50 == 0:
            q_values = agent.q_table[state]
            position_status = "HAS POSITION" if has_position else "NO POSITION"
            print(f"Day {i}: Price ${current_price:.2f}, {position_status}, Action: {action.name}, "
                  f"Q-values: {q_values}, Portfolio: ${current_portfolio:.2f}")

        # Execute trades
        if action == Actions.Buy and not has_position:
            held = (balance * (1 - commission)) / current_price
            balance = 0
            has_position = True
            buy_dates.append(i)
            print(f"BUY at ${current_price:.2f}, Portfolio Value: ${current_portfolio:.2f}")
        elif action == Actions.Sell and has_position:
            balance += held * current_price * (1 - commission)
            held = 0
            has_position = False
            sell_dates.append(i)
            print(f"SELL at ${current_price:.2f}, Portfolio Value: ${current_portfolio:.2f}")
        elif action == Actions.Hold:
            holdings.append(i)
            print(f"HOLD at ${current_price:.2f}, Portfolio Value: ${current_portfolio:.2f}")


        # Calculate reward (for tracking purposes only)
        next_portfolio_value = balance + (held * next_price)
        reward = (next_portfolio_value - current_portfolio)
        total_reward += reward

    final_value = balance + (held * df_test.iloc[-1]['Close'])
    roi = calculate_roi(initial_balance, final_value)

    print(f"\nFinal Balance: ${balance:.2f}")
    print(
        f"Final Holdings: {held:.6f} units at ${df_test.iloc[-1]['Close']:.2f} = ${held * df_test.iloc[-1]['Close']:.2f}")
    print(f"Final Portfolio Value: ${final_value:.2f}")
    print(f"Total ROI on test data: {roi:.2f}%")
    print(f"Total Reward on test data: {total_reward:.2f}")

    # Print action distribution
    # print("\nAction Distribution during Evaluation:")
    # for action, count in actions_taken.items():
    #     print(f"{action}: {count} times ({count / len(df_test) * 100:.2f}%)")

    # Plot performance
    plt.figure(figsize=(14, 7))
    plt.plot(eval_portfolio_values, label='Portfolio Value')

    # Plot buy/sell markers
    test_prices = df_test['Close'].values[:len(eval_portfolio_values)]
    for buy_idx in buy_dates:
        plt.scatter(buy_idx, eval_portfolio_values[buy_idx], color='green', marker='^', s=80,
                    label='Buy' if buy_idx == buy_dates[0] else "")
    for sell_idx in sell_dates:
        plt.scatter(sell_idx, eval_portfolio_values[sell_idx], color='red', marker='v', s=80,
                    label='Sell' if sell_idx == sell_dates[0] else "")

    for holding in holdings:
        plt.scatter(holding, eval_portfolio_values[holding], color='blue', marker='o', s=30,
                    label='Hold' if holding == holdings[0] else "")

    plt.title(f'Portfolio Value During Evaluation (ROI: {roi:.2f}%)')
    plt.xlabel('Trading Day')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.savefig('grafici/eval_portfolio_qtable.png')
    # plt.show()

    # For comparison, plot buy and hold strategy
    buy_hold_final = initial_balance / df_test.iloc[0]['Close'] * df_test.iloc[-1]['Close']
    buy_hold_roi = calculate_roi(initial_balance, buy_hold_final)
    print(f"\nBuy & Hold Strategy ROI: {buy_hold_roi:.2f}%")

    return agent, roi


# Main execution
if __name__ == "__main__":
    # Caricamento e preprocessamento dati
    df = pd.read_csv("data/crypto/ada-usd.csv")
    df["price_change"] = df["Close"].pct_change() * 100
    df["volume_change"] = df["Volume"].pct_change() * 100
    df.dropna(inplace=True)

    # Create agent with better parameters
    agent = QLearningAgent(
        price_bins=20,  # More granular price bins
        volume_bins=20,  # More granular volume bins
        alpha=0.2,  # Higher learning rate
        gamma=0.95,  # Slightly lower discount factor
        epsilon=1.0,  # Start with full exploration
        epsilon_min=0.1,  # Higher minimum exploration
        epsilon_decay=0.99  # Slower decay
    )

    # Train with more episodes
    total_episodes = 200  # More episodes for better learning
    initial_balance = 10000

    agent, roi = training(df, agent, total_episodes, initial_balance)