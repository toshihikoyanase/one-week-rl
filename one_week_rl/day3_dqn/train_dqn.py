from collections import deque
import random

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):
    def __init__(self, state_size: int, action_size: int) -> None:
        super().__init__()
        n_hidden = 64
        self.fc = nn.Sequential(
            nn.Linear(state_size, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, action_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class ReplayBuffer:
    def __init__(self, capacity: int=10_000):
        self.buffer = deque(maxlen=capacity)

    def push(
            self,
            state: np.ndarray,
            action: int,
            reward: float,
            next_state: np.ndarray,
            done: bool
        ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple[np.ndarray, ...]:
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self) -> int:
        return len(self.buffer)


def train() -> tuple[nn.Module, list[float]]:
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    q_network = DQN(state_size, action_size)
    target_network = DQN(state_size, action_size)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()

    optimizer = optim.Adam(q_network.parameters(), lr=1e-3)
    replay_buffer = ReplayBuffer()

    batch_size = 64
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    epsilon = epsilon_start
    target_update = 20

    def select_action(state: np.ndarray, q_network: DQN, epsilon: float) -> int:
        if random.random() < epsilon:
            return env.action_space.sample()
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = q_network(state)
            return q_values.argmax().item()

    reward_history = []
    num_episodes = 1000
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = select_action(state, q_network, epsilon)
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated

            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(replay_buffer) > batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions).unsqueeze(1)
                rewards = torch.FloatTensor(rewards).unsqueeze(1)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones).unsqueeze(1)

                q_values = q_network(states).gather(1, actions)
                with torch.no_grad():
                    next_q = target_network(next_states).max(1)[0].unsqueeze(1)
                    target = rewards + gamma * next_q * (1 - dones)

                loss = nn.MSELoss()(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if episode % target_update == 0:
            target_network.load_state_dict(q_network.state_dict())

        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")
        reward_history.append(total_reward)

    env.close()
    return q_network, reward_history


def visualize(rewards: list[float]) -> None:
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN Performance")
    plt.show()


if __name__ == "__main__":
    trained_model, rewards = train()
    visualize(rewards)
