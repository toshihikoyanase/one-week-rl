import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

""" REINFORCE (Monte Carlo Policy Gradient)
    - Policy Network
    - No Critic Network
    - No Replay Buffer
    - Update policy after each episode
"""


class PolicyNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int) -> None:
        super().__init__()
        n_hidden = 64
        self.fc = nn.Sequential(
            nn.Linear(state_size, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, action_size),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def train() -> tuple[nn.Module, list[float]]:
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    policy_network = PolicyNetwork(state_size, action_size)
    optimizer = optim.Adam(policy_network.parameters(), lr=0.001)
    gamma = 0.99
    n_episodes = 500
    scores = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        done = False
        episode_states = []
        episode_actions = []
        episode_rewards = []
        total_reward = 0

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs = policy_network(state_tensor).detach().numpy().squeeze()
            action = np.random.choice(action_size, p=action_probs)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            total_reward += reward
            state = next_state

        # Compute returns
        returns = []
        G = 0
        for reward in reversed(episode_rewards):
            G = reward + gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)  # Normalize returns

        # Convert to tensors
        states_tensor = torch.FloatTensor(episode_states)
        actions_tensor = torch.LongTensor(episode_actions)

        # Compute loss
        action_probs = policy_network(states_tensor)
        action_log_probs = torch.log(action_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze())
        loss = -torch.sum(action_log_probs * returns)

        # Update policy network
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scores.append(total_reward)
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    return policy_network, scores


def visualize(rewards: list[float]) -> None:
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("REINFORCE Performance")
    plt.show()


if __name__ == "__main__":
    trained_model, rewards = train()
    visualize(rewards)
