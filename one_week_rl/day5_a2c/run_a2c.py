import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

""" Advantage Actor-Critic (A2C)
    - Policy Network (Actor)
    - Value Network (Critic)
    - No Replay Buffer
    - Update after each episode
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
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class ValueNetwork(nn.Module):
    def __init__(self, state_size: int) -> None:
        super().__init__()
        n_hidden = 64
        self.fc = nn.Sequential(
            nn.Linear(state_size, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def train() -> tuple[nn.Module, nn.Module, list[float]]:
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    policy_network = PolicyNetwork(state_size, action_size)
    value_network = ValueNetwork(state_size)
    policy_optimizer = optim.Adam(policy_network.parameters(), lr=0.003)
    value_optimizer = optim.Adam(value_network.parameters(), lr=0.001)

    gamma = 0.99
    n_episodes = 500
    scores = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        states = []
        actions = []
        rewards = []

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            with torch.no_grad():
                action_probs = policy_network(state_tensor)
                action = torch.multinomial(action_probs, 1).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state

        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)

        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.LongTensor(actions)

        values = value_network(states_tensor).squeeze()

        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)  # Normalize advantages

        # Update Value Network (Critic).
        value_loss = nn.MSELoss()(values, returns)
        value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_network.parameters(), 0.5)
        value_optimizer.step()

        # Update Policy Network (Actor).
        action_probs = policy_network(states_tensor)
        log_probs_tensor = torch.log(action_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze() + 1e-8)
        policy_loss = -(log_probs_tensor * advantages).mean()
        policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_network.parameters(), 0.5)
        policy_optimizer.step()

        scores.append(total_reward)

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}, Score: {total_reward}")

    return policy_network, value_network, scores

if __name__ == "__main__":
    policy_net, value_net, reward_history = train()
    plt.plot(reward_history)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("A2C on CartPole-v1")
    plt.show()
