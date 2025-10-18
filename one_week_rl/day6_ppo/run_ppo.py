import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

""" Proximal Policy Optimization (PPO)
    - Policy Network (Actor)
    - Value Network (Critic)
    - Clipped Surrogate Objective
    - Multiple Epochs Update
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


class ValueNetwork(nn.Module):
    def __init__(self, state_size: int) -> None:
        super().__init__()
        n_hidden = 64
        self.fc = nn.Sequential(
            nn.Linear(state_size, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1),
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
    clip_ratio = 0.2
    ppo_epochs = 4
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
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-9
        )  # Normalize advantages

        # Store old log probabilities for PPO
        with torch.no_grad():
            old_action_probs = policy_network(states_tensor)
            old_log_probs = torch.log(
                old_action_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze() + 1e-8
            )

        # PPO Multiple Epochs Update
        for _ in range(ppo_epochs):
            # Update Value Network (Critic)
            current_values = value_network(states_tensor).squeeze()
            value_loss = nn.MSELoss()(current_values, returns)
            value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(value_network.parameters(), 0.5)
            value_optimizer.step()

            # Update Policy Network (Actor) with PPO Clipped Objective
            new_action_probs = policy_network(states_tensor)
            new_log_probs = torch.log(
                new_action_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze() + 1e-8
            )

            # Calculate probability ratio
            ratio = torch.exp(new_log_probs - old_log_probs)

            # PPO Clipped Surrogate Objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

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
    plt.title("PPO on CartPole-v1")
    plt.show()
