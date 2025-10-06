import random

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


# 1. Environment
env = gym.make("CartPole-v1", render_mode="human")
n_actions = env.action_space.n

# 2. Discretize state space
BINS = {
    "cart_position": np.linspace(-2.4, 2.4, 7),
    "cart_velocity": np.linspace(-3.0, 3.0, 7),
    "pole_angle": np.linspace(-0.21, 0.21, 9),
    "pole_velocity": np.linspace(-3.5, 3.5, 9),
}

def discretize(obs):
    cart_pos, cart_vel, pole_angle, pole_vel = obs
    state = (
        np.digitize(cart_pos, BINS["cart_position"]) -1,
        np.digitize(cart_vel, BINS["cart_velocity"]) -1,
        np.digitize(pole_angle, BINS["pole_angle"]) -1,
        np.digitize(pole_vel, BINS["pole_velocity"]) -1,
    )
    state = (
        np.clip(state[0], 0, len(BINS["cart_position"])-2),
        np.clip(state[1], 0, len(BINS["cart_velocity"])-2),
        np.clip(state[2], 0, len(BINS["pole_angle"])-2),
        np.clip(state[3], 0, len(BINS["pole_velocity"])-2),
    )
    return state


shape = (
    len(BINS["cart_position"])-1,
    len(BINS["cart_velocity"])-1,
    len(BINS["pole_angle"])-1,
    len(BINS["pole_velocity"])-1,
    n_actions,
)

# 3. Initialize Q-table
q_table = np.zeros(shape, dtype=np.float32)

# 4. Hyperparameters
n_episodes = 800
max_steps = 200
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995

rewards = []
for ep in range(n_episodes):
    obs, info = env.reset()
    s = discretize(obs)
    done = False
    total_r = 0.0

    step = 0
    while not done and step < max_steps:
        step += 1
        # Select action using epsilon-greedy policy
        if random.random() < epsilon:
            a = env.action_space.sample()  # Explore
        else:
            a = np.argmax(q_table[s])  # Exploit

        # one step
        obs_next, r, terminated, truncated, info = env.step(a)
        s_next = discretize(obs_next)
        done = terminated or truncated
        total_r += r

        # Bootstrap Q-learning update
        if terminated:
            best_next = 0.0
        else:
            best_next = np.max(q_table[s_next])  # max_{a'} Q(s', a')

        # TD target and error
        td_target = r + gamma * best_next    # r + γ * maxQ(s', a')
        td_error = td_target - q_table[s][a]  # TD error
        q_table[s][a] += alpha * td_error  # Q(s, a) update

        # Move to next state
        s = s_next

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    print(f"Episode {ep+1}: Total Reward: {total_r}, Epsilon: {epsilon:.3f}")
    rewards.append(total_r)

plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Q-Learning Performance")
plt.show()

env.close()
