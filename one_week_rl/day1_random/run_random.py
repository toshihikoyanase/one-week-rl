import gymnasium as gym
import matplotlib.pyplot as plt

rewards = []
env = gym.make("CartPole-v1", render_mode="human")

for episode in range(50):
    state, info = env.reset()
    total_reward = 0
    done = False
    print("Initial state:", state)

    while not done:
        # Random action.
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        print(f"action={action}, reward={reward}, next_state={next_state}")
        total_reward += reward
        done = terminated or truncated

    rewards.append(total_reward)

plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Random Policy Performance")
plt.show()

env.close()
