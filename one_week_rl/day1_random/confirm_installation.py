import gymnasium as gym


env = gym.make("CartPole-v1", render_mode="human")
state, info = env.reset()
print("Initial state:", state)

for _ in range(10):
    # Random action.
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)
    print(f"action={action}, reward={reward}, next_state={next_state}")
    if terminated or truncated:
        state, info = env.reset()

env.close()
