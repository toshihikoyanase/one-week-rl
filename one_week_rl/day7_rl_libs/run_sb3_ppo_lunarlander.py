import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

class RewardCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.
    Records episode rewards for plotting.
    """
    def __init__(self, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # Record episode reward when episode is done
        if self.locals.get('dones') is not None:
            for i, done in enumerate(self.locals['dones']):
                if done:
                    info = self.locals['infos'][i]
                    if 'episode' in info:
                        reward = info['episode']['r']
                        length = info['episode']['l']
                        self.episode_rewards.append(reward)
                        self.episode_lengths.append(length)
        return True


# Change environment from CartPole to LunarLander.
env = gym.make("LunarLander-v3")

# Create callback to record rewards
reward_callback = RewardCallback()

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=300_000, callback=reward_callback)
model.save("ppo_lunarlander")
env.close()

rewards = reward_callback.episode_rewards
plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("PPO on LunarLander-v3")
plt.show()


