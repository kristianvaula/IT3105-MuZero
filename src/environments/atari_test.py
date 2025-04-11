import gymnasium as gym
import ale_py

# Create the River Raid environment
gym.register_envs(ale_py)
env = gym.make("ALE/Riverraid-v5")

# Reset the environment
obs, info = env.reset()

done = False
while not done:
    action = env.action_space.sample()  # Select a random action
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated  # End episode condition

env.close()
