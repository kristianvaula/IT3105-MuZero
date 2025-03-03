import gymnasium as gym

# Create the River Raid environment
env = gym.make("ALE/Riverraid-v5", render_mode="human")

# Reset the environment
obs, info = env.reset()

done = False
while not done:
    action = env.action_space.sample()  # Select a random action
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated  # End episode condition

env.close()
