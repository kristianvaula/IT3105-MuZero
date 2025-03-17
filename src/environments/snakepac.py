import gymnasium as gym
import numpy as np
from src.config import Config

class SnakePacEnv(gym.Env):
  metadata = {"render.modes": ["human"]}

  def __init__(self, world_length=10, seed=None):
    super(SnakePacEnv, self).__init__()
    self.world_length = world_length
    
    # Two actions: 0 = no-op, 1 = left, 2 = right.
    self.action_space = gym.spaces.Discrete(3)
    
    # Observation: one-dimensional array of length 10.
    # We'll use 0 for empty, 1 for the agent, and 2 for the coin.
    self.observation_space = gym.spaces.Box(low=0, high=2, shape=(self.world_length,), dtype=np.int32)
    
    self.user_pos = None
    self.coin_pos = None
    
    self.seed(seed)
    
    self.reset()

  def seed(self, seed=None):
    self.np_random, _ = gym.utils.seeding.np_random(seed)
    self.internal_Seed = seed

  def reset(self, seed=None, options=None):
    # Optionally set the seed.
    if seed is not None:
      self.seed(seed)
    
    # Spawn the user at a random position.
    self.user_pos = self.np_random.integers(0, self.world_length)
    
    # Spawn the coin in a different random position.
    self.coin_pos = self.np_random.integers(0, self.world_length)
    while self.coin_pos == self.user_pos:
      self.coin_pos = self.np_random.integers(0, self.world_length)

    return self._get_obs(), {}

  def step(self, action):
    # Validate action (0: no-op, 1: left, 2: right).
    if action not in [0, 1, 2]:
        raise ValueError("Invalid Action. Action must be 0 (no-op), 1 (left) or 2 (right).")
    
    # Execute the action.
    if action == 1:
        self.user_pos = max(0, self.user_pos - 1)
    elif action == 2:
        self.user_pos = min(self.world_length - 1, self.user_pos + 1)
    # action==0: no-op (do nothing)
    
    reward = 0
    done = False  # This environment is endless.
    
    # Check if the user lands on the coin.
    if self.user_pos == self.coin_pos:
        reward = 1
        # Respawn the coin at a new location.
        new_coin_pos = self.np_random.integers(0, self.world_length)
        while new_coin_pos == self.user_pos:
            new_coin_pos = self.np_random.integers(0, self.world_length)
        self.coin_pos = new_coin_pos

    info = {}
    # Gymnasium's step should return: observation, reward, terminated, truncated, info.
    return self._get_obs(), reward, done, False, info

  def render(self, mode="human"):
    # Create a simple string representation of the world.
    world = np.full(self.world_length, " . ")
    world[self.user_pos] = " U "
    world[self.coin_pos] = " C "
    print("".join(world))

  def _get_obs(self):
    # Observation represented as a numpy array:
    # 0 means empty, 1 means user, 2 means coin.
    obs = np.zeros(self.world_length, dtype=np.int32)
    obs[self.user_pos] = 1
    obs[self.coin_pos] = 2
    return obs

  def close(self):
    pass

# Example usage:
if __name__ == "__main__":
  config = Config()
  env = SnakePacEnv(config.environment.world_length, config.environment.seed)
  obs, _ = env.reset()
  print("Initial state:")
  env.render()
  
  # Example step: move right.
  obs, reward, done, truncated, info = env.step(1)
  print("\nAfter taking action 'right':")
  env.render()
  print("Reward:", reward)