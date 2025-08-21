import gymnasium as gym
import numpy as np
from src.config import Config

class SnakePacEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, world_length=10, seed=None):
        super(SnakePacEnv, self).__init__()
        self.world_length = world_length
        
        # Three actions: 0 = no-op, 1 = left, 2 = right.
        self.action_space = gym.spaces.Discrete(3)
        
        # Observation: one-dimensional array of length world_length.
        # 0 for empty, 1 for the agent, and 2 for the coin.
        self.observation_space = gym.spaces.Box(low=0, high=2, shape=(self.world_length,), dtype=np.int32)
        
        self.user_pos = None
        self.coin_pos = None
        
        self.seed(seed)
        self.reset()

    def seed(self, seed=None):
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.internal_Seed = seed

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        
        # Spawn the user at a random position.
        #self.user_pos = self.np_random.integers(0, self.world_length)
        
        # Spawn the coin at a different random position.
        #self.coin_pos = self.np_random.integers(0, self.world_length)
        #while self.coin_pos == self.user_pos:
        #   self.coin_pos = self.np_random.integers(0, self.world_length)

        self.coin_pos = 0
        self.user_pos = self.world_length - 1
        # spawm the coin at the same pos, but not the user
        #self.user_pos = self.np_random.integers(0, self.world_length)
        #while self.coin_pos == self.user_pos:
        #   self.user_pos = self.np_random.integers(0, self.world_length)


        return self._get_obs(), {}

    def step(self, action):
        # Validate action (0: no-op, 1: left, 2: right).
        if action not in [0, 1, 2]:
            raise ValueError("Invalid Action. Action must be 0 (no-op), 1 (left) or 2 (right).")
        
        # Find the direction of the coin relative to the user.
        right_direction = False
        if self.coin_pos > self.user_pos:
            right_direction = True
        elif self.coin_pos < self.user_pos:
            right_direction = False
        
        # Execute the action.
        if action == 1:
            self.user_pos = max(0, self.user_pos - 1)
        elif action == 2:
            self.user_pos = min(self.world_length - 1, self.user_pos + 1)
        # For action 0, no change.

        done = False  # This environment is endless.

        # Reward shaping based on the movement direction relative to the coin.
        if self.user_pos == self.coin_pos:
            reward = 10  # Main reward for collecting the coin.
            # Respawn the coin at a new location different from the agent.
            self.reset()
        elif action == 0:
            reward = -1
        elif right_direction:
            if action == 2:
                reward = 2  # Moving right when the coin is to the right.
            elif action == 1:
                reward = -3  # Moving left is the wrong direction.
        else:
            if action == 1:
                reward = 2  # Moving left when the coin is to the left.
            elif action == 2:
                reward = -3  # Moving right is the wrong direction.

        info = {}
        # Gymnasium's step should return: observation, reward, terminated, truncated, info.
        return self._get_obs(), reward, done, False, info

    def render(self, mode="human", reward=False):
        world = np.full(self.world_length, " . ")
        world[self.user_pos] = " U "
        world[self.coin_pos] = " C "
        state = "".join(world)
        state = state + "+1" if reward else state
        print(state)

    def _get_obs(self):
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
    obs, reward, done, truncated, info = env.step(2)
    print("\nAfter taking action 'right':")
    env.render()
    print("Reward:", reward)
