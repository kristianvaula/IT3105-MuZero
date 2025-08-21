import gymnasium as gym
import ale_py

class RiverraidEnv(gym.Env):

    def __init__(self, seed, frame_skip):
        gym.register_envs(ale_py)
        
        self.env = gym.make('ALE/Riverraid-v5', full_action_space=True, frameskip=frame_skip)
        self.env.reset(seed=seed)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.seed = seed
        
    def reset(self):
        """Returns (obs, info)"""
        return self.env.reset(seed=self.seed)
        
   

# import gymnasium as gym
# import numpy as np
# from gymnasium.wrappers import AtariPreprocessing, FrameStack

# class RiverRaidEnv(gym.Env):
#     metadata = {"render_modes": ["human", "rgb_array"]}

#     def __init__(
#         self,
#         frame_skip: int = 4,
#         num_stack: int = 4,
#         seed: int | None = None,
#     ):
#         # 1) Create the raw ALE environment with frameskip
#         env = gym.make("ALE/Riverraid-v5", render_mode="rgb_array", frameskip=frame_skip)

#         # 2) Atari preprocessing: grayscale + resize to 84×84 + frame skipping
#         env = AtariPreprocessing(
#             env,
#             frame_skip=frame_skip,
#             grayscale_obs=True,
#             scale_obs=True,       # scales pixels to [0,1]
#             screen_size=84,
#         )

#         # 3) Stack the last `num_stack` frames into a single observation
#         env = FrameStack(env, num_stack=num_stack)

#         self.env = env
#         self.action_space = env.action_space
#         self.observation_space = env.observation_space

#         if seed is not None:
#             self.seed(seed)

#     def seed(self, seed: int):
#         self.env.reset(seed=seed)
#         np.random.seed(seed)

#     def reset(self, seed=None, options=None):
#         """Returns (obs, info)"""
#         return self.env.reset(seed=seed, options=options)

#     def step(self, action:int):
#         """
#         Returns (obs, reward, done, truncated, info).
#         `done` is True if either terminated or truncated.
#         """
#         obs, reward, terminated, truncated, info = self.env.step(action)
#         done = terminated or truncated
#         return obs, reward, done, truncated, info

#     def render(self, mode="human"):
#         # 'rgb_array' mode will return a H×W×3 array if you need to feed it to a network
#         return self.env.render()

#     def close(self):
#         self.env.close()
