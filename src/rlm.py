import random
import numpy as np
from src.environments.snakepac import SnakePacEnv
from src.gsm.gsm import GameStateManager
from src.self_play.uMCTS import uMCTS
from src.networks.neural_network_manager import NeuralNetManager
from src.storage.episode_buffer import EpisodeBuffer
from src.config import Config

class ReinforcementLearningManager():
  """
  Reinforcement Learning Manager (RLM) for MuZero.
  
  Runs the main training algorithm:
    1. Runs episodes in the environment.
    2. At each time step, gathers a history of real-game states (ϕₖ),
        computes the abstract state (σₖ) using the representation network (NNr),
        and then performs u-MCTS search from that state.
    3. Collects training data (state, value, policy, action, reward) for each time step.
    4. Every training_interval episodes, performs BPTT training using data from the episode buffer.
  """
  def __init__(self, env: SnakePacEnv, gsm: GameStateManager, monte_carlo: uMCTS, nnm: NeuralNetManager, episode_buffer: EpisodeBuffer, config: Config):
    self.env = env
    self.gsm = gsm
    self.monte_carlo = monte_carlo 
    self.nnm = nnm
    self.episode_buffer = episode_buffer

    self.num_episodes = config.environment.num_episodes             # Ne: total number of episodes
    self.num_episode_steps = config.environment.num_episode_steps       # Nes: max steps per episode
    self.training_interval = config.environment.training_interval     # It: training update interval
    self.minibatch_size = config.environment.batch_size           # mbs: minibatch size for BPTT
    self.history_length = config.environment.history_length          # q: number of past states required for φₖ