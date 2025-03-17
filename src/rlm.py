import random
import numpy as np
import torch
from src.environments.snakepac import SnakePacEnv
from src.gsm.gsm import GameStateManager
from src.self_play.uMCTS import uMCTS
from src.networks.neural_network_manager import NeuralNetManager
from src.storage.episode_buffer import EpisodeBuffer
from src.config import Config
from src.storage.episode_buffer import Episode

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
    self.config = config

    self.num_episodes = config.environment.num_episodes             # Ne: total number of episodes
    self.num_episode_steps = config.environment.num_episode_steps       # Nes: max steps per episode
    self.training_interval = config.environment.training_interval     # It: training update interval
    self.minibatch_size = config.environment.batch_size           # mbs: minibatch size for BPTT
    self.history_length = config.environment.network.state_window          # q: number of past states required for φₖ
    
  def train(self):
    """
    Runs the full training loop and returns the trained NeuralNetManager (nnm).
    """
    for episode in range(self.num_episodes):
      # (a) Reset env and obtain initial state s₀.
      current_state = self.env.reset()
      action = self.env.action_space.sample()
      
      episode_data = Episode()
      
      state_history = []
      action_history = []
      done = False
      
      for _ in range(self.num_episode_steps):
        state_history.append(current_state)
        action_history.append(action)

        phi_k_tensor = self._get_phi_k(state_history, action_history)
        abstract_state, _, _, _, _ = self.nnm.translate_and_evaluate(phi_k_tensor)
        
        # Initialize u-Tree with abtract state σ₀ and run uMCTS
        policy, root_value = self.monte_carlo.search(abstract_state)
        
        # Sample action aₖ₊₁ from the policy πₖ
        action = self.__sample_action(policy)

        # Simulate one timestep in the game: sₖ₊₁, rₖ₊₁
        next_state, reward, done, info = self.env.step(action)
        
        # Save the training data for this step.
        episode_data.add_step(
            state=current_state,
            value=root_value.item() if hasattr(root_value, "item") else root_value,
            policy=list(policy.values()),
            action=action,
            reward=reward
        )
        
        current_state = next_state
        if done:
          break
      
      # Add the completed episode to the episode buffer
      self.episode_buffer.add_episode(episode_data)
      
      # Every training_interval episodes, perform BPTT training
      if (episode + 1) % self.training_interval == 0:
        loss = self.nnm.bptt(self.episode_buffer)
        print(f"Episode {episode + 1} Loss: {loss}")
    
    return self.nnm
  
  def _get_phi_k(self, state_history, action_history):
    """
    Returns the history of states ϕₖ.
    """
    if self.config.environment == "snakepac":
      state_window = self.history_length       
      state_tensor = torch.tensor(state_history[:-state_window], dtype=torch.float32)
      action_tensor = torch.tensor(action_history[:-state_window], dtype=torch.float32)
      return torch.cat((state_tensor, action_tensor), dim=1)
    else:
      raise NotImplementedError("Only SnakePac environment is supported.")
  
  def _sample_action(self, policy):
    """
    Samples an action from the policy.
    """
    actions = list(policy.keys())
    probabilities = list(policy.values())
    return random.choices(actions, weights=probabilities, k=1)[0]