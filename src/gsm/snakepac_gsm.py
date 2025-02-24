import gymnasium as gym
import numpy as np
import copy
from src.gsm.gsm import GameStateManager

class SnakePacGSM(GameStateManager):
  """ 
  The Game State Manager (GSM) wraps a Gymnasium environment and provides: 
    • Generation of an initial state. 
    • Retrieval of legal actions. 
    • Simulation of state transitions given an action. 
    • A heuristic evaluation of a state. 
    • A simple policy suggestion based on the current state. 
  
  Args:
    env: Environment to be managed. Must follow Gymnasium structure.
  """
  def __init__(self, env):
    super().__init__(env)
        
  def get_initial_state(self):
    """ Returns the initial state of the environment. """
    state, _ = self.env.reset()
    return state
  
  def get_legal_actions(self, state):
    """
    Returns the legal actions for the given state.
    TODO: Consider changing the return type of this function
    Args:
      state: State of the environment.
    
    Returns:
      List of legal actions.
    """
    return list(range(self.env.action_space.n))
  
  def get_next_state_reward(self, state, action):
    """
    Given a state and action, simulate the environment to obtain the next state,
    reward and terminal status. We use deepcopy to clone the environment and 
    manually set its internal state.

    Args:
      state: Current state of the environment
      action: Action to be taken
      
    Returns:
      Tuple of (next_state, reward, done)
    """
    
    # Use cache if state-action pair has been simulated before
    cache_key = (state.toBytes(), action)
    if cache_key in self.cache:
      return self.cache[cache_key]
    
    env_clone = copy.deepcopy(self.env)
    
    try:
      user_idx = np.where(state == 1)[0][0]
      coin_idx = np.where(state == 2)[0][0]
    except IndexError:
      raise ValueError("Invalid state. State must contain one user and one coin.")
    
    env_clone.user_pos = int(user_idx)
    env_clone.coin_pos = int(coin_idx)
    
    next_state, reward, done, _, _ = env_clone.step(action)
    
    # Cache the result
    self.cache[cache_key] = (next_state, reward, done)
    
    return next_state, reward, done
  
  def evaluate_state(self, state):
    """
    Provide a heuristic evaluation for the given state.
    For example, in SnakePacEnv, a simple evaluation might be the negative distance
    between the user and the coin (closer is better).
    
    Args:
      state: State of the environment.
      
    Returns:
      Heuristic evaluation of the state.
    """
    try:
      user_idx = np.where(state == 1)[0][0]
      coin_idx = np.where(state == 2)[0][0]
    except IndexError:
      raise ValueError("Invalid state. State must contain one user and one coin.")
    
    return -abs(user_idx - coin_idx)
  
  def get_policy(self, state):
    """
    Return a simple probability distribution over actions.
    For SnakePacEnv, bias the action that moves the user toward the coin.
    
    Args:
      state: State of the environment.
      
    Returns:
      List of probabilities for each action.
    """
    
    try:
      user_idx = np.where(state == 1)[0][0]
      coin_idx = np.where(state == 2)[0][0]
    except IndexError:
      raise ValueError("Invalid state. State must contain one user and one coin.")
    
    legal_actions = self.get_legal_actions(state)
    
    policy = {action: 1.0 / len(legal_actions) for action in legal_actions}
    
    if coin_idx < user_idx:
      policy[0] = 0.8
      policy[1] = 0.2
    elif coin_idx > user_idx:
      policy[0] = 0.2
      policy[1] = 0.8
      
    return policy