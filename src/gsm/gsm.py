from abc import ABC, abstractmethod

class GameStateManager(ABC):
    """
    Abstract Game State Manager (GSM) that wraps around a Gymnasium environment.
    Provides an interface for managing game states, actions, and evaluations.

    Methods to implement:
      • get_initial_state()
      • get_legal_actions(state)
      • get_next_state_reward(state, action)
      • evaluate_state(state)
      • get_policy(state)
      • is_terminal(state)
    """

    def __init__(self, env):
        self.env = env
        self.cache = {}

    @abstractmethod
    def get_initial_state(self):
        """Returns the initial state of the environment."""
        pass

    @abstractmethod
    def get_legal_actions(self, state):
        """Returns a list of legal actions for the given state."""
        pass

    @abstractmethod
    def get_next_state_reward(self, state, action):
        """Simulates an action from a given state and returns (next_state, reward, done)."""
        pass

    @abstractmethod
    def evaluate_state(self, state):
        """Provides a heuristic evaluation of the state."""
        pass

    @abstractmethod
    def get_policy(self, state):
        """Returns a probability distribution over legal actions."""
        pass

    @abstractmethod
    def is_terminal(self, state):
        """Determines whether a given state is terminal."""
        pass
