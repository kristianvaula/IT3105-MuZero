import numpy as np
import copy
from .gsm import GameStateManager


class RiverraidGSM(GameStateManager):
    def __init__(self, env):
        super().__init__(env)
        self.initial_observation, _ = self.env.reset()
    
    def get_initial_state(self):
        """
        Returns the initial state of the environment.
        We use a deep copy to avoid unintended mutations.
        """
        obs, _ = self.env.reset()
        return copy.deepcopy(obs)

    def get_legal_actions(self, state):
        """
        Riverraid has a discrete fixed action space.
        """
        return list(range(self.env.action_space.n))

    def get_next_state_reward(self, state, action):
        """
        Simulates an action from a given state and returns (next_state, reward, done).
        We create a new environment instance to simulate the state transition safely.
        """
        temp_env = copy.deepcopy(self.env)
        temp_env.reset()
        temp_env.env._ale.setRAM(state.copy())
        
        next_obs, reward, terminated, truncated, _ = temp_env.step(action)
        done = terminated or truncated
        return copy.deepcopy(next_obs), reward, done

    def evaluate_state(self, state):
        """
        Heuristic evaluation of the state. For now, we just return 0.
        You can improve this by estimating score from pixel data.
        """
        return 0.0

    def get_policy(self, state):
        """
        Returns a uniform random policy.
        """
        legal_actions = self.get_legal_actions(state)
        policy = np.ones(len(legal_actions)) / len(legal_actions)
        return policy

    def is_terminal(self, state):
        """
        We can’t directly tell from the state. Placeholder method.
        Will need real game logic or internal signal for accuracy.
        """
        return False  # Optional: track episode length or life loss
