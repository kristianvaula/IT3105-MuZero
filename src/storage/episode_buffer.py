import random
import os
import pickle
from typing import List
from dataclasses import dataclass

@dataclass
class EpisodeStep:
    """ Represents a single step in an episode. """
    state: any
    value: float
    policy: List[float]
    action: any
    reward: float
  
class Episode:
    """ Stores a sequence of steps for a single episode. """
    def __init__(self):
          self.steps = []  # List of EpisodeStep objects

    def add_step(self, state, value, policy, action, reward):
        """ Adds a step (transition) to the episode. """
        self.steps.append(EpisodeStep(state, value, policy, action, reward))

    def get_steps(self):
        """ Returns all steps in the episode. """
        return self.steps

class EpisodeBuffer:
    def __init__(self, save_dir="episode_data", filename="episode_history.pkl"):
        self.episodes = []  # Stores all episodes
        self.save_dir = save_dir  # Directory to save episode history
        self.filename = filename  # Filename for saved history
        os.makedirs(save_dir, exist_ok=True)  # Ensure save directory exists
    
    def add_episode(self, episode: Episode):
        """ Adds a new episode (list of steps) to the buffer """
        self.episodes.append(episode)

    def sample_state(self, q=0):
        """ 
        Randomly selects an episode and a state within that episode.
        Returns a sequence of q+1 states ending at the selected state (for BPTT).
        """
        if not self.episodes:
            raise ValueError("No episodes in the buffer.")

        episode = random.choice(self.episodes)
        max_index = len(episode.steps) - 1
        
        if max_index < q:
            raise ValueError(f"Episode length ({max_index}) is less than q ({q}).")
        
        k = random.randint(q, max_index) # Choose random state ensuring enough look-back
        return episode.steps[k - q:k + 1] # Return [sb, k-q, ..., sb, k]
      
    def save_history(self):
        """ Saves all episodes to a file. """
        file_path = os.path.join(self.save_dir, self.filename)
        with open(file_path, "wb") as f:
            pickle.dump(self.episodes, f)
        print(f"Saved {len(self.episodes)} episodes to {file_path}.")

    def load_history(self):
        """ Loads episode history from a file if it exists. """
        file_path = os.path.join(self.save_dir, self.filename)
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                self.episodes = pickle.load(f)
            print(f"Loaded {len(self.episodes)} episodes from {file_path}.")
        else:
            print("No saved episode history found.")