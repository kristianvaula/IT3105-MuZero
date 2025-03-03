# EpisodeBuffer (EB)
# – Stores episode data (real states, actions, rewards, policies, and evaluations) to be used for training.
import random
import os
import pickle

class EpisodeBuffer:
    def __init__(self):
        self.episodes = []  # List of episodes, each episode is a list of steps
    
    def add_episode(self, episode):
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
        max_index = len(episode) - 1
        
        if max_index < q:
            raise ValueError(f"Episode length ({max_index}) is less than q ({q}).")
        
        k = random.randint(q, max_index) # Choose random state ensuring enough look-back
        return episode[k - q:k + 1] # Return {sb, k-q, ..., sb, k}
  
class EpisodeHistory:
  def __init__(self, save_dir="episode_data"):
    self.history = [] #Stores all episodes
    self.save_dir = save_dir #Directory to save the history
    self.filename = "episode_history.pkl" #Filename to save the history
    os.makedirs(save_dir, exist_ok=True) #Create the directory if it doesn't exist
    
    def add_episode(self, epidata):
      """ Adds a completed episode to the history. """
      # Format the episode data, can be altered to unformatted if desirable 
      formatted_episode = [
        {"state": step[0], "value": step[1], "policy": step[2], "action": step[3], "reward": step[4]}
        for step in epidata
      ]
      self.history.append(formatted_episode)

    def get_all_episodes(self):
      """ Returns all episodes in the history. """
      return self.history
    
    def sample_episodes(self, num_samples):
      """ Returns a random sample of past episodes from the history, for training. """
      return random.sample(self.history, min(num_samples, len(self.history)))

    def save_history(self):
      """ Saves the history to a file. """
      file_path = os.path.join(self.save_dir, self.filename)
      with open(file_path, "wb") as f:
        pickle.dump(self.history, f)


    def load_history(self):
      """ Loads the history from a file if it exists. """
      file_path = os.path.join(self.save_dir, self.filename)
      if os.path.exists(file_path):
        with open(file_path, "rb") as f:
          self.history = pickle.load(f)
          print(f"Loaded {len(self.history)} episodes from {file_path}.")
      else:
        print("No saved episode history found.")        
