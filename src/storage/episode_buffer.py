# EpisodeBuffer (EB)
# – Stores episode data (real states, actions, rewards, policies, and evaluations) to be used for training.
import random

class EpisodeBuffer:
  def __init__(self):
    self.episodes = []
    
  def add_episode(self, episode): 
    self.episodes.append(episode)
    
  def sample_episodes(self, num_episodes):
    return random.sample(self.episodes, num_episodes)