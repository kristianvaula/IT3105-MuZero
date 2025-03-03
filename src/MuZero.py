from .config import Config

""" Important parameters 

num_episodes: Number of episodes to run
num_simulations: Number of simulations to run in u-MCTS
max_depth: Maximum depth to expand in u-MCTS
ucb_constant: UCB constant for u-MCTS
mbs (Minibatch size): Number of episodes to sample for BPTT
I_t: Training interval for the trident network


"""

class MuZero: 
  def __init__(self, config: Config):
    # Load environments 
  
    # Load Game State Manager
    
    # Initialize neural networks with configurations 
    
    # Initialize neural nework manager (NNM)
    
    # Initialize abstract state manager (ASM) using representation network 
    
    # Intialize u-MCTS module 
    
    # Initialize episode buffer 
    
    # Initalize reinforcement learning manager (RLM) 
  

  def train(self):
    # Initialize episode history 
    
    # For episode in range of num_episodes:
      # Reset the video game to an initial state (s0)
      
      # Initialize episode data 
      
      # For k in range num_simulations:
        # Gather q+1 real game states. Fill blank states when k < q. 
        
        # Createa a new abstract state (s) using the NNr. 
      
        # Perform u-MCTS search starting from the current abstract state (s)
        
        # Select an action using the policy returned by u-MCTS search
        
        # Take the selected action in the video game and observe the next state (s')
        
        # Store the transition (s, a, r, s') in the episode data
        
        # If the game has ended, break
      
      # Add the episode data to the episode history
      
      # If episode modulo I_t == 0:
        # Back propagation through time (BPTT) using trident, EH, and MBS 
      
      # Return network
      return 0 

def main():
  
  # Load configurations 
  config = Config()
  
  # Initialize MuZero
  muzero = MuZero(config)
 
  # Run training loop 
  muzero.train()
  
if __name__ == "__main__":
  main()