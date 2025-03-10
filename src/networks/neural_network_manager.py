import torch.nn as nn
import torch

class NeuralNetManager():
    def __init__(self, representation: nn.Module, dynamics: nn.Module, prediction: nn.Module):
        self.representation = representation
        self.dynamics = dynamics
        self.prediction = prediction
    
    def NNr(self):
        # TODO implement method with in/out
        pass
    
    def NNd(self):
        # TODO implement method with in/out
        pass
    
    def NNp(self, state):
        return self.prediction(state)
    
    def translate_and_evaluate(self, game_state):
        # Translate game state to hidden state
        hidden_state = self.representation(game_state)
        
        # Evaluate hidden state
        policy, value = self.prediction(hidden_state)
        
        # Only translate, so reward is 0 (tensor)
        reward = torch.tensor(0, dtype=torch.float32)
        
        policy_p = policy[0]
        
        return [hidden_state, value, reward, [p for p in policy_p], policy]
    
    def transition_and_evaluate(self, hidden_state, action):
        # Transition hidden state with action
        next_hidden_state, reward = self.dynamics(hidden_state, action)
        
        # Evaluate hidden state
        policy, value = self.prediction(next_hidden_state)
        
        policy_p = policy[0]
        
        return [next_hidden_state, value, reward, [p for p in policy_p], policy]
         
    def get_weights(self):
        networks = [self.representation, self.dynamics, self.prediction]
        return [net.get_parameters() for net in networks]
    
    def bptt(self):
        # TODO implement method
        pass
        
        
        
    