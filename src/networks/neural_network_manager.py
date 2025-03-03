import torch.nn as nn

class NeuralNetManager():
    def __init__(self, representation: nn.Module, dynamics: nn.Module, prediction: nn.Module):
        self.representation = representation
        self.dynamics = dynamics
        self.prediction = prediction
    
    def NNr(self):
        pass
    
    def NNd(self):
        pass
    
    def NNp(self, state):
        return self.prediction(state)
    