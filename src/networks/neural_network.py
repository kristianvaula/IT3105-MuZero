import torch
import torch.nn as nn
import yaml
from src.networks.network_builder import NetworkBuilder
import os
import time

class NeuralNetwork():
    def __init__(self, layer_configs, device, build=True):
        """
        Build a network from layer configurations.
        Each configuration is a dict that includes:
          - "type": type of the layer ("linear", "conv2d", etc.)
          - Required parameters (e.g., in_features, out_features).
          - Optional "activation": Activation function name (e.g., "relu").
        """
        self.network = NetworkBuilder(layer_configs).build_network() if build else None
        self.device = device
        if self.network is not None:
            self.network.to(device)
        
    
    
    def save_model(self, subdir:int, model_name="representation_model.pth", dir="models"):
        """
        Save the model to a directory.
        
        Args:
            subdir (str): Subdirectory to save the model - > UNIX timestamp as int.
            model_name (str): Name of the model file, with type.
            dir (str): Directory to save the model.
        """
        subdir = str(subdir)
        dir_path = f"{dir}/{subdir}"
        os.makedirs(dir_path, exist_ok=True)
        
        model_path = f"{dir_path}/{model_name}"
        torch.save(self.network.state_dict(), model_path)
        
        
    def load_model(self, iteration=None, model_name="representation_model.pth", dir="models"):
        """
        Load the model from a directory.
        """
        subdirs = os.listdir(dir)
        if iteration is None:
            subdir = max([int(subdir) for subdir in subdirs])
            subdir = str(iteration)
        else:
            subdir = subdirs[iteration]

        model_path = f"{dir}/{subdir}/{model_name}"
        model = torch.load(model_path)
        
        self.network = model
            
    def preprocess(self, x):
        return x

                   
    def forward(self, x):
        return self.network(x)
    
    def postprocess(self, x):
        return x
    
    
def save_network():
    timestamp = int(time.time())
    representation_net.save_model(timestamp, "representation_model.pth")
    dynamics_net.save_model(timestamp, "dynamics_model.pth")
    prediction_net.save_model(timestamp, "prediction_model.pth")

def load_model(iteration=None):
    representation_net.load_model(iteration)
    dynamics_net.load_model(iteration)
    prediction_net.load_model(iteration)

if __name__ == "__main__":
    # Load the configuration from the YAML file.
    config_filename = "config.yaml"
    with open(config_filename, "r") as file:
        loaded_config = yaml.safe_load(file)

    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if loaded_config["network"]["iteration"] is None:
        # Initialize networks
        representation_net = NeuralNetwork(loaded_config["network"]["representation"], device=device)
        dynamics_net = NeuralNetwork(loaded_config["network"]["dynamics"], device=device)
        prediction_net = NeuralNetwork(loaded_config["network"]["prediction"], device=device)

    else: 
        representation_net = NeuralNetwork(loaded_config["network"]["representation"], device=device, build=False)
        dynamics_net = NeuralNetwork(loaded_config["network"]["dynamics"], device=device, build=False)
        prediction_net = NeuralNetwork(loaded_config["network"]["prediction"], device=device, build=False)
        
        representation_net.load_model(loaded_config["network"]["iteration"], "representation_model.pth")
        dynamics_net.load_model(loaded_config["network"]["iteration"], "dynamics_model.pth")
        prediction_net.load_model(loaded_config["network"]["iteration"], "prediction_model.pth")        
        
    
    
