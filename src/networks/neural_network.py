import torch
from src.networks.network_builder import NetworkBuilder
import os

def build_head(head_config):
    layers = NetworkBuilder([head_config]).build_layer(head_config)
    return torch.nn.Sequential(*layers) if len(layers) > 1 else layers[0]

class NeuralNetwork:
    def __init__(self, layer_configs, device, build=True, iteration=None):
        """
        Build a network from layer configurations.
        Each configuration is a dict that includes:
          - "type": type of the layer ("linear", "conv2d", etc.)
          - Required parameters (e.g., in_features, out_features).
          - Optional "activation": Activation function name (e.g., "relu").
        """
        self.layer_configs = layer_configs
        self.network = NetworkBuilder(layer_configs).build_network()
        self.device = device

        if self.network is not None:
            self.network.to(device)


    def __call__(self, x, **kwargs):
        raise NotImplementedError("Subclasses must implement this method.")
        
        
    def save_model(
        self, subdir: int, model_name="representation_model.pth", dir="models"
    ):
        """
        Save the model to a directory.

        Args:
            subdir (str): Subdirectory to save the model - > UNIX timestamp as int.
            model_name (str): Name of the model file, with type.
            dir (str): Directory to save the model.
        """
        if self.network is None:
            raise ValueError("Model has not been initialized. Cannot save.")

        subdir = str(subdir)
        dir_path = f"{dir}/{subdir}"
        os.makedirs(dir_path, exist_ok=True)

        model_path = f"{dir_path}/{model_name}"
        torch.save(self.network.state_dict(), model_path)

    def load_model(
        self, iteration=None, model_name="representation_model.pth", dir="models"
    ):
        """
        Load the model from a directory.
        """
        try:
            subdirs = os.listdir(dir)
            if not subdirs:
                raise FileNotFoundError("No models found in directory.")

            if iteration is None:
                subdir = str(max(int(s) for s in subdirs))
            else:
                subdir = str(iteration)

            model_path = os.path.join(dir, subdir, model_name)

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file {model_path} not found.")

            if self.network is None:
                print("Rebuilding the network before loading weights...")
                self.network = NetworkBuilder(self.layer_configs).build_network()
                self.network.to(self.device)

            self.network.load_state_dict(torch.load(model_path), strict=False)
            self.network.to(self.device)
            print(f"Model loaded from {model_path}")

        except Exception as e:
            print(f"Error loading model: {e}")

    def preprocess(self, x):
        return x

    def forward(self, x):
        return self.network(x)

    def postprocess(self, x):
        return x
    
    def get_parameters(self):
        return list(self.network.parameters())


class RepresentationNetwork(NeuralNetwork):
    def __init__(self, layer_configs, device, build=True):
        super().__init__(layer_configs, device, build)
    
    def __call__(self, x, **kwargs):
        x = self.preprocess(x)
        hidden_activation = self.forward(x)
        return self.postprocess(hidden_activation)


class DynamicsNetwork(NeuralNetwork):
    def __init__(self, layer_configs, device, build=True):
        head_layers = layer_configs[-2:]
        build_layers = layer_configs[:-2]
        super().__init__(build_layers, device, build)
        self.state_head = build_head(head_layers[0])
        self.reward_head = build_head(head_layers[1])
    
    def __call__(self, x, **kwargs):
        input = self.preprocess(x)
        hidden_activation = self.forward(input)
        return self.postprocess(hidden_activation)
    
    def preprocess(self, x, **kwargs):
        return x
    
    def postprocess(self, hidden_activation):
        # Return new hidden state and reward
        return self.state_head(hidden_activation), self.reward_head(hidden_activation)
    
    def get_parameters(self):
        return super().get_parameters() + list(self.state_head.parameters()) + list(self.reward_head.parameters())
    

class PredictionNetwork(NeuralNetwork):
    def __init__(self, layer_configs, device, build=True):
        head_layers = layer_configs[-2:]
        build_layers = layer_configs[:-2]
        super().__init__(build_layers, device, build)
        self.policy_head = build_head(head_layers[0])
        self.value_head = build_head(head_layers[1])
        
    def __call__(self, x):
        input = self.preprocess(x)
        hidden_activation = self.forward(input)
        return self.postprocess(hidden_activation)
        
    def postprocess(self, hidden_activation):
        return self.policy_head(hidden_activation), self.value_head(hidden_activation)
    
    def get_parameters(self):
        return super().get_parameters() + list(self.value_head.parameters()) + list(self.policy_head.parameters())
    
    def forward(self, x):
        return self.network(x)