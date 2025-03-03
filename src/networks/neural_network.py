import torch
import yaml
from src.networks.network_builder import NetworkBuilder
import os
import time


class NeuralNetwork:
    def __init__(self, layer_configs, device, build=True):
        """
        Build a network from layer configurations.
        Each configuration is a dict that includes:
          - "type": type of the layer ("linear", "conv2d", etc.)
          - Required parameters (e.g., in_features, out_features).
          - Optional "activation": Activation function name (e.g., "relu").
        """
        self.layer_configs = layer_configs
        self.network = NetworkBuilder(layer_configs).build_network() if build else None
        self.device = device

        if self.network is not None:
            self.network.to(device)

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
        representation_net = NeuralNetwork(
            loaded_config["network"]["representation"], device=device
        )
        dynamics_net = NeuralNetwork(
            loaded_config["network"]["dynamics"], device=device
        )
        prediction_net = NeuralNetwork(
            loaded_config["network"]["prediction"], device=device
        )

    else:
        representation_net = NeuralNetwork(
            loaded_config["network"]["representation"], device=device, build=False
        )
        dynamics_net = NeuralNetwork(
            loaded_config["network"]["dynamics"], device=device, build=False
        )
        prediction_net = NeuralNetwork(
            loaded_config["network"]["prediction"], device=device, build=False
        )

        representation_net.load_model(
            loaded_config["network"]["iteration"], "representation_model.pth"
        )
        dynamics_net.load_model(
            loaded_config["network"]["iteration"], "dynamics_model.pth"
        )
        prediction_net.load_model(
            loaded_config["network"]["iteration"], "prediction_model.pth"
        )
